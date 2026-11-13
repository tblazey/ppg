import json
import sys

import nibabel as nib
import numpy as np
import pytest

from ppg import util
from ppg.scripts import spectral_opt

from .conftest import replicate_with_noise, save_csv, save_nifti, save_pet_json

# Irreversible two-tissue truth, in 1/sec
TRUE_K1 = 0.0017
TRUE_K2 = 0.0021
TRUE_K3 = 0.001
TRUE_KI = TRUE_K1 * TRUE_K3 / (TRUE_K2 + TRUE_K3)


def _build_dataset(tmp_path, spatial_shape):
    t = np.arange(30, 3600, 60.0)
    aif_cnt = 100.0 * t * np.exp(-t / 60.0) + 5.0

    pet_cnt = util.spectral_basis(t, aif_cnt, [0.0, TRUE_K2 + TRUE_K3]) @ [
        TRUE_KI,
        TRUE_K1 - TRUE_KI,
    ]

    aif_path = save_csv(tmp_path / "aif.csv", t, aif_cnt)
    json_path = save_pet_json(
        tmp_path / "pet.json", t, duration=60.0, radionuclide="18F"
    )
    pet_data = replicate_with_noise(pet_cnt, spatial_shape)
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    return aif_path, pet_path, json_path


def test_spectral_opt_whole_brain_average(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (1, 1, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["spectral-opt", aif_path, pet_path, json_path, out_prefix, "-avg", "-trap"],
    )
    with pytest.raises(SystemExit):
        spectral_opt.main()

    for suffix in ["wb_plot.jpeg", "wb_spectrum.csv", "wb_spectrum.jpeg", "args.json"]:
        assert (tmp_path / f"out_{suffix}").exists()

    values = json.loads((tmp_path / "out_wb_params.json").read_text())
    assert list(values) == ["K1", "Ki", "Vt", "nrmse"]
    assert values["K1"]["value"] == pytest.approx(TRUE_K1 * 6000.0 / 1.05, rel=0.05)
    assert values["Ki"]["value"] == pytest.approx(TRUE_KI * 6000.0 / 1.05, rel=0.05)
    assert values["nrmse"]["value"] < 0.05


def test_spectral_opt_default_rates_follow_veronese(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (1, 1, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["spectral-opt", aif_path, pet_path, json_path, out_prefix, "-avg"],
    )
    with pytest.raises(SystemExit):
        spectral_opt.main()

    # Last frame ends at 3600 sec and the first frame is 60 sec long
    args = json.loads((tmp_path / "out_args.json").read_text())
    assert args["rates"] == pytest.approx([1.0 / (3.0 * 3600.0), 3.0 / 60.0])

    spectrum = np.loadtxt(tmp_path / "out_wb_spectrum.csv", delimiter=",")
    assert spectrum.shape == (100, 2)


def test_spectral_opt_nan_flag_replaces_pet_nans(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (2, 2, 1))

    # Put NaNs in one voxel
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    pet_data[0, 0, 0, 5:10] = np.nan
    nib.Nifti1Image(pet_data, pet_img.affine).to_filename(pet_path)

    # Without the flag the NaNs break the fit
    out_prefix = str(tmp_path / "fail")
    monkeypatch.setattr(
        sys, "argv", ["spectral-opt", aif_path, pet_path, json_path, out_prefix]
    )
    with pytest.raises(ValueError):
        spectral_opt.main()

    # With it, every voxel gets a finite fit
    out_prefix = str(tmp_path / "out")
    monkeypatch.setattr(
        sys,
        "argv",
        ["spectral-opt", aif_path, pet_path, json_path, out_prefix, "-nan", "0"],
    )
    spectral_opt.main()

    for name in ["K1", "Vt", "nrmse"]:
        img = nib.load(tmp_path / f"out_{name}.nii.gz").get_fdata()
        assert np.all(np.isfinite(img))
    assert json.loads((tmp_path / "out_args.json").read_text())["nan"] == [0.0]


def test_spectral_opt_pred_saves_fitted_curves(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (2, 2, 1))
    out_prefix = str(tmp_path / "out")

    # Censor a frame so the prediction has fewer frames than the input
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectral-opt",
            aif_path,
            pet_path,
            json_path,
            out_prefix,
            "-trap",
            "-pred",
            "-censor",
            "3",
        ],
    )
    spectral_opt.main()

    # Whole-brain csv has time, measured, and predicted, minus the censored frame
    wb_pred = np.loadtxt(tmp_path / "out_wb_pred.csv", delimiter=",")
    pet_data = nib.load(pet_path).get_fdata()
    assert wb_pred.shape == (pet_data.shape[-1] - 1, 3)
    assert np.allclose(wb_pred[:, 2], wb_pred[:, 1], rtol=0.05, atol=1.0)

    # Voxel predictions are close to the data on the same frames
    pred = nib.load(tmp_path / "out_pred.nii.gz").get_fdata()
    assert pred.shape == (2, 2, 1, wb_pred.shape[0])
    fit_data = np.delete(pet_data, 3, axis=-1)
    assert np.allclose(pred, fit_data, rtol=0.05, atol=0.05 * fit_data.max())


def test_spectral_opt_voxelwise(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (2, 2, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spectral-opt",
            aif_path,
            pet_path,
            json_path,
            out_prefix,
            "-trap",
            "-vb",
            "-hist",
        ],
    )
    spectral_opt.main()

    for name in ["K1", "Ki", "Vt", "Vb", "nrmse"]:
        assert (tmp_path / f"out_{name}.nii.gz").exists()
        assert (tmp_path / f"out_{name}_hist.tiff").exists()

    assert (tmp_path / "out_args.json").exists()

    # Spectrum has one volume per rate (100 log-spaced plus the trap term),
    # in the same order as the whole-brain spectrum csv
    spec = nib.load(tmp_path / "out_spectrum.nii.gz").get_fdata()
    wb_spec = np.loadtxt(tmp_path / "out_wb_spectrum.csv", delimiter=",")
    assert spec.shape == (2, 2, 1, 101)
    assert np.all(spec >= 0.0)

    # Summing each voxel's spectrum should reproduce its K1 image
    k1 = nib.load(tmp_path / "out_K1.nii.gz").get_fdata()
    assert np.allclose(spec.sum(axis=-1) * 6000.0 / 1.05, k1)
    assert wb_spec.shape[0] == spec.shape[-1]
