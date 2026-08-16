import sys

import numpy as np
import pytest

import ppg
from ppg.scripts import cmr_opt

from .conftest import replicate_with_noise, save_csv, save_nifti

# plasma=False (i.e. the -wb flag) so the aif isn't run through the
# whole-blood-to-plasma conversion, keeping the forward simulation simple
TRUE_THREE = np.array([0.0017, 0.79, 0.001, 0.04])  # K1, vd, k3, vb
TRUE_FOUR = np.array([0.0017, 0.79, 0.001, 0.00011, 0.04])  # K1, vd, k3, k4, vb


def _build_dataset(tmp_path, spatial_shape, true_params, model_cls, seed):
    t = np.arange(0, 200, 4.0)
    aif_cnt = 100.0 * t * np.exp(-t / 40.0) + 5.0

    aif = ppg.Tac(t, aif_cnt, dc=True, h_life=6582.0)
    dummy_pet = ppg.Tac(t, np.zeros_like(t), dc=True, h_life=6582.0)
    model = model_cls(aif, dummy_pet, plasma=False)
    pet_cnt = model.pred(true_params)

    aif_path = save_csv(tmp_path / "aif.csv", t, aif_cnt)
    time_path = tmp_path / "time.csv"
    np.savetxt(time_path, t, delimiter=",", fmt="%.6f")

    pet_data = replicate_with_noise(pet_cnt, spatial_shape, seed=seed, rel_scale=1e-3)
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    return aif_path, pet_path, str(time_path)


def test_cmr_opt_three_compartment_whole_brain(tmp_path, monkeypatch):
    aif_path, pet_path, time_path = _build_dataset(
        tmp_path, (1, 1, 1), TRUE_THREE, ppg.pet_model.FdgThree, seed=2
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["cmr-opt", aif_path, pet_path, time_path, out_prefix, "-avg", "-wb"],
    )
    with pytest.raises(SystemExit):
        cmr_opt.main()

    vals_path = tmp_path / "out_wb_vals.csv"
    assert vals_path.exists()
    assert (tmp_path / "out_wb_fit.tiff").exists()
    assert (tmp_path / "out_args.txt").exists()

    lines = vals_path.read_text().strip().split("\n")
    values = {row.split(",")[0]: float(row.split(",")[1]) for row in lines}

    K1, vd, k3, vb = TRUE_THREE
    k2 = K1 / vd - k3
    expected_K1 = K1 * 60.0 / 1.05 * 100.0
    assert values["K1"] == pytest.approx(expected_K1, rel=0.1)
    assert values["nrmse"] < 0.05
    assert "bic" in values


def test_cmr_opt_four_compartment_with_ca_and_voxels(tmp_path, monkeypatch):
    aif_path, pet_path, time_path = _build_dataset(
        tmp_path, (2, 2, 1), TRUE_FOUR, ppg.pet_model.FdgFour, seed=3
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cmr-opt",
            aif_path,
            pet_path,
            time_path,
            out_prefix,
            "-wb",
            "-k4",
            "-ca",
            "90",
            "-basin",
        ],
    )
    cmr_opt.main()

    vals_path = tmp_path / "out_wb_vals.csv"
    lines = vals_path.read_text().strip().split("\n")
    par_names = [row.split(",")[0] for row in lines]

    # Regression: par_names[-2:1] insertion trick must still land cmrglc/
    # influx/conc right before nrmse/bic, not silently drop them
    assert par_names == [
        "K1",
        "k2",
        "k3",
        "k4",
        "ki",
        "vt",
        "vb",
        "cmrglc",
        "influx",
        "conc",
        "nrmse",
        "bic",
    ]

    no_converge = int((tmp_path / "out_no_converge.txt").read_text())
    assert no_converge == 0

    for name in par_names[:-1]:  # bic isn't a voxelwise output
        assert (tmp_path / f"out_{name}.nii.gz").exists()
