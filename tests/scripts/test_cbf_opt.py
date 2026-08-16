import sys

import numpy as np
import pytest

from ppg import util
from ppg.scripts import cbf_opt

from .conftest import replicate_with_noise, save_csv, save_nifti, save_pet_json

TRUE_K1 = 0.0077
TRUE_K2 = 0.0108


def _build_dataset(tmp_path, spatial_shape):
    t = np.arange(0, 100, 2.0)
    aif_cnt = 100.0 * t * np.exp(-t / 20.0) + 5.0

    pet_cnt = util.exp_conv(t, aif_cnt, coef=[TRUE_K1], rate=[TRUE_K2])

    aif_path = save_csv(tmp_path / "aif.csv", t, aif_cnt)
    json_path = save_pet_json(
        tmp_path / "pet.json", t, duration=2.0, radionuclide="15O"
    )

    # Independent per-voxel noise; see replicate_with_noise's docstring for
    # why identical voxel copies make L-BFGS-B's fit falsely "fail".
    pet_data = replicate_with_noise(pet_cnt, spatial_shape)
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    return aif_path, pet_path, json_path


def test_cbf_opt_whole_brain_average(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (1, 1, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys, "argv", ["cbf-opt", aif_path, pet_path, json_path, out_prefix, "-avg"]
    )
    with pytest.raises(SystemExit):
        cbf_opt.main()

    vals_path = tmp_path / "out_wb_vals.csv"
    assert vals_path.exists()
    assert (tmp_path / "out_wb_fit.tiff").exists()
    assert (tmp_path / "out_wb_contour.tiff").exists()
    assert (tmp_path / "out_args.txt").exists()

    lines = vals_path.read_text().strip().split("\n")
    values = {row.split(",")[0]: float(row.split(",")[1]) for row in lines}

    expected_cbf = TRUE_K1 * 6000.0 / 1.05
    expected_k2 = TRUE_K2 * 60.0
    assert values["cbf"] == pytest.approx(expected_cbf, rel=0.05)
    assert values["k2"] == pytest.approx(expected_k2, rel=0.05)
    assert values["nrmse"] < 0.05


def test_cbf_opt_voxelwise(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (2, 2, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys, "argv", ["cbf-opt", aif_path, pet_path, json_path, out_prefix]
    )
    cbf_opt.main()

    assert (tmp_path / "out_no_converge.txt").exists()
    no_converge = int((tmp_path / "out_no_converge.txt").read_text())
    assert no_converge == 0

    for name in ["cbf", "k2", "lambda", "nrmse"]:
        img_path = tmp_path / f"out_{name}.nii.gz"
        hist_path = tmp_path / f"out_{name}_hist.tiff"
        assert img_path.exists()
        assert hist_path.exists()

    assert (tmp_path / "out_args.txt").exists()


def test_cbf_opt_whole_brain_always_uses_simpson_regardless_of_algo_flag(
    tmp_path, monkeypatch
):
    # The whole-brain fit is supposed to always use simpson, even if the
    # user asks for -algo trapz (that flag only governs the voxel loop).
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (1, 1, 1))

    results = {}
    for algo in ["trapz", "simpson"]:
        out_prefix = str(tmp_path / f"out_{algo}")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cbf-opt",
                aif_path,
                pet_path,
                json_path,
                out_prefix,
                "-avg",
                "-algo",
                algo,
            ],
        )
        with pytest.raises(SystemExit):
            cbf_opt.main()
        results[algo] = (tmp_path / f"out_{algo}_wb_vals.csv").read_text()

    assert results["trapz"] == results["simpson"]
