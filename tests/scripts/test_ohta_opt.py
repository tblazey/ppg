import sys

import numpy as np
import pytest

from ppg import util
from ppg.scripts import ohta_opt

from .conftest import replicate_with_noise, save_csv, save_nifti, save_pet_json

TRUE_K1 = 0.0045
TRUE_K2 = 0.02
TRUE_V0 = 0.02


def _build_dataset(tmp_path, spatial_shape):
    t = np.arange(0, 100, 2.0)
    aif_cnt = 100.0 * t * np.exp(-t / 20.0) + 5.0

    pet_cnt = (
        util.exp_conv(t, aif_cnt, coef=[TRUE_K1], rate=[TRUE_K2]) + aif_cnt * TRUE_V0
    )

    aif_path = save_csv(tmp_path / "aif.csv", t, aif_cnt)
    json_path = save_pet_json(
        tmp_path / "pet.json", t, duration=2.0, radionuclide="15O"
    )

    pet_data = replicate_with_noise(pet_cnt, spatial_shape, seed=1)
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    return aif_path, pet_path, json_path


def test_ohta_opt_whole_brain_average(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (1, 1, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys, "argv", ["ohta-opt", aif_path, pet_path, json_path, out_prefix, "-avg"]
    )
    with pytest.raises(SystemExit):
        ohta_opt.main()

    vals_path = tmp_path / "out_wb_vals.csv"
    assert vals_path.exists()
    assert (tmp_path / "out_wb_fit.tiff").exists()
    assert (tmp_path / "out_args.txt").exists()

    lines = vals_path.read_text().strip().split("\n")
    values = {row.split(",")[0]: float(row.split(",")[1]) for row in lines}

    expected_k1 = TRUE_K1 * 6000.0 / 1.05
    expected_v0 = TRUE_V0 * 105.0
    assert values["K1"] == pytest.approx(expected_k1, rel=0.1)
    assert values["v0"] == pytest.approx(expected_v0, rel=0.1)
    assert values["nrmse"] < 0.05


def test_ohta_opt_voxelwise(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, (2, 2, 1))
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys, "argv", ["ohta-opt", aif_path, pet_path, json_path, out_prefix]
    )
    ohta_opt.main()

    no_converge = int((tmp_path / "out_no_converge.txt").read_text())
    assert no_converge == 0

    for name in ["K1", "k2", "lambda", "v0", "nrmse"]:
        assert (tmp_path / f"out_{name}.nii.gz").exists()
        assert (tmp_path / f"out_{name}_hist.tiff").exists()
