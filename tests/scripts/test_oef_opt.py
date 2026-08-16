import sys

import numpy as np
import pytest

import ppg
from ppg.scripts import oef_opt

from .conftest import replicate_with_noise, save_csv, save_nifti

TRUE_FLOW = 0.0077  # mL/mL/sec
TRUE_K2 = 0.0108  # 1/sec
TRUE_VB = 0.04  # mL/mL
TRUE_OEF = 0.40


def _build_dataset(tmp_path, spatial_shape, seed):
    t = np.arange(0, 180, 2.0)
    aif_cnt = 100.0 * t * np.exp(-t / 20.0) + 5.0
    aif = ppg.Tac(t, aif_cnt, dc=True, h_life=122.24)

    aif_oxy, aif_water = ppg.util.iida_oxy_aif(aif, delta=20, prod=0.0012)
    dummy_pet = ppg.Tac(t, np.zeros_like(t), dc=True, h_life=122.24)
    model_gen = ppg.pet_model.OxyOne(
        aif_oxy, aif_water, dummy_pet, TRUE_FLOW, TRUE_K2, TRUE_VB
    )
    pet_cnt = model_gen.pred(TRUE_OEF)

    aif_path = save_csv(tmp_path / "aif.csv", t, aif_cnt)
    time_path = tmp_path / "time.csv"
    np.savetxt(time_path, t, delimiter=",", fmt="%.6f")

    pet_data = replicate_with_noise(pet_cnt, spatial_shape, seed=seed, rel_scale=1e-3)
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    # Physiological-unit images, inverse of oef_opt's own conversion back to
    # internal units (imgs[0]/6000*1.05, imgs[1]/60, imgs[2]/100*1.05)
    cbf_img = np.full(spatial_shape, TRUE_FLOW * 6000.0 / 1.05)
    k2_img = np.full(spatial_shape, TRUE_K2 * 60.0)
    cbv_img = np.full(spatial_shape, TRUE_VB * 100.0 / 1.05)
    cbf_path = save_nifti(tmp_path / "cbf.nii.gz", cbf_img)
    k2_path = save_nifti(tmp_path / "k2.nii.gz", k2_img)
    cbv_path = save_nifti(tmp_path / "cbv.nii.gz", cbv_img)

    return aif_path, pet_path, str(time_path), cbf_path, k2_path, cbv_path


def test_oef_opt_whole_brain_average(tmp_path, monkeypatch):
    aif_path, pet_path, time_path, cbf_path, k2_path, cbv_path = _build_dataset(
        tmp_path, (1, 1, 1), seed=4
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oef-opt",
            aif_path,
            pet_path,
            time_path,
            cbf_path,
            k2_path,
            cbv_path,
            out_prefix,
            "-avg",
        ],
    )
    with pytest.raises(SystemExit):
        oef_opt.main()

    vals_path = tmp_path / "out_wb_vals.csv"
    assert vals_path.exists()
    assert (tmp_path / "out_wb_fit.tiff").exists()
    assert (tmp_path / "out_wb_opt_error.tiff").exists()
    assert (tmp_path / "out_args.txt").exists()

    lines = vals_path.read_text().strip().split("\n")
    values = {row.split(",")[0]: float(row.split(",")[1]) for row in lines}
    assert values["oef"] == pytest.approx(TRUE_OEF, abs=0.02)
    assert values["nrmse"] < 0.05


def test_oef_opt_with_ca_and_voxels(tmp_path, monkeypatch):
    aif_path, pet_path, time_path, cbf_path, k2_path, cbv_path = _build_dataset(
        tmp_path, (2, 2, 1), seed=5
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oef-opt",
            aif_path,
            pet_path,
            time_path,
            cbf_path,
            k2_path,
            cbv_path,
            out_prefix,
            "-ca",
            "20",
        ],
    )
    oef_opt.main()

    vals_path = tmp_path / "out_wb_vals.csv"
    par_names = [row.split(",")[0] for row in vals_path.read_text().strip().split("\n")]
    assert par_names == ["oef", "cmro2", "nrmse"]

    no_converge = int((tmp_path / "out_no_converge.txt").read_text())
    assert no_converge == 0

    for name in par_names:
        assert (tmp_path / f"out_{name}.nii.gz").exists()
        assert (tmp_path / f"out_{name}_hist.tiff").exists()
