import json
import sys

import nibabel as nib
import numpy as np
import pytest

import ppg
from ppg.scripts import cmr_opt

from .conftest import replicate_with_noise, save_csv, save_nifti, save_pet_json

# hct is left at its None default (no whole-blood-to-plasma conversion),
# keeping the forward simulation simple
TRUE_THREE = np.array([0.0017, 0.79, 0.001, 0.04])  # K1, vd, k3, vb
TRUE_FOUR = np.array([0.0017, 0.79, 0.001, 0.00011, 0.04])  # K1, vd, k3, k4, vb
_F18_HALF_LIFE = ppg.io.RADIONUCLIDE_HALF_LIFE["18F"]


def _fdg_rates_to_ab(K1, vd, k3, k4=None):
    """
    Test-only inverse of ppg.pet_model.fdg_ab_to_rates: converts physically
    meaningful ground-truth rate constants into the alpha/beta params that
    Fdg.pred now expects, so tests can still express "truth" in readable
    physical units.
    """

    k2 = K1 / vd - k3

    if k4 is None:
        beta1 = k2 + k3
        alpha1 = K1 * k2 / beta1
        alpha2 = K1 * k3 / beta1
        return np.array([alpha1, alpha2, beta1])

    k_sum = k2 + k3 + k4
    k_sqrt = np.sqrt(k_sum**2 - 4.0 * k2 * k4)
    beta1 = (k_sum - k_sqrt) / 2.0
    beta2 = (k_sum + k_sqrt) / 2.0
    d = K1 / (beta2 - beta1)
    alpha1 = d * (k3 + k4 - beta1)
    alpha2 = d * (beta2 - k3 - k4)
    return np.array([alpha1, alpha2, beta1, beta2])


def _build_dataset(tmp_path, spatial_shape, true_params, k4, seed, hct=None):
    t = np.arange(0, 200, 4.0)
    aif_cnt = 100.0 * t * np.exp(-t / 40.0) + 5.0

    aif = ppg.Tac(t, aif_cnt, dc=True, h_life=_F18_HALF_LIFE)
    dummy_pet = ppg.Tac(t, np.zeros_like(t), dc=True, h_life=_F18_HALF_LIFE)
    model = ppg.pet_model.Fdg(aif, dummy_pet, k4=k4, hct=hct)

    if k4 is False:
        K1, vd, k3, vb = true_params
        ab_params = np.append(_fdg_rates_to_ab(K1, vd, k3), vb)
    else:
        K1, vd, k3, k4_rate, vb = true_params
        ab_params = np.append(_fdg_rates_to_ab(K1, vd, k3, k4_rate), vb)
    pet_cnt = model.pred(ab_params)

    aif_path = save_csv(tmp_path / "aif.csv", t, aif_cnt)
    json_path = save_pet_json(
        tmp_path / "pet.json", t, duration=4.0, radionuclide="18F"
    )

    pet_data = replicate_with_noise(pet_cnt, spatial_shape, seed=seed, rel_scale=1e-3)
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    return aif_path, pet_path, json_path


def test_cmr_opt_three_compartment_whole_brain(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (1, 1, 1), TRUE_THREE, k4=False, seed=2
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["cmr-opt", aif_path, pet_path, json_path, out_prefix, "-avg"],
    )
    with pytest.raises(SystemExit):
        cmr_opt.main()

    params_path = tmp_path / "out_wb_params.json"
    assert params_path.exists()
    assert (tmp_path / "out_wb_plot.jpeg").exists()
    assert (tmp_path / "out_args.json").exists()

    with open(params_path, encoding="utf-8") as f:
        wb_params = json.load(f)
    values = {name: entry["value"] for name, entry in wb_params.items()}

    K1, vd, k3, vb = TRUE_THREE
    k2 = K1 / vd - k3
    expected_K1 = K1 * 60.0 / 1.05 * 100.0
    assert values["K1"] == pytest.approx(expected_K1, rel=0.1)
    assert values["nrmse"] < 0.05
    assert "bic" in values


def test_cmr_opt_four_compartment_with_ca_and_voxels(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (2, 2, 1), TRUE_FOUR, k4=True, seed=3
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cmr-opt",
            aif_path,
            pet_path,
            json_path,
            out_prefix,
            "-k4",
            "-ca",
            "90",
            "-basin",
        ],
    )
    cmr_opt.main()

    params_path = tmp_path / "out_wb_params.json"
    with open(params_path, encoding="utf-8") as f:
        wb_params = json.load(f)
    par_names = list(wb_params.keys())

    # Regression: par_names[-2:1] insertion trick must still land CMRglc/
    # influx/conc right before nrmse/bic, not silently drop them
    assert par_names == [
        "K1",
        "k2",
        "k3",
        "k4",
        "Ki",
        "Vt",
        "Vb",
        "CMRglc",
        "influx",
        "conc",
        "nrmse",
        "bic",
    ]

    no_converge = int((tmp_path / "out_no_converge.txt").read_text())
    assert no_converge == 0

    for name in par_names[:-1]:  # bic isn't a voxelwise output
        assert (tmp_path / f"out_{name}.nii.gz").exists()


def test_cmr_opt_save_se(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (2, 2, 1), TRUE_THREE, k4=False, seed=7
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["cmr-opt", aif_path, pet_path, json_path, out_prefix, "-save_se"],
    )
    cmr_opt.main()

    se_names = ["K1", "k2", "k3", "Ki", "Vt", "Vb"]

    se_path = tmp_path / "out_wb_se.csv"
    assert se_path.exists()
    se_lines = se_path.read_text().strip().split("\n")
    wb_se = {row.split(",")[0]: float(row.split(",")[1]) for row in se_lines}
    assert set(wb_se) == set(se_names)
    assert all(np.isfinite(v) and v >= 0 for v in wb_se.values())

    for name in se_names:
        assert (tmp_path / f"out_{name}_se.nii.gz").exists()


def test_cmr_opt_four_compartment_save_se(tmp_path, monkeypatch):
    # k4=True + -save_se together exercises the "both terms are real
    # exponentials" branch of exp_conv's hess=True kernels (k4=False's
    # -save_se test only ever has one real term + one constant term) --
    # this combination wasn't covered by any other test.
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (2, 2, 1), TRUE_FOUR, k4=True, seed=8
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["cmr-opt", aif_path, pet_path, json_path, out_prefix, "-k4", "-save_se"],
    )
    cmr_opt.main()

    se_names = ["K1", "k2", "k3", "k4", "Ki", "Vt", "Vb"]

    se_path = tmp_path / "out_wb_se.csv"
    assert se_path.exists()
    se_lines = se_path.read_text().strip().split("\n")
    wb_se = {row.split(",")[0]: float(row.split(",")[1]) for row in se_lines}
    assert set(wb_se) == set(se_names)
    # k4 (reversible) fits can be genuinely weakly identified -- a NaN SE
    # for a specific output is an expected outcome, not a bug -- but every
    # value should be either finite-nonnegative or NaN, never negative
    assert all(np.isnan(v) or v >= 0 for v in wb_se.values())

    for name in se_names:
        assert (tmp_path / f"out_{name}_se.nii.gz").exists()


def test_cmr_opt_voxelwise_algo_simpson(tmp_path, monkeypatch):
    # Both the whole-brain and per-voxel fits default to algo="trapz" --
    # explicitly requesting "simpson" exercises exp_conv's simpson kernels
    # with deriv=True through the real per-voxel L-BFGS-B loop.
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (2, 2, 1), TRUE_THREE, k4=False, seed=9
    )
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["cmr-opt", aif_path, pet_path, json_path, out_prefix, "-algo", "simpson"],
    )
    cmr_opt.main()

    no_converge = int((tmp_path / "out_no_converge.txt").read_text())
    assert no_converge == 0

    K1, vd, k3, vb = TRUE_THREE
    expected_K1 = K1 * 60.0 / 1.05 * 100.0

    k1_img = nib.load(str(tmp_path / "out_K1.nii.gz")).get_fdata()
    assert np.all(np.isfinite(k1_img[k1_img != 0]))
    assert np.mean(k1_img[k1_img != 0]) == pytest.approx(expected_K1, rel=0.1)


def test_cmr_opt_censor_excludes_corrupted_frames(tmp_path, monkeypatch):
    # Corrupt two frames with a large outlier, then confirm the fit only
    # recovers K1 well when those exact frame indices are censored --
    # a real check that -censor removes the intended frames, not just
    # that the flag parses without crashing.
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (1, 1, 1), TRUE_THREE, k4=False, seed=4
    )
    K1, vd, k3, vb = TRUE_THREE
    expected_K1 = K1 * 60.0 / 1.05 * 100.0

    corrupted_frames = [5, 10]
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    pet_data[..., corrupted_frames] *= 20.0
    corrupted_path = str(tmp_path / "pet_corrupted.nii.gz")
    nib.Nifti1Image(pet_data, pet_img.affine).to_filename(corrupted_path)

    def fit_k1(out_name, extra_args):
        out_prefix = tmp_path / out_name
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cmr-opt",
                aif_path,
                corrupted_path,
                json_path,
                str(out_prefix),
                "-avg",
                *extra_args,
            ],
        )
        with pytest.raises(SystemExit):
            cmr_opt.main()
        with open(tmp_path / f"{out_name}_wb_params.json", encoding="utf-8") as f:
            wb_params = json.load(f)
        return wb_params["K1"]["value"]

    k1_censored = fit_k1(
        "censored", ["-censor", str(corrupted_frames[0]), str(corrupted_frames[1])]
    )
    k1_uncensored = fit_k1("uncensored", [])

    assert k1_censored == pytest.approx(expected_K1, rel=0.1)
    assert abs(k1_censored - expected_K1) < abs(k1_uncensored - expected_K1)


def test_cmr_opt_censor_aif_removes_corrupted_aif_frames(tmp_path, monkeypatch):
    # _build_dataset's aif already shares the pet's exact frame grid (an
    # image-derived input function). Corrupt the aif itself (not the pet)
    # and confirm: censoring the pet frames alone does nothing useful
    # (the corrupted aif samples still poison the convolution integral at
    # every later output point, not just the two matching frames) -- only
    # -censor_aif, which also drops those frames from the aif, fixes it.
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (1, 1, 1), TRUE_THREE, k4=False, seed=12
    )
    K1, vd, k3, vb = TRUE_THREE
    expected_K1 = K1 * 60.0 / 1.05 * 100.0

    corrupted_frames = [5, 10]
    aif_time, aif_cnt = np.loadtxt(aif_path, delimiter=",", unpack=True)
    aif_cnt[corrupted_frames] *= 20.0
    corrupted_aif_path = save_csv(tmp_path / "aif_corrupted.csv", aif_time, aif_cnt)

    def fit_k1(out_name, extra_args):
        out_prefix = tmp_path / out_name
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cmr-opt",
                corrupted_aif_path,
                pet_path,
                json_path,
                str(out_prefix),
                "-avg",
                *extra_args,
            ],
        )
        with pytest.raises(SystemExit):
            cmr_opt.main()
        with open(tmp_path / f"{out_name}_wb_params.json", encoding="utf-8") as f:
            wb_params = json.load(f)
        return wb_params["K1"]["value"]

    censor_args = ["-censor", str(corrupted_frames[0]), str(corrupted_frames[1])]
    k1_pet_only = fit_k1("pet_only", censor_args)
    k1_both = fit_k1("both", censor_args + ["-censor_aif"])

    assert k1_both == pytest.approx(expected_K1, rel=0.1)
    assert abs(k1_both - expected_K1) < abs(k1_pet_only - expected_K1)


def test_cmr_opt_hct_correction_recovers_k1(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(
        tmp_path, (1, 1, 1), TRUE_THREE, k4=False, seed=6, hct=0.45
    )
    K1, vd, k3, vb = TRUE_THREE
    expected_K1 = K1 * 60.0 / 1.05 * 100.0

    def fit_k1(out_name, extra_args):
        out_prefix = tmp_path / out_name
        monkeypatch.setattr(
            sys,
            "argv",
            ["cmr-opt", aif_path, pet_path, json_path, str(out_prefix), "-avg", *extra_args],
        )
        with pytest.raises(SystemExit):
            cmr_opt.main()
        with open(tmp_path / f"{out_name}_wb_params.json", encoding="utf-8") as f:
            wb_params = json.load(f)
        return wb_params["K1"]["value"]

    k1_with_hct = fit_k1("with_hct", ["-hct", "0.45"])
    k1_without_hct = fit_k1("without_hct", [])

    # Fitting with the correct hct should recover K1 closely, and be
    # noticeably closer to truth than ignoring the whole-blood-to-plasma
    # correction the data was actually generated with
    assert k1_with_hct == pytest.approx(expected_K1, rel=0.05)
    assert abs(k1_with_hct - expected_K1) < abs(k1_without_hct - expected_K1)
