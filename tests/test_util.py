import nibabel as nib
import numpy as np
import pytest

from ppg import util
from ppg.tac import Tac


def _save_nifti(tmp_path, name, data, affine=None):
    if affine is None:
        affine = np.eye(4)
    path = tmp_path / name
    nib.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


# ---- count_int ----------------------------------------------------------


def test_count_int_matches_analytic_integral():
    time = np.linspace(0, 10, 200)
    counts = np.full(time.shape[0], 3.0)
    result = util.count_int(time, counts, [0, 10])
    assert result == pytest.approx(30.0, rel=1e-3)


def test_count_int_2d_input():
    time = np.linspace(0, 10, 200)
    counts = np.stack([np.full(time.shape[0], 2.0), np.full(time.shape[0], 4.0)])
    result = util.count_int(time, counts, [0, 10])
    assert result.shape == (2,)
    assert result[0] == pytest.approx(20.0, rel=1e-3)
    assert result[1] == pytest.approx(40.0, rel=1e-3)


def test_count_int_too_few_points_raises():
    time = np.array([0.0, 1.0])
    counts = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        util.count_int(time, counts, [0, 1])


def test_count_int_bad_ndim_raises():
    time = np.linspace(0, 10, 20)
    counts = np.zeros((2, 2, time.shape[0]))
    with pytest.raises(ValueError):
        util.count_int(time, counts, [0, 10])


# ---- image dimension checks ---------------------------------------------


def test_check_img_dim_pass_and_fail(tmp_path):
    path = _save_nifti(tmp_path, "img.nii.gz", np.zeros((2, 2, 2, 5)))
    hdr = nib.load(path)

    util.check_img_dim(hdr, 4)  # should not raise
    with pytest.raises(ValueError):
        util.check_img_dim(hdr, 3, allow_single=False)


def test_comp_img_dim_pass_and_fail(tmp_path):
    path_a = _save_nifti(tmp_path, "a.nii.gz", np.zeros((2, 2, 2)))
    path_b = _save_nifti(tmp_path, "b.nii.gz", np.zeros((2, 2, 2)))
    path_c = _save_nifti(tmp_path, "c.nii.gz", np.zeros((3, 3, 3)))

    util.comp_img_dim(nib.load(path_a), nib.load(path_b))  # no raise
    with pytest.raises(ValueError):
        util.comp_img_dim(nib.load(path_a), nib.load(path_c))


# ---- conv_matrix ----------------------------------------------------------


def test_conv_matrix_matches_manual_construction():
    kernel = np.array([1.0, 2.0, 3.0])
    c_mat = util.conv_matrix(kernel)

    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1):
            if i - j < kernel.shape[0]:
                expected[i, j] = kernel[i - j]

    assert np.allclose(c_mat, expected)


def test_conv_matrix_with_padding_is_larger():
    kernel = np.array([1.0, 2.0])
    c_mat = util.conv_matrix(kernel, pad=2)
    assert c_mat.shape == (4, 4)


# ---- time masking / peak finding -----------------------------------------


def test_gen_time_mask_no_limit_is_all_true():
    tac = Tac(np.arange(0, 20, 1.0), np.zeros(20))
    mask = util.gen_time_mask(tac)
    assert np.all(mask)


def test_gen_time_mask_with_limit_restricts_range():
    t = np.arange(0, 60, 1.0)
    cnt = 100.0 * t * np.exp(-t / 10.0)
    tac = Tac(t, cnt)

    mask = util.gen_time_mask(tac, limit=5.0)
    assert mask.sum() < tac.n
    assert mask[0]


def test_bolus_approx_uses_first_point_when_peak_is_early():
    t = np.arange(0, 20, 1.0)
    cnt = np.exp(-t)  # peak at t=0
    tac = Tac(t, cnt)
    assert util.bolus_approx(tac) == pytest.approx(tac.time[0])


def test_bolus_approx_uses_derivative_peak_when_peak_is_late():
    t = np.arange(0, 60, 1.0)
    cnt = 100.0 * t * np.exp(-t / 10.0)
    tac = Tac(t, cnt)
    bolus_time = util.bolus_approx(tac)
    peak_time, _ = util.loc_peak(tac)
    assert 0 < bolus_time < peak_time


def test_loc_peak_finds_gaussian_maximum():
    t = np.linspace(0, 20, 400)
    cnt = np.exp(-((t - 8.0) ** 2) / 2.0)
    tac = Tac(t, cnt)

    t_peak, c_peak = util.loc_peak(tac)
    assert t_peak == pytest.approx(8.0, abs=0.1)
    assert c_peak == pytest.approx(1.0, abs=0.05)

    t_peak_raw, _ = util.loc_peak(tac, tac_interp=False)
    assert t_peak_raw == pytest.approx(8.0, abs=0.1)


def test_loc_deriv_peak_precedes_curve_peak():
    t = np.linspace(0, 20, 400)
    cnt = 100.0 * t * np.exp(-t / 3.0)
    tac = Tac(t, cnt)

    t_peak, _ = util.loc_peak(tac)
    t_deriv_peak, _ = util.loc_deriv_peak(tac)
    assert t_deriv_peak < t_peak


# ---- spline helpers ---------------------------------------------------


def test_knot_loc_rejects_too_few_knots():
    with pytest.raises(ValueError):
        util.knot_loc(np.arange(100.0), 2)


def test_knot_loc_rejects_too_many_knots():
    with pytest.raises(ValueError):
        util.knot_loc(np.arange(5.0), 10)


def test_knot_loc_matches_percentiles():
    x = np.arange(1000.0)
    knots = util.knot_loc(x, 3)
    assert np.allclose(knots, np.percentile(x, np.linspace(10, 90, 3)))


def test_knot_loc_custom_bounds():
    x = np.arange(1000.0)
    knots = util.knot_loc(x, 3, bounds=[20, 80])
    assert np.allclose(knots, np.percentile(x, np.linspace(20, 80, 3)))


def test_natural_spline_basis_shape_and_linear_columns():
    x = np.linspace(0, 10, 50)
    knots = util.knot_loc(x, 4)
    basis = util.natural_spline_basis(x, knots)

    assert basis.shape == (50, 4)
    assert np.allclose(basis[:, 0], 1.0)
    assert np.allclose(basis[:, 1], x)


def test_natural_spline_basis_derivative_matches_numeric_gradient():
    x = np.linspace(0.1, 10, 500)
    knots = util.knot_loc(x, 4)
    basis, basis_d = util.natural_spline_basis(x, knots, dot=1)

    numeric_d = np.gradient(basis[:, 2], x)
    # compare away from the boundaries, where np.gradient is less accurate
    interior = slice(10, -10)
    assert np.allclose(basis_d[interior, 2], numeric_d[interior], atol=1e-2)


def test_natural_spline_basis_bad_dot_raises():
    x = np.linspace(0, 10, 20)
    knots = util.knot_loc(x, 3)
    with pytest.raises(ValueError):
        util.natural_spline_basis(x, knots, dot=3)


# ---- to_2d ----------------------------------------------------------------


def test_to_2d_reshape():
    arr = np.arange(24.0).reshape(2, 3, 4)
    flat = util.to_2d(arr)
    assert flat.shape == (6, 4)
    assert np.allclose(flat[0], arr[0, 0])


# ---- iida_oxy_aif -----------------------------------------------------


def test_iida_oxy_aif_splits_into_oxygen_and_water(synth_aif):
    aif_oxy, aif_water = util.iida_oxy_aif(synth_aif)
    assert aif_oxy.n == synth_aif.n
    assert aif_water.n == synth_aif.n
    assert aif_oxy.dc is True
    assert aif_water.dc is True
    assert np.allclose(aif_oxy.cnt + aif_water.cnt, synth_aif.cnt)


# ---- image loading pipeline --------------------------------------------


def test_load_mask_defaults_to_all_true(tmp_path):
    pet_path = _save_nifti(tmp_path, "pet.nii.gz", np.zeros((2, 2, 2, 5)))
    msk_data, msk_hdr = util.load_mask(None, nib.load(pet_path))
    assert msk_hdr is None
    assert msk_data.shape == (8,)
    assert np.all(msk_data)


def test_load_pet_and_prep_model_with_extra_images(tmp_path):
    # Regression test for prep_model's NameError on vol_mskt/vol_sum
    # in the img_paths branch (they were only ever computed inside
    # load_pet's local scope, never returned or recomputed).
    shape = (2, 2, 2)
    n_frames = 20

    pet_data = np.zeros(shape + (n_frames,))
    for i in range(n_frames):
        pet_data[..., i] = i + 1.0
    pet_path = _save_nifti(tmp_path, "pet.nii.gz", pet_data)

    extra_data = np.full(shape, 5.0)
    extra_path = _save_nifti(tmp_path, "extra.nii.gz", extra_data)

    aif_path = tmp_path / "aif.txt"
    aif_time = np.arange(0, 25, 1.0)
    aif_cnt = np.full(aif_time.shape[0], 10.0)
    np.savetxt(aif_path, np.stack((aif_time, aif_cnt), axis=1), delimiter=",")

    result = util.prep_model(
        aif_path=str(aif_path),
        pet_path=pet_path,
        time_path=None,
        msk_path=None,
        vol_path=None,
        scale=1.0,
        limit=None,
        censor_path=None,
        h_life=1220.0,
        img_paths=[extra_path],
    )

    aif, pet_hdr, pet_mskt, msk_data, msk_hdr, mean_tac_mskt, imgs, avgs = result

    assert pet_mskt.shape[0] == 8
    assert len(imgs) == 1
    assert len(avgs) == 1
    assert avgs[0] == pytest.approx(5.0)
