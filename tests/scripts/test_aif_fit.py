import sys

import numpy as np
import pytest

from ppg.scripts import aif_fit

from .conftest import save_csv


def _feng_curve(t, t0=8.0):
    alpha1, alpha2, alpha3 = 5000.0, 2000.0, 800.0
    eig1, eig2, eig3 = 0.3, 0.03, 0.003
    tm = np.maximum(t - t0, 0)
    mask = t > t0
    cnt = np.zeros_like(t)
    exp1 = np.exp(-eig1 * tm[mask])
    cnt[mask] = (
        alpha1 * tm[mask] * exp1
        + alpha2 * (np.exp(-eig2 * tm[mask]) - exp1)
        + alpha3 * (np.exp(-eig3 * tm[mask]) - exp1)
    )
    return cnt


def _r_squared(orig, fit):
    ss_res = np.sum((orig - fit) ** 2)
    ss_tot = np.sum((orig - orig.mean()) ** 2)
    return 1 - ss_res / ss_tot


def test_aif_fit_feng_default(tmp_path, monkeypatch):
    t = np.arange(0, 300, 1.0)
    cnt = _feng_curve(t)
    aif_path = save_csv(tmp_path / "aif.csv", t, cnt)
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(sys, "argv", ["aif-fit", aif_path, out_prefix])
    aif_fit.main()

    assert (tmp_path / "out.tiff").exists()
    fitted = np.loadtxt(tmp_path / "out_fitted.csv", delimiter=",")
    assert _r_squared(cnt, fitted[:, 1]) > 0.85

    params_path = tmp_path / "out_params.csv"
    assert params_path.exists()
    par_names = [
        row.split(",")[0] for row in params_path.read_text().strip().split("\n")
    ]
    assert par_names == [
        "alpha_1",
        "alpha_2",
        "alpha_3",
        "eig_1",
        "eig_2",
        "eig_3",
        "t_zero",
        "scale",
    ]


def test_aif_fit_golish_model(tmp_path, monkeypatch):
    t = np.arange(0, 300, 1.0)
    cnt = _feng_curve(t)  # close enough in shape to fit reasonably
    aif_path = save_csv(tmp_path / "aif.csv", t, cnt)
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys, "argv", ["aif-fit", aif_path, out_prefix, "-model", "Golish"]
    )
    aif_fit.main()

    fitted = np.loadtxt(tmp_path / "out_fitted.csv", delimiter=",")
    assert np.all(np.isfinite(fitted))

    par_names = [
        row.split(",")[0]
        for row in (tmp_path / "out_params.csv").read_text().strip().split("\n")
    ]
    assert par_names == ["alpha", "beta", "c_max", "c_zero", "tau", "t_zero", "scale"]


def test_aif_fit_with_tau_kernel_deconvolution(tmp_path, monkeypatch):
    # Regression coverage for the aif_model kernel-resampling bugs: -tau
    # builds a kernel on the aif's own sampling grid and drives the
    # deconvolution branch in FengModel.pred.
    t = np.arange(0, 300, 1.0)
    cnt = _feng_curve(t)
    aif_path = save_csv(tmp_path / "aif.csv", t, cnt)
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(sys, "argv", ["aif-fit", aif_path, out_prefix, "-tau", "5.0"])
    aif_fit.main()

    fitted = np.loadtxt(tmp_path / "out_fitted.csv", delimiter=",")
    dcv = np.loadtxt(tmp_path / "out_dcv.csv", delimiter=",")
    assert np.all(np.isfinite(fitted))
    assert np.all(np.isfinite(dcv))
    assert _r_squared(cnt, fitted[:, 1]) > 0.8


def test_aif_fit_with_resample(tmp_path, monkeypatch):
    t = np.arange(0, 300, 1.0)
    cnt = _feng_curve(t)
    aif_path = save_csv(tmp_path / "aif.csv", t, cnt)
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys, "argv", ["aif-fit", aif_path, out_prefix, "-resample", "2.0"]
    )
    aif_fit.main()

    fitted = np.loadtxt(tmp_path / "out_fitted.csv", delimiter=",")
    diffs = np.diff(fitted[:, 0])
    assert np.allclose(diffs, 2.0)
