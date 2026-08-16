import numpy as np
import pytest

from ppg.aif_model import AifModel, FengModel, GolishModel
from ppg.tac import Tac


def _feng_params(t_zero=5.0):
    linear = np.array([50.0, 20.0, 10.0, 0.5, 0.05, 0.005])
    return np.concatenate((np.log(linear), [t_zero]))


def _golish_params(t_zero=5.0):
    linear = np.array([2.0, 20.0, 80.0, 5.0, 30.0])
    return np.concatenate((np.log(linear), [t_zero]))


def test_aif_model_cost_matches_manual_sse():
    aif = Tac(np.arange(10.0), np.arange(10.0) * 2.0)

    class Identity(AifModel):
        def pred(self, params, kernel=None):
            return params

    model = Identity(aif)
    pred = np.arange(10.0)
    cost = model.cost(pred)
    assert cost == pytest.approx(np.sum((aif.cnt - pred) ** 2))


def test_feng_pred_no_kernel():
    aif = Tac(np.arange(0, 200, 1.0), np.zeros(200))
    model = FengModel(aif)
    hat = model.pred(_feng_params())
    assert hat.shape == (200,)
    assert np.all(np.isfinite(hat))


def test_feng_pred_kernel_resampling_matches_native_sampling():
    # Regression: aif_model.py used to be missing the scipy.interpolate
    # import entirely, so this code path raised NameError whenever the
    # kernel's sampling differed from the aif's.
    aif = Tac(np.arange(0, 200, 1.0), np.zeros(200))
    model = FengModel(aif)
    params = _feng_params()

    kernel_native = Tac(aif.time.copy(), np.full(aif.n, 3.0))
    kernel_sparse = Tac(np.arange(0, 200, 2.0), np.full(100, 3.0))

    hat_native = model.pred(params, kernel=kernel_native)
    hat_resampled = model.pred(params, kernel=kernel_sparse)

    assert np.allclose(hat_native, hat_resampled, rtol=1e-6)


def test_feng_unit_conv_leaves_t_zero_unlogged():
    params = _feng_params(t_zero=7.5)
    conv = FengModel(Tac(np.arange(10.0), np.zeros(10))).unit_conv(params)
    assert np.allclose(conv[0:6], np.exp(params[0:6]))
    assert conv[6] == pytest.approx(7.5)


def test_golish_pred_no_kernel():
    aif = Tac(np.arange(0, 200, 1.0), np.zeros(200))
    model = GolishModel(aif)
    hat = model.pred(_golish_params())
    assert hat.shape == (200,)
    assert np.all(np.isfinite(hat))


def test_golish_pred_kernel_resampling_matches_native_sampling():
    # Regression: pred() used to convolve against the raw kernel.cnt
    # array instead of the resampled kernel_cnt, silently using the
    # wrong-length kernel whenever sampling differed.
    aif = Tac(np.arange(0, 200, 1.0), np.zeros(200))
    model = GolishModel(aif)
    params = _golish_params()

    kernel_native = Tac(aif.time.copy(), np.full(aif.n, 3.0))
    kernel_sparse = Tac(np.arange(0, 200, 2.0), np.full(100, 3.0))

    hat_native = model.pred(params, kernel=kernel_native)
    hat_resampled = model.pred(params, kernel=kernel_sparse)

    assert np.allclose(hat_native, hat_resampled, rtol=1e-6)


def test_golish_unit_conv_leaves_t_zero_unlogged():
    # Regression: unit_conv used to exponentiate all 6 elements of the
    # params array, including t_zero, which is never log-transformed.
    params = _golish_params(t_zero=7.5)
    conv = GolishModel(Tac(np.arange(10.0), np.zeros(10))).unit_conv(params)
    assert np.allclose(conv[0:5], np.exp(params[0:5]))
    assert conv[5] == pytest.approx(7.5)
