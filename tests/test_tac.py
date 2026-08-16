import numpy as np
import pytest

from ppg.tac import Tac


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        Tac(np.arange(5.0), np.arange(4.0))


def test_requires_1d_input():
    with pytest.raises(ValueError):
        Tac(np.zeros((5, 1)), np.zeros((5, 1)))


def test_uniform_sampling_detected_integer_interval():
    tac = Tac(np.arange(0, 10, 1.0), np.zeros(10))
    assert tac.unif is True
    assert tac.samp == pytest.approx(1.0)


def test_uniform_sampling_detected_fractional_interval():
    # Regression: rounding np.gradient to the nearest int used to
    # misclassify a 2.5s interval as uniform with the wrong samp value.
    t = np.arange(0, 25, 2.5)
    tac = Tac(t, np.zeros(t.shape[0]))
    assert tac.unif is True
    assert tac.samp == pytest.approx(2.5)


def test_nonuniform_sampling_detected():
    t = np.array([0, 1, 3, 7, 15], dtype=float)
    tac = Tac(t, np.zeros(t.shape[0]))
    assert tac.unif is False


def test_decay_flip_roundtrip():
    t = np.arange(0, 10, 1.0)
    cnt = np.full(t.shape[0], 10.0)
    tac = Tac(t, cnt.copy(), dc=True, h_life=5.0)

    tac.decay_flip()
    assert tac.dc is False
    assert not np.allclose(tac.cnt, cnt)

    tac.decay_flip()
    assert tac.dc is True
    assert np.allclose(tac.cnt, cnt)


def test_decay_flip_noop_without_h_life():
    t = np.arange(0, 10, 1.0)
    cnt = np.full(t.shape[0], 10.0)
    tac = Tac(t, cnt.copy())

    tac.decay_flip()
    assert np.allclose(tac.cnt, cnt)
