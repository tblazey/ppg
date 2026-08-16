import numpy as np
import pytest

from ppg.pet_model import (
    FdgFour,
    FdgThree,
    FlowTwo,
    OhtaTwo,
    OneComp,
    OxyOne,
    PetModel,
    TwoComp,
)
from ppg.tac import Tac


def test_pet_model_requires_matching_decay_status(synth_aif, synth_pet):
    synth_pet.dc = False
    with pytest.raises(ValueError):
        PetModel(synth_aif, synth_pet)


def test_flow_two_recovers_known_truth(synth_aif, synth_pet, flow_two_truth):
    model = FlowTwo(synth_aif, synth_pet)
    hat = model.pred(flow_two_truth)
    # synth_pet was built with this exact model/params, on the same grid
    assert np.allclose(hat, synth_pet.cnt, rtol=1e-3, atol=1e-6)
    assert model.cost(flow_two_truth) < 1e-6


def test_flow_two_unit_conv():
    model = FlowTwo.__new__(FlowTwo)  # unit_conv doesn't touch self
    meas = model.unit_conv(np.array([0.5, 0.05]))
    cbf, k2, lmbda = meas
    assert cbf == pytest.approx(0.5 * 6000.0 / 1.05)
    assert k2 == pytest.approx(0.05 * 60.0)
    assert lmbda == pytest.approx(cbf / k2 / 100.0)


def test_ohta_two_reduces_to_flow_two_when_v0_zero(
    synth_aif, synth_pet, flow_two_truth
):
    flow = FlowTwo(synth_aif, synth_pet)
    ohta = OhtaTwo(synth_aif, synth_pet)

    params = np.concatenate((flow_two_truth, [0.0]))
    assert np.allclose(ohta.pred(params), flow.pred(flow_two_truth), rtol=1e-6)


def test_ohta_two_unit_conv_with_and_without_art():
    model = OhtaTwo.__new__(OhtaTwo)
    params = np.array([0.5, 0.05, 0.02])
    no_art = model.unit_conv(params)
    with_art = model.unit_conv(params, art=10.0)
    assert with_art.shape[0] == no_art.shape[0] + 1
    assert with_art[-1] == pytest.approx(with_art[0] * 10.0)


def test_one_comp_recovers_known_truth(synth_aif, synth_pet, flow_two_truth):
    model = OneComp(synth_aif, synth_pet)
    coefs = model.coef(0.0)
    assert coefs == pytest.approx(flow_two_truth, rel=0.1)


def test_one_comp_pred_matches_masked_pet(synth_aif, synth_pet):
    model = OneComp(synth_aif, synth_pet)
    hat = model.pred(0.0)
    assert hat.shape == synth_pet.cnt[model.pet.mask].shape
    assert np.all(np.isfinite(hat))


def test_one_comp_with_blood_volume_runs(synth_aif, synth_pet):
    model = OneComp(synth_aif, synth_pet, vol=True)
    coefs = model.coef(0.0)
    assert coefs.shape == (3,)
    assert np.all(coefs >= 0)


def test_two_comp_runs_and_matches_shapes(synth_aif, synth_pet):
    model = TwoComp(synth_aif, synth_pet)
    hat = model.pred(0.0)
    assert hat.shape == synth_pet.cnt[model.pet.mask].shape
    assert np.all(np.isfinite(hat))


def test_two_comp_with_blood_volume_runs(synth_aif, synth_pet):
    model = TwoComp(synth_aif, synth_pet, vol=True)
    coefs = model.coef(0.0)
    assert coefs.shape == (5,)


def test_fdg_four_pred_and_components(synth_aif, synth_pet):
    model = FdgFour(synth_aif, synth_pet, plasma=False)
    params = np.array([0.02, 0.5, 0.05, 0.01, 0.03])
    hat = model.pred(params)
    assert hat.shape == synth_pet.time.shape
    assert np.all(np.isfinite(hat))

    comps = model.comp(params)
    assert comps.shape == (synth_pet.time.shape[0], 3)


def test_fdg_four_unit_conv_shapes():
    model = FdgFour.__new__(FdgFour)
    params = np.array([0.02, 0.5, 0.05, 0.01, 0.03])
    meas = model.unit_conv(params)
    assert meas.shape == (7,)

    meas_glu = model.unit_conv(params, glu=90.0)
    assert meas_glu.shape == (10,)


def test_fdg_three_pred_and_components(synth_aif, synth_pet):
    model = FdgThree(synth_aif, synth_pet, plasma=False)
    params = np.array([0.02, 0.5, 0.05, 0.03])
    hat = model.pred(params)
    assert hat.shape == synth_pet.time.shape
    assert np.all(np.isfinite(hat))

    comps = model.comp(params)
    assert comps.shape == (synth_pet.time.shape[0], 3)


def test_oxy_one_pred_and_unit_conv(synth_aif):
    water = Tac(synth_aif.time, synth_aif.cnt * 0.3, dc=True, h_life=122.24)
    oxy = Tac(synth_aif.time, synth_aif.cnt * 0.7, dc=True, h_life=122.24)
    pet = Tac(synth_aif.time, synth_aif.cnt * 0.5, dc=True, h_life=122.24)

    model = OxyOne(oxy, water, pet, flow=0.005, k2=0.001, vb=0.04)
    hat = model.pred(0.4)
    assert hat.shape == pet.time.shape
    assert np.all(np.isfinite(hat))

    assert model.unit_conv(0.4).shape == (1,)
    meas = model.unit_conv(0.4, ca=20.0)
    assert meas[0] == pytest.approx(0.4)
    assert meas[1] == pytest.approx(0.4 * model.flow * 20.0)


def test_oxy_one_requires_matching_decay_status(synth_aif):
    water = Tac(synth_aif.time, synth_aif.cnt * 0.3, dc=True, h_life=122.24)
    oxy = Tac(synth_aif.time, synth_aif.cnt * 0.7, dc=False, h_life=122.24)
    pet = Tac(synth_aif.time, synth_aif.cnt * 0.5, dc=True, h_life=122.24)

    with pytest.raises(ValueError):
        OxyOne(oxy, water, pet, flow=0.005, k2=0.001, vb=0.04)
