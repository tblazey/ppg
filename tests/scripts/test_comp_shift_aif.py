import json
import sys

import numpy as np
import pytest

from ppg import util
from ppg.scripts import comp_shift_aif

from .conftest import replicate_to_4d, save_csv, save_nifti, save_pet_json

# Measured aif arrives this many seconds after the input at the tissue
TRUE_SHIFT = 12.0
TRUE_K1 = 0.008
TRUE_K2 = 0.009
TRUE_K3 = 0.002


def _input(t):
    t = np.clip(t - 20.0, 0.0, None)
    return 1000.0 * t * np.exp(-t / 8.0) + 50.0 * (1.0 - np.exp(-t / 30.0))


def _build_dataset(tmp_path, comp):
    # Aif on a 1 sec grid, delayed relative to the tissue input
    t = np.arange(0, 300, 1.0)
    tissue_input = _input(t)

    if comp == "one":
        pet_cnt = util.exp_conv(t, tissue_input, [TRUE_K1], [TRUE_K2])
    else:
        rate = TRUE_K2 + TRUE_K3
        pet_cnt = util.exp_conv(
            t,
            tissue_input,
            [TRUE_K1 * TRUE_K2 / rate, TRUE_K1 * TRUE_K3 / rate],
            [rate, 0.0],
        )

    # PET frames every 4 sec, as a 4D image with a BIDS sidecar
    pet_time = np.arange(2, 290, 4.0)
    aif_path = save_csv(tmp_path / "aif.csv", t, _input(t - TRUE_SHIFT))
    json_path = save_pet_json(
        tmp_path / "pet.json", pet_time, duration=4.0, radionuclide="18F"
    )
    pet_data = replicate_to_4d(np.interp(pet_time, t, pet_cnt), (2, 2, 1))
    pet_path = save_nifti(tmp_path / "pet.nii.gz", pet_data)

    return aif_path, pet_path, json_path


def _read_vals(path):
    lines = path.read_text().strip().split("\n")
    return {row.split(",")[0]: float(row.split(",")[1]) for row in lines}


@pytest.mark.parametrize("comp", ["one", "two"])
def test_comp_shift_aif_recovers_shift(tmp_path, monkeypatch, comp):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, comp)
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["comp-shift-aif", aif_path, pet_path, json_path, out_prefix, "-comp", comp],
    )
    comp_shift_aif.main()

    for suffix in [".csv", "_vals.csv", ".tiff", "_args.json"]:
        assert (tmp_path / f"out{suffix}").exists()

    values = _read_vals(tmp_path / "out_vals.csv")
    assert values["shift"] == pytest.approx(TRUE_SHIFT, abs=0.1)
    assert values["K1"] == pytest.approx(TRUE_K1 * 6000.0 / 1.05, rel=0.01)
    assert values["k2"] == pytest.approx(TRUE_K2 * 60.0, rel=0.01)
    if comp == "two":
        assert values["k3"] == pytest.approx(TRUE_K3 * 60.0, rel=0.01)

    # Shifted aif is the input aif with its times moved back by the shift
    aif_in = np.loadtxt(aif_path, delimiter=",")
    aif_out = np.loadtxt(tmp_path / "out.csv", delimiter=",")
    assert np.allclose(aif_out[:, 0], aif_in[:, 0] - values["shift"])
    assert np.allclose(aif_out[:, 1], aif_in[:, 1])

    assert json.loads((tmp_path / "out_args.json").read_text())["comp"] == [comp]


def test_comp_shift_aif_vb_adds_blood_volume(tmp_path, monkeypatch):
    aif_path, pet_path, json_path = _build_dataset(tmp_path, "one")
    out_prefix = str(tmp_path / "out")

    monkeypatch.setattr(
        sys,
        "argv",
        ["comp-shift-aif", aif_path, pet_path, json_path, out_prefix, "-vb"],
    )
    comp_shift_aif.main()

    values = _read_vals(tmp_path / "out_vals.csv")
    assert list(values) == ["shift", "K1", "k2", "lambda", "Vb"]
    assert values["shift"] == pytest.approx(TRUE_SHIFT, abs=0.5)
