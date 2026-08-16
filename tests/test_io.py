import argparse

import nibabel as nib
import numpy as np
import pytest

from ppg import io
from ppg.tac import Tac


def test_tac_to_txt_and_back_roundtrip(tmp_path):
    tac = Tac(np.arange(0, 10, 1.0), np.arange(10, 20, 1.0))
    path = tmp_path / "tac.txt"

    io.tac_to_txt(tac, str(path))
    loaded = io.txt_to_tac(str(path))

    assert np.allclose(loaded.time, tac.time)
    assert np.allclose(loaded.cnt, tac.cnt, atol=1e-5)


def test_txt_to_tac_rejects_1d_input(tmp_path):
    path = tmp_path / "bad.txt"
    np.savetxt(path, np.arange(5.0))

    with pytest.raises(ValueError):
        io.txt_to_tac(str(path))


def test_txt_to_tac_interpolates_to_uniform_sampling(tmp_path):
    path = tmp_path / "nonuniform.txt"
    data = np.array([[0, 0], [1, 1], [3, 3], [7, 7]], dtype=float)
    np.savetxt(path, data, delimiter=",")

    tac = io.txt_to_tac(str(path), unif=True)
    assert tac.unif is True
    assert tac.samp == pytest.approx(1.0)


def test_write_str(tmp_path):
    path = tmp_path / "out.txt"
    io.write_str("hello", str(path))
    assert path.read_text() == "hello"


def test_write_args_handles_lists_and_empty_lists(tmp_path):
    path = tmp_path / "args.txt"
    ns = argparse.Namespace(a=[], b=[1, 2, 3], c="x")

    io.write_args(ns, str(path))
    contents = path.read_text()

    assert "a: \n" in contents
    assert "b: 1,2,3\n" in contents
    assert "c: x\n" in contents


def test_write_pars(tmp_path):
    path = tmp_path / "pars.csv"
    io.write_pars([1.0, 2.5], ["K1", "k2"], ["mL/min", "1/min"], str(path))

    lines = path.read_text().strip().split("\n")
    assert lines[0].startswith("K1,1.0000000000,mL/min")
    assert lines[1].startswith("k2,2.5000000000,1/min")


def test_write_pars_length_mismatch_raises(tmp_path):
    path = tmp_path / "pars.csv"
    with pytest.raises(ValueError):
        io.write_pars([1.0], ["K1", "k2"], ["mL/min", "1/min"], str(path))


def test_write_img_and_load_roundtrip(tmp_path):
    shape = (2, 2, 2)
    affine = np.eye(4)
    img_data = np.arange(8.0)

    out_path = tmp_path / "img"
    io.write_img(img_data, shape, affine, str(out_path))

    loaded = nib.load(str(out_path) + ".nii.gz")
    assert loaded.shape == shape
    assert np.allclose(loaded.get_fdata().flatten(), img_data)


def test_write_img_with_mask(tmp_path):
    shape = (2, 2, 2)
    affine = np.eye(4)
    msk = np.zeros(8, dtype=bool)
    msk[[0, 3, 5]] = True
    img_data = np.array([10.0, 20.0, 30.0])

    out_path = tmp_path / "masked"
    io.write_img(img_data, shape, affine, str(out_path), msk=msk)

    loaded = nib.load(str(out_path) + ".nii.gz").get_fdata().flatten()
    assert loaded[0] == pytest.approx(10.0)
    assert loaded[3] == pytest.approx(20.0)
    assert loaded[5] == pytest.approx(30.0)
    assert loaded[1] == pytest.approx(0.0)


def test_write_img_bad_shape_raises(tmp_path):
    with pytest.raises(ValueError):
        io.write_img(np.arange(8.0), (3, 3, 3), np.eye(4), str(tmp_path / "bad"))


def test_write_imgs_name_count_mismatch_raises(tmp_path):
    img_data = np.zeros((8, 2))
    with pytest.raises(ValueError):
        io.write_imgs(img_data, (2, 2, 2), np.eye(4), [str(tmp_path / "one")])
