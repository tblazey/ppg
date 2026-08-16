#!/usr/bin/python
"""
Input/output functions for the ppg module
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Load needed libraries
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.interpolate as interp
from .tac import Tac


def tac_to_txt(tac, path):
    """
    Outputs a text file given a tac file

    Parameters
    ----------
    tac: Tac Object
        Tac object to write out
    path: string
        Path to output file.
    """

    np.savetxt(path, np.stack((tac.time, tac.cnt), axis=1), delimiter=",", fmt="%.5f")


def txt_to_tac(
    path,
    h_life=None,
    dc=None,
    t_unit="second",
    c_unit="Bq/mL",
    delimiter=",",
    unif=False,
):
    """
    Creates a tac object from text file input

    Parameters
    ----------
    path: string
        Path to text file.
        Assumes time is in the first column and counts in the second
    h_life: float
        Half life for tracer
    dc: bool
        True indicates tracer has been decay corrected
    t_unit: str
        Time unit
    c_unit:
        Activity unit
    delimiter: string
        Text delimiter. If none, it uses all whitespace
    unif: boolean
        Interpolates tac to uniform sampling if True

    Returns
    -------
    tac: object
        A tac object containing the data
    """

    # Load in text file
    data = np.loadtxt(path, delimiter=delimiter)

    # Make sure it has reasonable dimensions
    if data.ndim != 2:
        raise ValueError(f"Data at {path} must have two columns")

    # Interpolation logic
    if unif is True:
        # Interpolate tac to uniform sampling
        samp = np.min(np.diff(data[:, 0]))
        i_tac_time = np.arange(data[0, 0], data[-1, 0], samp)
        i_tac_cnt = interp.interp1d(data[:, 0], data[:, 1], kind="linear")(i_tac_time)

        # Make object for interpolated tac
        tac = Tac(
            i_tac_time, i_tac_cnt, h_life=h_life, dc=dc, t_unit=t_unit, c_unit=c_unit
        )

    else:
        # Make raw Tac object
        tac = Tac(
            data[:, 0], data[:, 1], h_life=h_life, dc=dc, t_unit=t_unit, c_unit=c_unit
        )

    return tac


def write_str(string, path):
    """
    Write out a string to text

    Parameters
    ----------
    string: str
        String to write out
    path: str
        Name of file output with extension
    """

    # Write out string
    try:
        with open(path, "w") as str_out:
            str_out.write(string)
    except IOError:
        raise IOError(f"Cannot write file {path}")


def write_args(args, path):
    """
    Write out an argparse argument list to text

    Parameters
    ----------
    args: Namespace
        argparse Namespace containing arguments
    path: string
        Path to write file to with extension
    """

    # Make string with all arguments
    arg_string = ""
    for arg, value in sorted(vars(args).items()):
        if type(value) is list:
            if len(value) > 0:
                value = ",".join(map(str, value))
            else:
                value = ""
        arg_string += f"{arg}: {value}\n"

    # Write out arguments string
    write_str(arg_string, path)


def write_pars(pars, names, units, path):
    """
    Writes out parameter vector to csv

    Parameters
    ----------
    pars: array
        A array of length p containing parameter values
    names: list
        A list of length p with parameter names
    units: list
        A list of length p with parameter units
    path: str
        Name for output file (with extension)
    """

    # Make sure dimensions match
    if not (len(pars) == len(names) == len(units)):
        raise ValueError("Lengths of pars, names, and units must be the same")

    # Convert paramter vector to comma seperated string
    par_str = ""
    for i in range(len(pars)):
        par_str += f"{names[i]},{pars[i]:.10f},{units[i]}\n"

    # Write to file
    write_str(par_str, path)


def write_img(img_data, shape, affine, path, msk=None):
    """
    Write out array to Nifti image

    Parameters
    ----------
    img_data: array
        An array of n voxels
    shape: array
        An array of length 3 containing values to reshape to
    affine: array
        A 4 by 4 matrix containing the affine transform to save in img header
    path: str
        Path (no extension) to save image at
    msk: array
        Optional mask of size m x 1 containing n valid values
    """

    # Mask switch
    if msk is None:
        # Make sure shape is sufficient
        if np.prod(shape) != img_data.shape[0]:
            raise ValueError("Image cannot be reshaped to requested size")

        # Create Nifti image instance
        img = nib.Nifti1Image(img_data.reshape(shape), affine)

    else:
        # Make sure mask can be reshaped to specified dimensions
        if np.prod(shape) != msk.shape[0]:
            raise ValueError("Image cannot be reshaped to requested size")

        # Make sure mask contains enough valid values
        if np.sum(msk) != img_data.shape[0]:
            raise ValueError("Mask does not contain enough valid values")

        # Make unmasked data array
        out_data = np.zeros(msk.shape)
        out_data[msk] = img_data

        # Create Nifti image from "unmasked" values
        img = nib.Nifti1Image(out_data.reshape(shape), affine)

    # Try to save it
    out_path = f"{path}.nii.gz"
    try:
        img.to_filename(out_path)
    except IOError:
        raise IOError(f"Cannot save image at {out_path}")


def write_imgs(img_data, shape, affine, paths, msk=None):
    """
    Writes out each frame of a 2d array to a seperate 3d volume

    Parameters
    ----------
    img_data: array
        An array of n voxels by p images
    shape: array
        An array of length 3 containing values to reshape to
    affine: array
        4 x 4 containing the affine transform to save in img header
    paths: list
        A p length list of image names
    msk: array
        Optional mask of size m x 1 containing n valid values
    """

    # Make sure we have enough names
    if len(paths) != img_data.shape[1]:
        raise ValueError("Size of output names does not equal size of images")

    # Loop through images
    for i in range(img_data.shape[1]):
        # Save image
        write_img(img_data[:, i], shape, affine, paths[i], msk)
