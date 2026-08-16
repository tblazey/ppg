#!/usr/bin/python
"""
Convenience functions for ppg module
"""

# Load needed libraries
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import scipy.interpolate as interp
import scipy.ndimage as nd
from . import io
from .tac import Tac


def bolus_approx(tac):
    """
    Approximates the time of bolus arrival time

    Parameters
    ----------
    tac: Tac object to find bolus arrival for

    Returns
    --------
    bolus_time: float
        Estimated bolus arrival time

    If peak of tac is near the beginning of the
    time series, program will use first time point as bolus
    """

    # Make sure peak is not close to the tac start
    peak_time, _ = loc_peak(tac)
    if peak_time >= tac.time[4]:
        # Peak of derivative is approximate bolus
        bolus_time, _ = loc_deriv_peak(tac)

    else:
        # Use first timepoint
        bolus_time = tac.time[0]

    return bolus_time


def count_int(time, counts, limits):
    """
    Function to integrate data

    Parameters
    ----------
    time: array
        An n x 1 array containing time points
    counts: array
        An p x n array containing data to integrate
    limits: list
        A list containing minimum and maximum values for integration

    Returns
    -------
    counts_int: array
        An p x 1 array containing integration
    """

    # Make integration mask
    int_msk = np.logical_and(time >= limits[0], time <= limits[1])

    # Insure we have enough data points
    if np.sum(int_msk) < 3:
        raise ValueError("Not enough data points for integration")

    # Apply integration mask to data
    if counts.ndim == 1:
        counts_mskt = counts[int_msk]
    elif counts.ndim == 2:
        counts_mskt = counts[:, int_msk]
    else:
        raise ValueError("Unexpected number of dimensions during integration")

    # Run integration
    return np.trapezoid(counts_mskt, time[int_msk], axis=-1)


def check_img_dim(hdr, dim, allow_single=True):
    """
    Check to see if image is the specified numbe of dimensions

    Parameters
    ----------
    hdr: Nibabel image header
        Image to check
    dim: int
        Number of dimensions
    allow_single: bool
        Ignores signleton dimensions

    If dimensions of hdr do not match dim, a ValueError is raised
    """

    # Determine number of dimensions to check
    if allow_single is False:
        n_dim = len(hdr.shape) - np.sum(np.array(hdr.shape) == 1)
    else:
        n_dim = len(hdr.shape)

    # Check dimensions
    if n_dim == dim:
        pass
    else:
        msg = "%s does not have required dimensions of %i" % (hdr.get_filename(), dim)
        raise ValueError(msg)


def comp_img_dim(hdr_one, hdr_two, dim_end=None):
    """
    Checks to see if two images have the same dimensions

    Parameters
    ----------
    hdr_one: Nibabel image header
        First image to compare
    hdr_two: Niabel image header
        Second image to compare
    dim_end: int
        Compares image up to this dimension. By default it compares all dimensions

    If dimensions do not match, it raises a ValueError
    """

    if hdr_one.shape[0:dim_end] != hdr_two.shape[0:dim_end]:
        name_one = hdr_one.get_filename()
        name_two = hdr_two.get_filename()
        msg = f"{name_one} and {name_two} do not have the same dimensions"
        raise ValueError(msg)


def conv_matrix(kernel, pad=0):
    """
    Compute convolution matrix

    Parameters
    ----------
    kernel : array
        Convolution kernel of shape n
    pad: int
        Number of elements to pad the kernel by

    Returns
    -------
    c_mat : matrix
        A  n+p by n+p matrix representing the convolution kernel
    """

    # Make empty matrix
    n = kernel.shape[0] + pad
    c_mat = np.zeros((n, n))

    # Fill it up
    for i in range(n):
        for j in range(i + 1):
            if i - j < kernel.shape[0]:
                c_mat[i, j] = kernel[i - j]

    return c_mat


def exp_conv(aif_time, aif_cnt, coef, rate):
    """
    Analytically convolves an aif with a sum of decaying exponentials

    For a kernel sum_i coef[i] * exp(-rate[i] * t), evaluates the
    convolution integral at each point in aif_time by pulling the
    exponential out of the integral:

        exp(-rate[i] * t) * integral_0^t exp(rate[i] * tau) * aif(tau) dtau

    and computing the remaining integral as a running (cumulative
    trapezoidal) integral instead of a discrete convolution. Unlike
    np.convolve, this does not require aif_time to be uniformly sampled.

    Parameters
    ----------
    aif_time: array
        A n length array of aif sampling times
    aif_cnt: array
        A n length array of aif counts
    coef: array
        Coefficients for each exponential term
    rate: array
        Decay rate for each exponential term. A rate of 0 is a valid
        (constant) term.

    Returns
    -------
    hat: array
        A n length array with the convolved prediction, evaluated at
        aif_time
    """

    # Add up the contribution from each exponential term
    hat = np.zeros_like(aif_time)
    for c, r in zip(coef, rate):
        if r == 0:
            # No reweighting needed for a constant term
            hat += c * integ.cumulative_trapezoid(aif_cnt, aif_time, initial=0.0)
        else:
            weighted = aif_cnt * np.exp(r * aif_time)
            integral = integ.cumulative_trapezoid(weighted, aif_time, initial=0.0)
            hat += c * np.exp(-r * aif_time) * integral

    return hat


def gen_time_mask(pet, limit=None):
    """
    Generates 1D time mask

    Parameters
    ----------
    pet: Tac object
        Tac object for pet
    limit: float
        Limits analysis to specified seconds after bolus

    Returns
    -------
    time_msk: array
        An n x 1 array containing time mask
    """

    # Limit if specified seconds after bolus
    if limit is not None:
        # Determine bolus arrival time
        bolus_time = bolus_approx(pet)

        # Create mask using bolus
        time_msk = pet.time <= (bolus_time + limit)

    else:
        # Create all valid mask
        time_msk = np.ones(pet.n, dtype=bool)

    return time_msk


def iida_oxy_aif(aif, delta=20, prod=0.0012):
    """
    Extract oxygen and water components from aif using Iida et al., 1993 model

    Parameters
    -----------
    aif: Tac object
        Tac object containing aif samples
    delta: float
        Delay paramter for Iida convolution model
    prod: float
        Production rate constant for Iida convolution model

    Returns
    -------
    aif_oxy: Tac object
        Tac object containing oxygen aif samples
    aif_water: Tac object
        Tac object containing wwater aif samples
    """

    # Shift the input function
    aif_func = interp.interp1d(
        aif.time, aif.cnt, kind="cubic", fill_value=0.0, bounds_error=False
    )
    aif_shift = aif_func(aif.time - delta)

    # Analytically convolve the shifted input function with exp(-prod*t)
    # to extract the water aif
    aif_conv = exp_conv(aif.time, aif_shift, coef=[1.0], rate=[prod])
    aif_water = Tac(aif.time, prod * aif_conv, dc=True, h_life=122.24)

    # Extract oxygen portion of input function
    aif_oxy = Tac(aif.time, aif.cnt - aif_water.cnt, dc=True, h_life=122.24)

    # Return extracted aifs
    return aif_oxy, aif_water


def knot_loc(x, n_k, bounds=None):
    """
    Calculates location for knots based on sample quantiles

    Parameters
    ----------
    x : array
       A n length array containing the x-values
    n_k: interger
      Number of knots
    bounds: array
        A containing percentile lower and upper bound
        If None, uses algorithm described below

    Returns
    -------
    knots : array
        An array of knot locations

    Notes
    -----
    If bounds is not set, then function percentiles
    in the follwing manner:
        3 knots: 10 and 90%
        4-6 knots: 5% and 95%
        > 6 knots: 2.5% and 97.5%

    This is the same method as the Hmisc package in R
    """

    # Set boundary knot percentiles
    if n_k <= 2:
        raise ValueError("Number of knots must be greater than 2")
    elif bounds is not None:
        b_k = [bounds[0], bounds[1]]
    elif n_k == 3:
        b_k = [10, 90]
    elif n_k <= 6:
        b_k = [5, 95]
    elif n_k > 6 and n_k <= x.shape[0]:
        b_k = [2.5, 97.5]
    else:
        raise ValueError("# of knots must be less than # datapoints")

    # Get percentiles for all knots
    k_per = np.linspace(b_k[0], b_k[1], n_k)

    # Get actual knot locations based upon percentiles
    knots = np.percentile(x, k_per)

    return knots


def load_mask(msk_path, img_hdr):
    """
    Load 3d image mask. Must match image dimensions

    Parameters
    ----------
    msk_path: string
        Path to mask image
    img_hdr: Nibabel image header
        Header for image that mask is applied to

    Returns
    -------
    msk_data: array
        An n x 1 array containing mask
    img_hdr: Nibabel image header
        Header for image to be masked
    """

    # Mask creation logic
    if msk_path is None:
        # Create mask where every voxel is good if we aren't supplied a path
        msk_data = np.ones(np.prod(img_hdr.shape[0:3]), dtype=bool)
        msk_hdr = None

    else:
        # Load in mask image
        msk_hdr = nib.load(msk_path)

        # Check that dimensions are ok
        check_img_dim(msk_hdr, 3, allow_single=False)
        comp_img_dim(img_hdr, msk_hdr, 3)

        # Get mask data as a 1d array
        msk_data = msk_hdr.get_fdata().flatten() == 1.0

    # Return mask and header
    return msk_data, msk_hdr


def load_pet(
    pet_path,
    json_path,
    censor_path=None,
    msk_path=None,
    limit=None,
    scale=1.0 / 0.8657,
    vol_path=None,
):
    """
    Loads in PET data

    Parameters
    ----------
    pet_path: string
        Path to PET Nifti file
    json_path: string
        Path to BIDS PET JSON sidecar (FrameTimesStart, FrameDuration,
        and TracerRadionuclide fields are used)
    censor_path: string
        Path to file indicating time points to censor
    msk_path: string
        Path to mask Nifti file
    limit: float
        Time past bolus to limit analysis to
    scale: float
        Scale factor to convert PET to Well Bq/mL
    vol_path: string
        Path to volume weights

    Returns
    -------
    pet_hdr: Nibabel header
        Header for PET image
    pet_mskt: array
        A n x t array containing the PET image data
    msk_data: array
        A n x 1 array boolean array indication valid (True) voxels
    msk_hdr: Nibabel header
        Header for mask image
    h_life: float
        Physical half-life of the tracer, from the JSON sidecar
    """

    # Load in pet image header
    pet_hdr = nib.load(pet_path)

    # Check that pet is 4d
    check_img_dim(pet_hdr, 4)

    # Load frame timing and half-life from the BIDS PET JSON sidecar
    pet_time, h_life = io.load_pet_json(json_path)

    # Make sure we have correct number of times
    if pet_time.shape[0] < pet_hdr.shape[3]:
        raise ValueError(f"PET JSON sidecar {json_path} does not have enough frames")
    pet_time = pet_time[0 : pet_hdr.shape[3]]

    # Create or load mask
    msk_data, msk_hdr = load_mask(msk_path, pet_hdr)

    # Create of load volume weights
    vol_mskt, vol_sum = load_volume(vol_path, pet_hdr, msk_data)

    # Create masked 2d pet file
    pet_data = pet_hdr.get_fdata()
    pet_mskt = to_2d(pet_data)[msk_data, :] * scale

    # Create a nice time mask
    mean_pet = np.sum(pet_mskt * vol_mskt[:, np.newaxis], axis=0) / vol_sum
    mean_tac = Tac(pet_time, mean_pet)
    time_msk = gen_time_mask(mean_tac, limit=limit)

    # Censor logic
    if censor_path is not None:
        # Load in censor file
        censor_msk = np.loadtxt(censor_path)

        # Update time mask to exclude censored time points
        time_msk = np.logical_and(time_msk, censor_msk == 1)

    # Make sure we have enough pet data
    if np.sum(time_msk) < 5:
        raise ValueError("At least five pet data point are needed")

    # Compute mean pet tac object
    mean_tac_mskt = Tac(pet_time[time_msk], mean_pet[time_msk], dc=True, h_life=h_life)

    # Return everything we need
    return pet_hdr, pet_mskt[:, time_msk], msk_data, msk_hdr, mean_tac_mskt, h_life


def load_volume(vol_path, img_hdr, msk_data):
    """
    Loads image containing voxel volumes. Must match dimensions.

    Parameters
    ----------
    vol_path: string
        Path to volume image
    img_hdr: Nibabel image header
        Header for reference image
    msk_data: array
        Nifti image containing mask data

    Returns
    -------
    vol_mskt: array
        Array containing volume at each voxel with masked applied
    vol_sum: float
        Number of valid voxels
    """

    # Volume image creation logic
    if vol_path is not None:
        # Load image header
        vol_hdr = nib.load(vol_path)

        # Check that dimensions are ok
        check_img_dim(vol_hdr, 3, allow_single=False)
        comp_img_dim(img_hdr, vol_hdr, 3)

        # Load in image data
        vol_mskt = vol_hdr.get_fdata().flatten()[msk_data]

    else:
        # Make uniform volume weights
        vol_mskt = np.ones(np.sum(msk_data))

    # Get number of valid voxels
    vol_sum = np.sum(vol_mskt)

    return vol_mskt, vol_sum


def loc_deriv_peak(tac):
    """
    Estimates the time and value of derivative at its peak

    Parameters
    ----------
    tac: Tac object to find peak for

    Returns
    -------
    t_peak_d: float
        Time when derivative is at its peak
    y_peak_d: float
        Value of derivative at its peak
    """

    # Resample to fine grid
    tac_time_i = np.arange(tac.time[0], tac.time[-1], 0.01)
    tac_cnt_i = interp.interp1d(tac.time, tac.cnt, kind="cubic")(tac_time_i)

    # Smooth the interpolated signal
    y_sm = nd.gaussian_filter1d(tac_cnt_i, 150)

    # Compute the derivative of smoothed signal
    y_sm_d = np.gradient(y_sm, 0.01)

    # Return peak derivative time as its value
    idx = np.argmax(y_sm_d)
    return tac_time_i[idx], y_sm_d[idx]


def loc_peak(tac, tac_interp=True):
    """
    Estimate the peak value/time of a tac

    Parameters
    ----------
    tac: object
        Tac object to find peak for
    tac_interp: bool
        If True, interpolated values are used

    Returns
    -------
    t_peak: float
        Time at peak
    c_peak: float
        Counts at peak
    """

    # Interpolation switch
    if tac_interp is True:
        # Interpolate to very fine sampling
        tac_time_i = np.arange(tac.time[0], tac.time[-1], 0.01)
        tac_cnt_i = interp.interp1d(tac.time, tac.cnt, kind="cubic")(tac_time_i)

        # Determine max values suing interpolation values
        idx = np.argmax(tac_cnt_i)
        t_peak = tac_time_i[idx]
        c_peak = tac_cnt_i[idx]

    else:
        # Use non-interpolated values
        idx = np.argmax(tac.cnt)
        t_peak = tac.time[idx]
        c_peak = tac.cnt[idx]

    # Return peak time and value
    return t_peak, c_peak


def natural_spline_basis(x, knots, dot=None):
    """
    Computes a natural cubic spline basis
    See Elements of Statistical Learning: Equations 5.4 and 5.4

    Parameters
    ----------
    x: array
        An array of m x values defining the spline
    knots: array
        An n length array of knots for spline basis
    dot: int
        Order of spline derivative. Takes 1 or 2

    Returns
    -------
    basis: array
        An m x n array containing the spline basis functions
    deriv: array
        An m x n array with the basis for the spline's derivative
        Only returned if dot is not none
    """

    # Make sure dot is set correctly
    if dot != 1 and dot != 2 and dot is not None:
        raise ValueError("If set dot keyword arg must be either 1 or 2")

    # Make empty matrices
    n_k = knots.shape[0]
    basis = np.zeros((x.shape[0], n_k))
    if dot is not None:
        basis_d = np.zeros_like(basis)

    # Fill in non cubic terms
    basis[:, 0] = 1
    basis[:, 1] = x

    # Compute parts in all cubic terms
    last_spline = np.power(np.maximum(0, x - knots[-1]), 3)
    offset = (np.power(np.maximum(0, x - knots[n_k - 2]), 3) - last_spline) / (
        knots[-1] - knots[n_k - 2]
    )

    # Set derivative specific values
    if dot is not None:
        if dot == 1:
            # Derivative of x-term
            basis_d[:, 1] = 1

            # Set coefficients and power for splines
            dot_coef = 3.0
            dot_power = 2

        elif dot == 2:
            # Coefficients and power for splines
            dot_coef = 6.0
            dot_power = 1

        # Compute parts in all cubic terms
        last_spline_d = dot_coef * np.power(np.maximum(0, x - knots[-1]), dot_power)
        offset_d = (
            dot_coef * np.power(np.maximum(0, x - knots[n_k - 2]), dot_power)
            - last_spline_d
        ) / (knots[-1] - knots[n_k - 2])

    # Compute cubic terms
    for i in range(0, n_k - 2):
        # Add spline to basis
        basis[:, i + 2] = (
            (np.power(np.maximum(0, x - knots[i]), 3) - last_spline)
            / (knots[-1] - knots[i])
        ) - offset

        # Add derivative term if necessary
        if dot is not None:
            basis_d[:, i + 2] = (
                (
                    dot_coef * np.power(np.maximum(0, x - knots[i]), dot_power)
                    - last_spline_d
                )
                / (knots[-1] - knots[i])
            ) - offset_d

    # Return basis functions
    if dot is None:
        return basis
    else:
        return basis, basis_d


def prep_model(
    aif_path,
    pet_path,
    json_path,
    msk_path,
    vol_path,
    scale,
    limit,
    censor_path,
    img_paths=None,
    unif=False,
):
    """
    Runs common code needed to setup metabolic processing scripts

    Parameters
    ----------
    aif_path: string
        Path to aif text file
    pet_path: string
        Path to PET Nifti file
    json_path: string
        Path to BIDS PET JSON sidecar (FrameTimesStart, FrameDuration,
        and TracerRadionuclide fields are used)
    msk_path: string
        Path to mask Nifti file
    vol_path: string
        Path to voxel volume Nifti file
    scale: float
        Scale factor to convert PET to Well Bq/mL
    limit: float
        Time past bolus to limit analysis to
    censor_path: string
        Path to file indicating time points to censor
    img_path: list
        List of extra images to load in
    unif: boolean
        If True, interpolates the AIF to uniform sampling time

    Returns
    -------
    aif: Tac object
        Tac object containing aif data
    pet_hdr: Nibabel header
        Header for PET image
    pet_mskt: array
        A n x t array containing the PET image data
    msk_data: array
        A n x 1 array boolean array indication valid (True) voxels
    msk_hdr: Nibabel header
        Header for mask image
    mean_tac_mskt: Tac object
        Tac object containing valid whole-brain PET data and times
    h_life: float
        Physical half-life of the tracer, from the JSON sidecar

    If img_path is set function will also return:

    imgs: list
        A list containing masked image data for each path in list
    avgs: list
        A list containing masked average values for each image
    """

    # Load in image related data, including half-life from the JSON sidecar
    pet_hdr, pet_mskt, msk_data, msk_hdr, mean_tac_mskt, h_life = load_pet(
        pet_path,
        json_path,
        censor_path=censor_path,
        msk_path=msk_path,
        limit=limit,
        scale=scale,
        vol_path=vol_path,
    )

    # Load in aif into tac object
    aif = io.txt_to_tac(aif_path, dc=True, h_life=h_life, unif=unif)

    # Limit analysis to time points within aif
    time_msk = np.logical_and(
        mean_tac_mskt.time >= aif.time[0], mean_tac_mskt.time <= aif.time[-1]
    )

    # Apply new time mask
    if np.sum(time_msk == False) > 0:
        pet_mskt = pet_mskt[:, time_msk]
        mean_tac_mskt = Tac(
            mean_tac_mskt.time[time_msk],
            mean_tac_mskt.cnt[time_msk],
            dc=True,
            h_life=h_life,
        )

    # Logic for calls when we have extra images
    if img_paths is None:
        # Return everything we need
        return (aif, pet_hdr, pet_mskt, msk_data, msk_hdr, mean_tac_mskt, h_life)

    else:
        # Make empty lists
        imgs = []
        avgs = []

        # Get volume weights for averaging
        vol_mskt, vol_sum = load_volume(vol_path, pet_hdr, msk_data)

        # Loop through images
        for i in range(len(img_paths)):
            # Load in image header
            img_hdr = nib.load(img_paths[i])

            # Check that dimensions are ok
            check_dim = 4 - np.sum(np.array(pet_hdr.shape) == 1) - 1
            check_img_dim(img_hdr, check_dim, allow_single=False)
            comp_img_dim(img_hdr, pet_hdr, check_dim)

            # Load in image and apply mask
            img_mskt = img_hdr.get_fdata().flatten()[msk_data]

            # Compute mean pet tac object
            img_avg = np.sum(img_mskt * vol_mskt) / vol_sum

            # Add to lists
            imgs.append(img_mskt)
            avgs.append(img_avg)

        # Return everything with extra images
        return (
            aif,
            pet_hdr,
            pet_mskt,
            msk_data,
            msk_hdr,
            mean_tac_mskt,
            h_life,
            imgs,
            avgs,
        )


def tac_plot(tac, hats=None, labels=None, title=None, out_path=None, t_hat=None):
    """
    Plot showing time activity curve and possible model fits

    Parameter
    ---------
    tac: Tac object
        Time activity curve object storing original data
    hats: list
        List containing predictions vectors
    labels: list
        List containing names for vectors in hats
    title: str
        Title for plot
    out_path: str
        Name for output file with extension. If none plot is output to screen.
    t_hat: array
        Time points for prediction vectors. Uses tac times if not specificed
    """

    # Make scatter plot
    plt.style.use("ggplot")
    plt.figure(0)
    plt.scatter(tac.time, tac.cnt, color="black", s=25)
    plt.xlabel(f"Time ({tac.t_unit})", color="black", weight="bold", size=10)
    plt.ylabel(f"Activity ({tac.c_unit})", color="black", weight="bold", size=10)
    plt.title(title, weight="bold")

    # Make sure models and pars lists match
    if (hats and labels) is not None:
        if len(hats) != len(labels):
            raise ValueError("Number of prediction vectors must equal number of labels")

        # Loop through predictions
        for i in range(len(hats)):
            # Add line plot
            if t_hat is None:
                plt.plot(tac.time, hats[i], lw=1.75, label=labels[i])
            else:
                plt.plot(t_hat, hats[i], lw=1.75, label=labels[i])

        # Add legend to plot
        plt.legend()

    # Display plot correclty
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
    plt.close()


def to_2d(nd):
    """
    Reshapes a nd array into 2d array

    Parameters
    -----------
    nd: array
        A nd array to reshape

    Returns
    -------
    td: array
        A 2d array where the last dimension of array_nd is maintained
    """

    return nd.reshape(-1, nd.shape[-1])


def vox_hist(vox_data, name, unit, out_path=None):
    """
    Make a voxel parameter histogram

    Parameters
    ----------
    vox_data: array
        A array containing parmeter value for n voxels
    name: str
        Name for parameter
    unit: str
        Units for paramter
    out_path: str
        Name for output file with extension. If none plot is output to screen
    """

    # Make a voxel histogram
    plt.style.use("ggplot")
    plt.figure(0)
    plt.hist(vox_data, bins=150, density=True, edgecolor="black", color="#006BB6")
    plt.xlabel(f"{name} ({unit})", color="black", weight="bold", size=10)
    plt.ylabel("Density", color="black", weight="bold", size=10)
    plt.title(f"{name} Voxel Histogram", weight="bold")

    # Show it the right way
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
    plt.close()
