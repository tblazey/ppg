#!/usr/bin/python
"""
Fits a spectral analysis model (Cunningham and Jones, 1993) to dynamic PET data.
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Load up some libraries
import argparse
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

import ppg


def main():
    # Define parser
    parser = argparse.ArgumentParser(
        description="Compute spectral analysis parameters with:",
        epilog="Fits a non-negative sum of exponentials convolved with the aif.",
    )
    parser.add_argument("aif", type=str, nargs=1, help="Aif csv file")
    parser.add_argument("pet", type=str, nargs=1, help="4D PET image")
    parser.add_argument(
        "pet_json", type=str, nargs=1, help="BIDS PET JSON sidecar (*_pet.json)"
    )
    parser.add_argument("out", type=str, nargs=1, help="Name for file output")
    parser.add_argument(
        "-avg",
        action="store_const",
        const=[1],
        default=[0],
        help="Only fit the image average curve.",
    )
    parser.add_argument(
        "-censor",
        type=int,
        nargs="+",
        default=None,
        metavar="frame",
        help="0-based indices of PET frames to exclude (e.g. -censor 3 7 12)",
    )
    parser.add_argument(
        "-censor_aif",
        action="store_const",
        const=[True],
        default=[False],
        help="Also remove -censor's frames from the aif -- for an"
        + " image-derived input function on the same frame grid as the PET"
        + " data",
    )
    parser.add_argument(
        "-hist",
        action="store_const",
        const=[1],
        default=[0],
        help="Output parameter histograms",
    )
    parser.add_argument(
        "-limit",
        type=float,
        nargs=1,
        default=[None],
        help="Limit modeling to specified number of seconds past"
        + " bolus arrival. Default is to use entire scan.",
    )
    parser.add_argument(
        "-mask",
        type=str,
        nargs=1,
        metavar="nii",
        default=[None],
        help="3D binary mask image",
    )
    parser.add_argument(
        "-n_rates",
        type=int,
        nargs=1,
        default=[100],
        metavar="int",
        help="Number of log-spaced rates in the spectral basis. Default is 100",
    )
    parser.add_argument(
        "-rates",
        type=float,
        nargs=2,
        default=None,
        metavar=("min", "max"),
        help="Smallest and largest rates (1/sec) in the spectral basis."
        + " Default is 1/(3*Tend) and 3/Tin, where Tend is the end of the"
        + " last frame and Tin is the duration of the first frame"
        + " (Veronese et al., 2016)",
    )
    parser.add_argument(
        "-scale",
        type=float,
        nargs=1,
        default=[1.0],
        metavar="float",
        help="Scale factor to convert Pet activity to Well Bq/mL." + " Default is 1.0",
    )
    parser.add_argument(
        "-trap",
        action="store_const",
        const=[True],
        default=[False],
        help="Add a trapping (rate 0) term to the basis and estimate Ki",
    )
    parser.add_argument(
        "-vb",
        action="store_const",
        const=[True],
        default=[False],
        help="Add a blood volume term to the basis",
    )
    parser.add_argument(
        "-vol",
        type=str,
        nargs=1,
        metavar="nii",
        default=[None],
        help="Volume for each voxel in input images."
        + " Used for weighting whole-brain average",
    )
    args = parser.parse_args()

    # Define parameters that will always be estimated -- names match the
    # K1/Ki/Vt/Vb casing convention used by cmr_opt
    par_names = ["K1", "Vt", "nrmse"]
    par_units = ["mL/hg/min", "mL/hg", "NA"]

    # Add in trapping and blood volume if necessary
    if args.trap[0] is True:
        par_names.insert(1, "Ki")
        par_units.insert(1, "mL/hg/min")
    if args.vb[0] is True:
        par_names.insert(-1, "Vb")
        par_units.insert(-1, "%")

    # Load up all the data
    aif, pet_hdr, pet_mskd, msk_data, msk_hdr, mean_pet, h_life = ppg.util.prep_model(
        args.aif[0],
        args.pet[0],
        args.pet_json[0],
        args.mask[0],
        args.vol[0],
        args.scale[0],
        args.limit[0],
        args.censor,
        censor_aif=args.censor_aif[0],
    )

    # Default rate limits from Veronese et al., 2016. Uses the frames left
    # after censoring/limiting, since those are the ones being fit
    if args.rates is None:
        frame_times, _, frame_durs = ppg.io.load_pet_json(
            args.pet_json[0], durations=True
        )
        fit_durs = frame_durs[np.isin(frame_times, mean_pet.time)]
        t_end = mean_pet.time[-1] + fit_durs[-1] / 2.0
        args.rates = [1.0 / (3.0 * t_end), 3.0 / fit_durs[0]]

    # Make log-spaced rates for the spectral basis
    rates = np.logspace(
        np.log10(args.rates[0]), np.log10(args.rates[1]), args.n_rates[0]
    )
    if args.trap[0] is True:
        rates = np.insert(rates, 0, 0.0)

    # Setup model. The basis only depends on the aif and frame times, so
    # this model is reused for every voxel below
    mean_model = ppg.pet_model.Spectral(aif, mean_pet, rates, vb=args.vb[0])

    # Fit the mean pet tac
    mean_coef = mean_model.coef()

    # Compute nrmse for whole-brain fit
    mean_sse = mean_model.cost(mean_coef)
    mean_nrmse = np.sqrt(mean_sse / mean_pet.n) / np.mean(mean_pet.cnt)

    # Write out mean pet tac parameter estimates
    mean_pars = np.append(mean_model.unit_conv(mean_coef), mean_nrmse)
    ppg.io.write_pars_json(
        mean_pars, par_names, par_units, f"{args.out[0]}_wb_params.json"
    )

    # Make a plot showing fitted pet
    ppg.util.tac_plot(
        mean_pet,
        hats=[mean_model.pred(mean_coef)],
        labels=["Model Fit"],
        title="Spectral Model Fit: Mean Tac",
        out_path=f"{args.out[0]}_wb_plot.jpeg",
    )

    # Save whole-brain spectrum (blood volume, if present, is the last coef)
    mean_alpha = mean_coef[0 : rates.shape[0]]
    np.savetxt(
        f"{args.out[0]}_wb_spectrum.csv",
        np.stack((rates, mean_alpha), axis=1),
        delimiter=",",
    )

    # Make a spectrum plot. Rate 0 can't go on a log axis, so only the
    # reversible components are shown
    rev = rates > 0
    plt.style.use("ggplot")
    plt.figure(0)
    plt.stem(rates[rev], mean_alpha[rev], basefmt=" ")
    plt.xscale("log")
    plt.xlabel("Rate (1/sec)", color="black", weight="bold", size=10)
    plt.ylabel("Coefficient (1/sec)", color="black", weight="bold", size=10)
    plt.title("Spectrum: Mean Tac", weight="bold")

    # Report the terms that aren't on the plot in the legend. Empty plots
    # give legend entries without markers, and "best" keeps them off peaks
    if args.trap[0] is True or args.vb[0] is True:
        if args.trap[0] is True:
            trap_str = f"Trapping (rate 0): {mean_alpha[~rev][0]:.3g} 1/sec"
            plt.plot([], [], " ", label=trap_str)
        if args.vb[0] is True:
            plt.plot([], [], " ", label=f"Blood volume: {mean_coef[-1] * 100.0:.3g} %")
        plt.legend(loc="best", handlelength=0, fontsize=9)
    plt.savefig(f"{args.out[0]}_wb_spectrum.jpeg")
    plt.close()

    # Quit if we don't want to do voxels
    if args.avg[0] == 1:
        # Save arguments and exit
        ppg.io.write_args(args, f"{args.out[0]}_args.json")
        sys.exit()

    # Make empty array for storing voxelwise parameters
    n_vox = pet_mskd.shape[0]
    vox_params = np.zeros((n_vox, len(par_names)))
    vox_spec = np.zeros((n_vox, rates.shape[0]))

    # Loop through voxels
    for i in tqdm(range(n_vox)):
        # Fit current voxel using the whole-brain basis
        vox_cnt = pet_mskd[i, :]
        vox_coef = mean_model.coef(vox_cnt)

        # Store parameter estimates and spectrum
        vox_params[i, 0:-1] = mean_model.unit_conv(vox_coef)
        vox_spec[i, :] = vox_coef[0 : rates.shape[0]]

        # Compute normalized rmse
        vox_sse = np.sum(np.power(vox_cnt - mean_model.pred(vox_coef), 2))
        vox_params[i, -1] = np.sqrt(vox_sse / mean_pet.n) / np.mean(vox_cnt)

    # Save voxelwise parameters
    img_names = [f"{args.out[0]}_{name}" for name in par_names]
    ppg.io.write_imgs(
        vox_params, pet_hdr.shape[0:3], pet_hdr.affine, img_names, msk=msk_data
    )

    # Save voxelwise spectrum as a 4D image, one volume per rate. Rates for
    # each volume are in the first column of the whole-brain spectrum csv
    spec_shape = pet_hdr.shape[0:3] + (rates.shape[0],)
    spec_data = np.zeros(spec_shape).reshape(-1, rates.shape[0])
    spec_data[msk_data, :] = vox_spec
    spec_hdr = nib.Nifti1Image(spec_data.reshape(spec_shape), pet_hdr.affine)
    spec_hdr.to_filename(f"{args.out[0]}_spectrum.nii.gz")

    # Make a histogram plot for each parameter
    if args.hist[0] == 1:
        for i in range(vox_params.shape[1]):
            ppg.util.vox_hist(
                vox_params[:, i],
                par_names[i],
                par_units[i],
                out_path=f"{img_names[i]}_hist.tiff",
            )

    # Save arguments and go home
    ppg.io.write_args(args, f"{args.out[0]}_args.json")


if __name__ == "__main__":
    main()
