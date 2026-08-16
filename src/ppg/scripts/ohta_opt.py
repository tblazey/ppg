#!/usr/bin/python
"""
Fits the Ohta two-compartment model to dynamic PET data.
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Load up some libraries
import argparse
import sys

import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

import ppg


def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Fits Ohta. two comparment model")
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
        "-ca",
        type=float,
        nargs=1,
        default=[None],
        help="Arterial blood content in uMol/hg."
        + " If set, metabolic rate is computed",
    )
    parser.add_argument(
        "-censor",
        type=str,
        nargs=1,
        default=[None],
        help="Mask of PET time points to remove",
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
        "-scale",
        type=float,
        nargs=1,
        default=[1.0],
        metavar="float",
        help="Scale factor to convert Pet activity to Well Bq/mL." + " Default is 1.0",
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

    # Define names and units for parameters to estimate
    par_names = ["K1", "k2", "lambda", "v0", "nrmse"]
    par_units = ["mL/hg/min", "1/min", "mL/g", "mL/hg", "NA"]

    # Add in metabolic rate if necessary
    if args.ca[0] is not None:
        par_names.insert(4, "cmr")
        par_units.insert(4, "uMol/hg/min")
    n_params = len(par_names)

    # Load up all the data
    aif, pet_hdr, pet_mskd, msk_data, msk_hdr, mean_pet, h_life = ppg.util.prep_model(
        args.aif[0],
        args.pet[0],
        args.pet_json[0],
        args.mask[0],
        args.vol[0],
        args.scale[0],
        args.limit[0],
        args.censor[0],
    )

    # Prep for optmization
    mean_init = np.array([0.0045, 0.02, 0.02])
    mean_bounds = np.stack((mean_init / 5.0, mean_init * 5.0), axis=1)
    mean_model = ppg.pet_model.OhtaTwo(aif, mean_pet)

    # Optimize the mean pet tac
    mean_opt = opt.minimize(
        mean_model.cost, mean_init, method="L-BFGS-B", bounds=mean_bounds
    )

    # Convergence check
    if mean_opt.success is False:
        raise ValueError("Whole brain fit did not converge. Exiting...")

    # Compute nrmse for whole-brain fit
    mean_nrmse = np.sqrt(mean_opt.fun / mean_pet.n) / np.mean(mean_pet.cnt)

    # Write out mean pet tac parameter estimates
    mean_pars = mean_model.unit_conv(mean_opt.x, art=args.ca[0])
    mean_pars = np.append(mean_pars, mean_nrmse)
    ppg.io.write_pars(mean_pars, par_names, par_units, f"{args.out[0]}_wb_vals.csv")

    # Make a plot showing fitted pet
    ppg.util.tac_plot(
        mean_pet,
        hats=[mean_model.pred(mean_opt.x)],
        labels=["Model Fit"],
        title="Ohta Model Fit: Mean Tac",
        out_path=f"{args.out[0]}_wb_fit.tiff",
    )

    # Quit if we don't want to do voxels
    if args.avg[0] == 1:
        # Save arguments and exit
        ppg.io.write_args(args, f"{args.out[0]}_args.txt")
        sys.exit()

    # Make empty array for storing voxelwise parameters
    n_vox = pet_mskd.shape[0]
    vox_params = np.zeros((n_vox, n_params))

    # Use the whole-brain estimate to initilize/bound the voxel optimizations
    vox_init = mean_opt.x
    vox_bounds = np.stack((mean_opt.x / 3.0, mean_opt.x * 3.0), axis=1)

    # Loop through voxels
    no_c = 0
    for i in tqdm(range(n_vox)):
        # Construct tac object for current voxel
        vox_pet = ppg.Tac(mean_pet.time, pet_mskd[i, :], dc=True, h_life=h_life)

        # Make model object for current voxel
        vox_model = ppg.pet_model.OhtaTwo(aif, vox_pet)

        # Optimize the voxel pet tac
        vox_opt = opt.minimize(
            vox_model.cost,
            vox_init,
            method="L-BFGS-B",
            bounds=vox_bounds,
            options={"ftol": 1e-5},
        )

        # Convergence check
        if vox_opt.success is False:
            # Update count of voxels that didn't converge
            no_c += 1

        else:
            # Store parameter estimates
            vox_pars = vox_model.unit_conv(vox_opt.x, art=args.ca[0])
            vox_params[i, 0:-1] = vox_pars

            # Compute normalized rmse
            vox_params[i, -1] = np.sqrt(vox_opt.fun / np.sqrt(vox_pet.n)) / np.mean(
                vox_pet.cnt
            )

            # Initialize next voxel with previous optimization
            vox_init = vox_opt.x

    # Write out number of voxels that did not converge
    ppg.io.write_str(f"{no_c}", f"{args.out[0]}_no_converge.txt")

    # Save voxelwise parameters
    img_names = [f"{args.out[0]}_{name}" for name in par_names]
    ppg.io.write_imgs(
        vox_params, pet_hdr.shape[0:3], pet_hdr.affine, img_names, msk=msk_data
    )

    # Make a histogram plot for each parameter
    for i in range(vox_params.shape[1]):
        ppg.util.vox_hist(
            vox_params[:, i],
            par_names[i],
            par_units[i],
            out_path=f"{img_names[i]}_hist.tiff",
        )

    # Save arguments and go home
    ppg.io.write_args(args, f"{args.out[0]}_args.txt")


if __name__ == "__main__":
    main()
