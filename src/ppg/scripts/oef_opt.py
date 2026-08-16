#!/usr/bin/python
"""
Fits the Mintun oxygen extraction fraction (OEF) model to dynamic PET data.
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Load up some libraries
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

import ppg


def main():
    # Define parser
    parser = argparse.ArgumentParser(
        description="Computes oxygen extraction fraction with:",
        epilog="Fits Mintun O15 model to dynamic data",
    )
    parser.add_argument("aif", type=str, nargs=1, help="Aif csv file")
    parser.add_argument("pet", type=str, nargs=1, help="4D PET image")
    parser.add_argument(
        "pet_json", type=str, nargs=1, help="BIDS PET JSON sidecar (*_pet.json)"
    )
    parser.add_argument(
        "cbf", type=str, nargs=1, help="Cerebral blood flow image in mL/hg/min"
    )
    parser.add_argument("k2", type=str, nargs=1, help="Water efflux image in 1/min")
    parser.add_argument(
        "cbv", type=str, nargs=1, help="Cerebral blood volume image in mL/hg"
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
        help="Oxygen content of arterial blood in mL/dL"
        + " If set, CMRO2 is calculated",
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
    if args.ca[0] is not None:
        par_names = ["oef", "cmro2", "nrmse"]
        par_units = ["NA", "uMol/hg/min", "NA"]
    else:
        par_names = ["oef", "nrmse"]
        par_units = ["NA", "NA"]

    # Load up all the data
    img_list = [args.cbf[0], args.k2[0], args.cbv[0]]
    aif, pet_hdr, pet_mskd, msk_data, msk_hdr, mean_pet, h_life, imgs, avgs = (
        ppg.util.prep_model(
            args.aif[0],
            args.pet[0],
            args.pet_json[0],
            args.mask[0],
            args.vol[0],
            args.scale[0],
            args.limit[0],
            args.censor[0],
            img_paths=img_list,
        )
    )

    # Convert metabolic images to proper units
    cbf_mskd = imgs[0] / 6000.0 * 1.05
    k2_mskd = imgs[1] / 60.0
    cbv_mskd = imgs[2] / 100.0 * 1.05

    # Convert average values
    mean_cbf = avgs[0] / 6000.0 * 1.05
    mean_k2 = avgs[1] / 60.0
    mean_cbv = avgs[2] / 100.0 * 1.05

    # Extract water and oxygen components of input funciton
    aif_oxy, aif_water = ppg.util.iida_oxy_aif(aif, delta=20, prod=0.0012)

    # Create model for optimization
    mean_model = ppg.pet_model.OxyOne(
        aif_oxy, aif_water, mean_pet, mean_cbf, mean_k2, mean_cbv
    )

    # Optimize the mean pet tac
    mean_opt = opt.brute(mean_model.cost, (slice(0.01, 1, 0.01),), full_output=True)

    # Get the model predictions
    mean_pet_hat = mean_model.pred(mean_opt[0])

    # Make a plot showing fitted pet
    ppg.util.tac_plot(
        mean_pet,
        hats=[mean_pet_hat],
        labels=["Model Fit"],
        title="OEF Model Fit: Mean Tac",
        out_path=f"{args.out[0]}_wb_fit.tiff",
    )

    # Make a plot showing optmization test values
    plt.style.use("ggplot")
    plt.figure(0)
    plt.scatter(mean_opt[2], mean_opt[3], s=25, color="black")
    plt.scatter(mean_opt[0], mean_opt[1], s=35, color="#006BB6")
    plt.xlabel("OEF", color="black", weight="bold", size=10)
    plt.ylabel("Sum of Squares Error", color="black", weight="bold", size=10)
    plt.title("Optimization Error: Mean Tac")

    # Save contour plot
    plt.savefig(f"{args.out[0]}_wb_opt_error.tiff")
    plt.close()

    # Compute nomralized root mean square error
    mean_nrmse = np.sqrt(mean_opt[1] / mean_pet.n) / np.mean(mean_pet.cnt)

    # Write out parameters
    pars_out = np.append(mean_model.unit_conv(mean_opt[0], ca=args.ca[0]), mean_nrmse)
    ppg.io.write_pars(pars_out, par_names, par_units, f"{args.out[0]}_wb_vals.csv")

    # Quit if we don't want to do voxels
    if args.avg[0] == 1:
        # Save arguments and exit
        ppg.io.write_args(args, f"{args.out[0]}_args.txt")
        sys.exit()

    # Make empty array for storing voxelwise parameters
    n_vox = pet_mskd.shape[0]
    vox_params = np.zeros((n_vox, pars_out.shape[0]))

    # Loop through voxels
    no_c = 0
    for i in tqdm(range(n_vox)):
        # Construct tac object for current voxel
        vox_pet = ppg.Tac(mean_pet.time, pet_mskd[i, :], dc=True, h_life=h_life)

        # Make model object for current voxel
        vox_model = ppg.pet_model.OxyOne(
            aif_oxy, aif_water, vox_pet, cbf_mskd[i], k2_mskd[i], cbv_mskd[i]
        )

        # Optimize the voxel pet tac
        vox_opt = opt.minimize_scalar(
            vox_model.cost, bounds=np.array([0, 1]), method="bounded"
        )

        # Convergence check
        if vox_opt.success is False:
            # Update count of voxels that didn't converge
            no_c += 1

        else:
            # Store parameter estimates
            vox_params[i, 0:-1] = vox_model.unit_conv(vox_opt.x)

            # Compute normalized rmse
            vox_params[i, -1] = np.sqrt(vox_opt.fun / np.sqrt(vox_pet.n)) / np.mean(
                vox_pet.cnt
            )

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
