#!/usr/bin/python
"""
Fits a two-compartment FDG model to dynamic PET data to estimate CMRglc.
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Load up all the other libraries
import argparse
import sys

import nibabel as nib
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

import ppg


def main():
    # Define parser
    parser = argparse.ArgumentParser(
        description="Compute cerebral metabolic rate of glucose with:",
        epilog="Fits two compartment model to dynamic data.",
    )
    parser.add_argument("aif", type=str, nargs=1, help="Aif csv file")
    parser.add_argument("pet", type=str, nargs=1, help="4D PET image")
    parser.add_argument(
        "pet_json", type=str, nargs=1, help="BIDS PET JSON sidecar (*_pet.json)"
    )
    parser.add_argument("out", type=str, nargs=1, help="Name for file output")
    parser.add_argument(
        "-algo",
        type=str,
        nargs=1,
        default=["trapz"],
        choices=["trapz", "simpson"],
        help="Integration rule for the voxelwise fits. 'trapz' (default) is"
        + " faster; 'simpson' is more accurate, especially for sparsely"
        + " sampled AIFs, but ~3-4x slower per voxel. The whole-brain fit"
        + " always uses simpson regardless of this flag.",
    )
    parser.add_argument(
        "-avg",
        action="store_const",
        const=[1],
        default=[0],
        help="Only fit the image average curve.",
    )
    parser.add_argument(
        "-basin",
        action="store_const",
        const=[True],
        default=[False],
        help="Refine optimization with basinhopping algorithm",
    )
    parser.add_argument(
        "-ca", type=float, nargs=1, help="Plasma blood glucose in mg/dL"
    )
    parser.add_argument(
        "-censor",
        type=str,
        nargs=1,
        default=[None],
        help="Mask of PET time points to remove",
    )
    parser.add_argument(
        "-comps",
        action="store_const",
        const=[True],
        default=[False],
        help="Save predictions for individual compartments",
    )
    parser.add_argument(
        "-hist",
        action="store_const",
        const=[1],
        default=[0],
        help="Output parameter histograms",
    )
    parser.add_argument(
        "-init",
        type=float,
        nargs="*",
        help="Initial values for K1, K1/(k2+k3), k3, [k4], and Vb.",
    )
    parser.add_argument(
        "-k4",
        action="store_const",
        const=[True],
        default=[False],
        help="Include a k4 term in model",
    )
    parser.add_argument(
        "-lc",
        type=float,
        nargs=1,
        help="Value for the lumped constant. Default is 0.65"
        + "  without k4 and 0.81 with k4."
        + " If negative, LC is estimated from data.",
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
    parser.add_argument(
        "-wb",
        action="store_const",
        const=[False],
        default=[True],
        help="Use whole-blood input function for modeling",
    )
    args = parser.parse_args()

    # Define parameters that will always be estimated
    par_names = ["K1", "k2", "k3", "ki", "vt", "vb", "nrmse", "bic"]
    par_units = ["mL/hg/min", "1/min", "1/min", "mL/hg/min", "mL/hg", "%", "NA", "NA"]
    n_par = 4

    # Add in k4 if necessary
    if args.k4[0] is True:
        n_par += 1
        par_names.insert(3, "k4")
        par_units.insert(3, "1/min")

    # Add in params where we need ca
    if args.ca is not None:
        # unit_conv appends [cmrglc, influx, conc] right before the
        # trailing nrmse/bic that this script appends below
        insert_idx = len(par_names) - 2
        par_names[insert_idx:insert_idx] = ["cmrglc", "influx", "conc"]
        par_units[insert_idx:insert_idx] = ["uMol/hg/min", "uMol/hg/min", "uMol/hg"]

        # Determine the correct value for the lc
        if args.lc is None:
            if args.k4[0] is True:
                args.lc = [0.81]
            else:
                args.lc = [0.65]

    # Load up all the data
    aif, pet_hdr, pet_mskd, msk_data, msk_hdr, mean_pet, h_life = ppg.util.prep_model(
        args.aif[0],
        args.pet[0],
        args.pet_json[0],
        args.mask[0],
        args.vol[0],
        args.scale[0],
        None,
        args.censor[0],
    )

    # Create initial values
    if args.init is not None:
        if n_par != len(args.init):
            raise ValueError(
                "Number of initial values does not equal number of parameters"
            )
        mean_init = np.array(args.init)
    else:
        mean_init = np.array([0.0017, 0.79, 0.001, 0.04])
        if args.k4[0] is True:
            mean_init = np.insert(mean_init, 3, 0.00011)

    # Setup model
    if args.k4[0] is False:
        mean_model = ppg.pet_model.FdgThree(
            aif, mean_pet, plasma=args.wb[0], algo="simpson"
        )
    else:
        mean_model = ppg.pet_model.FdgFour(
            aif, mean_pet, plasma=args.wb[0], algo="simpson"
        )

    # Setup inits
    mean_bounds = np.stack((mean_init / 5.0, mean_init * 5.0), axis=1)

    # Optimize the mean pet tac
    mean_opt = opt.minimize(
        mean_model.cost, mean_init, method="L-BFGS-B", bounds=mean_bounds
    )

    # Convergence check
    if mean_opt.success is False and args.basin[0] is False:
        raise ValueError("Whole brain fit did not converge. Exiting...")

    # Refine estimation with basin hopping if necessary
    if args.basin[0] is True:
        mean_b_bounds = np.stack((mean_opt.x / 2.0, mean_opt.x * 2.0), axis=1)
        mean_opt = opt.basinhopping(
            mean_model.cost,
            mean_opt.x,
            minimizer_kwargs={"bounds": mean_b_bounds, "options": {"ftol": 1e-5}},
        )

    # Compute nrmse and bic for whole-brain fit
    mean_nrmse = np.sqrt(mean_opt.fun / mean_pet.n) / np.mean(mean_pet.cnt)
    mean_bic = mean_pet.n * np.log(mean_opt.fun / mean_pet.n) + mean_opt.x.shape[
        0
    ] * np.log(mean_pet.n)

    # Write out mean pet tac parameter estimates
    if args.ca is None:
        mean_pars = mean_model.unit_conv(mean_opt.x)
    else:
        mean_pars = mean_model.unit_conv(mean_opt.x, args.ca[0], args.lc[0])
    mean_pars = np.append(np.append(mean_pars, mean_nrmse), mean_bic)
    ppg.io.write_pars(mean_pars, par_names, par_units, f"{args.out[0]}_wb_vals.csv")

    # Make a plot showing fitted pet
    mean_hat = mean_model.pred(mean_opt.x)
    ppg.util.tac_plot(
        mean_pet,
        hats=[mean_hat],
        labels=["Model Fit"],
        title="FDG Model Fit: Mean Tac",
        out_path=f"{args.out[0]}_wb_fit.tiff",
    )

    # Save whole-brain compartments if necessary
    if args.comps[0] is True:
        # Set title for compartmental plot
        if args.k4[0] is False:
            comp_title = "FDG Model Component: No k4"
        else:
            comp_title = "FDG Model Component: With k4"

        # Make plot with compartmental predictions
        mean_comp = mean_model.comp(mean_opt.x)
        ppg.util.tac_plot(
            mean_pet,
            hats=[mean_hat, mean_comp[:, 0], mean_comp[:, 1], mean_comp[:, 2]],
            labels=["Model Fit", "Ca", "Ce", "Cm"],
            title=comp_title,
            out_path=f"{args.out[0]}_wb_comps.tiff",
        )

        # Save compartmental predictions
        np.savetxt(
            f"{args.out[0]}_wb_comps.csv",
            np.hstack((mean_pet.time[:, np.newaxis], mean_comp)),
            delimiter=",",
        )

    # Quit if we don't want to do voxels
    if args.avg[0] == 1:
        # Save arguments and exit
        ppg.io.write_args(args, f"{args.out[0]}_args.txt")
        sys.exit()

    # Remove bic from parameter list
    del par_names[-1]
    del par_units[-1]

    # Make empty array for storing voxelwise parameters
    n_vox = pet_mskd.shape[0]
    vox_params = np.zeros((n_vox, len(par_names)))
    if args.comps[0] is True:
        vox_comps = np.zeros((n_vox, mean_pet.n, 3))

    # Use the whole-brain estimate to initilize/bound the voxel optimizations
    vox_init = mean_opt.x
    vox_bounds = np.stack((mean_opt.x / 3.0, mean_opt.x * 3.0), axis=1)

    # Loop through voxels
    no_c = 0
    for i in tqdm(range(n_vox)):
        # Construct tac object for current voxel
        vox_pet = ppg.Tac(mean_pet.time, pet_mskd[i, :], dc=True, h_life=h_life)

        # Make model object for current voxel
        if args.k4[0] is False:
            vox_model = ppg.pet_model.FdgThree(
                aif, vox_pet, plasma=args.wb[0], algo=args.algo[0]
            )
        else:
            vox_model = ppg.pet_model.FdgFour(
                aif, vox_pet, plasma=args.wb[0], algo=args.algo[0]
            )

        # Optimize the voxel pet tac
        vox_opt = opt.minimize(
            vox_model.cost,
            vox_init,
            method="L-BFGS-B",
            bounds=vox_bounds,
            options={"ftol": 1e-5},
        )

        # Convergence check
        if args.basin[0] is True:
            vox_b_bounds = np.stack((vox_opt.x / 2.0, vox_opt.x * 2.0), axis=1)
            vox_opt = opt.basinhopping(
                vox_model.cost,
                vox_opt.x,
                minimizer_kwargs={"bounds": vox_b_bounds, "options": {"ftol": 1e-5}},
            )
        elif vox_opt.success is False:
            no_c += 1
            continue

        # Store parameter estimates
        if args.ca is None:
            vox_params[i, 0:-1] = vox_model.unit_conv(vox_opt.x)
        else:
            vox_params[i, 0:-1] = vox_model.unit_conv(vox_opt.x, args.ca[0], args.lc[0])

        # Compute normalized rmse
        vox_params[i, -1] = np.sqrt(vox_opt.fun / np.sqrt(vox_pet.n)) / np.mean(
            vox_pet.cnt
        )
        vox_init = vox_opt.x

        # Get and save voxel compartment tacs
        if args.comps[0] is True:
            vox_comps[i, :] = vox_model.comp(vox_opt.x)

    # Write out number of voxels that did not converge
    ppg.io.write_str(f"{no_c}", f"{args.out[0]}_no_converge.txt")

    # Save voxelwise parameters
    img_names = [f"{args.out[0]}_{name}" for name in par_names]
    ppg.io.write_imgs(
        vox_params, pet_hdr.shape[0:3], pet_hdr.affine, img_names, msk=msk_data
    )

    # Save voxelwise components if necessary
    if args.comps[0] is True:
        for idx, comp in enumerate(["cp", "ce", "cm"]):
            comp_data = np.zeros(pet_hdr.shape).reshape(-1, pet_hdr.shape[-1])
            comp_data[msk_data, :] = vox_comps[:, :, idx]
            comp_hdr = nib.Nifti1Image(comp_data.reshape(pet_hdr.shape), pet_hdr.affine)
            comp_hdr.to_filename(f"{args.out[0]}_{comp}.nii.gz")

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
    ppg.io.write_args(args, f"{args.out[0]}_args.txt")


if __name__ == "__main__":
    main()
