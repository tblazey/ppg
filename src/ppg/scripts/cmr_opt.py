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
        type=int,
        nargs="+",
        default=None,
        metavar="frame",
        help="0-based indices of PET frames to exclude (e.g. -censor 3 7 12)",
    )
    parser.add_argument(
        "-comps",
        action="store_const",
        const=[True],
        default=[False],
        help="Save predictions for individual compartments",
    )
    parser.add_argument(
        "-hct",
        type=float,
        nargs=1,
        default=[None],
        help="Subject hematocrit (0-1). If given, converts the aif from"
        + " whole blood to plasma via a time-varying RBC-to-plasma ratio"
        + " (Phelps et al., 1979). If omitted, no conversion is applied.",
    )
    parser.add_argument(
        "-hist",
        action="store_const",
        const=[1],
        default=[0],
        help="Output parameter histograms",
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
        "-save_se",
        action="store_const",
        const=[True],
        default=[False],
        help="Save standard error estimates for parameters",
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

    # Define parameters that will always be estimated
    par_names = ["K1", "k2", "k3", "ki", "vt", "vb", "nrmse", "bic"]
    par_units = ["mL/hg/min", "1/min", "1/min", "mL/hg/min", "mL/hg", "%", "NA", "NA"]

    # Add in k4 if necessary
    if args.k4[0] is True:
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

    # Extra keyword arguments unit_conv (and se, which wraps it) need
    unit_conv_kwargs = {} if args.ca is None else {"glu": args.ca[0], "lc": args.lc[0]}

    # Load up all the data
    aif, pet_hdr, pet_mskd, msk_data, msk_hdr, mean_pet, h_life = ppg.util.prep_model(
        args.aif[0],
        args.pet[0],
        args.pet_json[0],
        args.mask[0],
        args.vol[0],
        args.scale[0],
        None,
        args.censor,
    )

    # Default init, in the alpha/beta parameterization fit by Fdg --
    # roughly K1=0.0017, vd=0.79, k3=0.001, vb=0.04 (plus k4=0.00011 for
    # the k4 model) in physical units
    if args.k4[0] is False:
        mean_init = np.array([0.00091, 0.00079, 0.00215, 0.04])
    else:
        mean_init = np.array([0.000833, 0.000867, 5.75e-05, 0.0022, 0.04])

    # aif/pet's time grids are the same for every voxel (every vox_pet
    # below is built on mean_pet.time), so detect same_grid/uniform_grid
    # once here instead of separately for the whole-brain fit and every
    # one of the (potentially 100k+) per-voxel Fdg instances
    same_grid = ppg.util.is_same_grid(aif.time, mean_pet.time)
    uniform_grid = ppg.util.is_uniform_grid(aif.time)

    # Setup model
    mean_model = ppg.pet_model.Fdg(
        aif,
        mean_pet,
        k4=args.k4[0],
        hct=args.hct[0],
        algo="simpson",
        same_grid=same_grid,
        uniform_grid=uniform_grid,
    )

    # Setup inits
    mean_bounds = np.stack((mean_init / 5.0, mean_init * 5.0), axis=1)

    # Optimize the mean pet tac
    mean_opt = opt.minimize(
        lambda x: mean_model.cost(x, deriv=True),
        mean_init,
        method="L-BFGS-B",
        jac=True,
        bounds=mean_bounds,
    )

    # Convergence check
    if mean_opt.success is False and args.basin[0] is False:
        raise ValueError("Whole brain fit did not converge. Exiting...")

    # Refine estimation with basin hopping if necessary
    if args.basin[0] is True:
        mean_b_bounds = np.stack((mean_opt.x / 2.0, mean_opt.x * 2.0), axis=1)
        mean_opt = opt.basinhopping(
            lambda x: mean_model.cost(x, deriv=True),
            mean_opt.x,
            minimizer_kwargs={
                "jac": True,
                "bounds": mean_b_bounds,
                "options": {"ftol": 1e-5},
            },
        )

    # Compute nrmse and bic for whole-brain fit
    mean_nrmse = np.sqrt(mean_opt.fun / mean_pet.n) / np.mean(mean_pet.cnt)
    mean_bic = mean_pet.n * np.log(mean_opt.fun / mean_pet.n) + mean_opt.x.shape[
        0
    ] * np.log(mean_pet.n)

    # Write out mean pet tac parameter estimates
    mean_pars = mean_model.unit_conv(mean_opt.x, **unit_conv_kwargs)
    mean_pars = np.append(np.append(mean_pars, mean_nrmse), mean_bic)
    ppg.io.write_pars(mean_pars, par_names, par_units, f"{args.out[0]}_wb_vals.csv")

    # Write out whole-brain standard errors if necessary (nrmse/bic aren't
    # part of unit_conv's output, so they're excluded here)
    if args.save_se[0] is True:
        mean_se = mean_model.se(mean_opt.x, unit_conv_kwargs=unit_conv_kwargs)
        ppg.io.write_pars(
            mean_se, par_names[:-2], par_units[:-2], f"{args.out[0]}_wb_se.csv"
        )

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
    if args.save_se[0] is True:
        # nrmse (par_names' last entry) isn't part of unit_conv's output
        vox_se = np.full((n_vox, len(par_names) - 1), np.nan)

    # Use the whole-brain estimate to initilize/bound the voxel optimizations
    vox_init = mean_opt.x
    vox_bounds = np.stack((mean_opt.x / 3.0, mean_opt.x * 3.0), axis=1)

    # Loop through voxels
    no_c = 0
    for i in tqdm(range(n_vox)):
        # Construct tac object for current voxel
        vox_pet = ppg.Tac(mean_pet.time, pet_mskd[i, :], dc=True, h_life=h_life)

        # Make model object for current voxel
        vox_model = ppg.pet_model.Fdg(
            aif,
            vox_pet,
            k4=args.k4[0],
            hct=args.hct[0],
            algo=args.algo[0],
            same_grid=same_grid,
            uniform_grid=uniform_grid,
        )

        # Warm-start from this voxel's own LLS estimate when it's usable
        # (both non-degenerate and inside the fixed bounds below -- LLS can
        # return a value at/near 0 that would collapse a bounds/N..bounds*N
        # window), otherwise fall back to the previous voxel's fit
        lls_init = vox_model.init_par(vox_pet.cnt)
        if lls_init is not None and np.all(lls_init >= vox_bounds[:, 0]) and np.all(
            lls_init <= vox_bounds[:, 1]
        ):
            cur_init = lls_init
        else:
            cur_init = vox_init

        # Optimize the voxel pet tac
        vox_opt = opt.minimize(
            lambda x: vox_model.cost(x, deriv=True),
            cur_init,
            method="L-BFGS-B",
            jac=True,
            bounds=vox_bounds,
            options={"ftol": 1e-5},
        )

        # Convergence check
        if args.basin[0] is True:
            vox_b_bounds = np.stack((vox_opt.x / 2.0, vox_opt.x * 2.0), axis=1)
            vox_opt = opt.basinhopping(
                lambda x: vox_model.cost(x, deriv=True),
                vox_opt.x,
                minimizer_kwargs={
                    "jac": True,
                    "bounds": vox_b_bounds,
                    "options": {"ftol": 1e-5},
                },
            )
        elif vox_opt.success is False:
            no_c += 1
            continue

        # Store parameter estimates
        vox_params[i, 0:-1] = vox_model.unit_conv(vox_opt.x, **unit_conv_kwargs)

        # Compute normalized rmse
        vox_params[i, -1] = np.sqrt(vox_opt.fun / np.sqrt(vox_pet.n)) / np.mean(
            vox_pet.cnt
        )
        vox_init = vox_opt.x

        # Get and save voxel compartment tacs
        if args.comps[0] is True:
            vox_comps[i, :] = vox_model.comp(vox_opt.x)

        # Get voxel standard errors
        if args.save_se[0] is True:
            vox_se[i, :] = vox_model.se(vox_opt.x, unit_conv_kwargs=unit_conv_kwargs)

    # Write out number of voxels that did not converge
    ppg.io.write_str(f"{no_c}", f"{args.out[0]}_no_converge.txt")

    # Save voxelwise parameters
    img_names = [f"{args.out[0]}_{name}" for name in par_names]
    ppg.io.write_imgs(
        vox_params, pet_hdr.shape[0:3], pet_hdr.affine, img_names, msk=msk_data
    )

    # Save voxelwise standard errors if necessary (nrmse isn't part of
    # unit_conv's output, so it's excluded here)
    if args.save_se[0] is True:
        se_names = [f"{args.out[0]}_{name}_se" for name in par_names[:-1]]
        ppg.io.write_imgs(
            vox_se, pet_hdr.shape[0:3], pet_hdr.affine, se_names, msk=msk_data
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
