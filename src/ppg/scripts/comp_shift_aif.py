#!/usr/bin/python
"""
Computes the time shift for an input function using a generic compartmental model.
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Load libraries
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

import ppg


def main():
    # Define parser
    parser = argparse.ArgumentParser(
        description="Computes shift for input function"
        + " using a generic compartmental model"
    )
    parser.add_argument("aif", type=str, nargs=1, help="Input function csv file")
    parser.add_argument("pet", type=str, nargs=1, help="PET csv file")
    parser.add_argument(
        "comp",
        type=str,
        nargs=1,
        choices=["one", "two"],
        help="Number of compartments to include",
    )
    parser.add_argument("out", type=str, nargs=1, help="Root for file output")
    parser.add_argument(
        "-fdg",
        action="store_const",
        const=[True],
        default=[False],
        help="Convert aif from whole-blood FDG to plasma FDG."
        + " Only applies for two-compartment model",
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
        "-vol",
        action="store_const",
        const=[True],
        default=[False],
        help="Add term for blood volume correction. Default is no correction.",
    )
    args = parser.parse_args()

    # Use ggplot style for all plots
    plt.style.use("ggplot")

    # Define parameters to write out
    if args.comp[0] == "one":
        par_names = ["shift", "K1", "k2", "lambda"]
        par_units = ["sec", "mL/hg/min", "1/min", "mL/g"]
    else:
        par_names = ["shift", "K1", "k2", "k3", "k4"]
        par_units = ["sec", "mL/hg/min", "1/min", "1/min", "1/min"]

    # Add in blood volume
    if args.vol[0] is True:
        par_names.append("Vb")
        par_units.append("mL/hg")

    # Load in aif and tac data
    aif = ppg.io.txt_to_tac(args.aif[0])
    pet = ppg.io.txt_to_tac(args.pet[0])

    # Limit analysis to valid times
    time_msk = ppg.util.gen_time_mask(pet, limit=args.limit[0])
    pet = ppg.Tac(pet.time[time_msk], pet.cnt[time_msk])

    # Compute peak times for aif and pet bolus
    aif_pt, aif_p = ppg.util.loc_peak(aif, tac_interp=False)
    pet_bolus = ppg.util.bolus_approx(pet)

    # Estimate rough shift
    init_shift = aif_pt - pet_bolus

    # Figure out minimum and maximum shift
    if args.fdg[0] is True:
        # For FDG, use a pretty tight window
        min_shift = init_shift
        max_shift = init_shift + 180

    else:
        # Get pet peak time
        pet_pt, pet_p = ppg.util.loc_peak(pet)

        # Define offset from shift to search from
        if pet_bolus == 0:
            init_off = 60
        else:
            init_off = 20

        # Set minimum/max values so that aif can't be to the right of the tac bolus
        if aif_pt <= pet_pt:
            min_shift = aif_pt - pet_bolus
            max_shift = init_shift + init_off
        else:
            min_shift = init_shift - init_off
            max_shift = aif_pt - pet_bolus

    # Create model for shift optimization
    if args.comp[0] == "one":
        comp_model = ppg.pet_model.OneComp(aif, pet, vol=args.vol[0])
    else:
        comp_model = ppg.pet_model.TwoComp(aif, pet, vol=args.vol[0], fdg=args.fdg[0])

    # Search for best shift on a grid with 1 second spacing
    rough_grid = (slice(min_shift, max_shift, 1),)
    rough_shift = opt.brute(comp_model.cost, rough_grid, finish=None)

    # Run optimization on a much finer grid
    fine_grid = (slice(rough_shift - 1.5, rough_shift + 1.5, 0.01),)
    shift = opt.brute(comp_model.cost, fine_grid, finish=None)

    # Write out shifted input function (just shift the times back)
    aif_out = np.stack((aif.time - shift, aif.cnt), axis=1)
    np.savetxt(f"{args.out[0]}.csv", aif_out, delimiter=",")

    # Write out parameter estimates
    coef_opt = comp_model.coef(shift)
    par_out = np.insert(comp_model.unit_conv(coef_opt), 0, shift)
    ppg.io.write_pars(par_out, par_names, par_units, f"{args.out[0]}_vals.csv")

    # Get model predictions
    pet_hat = comp_model.pred(shift)

    # Make figure
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(14, 6))

    # Make model fit figure
    ax_1.scatter(pet.time, pet.cnt, s=40, c="black")
    ax_1.plot(pet.time[comp_model.pet.mask], pet_hat, lw=2, c="#006bb6")
    ax_1.set_xlabel("Time (seconds)", fontsize=10, color="black", fontweight="bold")
    ax_1.set_ylabel(
        f"Pet Activity ({pet.c_unit})", fontsize=10, color="black", fontweight="bold"
    )
    ax_1.tick_params(labelsize=9)
    ax_1.set_title("Fit", fontsize=12, fontweight="bold")

    # Make a plot showing normalized tac with shifted aif
    ax_2.plot(pet.time, pet.cnt / np.max(pet.cnt), lw=2, c="#006bb6", label="Tac")
    ax_2.plot(
        aif.time - shift,
        aif.cnt / np.max(aif.cnt),
        lw=2,
        c="#b6006b",
        label="Shifted Aif",
    )
    ax_2.set_xlabel("Time (seconds)", fontsize=10, color="black", fontweight="bold")
    ax_2.set_ylabel(
        "Peak Normalized Activity", fontsize=10, color="black", fontweight="bold"
    )
    ax_2.set_ylim([-0.1, 1.1])
    ax_2.tick_params(labelsize=9)
    ax_2.set_title("Activity Comparison", fontsize=12, fontweight="bold")
    ax_2.legend(fontsize=9)

    # Add overall title
    plt.suptitle(
        f"{os.path.basename(args.out[0])}: Aif Shift", fontsize=14, fontweight="bold"
    )

    # Save figure
    fig.savefig(f"{args.out[0]}.tiff", bbox_inches="tight")
    plt.close()

    # Save arguments and go home
    ppg.io.write_args(args, f"{args.out[0]}_args.json")


if __name__ == "__main__":
    main()
