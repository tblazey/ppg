#!/usr/bin/python
"""
Fits an AIF model (Feng or Golish) to a blood curve, optionally
deconvolving a measurement kernel.
"""

# Use agg backend so plots can be made in background
import matplotlib

matplotlib.use("Agg")

# Import libraries
import argparse

import numpy as np
import scipy.optimize as opt

import ppg


def main():
    # Get to parsing
    parser = argparse.ArgumentParser(description="Fits AIF model to blood curve:")
    parser.add_argument("aif", nargs=1, type=str, help="Input function csv file")
    parser.add_argument("out", nargs=1, type=str, help="Root for output files")
    parser.add_argument(
        "-kernel",
        nargs=1,
        type=str,
        metavar="path",
        help="Convolution kernel csv file. Sampling rate must match <aif>",
    )
    parser.add_argument(
        "-model",
        nargs=1,
        type=str,
        default=["Feng"],
        choices=["Feng", "Golish"],
        help="Aif model to fit. Default is Feng",
    )
    parser.add_argument(
        "-tau",
        nargs=1,
        type=float,
        help="Decay constant for negative exponential kernel." + " Overrides -kernel.",
    )
    parser.add_argument(
        "-resample",
        nargs=1,
        type=float,
        default=None,
        help="Interpolate to uniform sampling of <resample> seconds.",
    )
    args = parser.parse_args()

    # Load in input function
    aif = ppg.io.txt_to_tac(args.aif[0])

    # Kernel setup logic
    if args.tau is not None:
        # Construct negative exponential kernel
        kernel_time = np.arange(0, aif.n * aif.samp, aif.samp)
        kernel_cnt = np.exp(-kernel_time / args.tau[0]) / args.tau[0]

        # Make it into a tac object
        krn = ppg.Tac(kernel_time, kernel_cnt)

        # Set initial time offset
        t_init = 5

    elif args.kernel is not None:
        # Load in kernel
        krn = ppg.io.txt_to_tac(args.kernel[0], delimiter=",")

        # Make sure sampling rates match
        if aif.samp != krn.samp:
            raise IOError("Kernel sampling rate does not match that of aif")

        # Set initial time offset
        t_init = 25

    else:
        # Pass none instead of kernel
        krn = None

        # Set initial time offset
        t_init = 0

    # Normalize aif by its sum prior to fitting
    aif_scale = np.sum(aif.cnt)
    aif_norm = ppg.Tac(aif.time, aif.cnt / aif_scale)

    # Get aif peak for initial
    aif_pt, _ = ppg.util.loc_deriv_peak(aif)

    # Decide which model to use
    if args.model[0] == "Feng":
        opt_init = np.array([-3.00, -3.70, -6.00, 8.00, -3.25, -7.50, aif_pt - t_init])
        aif_model = ppg.aif_model.FengModel(aif_norm)
    else:
        opt_init = np.array([-3.27, 2.56, -3.32, -5.30, -0.11, aif_pt - t_init])
        aif_model = ppg.aif_model.GolishModel(aif_norm)

    # Fit aif model to data
    opt_args = (krn,)
    opt_model = opt.minimize(aif_model.cost, opt_init, args=opt_args, method="L-BFGS-B")

    # Do a global optimization to refine answer
    opt_model = opt.basinhopping(
        aif_model.cost,
        opt_model.x,
        niter=100,
        minimizer_kwargs={"args": opt_args},
        stepsize=0.1,
        T=0.1,
    )

    if args.resample is not None:
        t_new = np.arange(
            aif_model.aif.time[0],
            aif_model.aif.time[-1] + args.resample[0],
            args.resample[0],
        )
        t_out = t_new
    else:
        t_new = None
        t_out = aif.time

    # Compute model predictions
    hats = [aif_model.pred(opt_model.x, kernel=krn, t_new=t_new) * aif_scale]
    labels = ["Model Fit"]
    if args.kernel is not None or args.tau is not None:
        hats.append(aif_model.pred(opt_model.x, t_new=t_new) * aif_scale)
        labels.append("Deconvolved")

    # Make plot showing fit
    plot_path = f"{args.out[0]}.tiff"
    ppg.util.tac_plot(
        aif,
        hats=hats,
        labels=labels,
        t_hat=t_new,
        title=f"{aif_model.name} Aif Fit",
        out_path=plot_path,
    )

    # Save fitted and deconvolved values
    hat_out = np.stack((t_out, hats[0]), axis=1)
    np.savetxt(f"{args.out[0]}_fitted.csv", hat_out, delimiter=",", fmt="%.8f")
    if args.kernel is not None or args.tau is not None:
        dcv_out = np.stack((t_out, hats[1]), axis=1)
        np.savetxt(f"{args.out[0]}_dcv.csv", dcv_out, delimiter=",", fmt="%.8f")

    # Save parameter values
    par_out = np.append(aif_model.unit_conv(opt_model.x), aif_scale)
    aif_model.par_names.append("scale")
    aif_model.par_units.append("Bq/mL")
    par_path = f"{args.out[0]}_params.csv"
    ppg.io.write_pars(par_out, aif_model.par_names, aif_model.par_units, par_path)


if __name__ == "__main__":
    main()
