#!/usr/bin/python
"""
Model classes  pet time activity curves in the ppg module
"""

# Load libraries
import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import scipy.optimize as opt
from . import util


class PetModel:
    """
    Defines generic PET model class
    """

    def __init__(self, aif, pet, name=None, algo="trapz"):
        """
        Initialize generic PET model

        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        algo: str
            Integration rule for models that use util.exp_conv: "trapz"
            (default, fast) or "simpson" (more accurate, ~3-4x slower per
            call -- worth it for a single whole-brain fit, usually not
            worth it inside a per-voxel optimization loop)
        """

        # Make sure decay correction status is the same
        if aif.dc != pet.dc:
            raise ValueError("Inputs must have the same decay status")

        # Add input to model objection
        self.aif = aif
        self.pet = pet
        self.name = name
        self.algo = algo

    def cost(self, params, deriv=False, hess=False):
        """
        Computes cost of model given parameters

        Parameters
        ----------
        params: array
            A n x 1 array containing model parameters
        deriv: bool
            If True, also return the analytical gradient of cost with
            respect to params, via the model's pred (only implemented by
            models that support it, e.g. Fdg)
        hess: bool
            If True (requires deriv=True), also return the analytical
            Hessian of cost with respect to params, via the model's pred --
            used to build a standard-error estimate for the fitted params

        Returns
        -------
        cost: float
            Sum of sqaures errors given parameters
        grad: array
            Only returned if deriv is True. Gradient of cost wrt params
        hess: array
            Only returned if hess is True. Hessian of cost wrt params
        """

        if hess is True and deriv is False:
            raise ValueError("hess=True requires deriv=True")

        # Get model prediction, along with its Jacobian/Hessian if
        # requested
        if hess is True:
            self.hat, jac, model_hess = self.pred(params, deriv=True, hess=True)
        elif deriv is True:
            self.hat, jac = self.pred(params, deriv=True)
        else:
            self.hat = self.pred(params)

        # Compute residuals
        if hasattr(self.pet, "mask") is True:
            self.resid = self.pet.cnt[self.pet.mask] - self.hat
        else:
            self.resid = self.pet.cnt - self.hat

        # Compute sse
        cost = np.sum(np.power(self.resid, 2))

        if deriv is False:
            return cost

        grad = -2.0 * self.resid @ jac

        if hess is False:
            return cost, grad

        # d^2(sse)/d(params)^2 = 2*(J^T J - sum_k(resid_k * model_hess_k))
        cost_hess = 2.0 * (
            jac.T @ jac - np.einsum("k,kij->ij", self.resid, model_hess)
        )
        return cost, grad, cost_hess

    def se(self, params, unit_conv_kwargs=None):
        """
        Estimates standard errors for unit_conv's output parameters via
        the delta method, propagated from the fit's parameter covariance
        2*sigma^2*inv(cost's analytical Hessian). Only usable for models
        whose pred implements hess=True (currently just Fdg).

        Parameters
        ----------
        params: array
            Converged fit parameters
        unit_conv_kwargs: dict
            Extra keyword arguments to pass through to unit_conv (e.g.
            glu, lc)

        Returns
        -------
        se: array
            Standard errors, in the same order/units as unit_conv's
            output. NaN if the parameter covariance couldn't be estimated
            (a singular Hessian, e.g. for a degenerate fit)
        """

        if unit_conv_kwargs is None:
            unit_conv_kwargs = {}

        base = self.unit_conv(params, **unit_conv_kwargs)

        sse, _, hess = self.cost(params, deriv=True, hess=True)
        n_par = len(params)
        sigma_sq = sse / (self.resid.shape[0] - n_par)

        try:
            cov = 2.0 * sigma_sq * np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return np.full(base.shape[0], np.nan)

        # Propagate to unit_conv's output space via the delta method. A
        # near-singular Hessian -- common for a weakly identified
        # parameter, e.g. k4 in the reversible model over a short scan --
        # can invert to a matrix that isn't quite a valid (positive
        # semi-definite) covariance, since inv() doesn't raise on a merely
        # ill-conditioned matrix; propagated through a PSD covariance, no
        # output's variance can come out negative, so a negative one below
        # just means that particular output inherited the bad direction --
        # np.sqrt leaves it (silently) NaN while unaffected outputs still
        # get a real number.
        eps = 1e-8
        jac = np.zeros((base.shape[0], n_par))
        for i in range(n_par):
            p_plus = np.array(params, dtype=float)
            p_plus[i] += eps
            p_minus = np.array(params, dtype=float)
            p_minus[i] -= eps
            jac[:, i] = (
                self.unit_conv(p_plus, **unit_conv_kwargs)
                - self.unit_conv(p_minus, **unit_conv_kwargs)
            ) / (2.0 * eps)

        cov_out = jac @ cov @ jac.T
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(cov_out))


def fdg_ab_to_rates(alpha1, alpha2, beta1, beta2=0.0):
    """
    Converts the alpha/beta (coefficient/rate) parameterization fit by Fdg
    back to physical FDG two-compartment rate constants.

    Parameters
    ----------
    alpha1, alpha2, beta1: float
        Parameterization fit by Fdg
    beta2: float
        Second rate, fit by Fdg when k4 is True. Omit (or 0.0) for the no
        k4 model -- k2*k4=0 and k2+k3+k4=beta1+beta2 both still hold with
        beta2=0, so the same formula recovers k4=0.0 without a special case.

    Returns
    -------
    K1, k2, k3, k4: float
        Physical rate constants
    """

    K1 = alpha1 + alpha2
    k2 = (alpha1 * beta1 + alpha2 * beta2) / K1
    k4 = beta1 * beta2 / k2
    k3 = beta1 + beta2 - k2 - k4

    return K1, k2, k3, k4


class Fdg(PetModel):
    """
    Defines the 2 compartment fdg model with blood volume, with or without
    a k4 (reversible) term

    Fit in the alpha/beta (coefficient/rate) parameterization -- params
    are [alpha1, alpha2, beta1, beta2, vb] (k4 True) or
    [alpha1, alpha2, beta1, vb] (k4 False), not the physical rate
    constants K1/k2/k3/k4. This keeps pred linear in alpha/exponential-in-
    beta, so the optimizer's Jacobian is a short
    closed-form expression instead of a finite difference through the
    physical -> eigenvalue solve. See fdg_ab_to_rates for conversion to
    K1, k2, k3, k4.
    """

    def __init__(self, aif, pet, k4=False, hct=None, algo="trapz"):
        """
        Initialize model object for two compartment model

        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        k4: boolean
            True fits a k4 (reversible) term; params are then
            [alpha1, alpha2, beta1, beta2, vb] instead of
            [alpha1, alpha2, beta1, vb]
        hct: float
            Subject hematocrit (0-1). If given, converts the whole-blood
            aif to plasma (Phelps et al., 1979) for the tissue-uptake
            term; the blood-volume term always uses whole blood. If None,
            no conversion is applied.
        algo: str
            Integration rule for util.exp_conv: "trapz" or "simpson"
        """

        # Add input to model objection
        name = "FDG with k4" if k4 is True else "FDG without k4"
        PetModel.__init__(self, aif, pet, name=name, algo=algo)
        self.k4 = k4
        self.hct = hct

        # Convert whole blood to plasma for the tissue-uptake term if
        # given a hematocrit; the blood-volume term always uses whole blood
        if self.hct is not None:
            t_min = self.aif.time / 60.0
            rbc_to_plasma = (
                0.814101
                + 0.000680 * t_min
                + 0.103307 * (1.0 - np.exp(-t_min / 50.052431))
            )
            self.aif.plasma = self.aif.cnt / (
                self.hct * rbc_to_plasma + (1.0 - self.hct)
            )
        else:
            self.aif.plasma = self.aif.cnt

    def _split(self, params):
        """
        Splits params into alpha1, alpha2, beta1, beta2, vb, with beta2
        fixed at 0.0 (the trapped compartment's constant, rate-less term)
        when this model doesn't fit a k4
        """

        if self.k4 is True:
            alpha1, alpha2, beta1, beta2, vb = params
        else:
            alpha1, alpha2, beta1, vb = params
            beta2 = 0.0

        return alpha1, alpha2, beta1, beta2, vb

    def pred(self, params, deriv=False, hess=False):
        """
        Generates predictions for the 2 compartment fdg model, optionally
        along with their Jacobian and Hessian with respect to params.

        Of hat's second partials, only four kinds are ever nonzero (every
        alpha-alpha, beta-beta cross, and vb-vb partial is exactly 0 since
        each exponential term only depends on its own coef/rate, and hat
        is linear in vb): d/dalpha_i d/dbeta_i, d/dbeta_i d/dbeta_i,
        d/dalpha_i d/dvb, and d/dbeta_i d/dvb. All but the beta_i-beta_i
        one come straight out of the Jacobian pieces below; the
        beta_i-beta_i one comes from exp_conv's hess=True output.

        Parameters
        ----------
        params: array
            [alpha1, alpha2, beta1, beta2, vb] (k4 True) or
            [alpha1, alpha2, beta1, vb] (k4 False) -- see fdg_ab_to_rates
            for the physical rate constants these correspond to
        deriv: bool
            If True, also return the Jacobian of hat wrt params
        hess: bool
            If True (requires deriv=True), also return the Hessian of hat
            wrt params

        Returns
        -------
        pred: array
            A vector of model predictions at pet times
        jac: array
            Only returned if deriv is True. A n_pet x len(params) array
        hess: array
            Only returned if hess is True. A n_pet x len(params) x
            len(params) array
        """

        if hess is True and deriv is False:
            raise ValueError("hess=True requires deriv=True")

        alpha1, alpha2, beta1, beta2, vb = self._split(params)

        # Analytically convolve the plasma input function with the
        # exponential kernel alpha1*exp(-beta1*t) + alpha2*exp(-beta2*t);
        # beta2=0 (no k4) makes the second term the trapped compartment's
        # constant contribution instead of a second exponential
        if hess is True:
            hat0, hat0_jac, hat0_rate_hess = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[alpha1, alpha2],
                rate=[beta1, beta2],
                algo=self.algo,
                deriv=True,
                hess=True,
            )
        elif deriv is True:
            hat0, hat0_jac = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[alpha1, alpha2],
                rate=[beta1, beta2],
                algo=self.algo,
                deriv=True,
            )
        else:
            hat0 = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[alpha1, alpha2],
                rate=[beta1, beta2],
                algo=self.algo,
            )

        hat_full = (1.0 - vb) * hat0 + self.aif.cnt * vb
        hat = util.resample(self.aif.time, hat_full, self.pet.time)

        if deriv is False:
            return hat

        # hat0_jac columns are [dalpha1, dalpha2, dbeta1, dbeta2]. When
        # there's no k4, beta2 isn't a free parameter (it's fixed at 0) so
        # its always-zero column is dropped.
        n_par = len(params)
        n_rate = 4 if self.k4 is True else 3
        jac_full = np.empty((hat0.shape[0], n_par))
        jac_full[:, 0:n_rate] = (1.0 - vb) * hat0_jac[:, 0:n_rate]
        jac_full[:, -1] = self.aif.cnt - hat0
        jac = np.stack(
            [
                util.resample(self.aif.time, jac_full[:, i], self.pet.time)
                for i in range(n_par)
            ],
            axis=1,
        )

        if hess is False:
            return hat, jac

        # d/dalpha_i d/dvb = -T_i, for both alpha1 and alpha2 regardless of
        # whether alpha2's rate (beta2) is a fit parameter
        alpha = (alpha1, alpha2)
        n_terms = 2 if self.k4 is True else 1  # number of *fit* rate terms
        beta_idx = 2  # index of beta1 in params -- same for k4 True/False
        hess_full = np.zeros((hat0.shape[0], n_par, n_par))
        for i in range(2):
            hess_full[:, i, -1] = hess_full[:, -1, i] = -hat0_jac[:, i]

        for i in range(n_terms):
            b = beta_idx + i
            t_i_prime_scaled = hat0_jac[:, 2 + i]  # = alpha_i * T_i'

            hess_full[:, i, b] = hess_full[:, b, i] = (1.0 - vb) * (
                t_i_prime_scaled / alpha[i]
            )
            hess_full[:, b, b] = (1.0 - vb) * hat0_rate_hess[:, i]
            hess_full[:, b, -1] = hess_full[:, -1, b] = -t_i_prime_scaled

        hess_out = np.zeros((self.pet.time.shape[0], n_par, n_par))
        for i in range(n_par):
            for j in range(i, n_par):
                if np.any(hess_full[:, i, j] != 0.0):
                    resampled = util.resample(
                        self.aif.time, hess_full[:, i, j], self.pet.time
                    )
                    hess_out[:, i, j] = resampled
                    hess_out[:, j, i] = resampled

        return hat, jac, hess_out

    def init_par(self, y):
        """
        Computes initial alpha/beta parameter estimates via the linearized
        operational equation (Feng et al., 1995, IEEE TMI), solved with
        NNLS. Meant as a per-voxel warm start that's usually much better
        than reusing a neighboring voxel's converged fit.

        Parameters
        ----------
        y: array
            Observed PET time activity curve, at pet.time sampling

        Returns
        -------
        init: array or None
            [alpha1, alpha2, beta1, beta2, vb] (k4 True) or
            [alpha1, alpha2, beta1, vb] (k4 False), or None if the
            linearized solve produced a non-physical result
        """

        # Resample the input functions onto the pet sampling grid, since y
        # is only observed there
        cp = util.resample(self.aif.time, self.aif.plasma, self.pet.time)
        cb = util.resample(self.aif.time, self.aif.cnt, self.pet.time)

        # Build the design matrix for the (twice-integrated) operational
        # equation: y = theta1*int(cp) + theta2*int2(cp) - theta3*int(y)
        # [- theta4*int2(y)] + theta5*cb, with all thetas >= 0
        int_cp = integ.cumulative_trapezoid(cp, self.pet.time, initial=0.0)
        int2_cp = integ.cumulative_trapezoid(int_cp, self.pet.time, initial=0.0)
        int_y = integ.cumulative_trapezoid(y, self.pet.time, initial=0.0)

        if self.k4 is True:
            int2_y = integ.cumulative_trapezoid(int_y, self.pet.time, initial=0.0)
            design = np.stack((int_cp, int2_cp, -int_y, -int2_y, cb), axis=1)
        else:
            design = np.stack((int_cp, int2_cp, -int_y, cb), axis=1)

        theta, _ = opt.nnls(design, y)
        theta1, vb = theta[0], theta[-1]

        # theta1 = (1-vb)*K1 -- bail out to None on any non-physical solve
        if theta1 <= 0 or vb >= 1.0:
            return None

        # Unlike this model's alpha1/alpha2, theta1 above is scaled by
        # (1-vb) -- divide it back out to match pred's params
        K1 = theta1 / (1.0 - vb)
        k34_sum = theta[1] / theta1

        if self.k4 is True:
            # theta3 = k2+k3+k4, theta4 = k2*k4
            disc = np.power(theta[2], 2) - 4.0 * theta[3]
            if disc < 0:
                return None

            beta1 = (theta[2] - np.sqrt(disc)) / 2.0
            beta2 = (theta[2] + np.sqrt(disc)) / 2.0
            if beta2 - beta1 < 1e-12:
                return None

            d = K1 / (beta2 - beta1)
            alpha1 = d * (k34_sum - beta1)
            alpha2 = d * (beta2 - k34_sum)
            if alpha1 <= 0 or alpha2 <= 0:
                return None

            return np.array([alpha1, alpha2, beta1, beta2, vb])

        # theta3 = k2+k3 (=beta1)
        beta1 = theta[2]
        if beta1 <= 0:
            return None

        k3 = k34_sum
        k2 = beta1 - k3
        if k2 <= 0:
            return None

        alpha1 = K1 * k2 / beta1
        alpha2 = K1 * k3 / beta1

        return np.array([alpha1, alpha2, beta1, vb])

    def comp(self, params):
        """
        Generates predictions for individual model components

        Parameters
        ----------
        params: array
            [alpha1, alpha2, beta1, beta2, vb] (k4 True) or
            [alpha1, alpha2, beta1, vb] (k4 False)

        Returns
        -------
        comps: list
           A list containing model component vectors
        """

        alpha1, alpha2, beta1, beta2, vb = self._split(params)

        # Compute blood volume piece
        c_p = self.aif.cnt * vb

        if self.k4 is True:
            # Analytically convolve the plasma input function with the
            # kernel for each compartment
            K1, _, k3, k4 = fdg_ab_to_rates(alpha1, alpha2, beta1, beta2)
            d = K1 / (beta2 - beta1)
            coef_e1 = d * (k4 - beta1)
            coef_e2 = d * (beta2 - k4)
            c_e = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[coef_e1, coef_e2],
                rate=[beta1, beta2],
                algo=self.algo,
            )

            coef_m = d * k3
            c_m = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[coef_m, -coef_m],
                rate=[beta1, beta2],
                algo=self.algo,
            )
        else:
            # Ce is the full K1=(alpha1+alpha2) exponential; Cm is what's
            # left once Ce is subtracted from the total prediction
            c_e = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[alpha1 + alpha2],
                rate=[beta1],
                algo=self.algo,
            )
            c_m = util.exp_conv(
                self.aif.time,
                self.aif.plasma,
                coef=[alpha2, -alpha2],
                rate=[0.0, beta1],
                algo=self.algo,
            )

        # Interpolate all the parts
        c_p_i = util.resample(self.aif.time, c_p, self.pet.time)
        c_e_i = util.resample(self.aif.time, c_e, self.pet.time)
        c_m_i = util.resample(self.aif.time, c_m, self.pet.time)

        # Return list with components
        return np.stack((c_p_i, c_e_i * (1.0 - vb), c_m_i * (1.0 - vb)), axis=1)

    def unit_conv(self, params, glu=None, lc=0.65):
        """
        Converts model parameters to physiological measurements

        Parameters
        ----------
        params: array
            [alpha1, alpha2, beta1, beta2, vb] (k4 True) or
            [alpha1, alpha2, beta1, vb] (k4 False)
        glu: float
            Plasma glucose level in mg/dL
        lc: float
            Lumped constant for FDG

        Returns
        -------
        meas: array
            A vector of metabolic parameters
        """

        alpha1, alpha2, beta1, beta2, vb = self._split(params)
        K1, k2, k3, k4 = fdg_ab_to_rates(alpha1, alpha2, beta1, beta2)

        # Convert rate constants and volumes to standard units
        K1 = K1 * 60.0 / 1.05 * 100.0
        k2 = k2 * 60.0
        k3 = k3 * 60.0
        k4 = k4 * 60.0
        vb = vb * 100.0
        ki = (K1 * k3) / (k2 + k3)

        # Make parameter list for output
        if self.k4 is True:
            vt = (K1 / k2) * (1 + (k3 / k4))
            meas = np.array([K1, k2, k3, k4, ki, vt, vb])
        else:
            vt = K1 / k2
            meas = np.array([K1, k2, k3, ki, vt, vb])

        # Parameters that require plasma glucose level
        if glu is not None:
            # Convert plasma glucose level to uMol/ml
            glu_conv = glu / 18.0156

            # See if we need to compute lc
            if lc < 0:
                lc = 0.39 + (1.48 - 0.39) * (ki / K1)

            # Compute additional parameters
            cmr = ki * glu_conv / lc
            influx = K1 * glu_conv
            conc = K1 / k2 * glu_conv

            # Add terms
            meas = np.append(meas, [cmr, influx, conc])

        return meas


class FlowTwo(PetModel):
    """
    Defines 2 paramter, 1 compartment blood flow model
    """

    def __init__(self, aif, pet, algo="trapz"):
        """
        Initialize model object for two compartment model

        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        algo: str
            Integration rule for util.exp_conv: "trapz" or "simpson"
        """

        # Initialize model
        PetModel.__init__(self, aif, pet, name="Flow Model", algo=algo)

    def pred(self, params):
        """
        Generates 2 param, 1 compartment model predictions

        Parameters
        ----------
        params: array
            A 2 x 1 array contaning K1 and k2

        Returns
        -------
        pred: array
            A vector of model predictions at pet times
        """

        # Rename paramters
        K1 = params[0]
        k2 = params[1]

        # Analytically convolve the input function with K1*exp(-k2*t)
        hat = util.exp_conv(
            self.aif.time, self.aif.cnt, coef=[K1], rate=[k2], algo=self.algo
        )

        # Interpolate the model prediction at tac sampling time
        return util.resample(self.aif.time, hat, self.pet.time)

    def unit_conv(self, params):
        """
        Converts model parameters to physiological measurements

        Parameters
        ----------
        params: array
            A 2 x 1 array contaning K1 and k2

        Returns
        -------
        meas: array
            A vector of metabolic parameters
        """

        # Compute cbf, k2, and the blood-brain partion coefficient
        cbf = params[0] * 6000.0 / 1.05  # mL/hg/min
        k2 = params[1] * 60.0  # 1/min
        lmbda_w = cbf / k2 / 100.0  # mL/g

        return np.array([cbf, k2, lmbda_w])


class OhtaTwo(PetModel):
    """
    Defines Ohta two compartment model
    """

    def __init__(self, aif, pet, algo="trapz"):
        """
        Initialize model object for Ohta model

        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        algo: str
            Integration rule for util.exp_conv: "trapz" or "simpson"
        """

        # Initialize model
        PetModel.__init__(self, aif, pet, name="Ohta Two Compartment", algo=algo)

    def pred(self, params):
        """
        Generates predictions for Ohta model

        Parameters
        ----------
        params: array
            A 3 x 1 array contaning K1, k2, and v0

        Returns
        -------
        pred: array
            A vector of model predictions at pet times
        """

        # Rename paramters
        K1 = params[0]
        k2 = params[1]
        v0 = params[2]

        # Analytically convolve the input function with K1*exp(-k2*t)
        hat = util.exp_conv(
            self.aif.time, self.aif.cnt, coef=[K1], rate=[k2], algo=self.algo
        )
        hat += self.aif.cnt * v0

        # Interpolate the model prediction at tac sampling time
        return util.resample(self.aif.time, hat, self.pet.time)

    def unit_conv(self, params, art=None):
        """
        Converts model parameters to physiological measurements

        Parameters
        ----------
        params: array
            A 3 x 1 array K1, k2, and v0
        art: float
            Arterial blood concentration in uMol/mL

        Returns
        -------
        meas: array
            A vector of metabolic parameters
        """

        # Compute cbf, k2, blood-brain partion coefficient, and blood volume
        K1 = params[0] * 6000.0 / 1.05  # mL/hg/min
        k2 = params[1] * 60.0  # 1/min
        lmbda = K1 / k2 / 100.0  # mL/g
        v0 = params[2] * 105.0  # mL/hg

        # Add in cmr if necessary
        if art is None:
            return np.array([K1, k2, lmbda, v0])
        else:
            return np.array([K1, k2, lmbda, v0, K1 * art])


class OneComp(PetModel):
    """
    Defines 2 paramter, 1 compartment with shift
    """

    def __init__(self, aif, pet, vol=False):
        """
        Initialize model object for one compartment model

        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        vol: boolean
            If true, add in a term for blood volume correction
        """

        # Initialize model
        PetModel.__init__(self, aif, pet, name="One Compartment")
        self.vol = vol

        # Integration of aif and pet
        self.aif.int = integ.cumulative_trapezoid(
            self.aif.cnt, self.aif.time, initial=0.0
        )
        self.pet.int = integ.cumulative_trapezoid(
            self.pet.cnt, self.pet.time, initial=0.0
        )

        # Create interpolation function for input function and its integral
        self.aif.func_int = interp.interp1d(self.aif.time, self.aif.int, kind="cubic")
        if self.vol is True:
            self.aif.func = interp.interp1d(self.aif.time, self.aif.cnt, kind="cubic")

    def coef(self, shift):
        """
        Compute coefficients for linear version of one compartment model

        Parameters
        ----------
        shift: float
            Value to shift input function by

        Returns
        -------
        coefs: array
            Values for K1, k2, and blood volume if necssary
        """

        # Determine times to shift function to
        self.aif.shift_time = self.aif.time - shift
        self.pet.shift_time = self.pet.time + shift

        # Make interpolation masks
        self.pet.shift_mask = np.logical_and(
            self.pet.shift_time >= self.aif.time[0],
            self.pet.shift_time <= self.aif.time[-1],
        )
        self.pet.mask = np.logical_and(
            self.pet.time >= self.aif.shift_time[0],
            self.pet.time <= self.aif.shift_time[-1],
        )

        # Interpolate the aif and its integral to pet sampling with shift
        self.aif.int_i = self.aif.func_int(self.pet.shift_time[self.pet.shift_mask])

        # Add in blood volume if necessary
        if self.vol is False:
            # Make design matrix without blood volume
            self.x = np.stack(
                (self.aif.int_i, -1.0 * self.pet.int[self.pet.mask]), axis=1
            )

        else:
            # Interpolate input function to pet sampling with shift
            self.aif.cnt_i = self.aif.func(self.pet.shift_time[self.pet.shift_mask])

            # Make design matrix with blood volume
            self.x = np.stack(
                (self.aif.int_i, -1.0 * self.pet.int[self.pet.mask], self.aif.cnt_i),
                axis=1,
            )

        # Get non-negative least squares solution
        coefs, _ = opt.nnls(self.x, self.pet.cnt[self.pet.mask])

        return coefs

    def pred(self, shift):
        """
        Generates predicition for one compartment linear model

        Parameters
        ----------
        shift: float
            Value to shift input function by

        Returns
        -------
        pred: array
            A vector of model predictions at pet times
        """

        # Compute coefficients
        beta = self.coef(shift)

        # Compute model prediction
        return self.x.dot(beta)

    def unit_conv(self, params):
        """
        Converts model parameters to physiological measurements

        Parameters
        ----------
        params: array
            An array containing K1, k2, and possibly vb

        Returns
        -------
        meas: array
            A vector of metabolic parameters
        """

        # Compute cbf, k2, and the blood-brain partion coefficient
        K1 = params[0] * 6000.0 / 1.05  # mL/hg/min
        k2 = params[1] * 60.0  # 1/min
        lmbda_w = K1 / k2 / 100.0  # mL/g

        # Deal with possible blood volume
        if self.vol is True:
            return np.array([K1, k2, lmbda_w, params[2] / 1.05 * 100])
        else:
            return np.array([K1, k2, lmbda_w])


class OxyOne(PetModel):
    """
    Defines 1 paramter, 2 Mintun oxygen consumption model
    """

    def __init__(self, aif_oxy, aif_water, pet, flow, k2, vb, algo="trapz"):
        """
        Initialize model object for oxygen consumpution model

        Parameters
        ----------
        aif_oxy: Tac object
            Tac object containing samples for oxygen input function
        aif_water: Tac object
            Tac object containing samples for water input function
        pet: Tac object
            Tac object containing the pet data
        flow: float
            Blood flow in mL/mL/sec
        k2: float
            Efflux term in 1/sec
        vb: float
            Blood volume in mL/mL
        algo: str
            Integration rule for util.exp_conv: "trapz" or "simpson"
        """

        # Make sure decay correction status is the same
        if aif_oxy.dc != aif_water.dc:
            raise ValueError("Inputs must have the same decay status")

        # Add input to model objection
        PetModel.__init__(self, aif_oxy, pet, name="Oxygen Model", algo=algo)
        self.aif_oxy = aif_oxy
        self.aif_water = aif_water
        self.flow = flow
        self.k2 = k2
        self.vb = vb
        self.ratio = 0.85

        # Analytically convolve water and oxygen input functions with the
        # fixed flow*exp(-k2*t) kernel
        conv_water = util.exp_conv(
            self.aif_water.time,
            self.aif_water.cnt,
            coef=[self.flow],
            rate=[self.k2],
            algo=self.algo,
        )
        conv_oxy = util.exp_conv(
            self.aif_oxy.time,
            self.aif_oxy.cnt,
            coef=[self.flow],
            rate=[self.k2],
            algo=self.algo,
        )

        # Generate blood volume term
        b_vol = self.ratio * self.vb * self.aif_oxy.cnt

        # Interpolate the model terms
        self.conv_water_i = util.resample(self.aif_oxy.time, conv_water, self.pet.time)
        self.conv_oxy_i = util.resample(self.aif_oxy.time, conv_oxy, self.pet.time)
        self.b_vol_i = util.resample(self.aif_oxy.time, b_vol, self.pet.time)

    def pred(self, oef):
        """
        Generates prediction for 1 parameter, 2 compartment oxygen model

        Parameters
        ----------
        oef: float
            Oxygen extraction fraction

        Returns
        -------
        pred: array
            A vector of model predictions at pet times
        """

        # Compute model prediction
        hat = (
            self.conv_water_i
            + self.conv_oxy_i * oef
            + (1.0 - 0.835 * oef) * self.b_vol_i
        )

        # Interpolate the model prediction at tac sampling time
        return hat

    def unit_conv(self, oef, ca=None):
        """
        Generates prediction for 1 parameter, 2 compartment oxygen model

        Parameters
        ----------
        oef: float
            Oxygen extraction fraction
        ca: float
            Oxygen content of arterial blood in mL/dL

        Returns
        -------
        meas: array
            A vector of metabolic parameters
        """

        # Compute cmro2 if given oxygen concentration in arterial blood
        if ca is not None:
            return np.array([oef, oef * self.flow * ca])
        else:
            return np.array([oef])


class TwoComp(PetModel):
    """
    Defines 4 paramter, 2 compartment with shift
    """

    def __init__(self, aif, pet, vol=False, fdg=False):
        """
        Initialize model object for two compartment model

        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        vol: boolean
            If true, correct for blood volume
        fdg: boolean
            If true, convert from whole blood to plasma
        """

        # Add input to model objection
        PetModel.__init__(self, aif, pet, name="Two Compartment")
        self.fdg = fdg
        self.vol = vol

        # Create scale to convert to plasma
        if self.fdg is True:
            self.aif.scale = 1.071966 - 1.07294e-5 * self.aif.time
        else:
            self.aif.scale = 1.0

        # Integration of aif and pet
        self.aif.int = integ.cumulative_trapezoid(
            self.aif.cnt * self.aif.scale, self.aif.time, initial=0.0
        )
        self.pet.int = integ.cumulative_trapezoid(
            self.pet.cnt, self.pet.time, initial=0.0
        )

        # Double integrations
        self.aif.int_dbl = integ.cumulative_trapezoid(
            self.aif.int * self.aif.scale, self.aif.time, initial=0.0
        )
        self.pet.int_dbl = integ.cumulative_trapezoid(
            self.pet.int, self.pet.time, initial=0.0
        )

        # Create aif interpolation functions
        self.aif.func_int = interp.interp1d(self.aif.time, self.aif.int, kind="cubic")
        self.aif.func_int_dbl = interp.interp1d(
            self.aif.time, self.aif.int_dbl, kind="cubic"
        )
        if self.vol is True:
            self.aif.func = interp.interp1d(self.aif.time, self.aif.cnt, kind="cubic")

    def coef(self, shift):
        """
        Compute coefficients for linear version of two compartment model

        Parameters
        ----------
        shift: float
            Value to shift input function by

        Returns
        -------
        coefs: array
            Values for p1, p2, p3, p4, and possibly vb
        """

        # Determine times to shift function to
        self.aif.shift_time = self.aif.time - shift
        self.pet.shift_time = self.pet.time + shift

        # Make interpolation masks
        self.pet.shift_mask = np.logical_and(
            self.pet.shift_time >= self.aif.time[0],
            self.pet.shift_time <= self.aif.time[-1],
        )
        self.pet.mask = np.logical_and(
            self.pet.time >= self.aif.shift_time[0],
            self.pet.time <= self.aif.shift_time[-1],
        )

        # Interpolate the aif integrals
        time_mskd = self.pet.shift_time[self.pet.shift_mask]
        self.aif.int_i = self.aif.func_int(time_mskd)
        self.aif.int_dbl_i = self.aif.func_int_dbl(time_mskd)

        # Account for blood volume if necessary
        if self.vol is False:
            # Make design matrix without blood
            self.x = np.stack(
                (
                    self.aif.int_i,
                    self.aif.int_dbl_i,
                    -self.pet.int[self.pet.mask],
                    -self.pet.int_dbl[self.pet.mask],
                ),
                axis=1,
            )

        else:
            # Interpolate input function to pet sampling with shift
            self.aif.cnt_i = self.aif.func(time_mskd)

            # Make design matrix with blood
            self.x = np.stack(
                (
                    self.aif.int_i,
                    self.aif.int_dbl_i,
                    -self.pet.int[self.pet.mask],
                    -self.pet.int_dbl[self.pet.mask],
                    self.aif.cnt_i,
                ),
                axis=1,
            )

        # Get non-negative least squares solution
        coefs, _ = opt.nnls(self.x, self.pet.cnt[self.pet.mask])

        return coefs

    def pred(self, shift):
        """
        Generates predicition for two compartment linear model

        Parameters
        ----------
        shift: float
            Value to shift input function by

        Returns
        -------
        pred: array
            A vector of model predictions at pet times
        """

        # Compute coefficients
        beta = self.coef(shift)

        # Compute model prediction
        return self.x.dot(beta)

    def unit_conv(self, params):
        """
        Converts model parameters to physiological measurements

        Parameters
        ----------
        params: array
            An array containing K1, k2, k3, k4, and possibly vb

        Returns
        -------
        meas: array
            A vector of metabolic parameters
        """

        # Compute rate constants
        K1 = params[0] * 6000.0 / 1.05
        k2 = params[2] - params[1] / params[0]
        k4 = params[3] / k2
        k3 = params[2] - k2 - k4

        # Convert rate constants to minutes
        k2 *= 60.0
        k3 *= 60.0
        k4 *= 60.0

        # Deal with possible blood volume
        if self.vol is True:
            return np.array([K1, k2, k3, k4, params[4] / 1.05 * 100])
        else:
            return np.array([K1, k2, k3, k4])
