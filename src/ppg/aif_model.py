#!/usr/bin/python
"""
Model classes for aif time activity curves in the ppg module
"""

#Load libraries
import numpy as np
import scipy.interpolate as interp

class AifModel():
    """
    Class to create generic aif model
    """

    def __init__(self, aif, name=None):
        """
        Initialize aif object

        Parameters
        ----------
        aif: Tac object
            Tac object containing aif
        """

        #Add model parameters
        self.aif = aif
        self.name = name

    def cost(self, params, kernel=None):
        """
        Generates cost for model

        Parameters
        ----------
        params: array
            A n x 1 array containing model parameters
         kernel: Tac object
            Tac object containing convolution kernel

        Returns
        -------
        sse: float
            Sum of squares error for model
        """

        #Compute residuals
        resid = self.aif.cnt - self.pred(params, kernel=kernel)

        #Return error
        return np.sum(np.power(resid, 2))

class FengModel(AifModel):
    """
    Class for fitting Feng et al. 1993 input function model
    """

    def __init__(self, aif):
        """
        Create initial model objection

        Parameters
        ----------
        aif: Tac object
            Tac object containing aif
        """

        #Initialize model
        AifModel.__init__(self, aif, name='Feng Aif Model')
        self.par_names = ['alpha_1', 'alpha_2', 'alpha_3',
                          'eig_1', 'eig_2', 'eig_3', 't_zero']
        self.par_units = ['Bq/mL', 'Bq/mL', 'Bq/mL',
                          '1/sec', '1/sec', '1/sec', 't_zero']

    def pred(self, params, kernel=None, t_new=None):
        """
        Generates predictions for Feng model

        Parameters
        ----------
        params: array
            A array contaning natural log of Feng parameters and a shift term
            (alpha_1, alpha_2, alpha_3, eig_1, eig_2, eig_3, and t_zero)
        kernel: Tac object
            Tac object containing convolution kernel
        t_new: array
            An array for new timepoints for predictions

        Returns
        -------
        hat: array
            Array of model predictions
        """

        #Extract parameters
        alpha_1 = np.exp(params[0])
        alpha_2 = np.exp(params[1])
        alpha_3 = np.exp(params[2])
        eig_1 = np.exp(params[3])
        eig_2 = np.exp(params[4])
        eig_3 = np.exp(params[5])
        t_zero = params[6]

        #Create array to store predictions
        if t_new is None:
            t = self.aif.time
            n = self.aif.n
        else:
            t = t_new
            n = t_new.shape[0]
        samp = t[1] - t[0]
        f_pred = np.zeros(n)

        if kernel is not None:
            if kernel.samp != samp:
                kernel_cnt = interp.interp1d(
                    kernel.time, kernel.cnt, kind='linear', fill_value='extrapolate'
                )(t)
            else:
                kernel_cnt = kernel.cnt

        #Ignore values where t is less than t_zero
        t_mask = t > t_zero
        t_shift = t[t_mask] - t_zero

        #Compute model predictions
        exp_1 = np.exp(-eig_1 * t_shift)
        f_pred[t_mask] = alpha_1 * t_shift * exp_1
        f_pred[t_mask] += alpha_2 * (np.exp(-eig_2 * t_shift) - exp_1)
        f_pred[t_mask] += alpha_3 * (np.exp(-eig_3 * t_shift) - exp_1)

        #Convolve with kernel if necessary
        if kernel is not None:
            return np.convolve(f_pred, kernel_cnt)[0:n] * samp
        else:
            return f_pred

    def cost(self, params, kernel=None):
        """
        Generates cost for Feng model

        Parameters
        ----------
        params: array
            A array contaning natural log of Feng parameters and a shift term
            (alpha_1, alpha_2, alpha_3, eig_1, eig_2, eig_3, and t_zero)
         kernel: Tac object
            Tac object containing convolution kernel

        Returns
        -------
        sse: float
            Sum of squares error for model
        """

        #Compute residauls
        resid = self.aif.cnt - self.pred(params, kernel=kernel)

        #Return error
        return np.sum(np.power(resid, 2))

    def unit_conv(self, params):
        """
        Converts model units to original scale

        Parameters
        ----------
        params: array
            A array contaning natural log of Feng model parameters and a shift term
            (alpha_1, alpha_2, alpha_3, eig_1, eig_2, eig_3, and t_zero)

        Returns
        -------
        meas: array
            Feng model parameters in original scale
        """

        #Get parameters back from log land
        conv_params = np.copy(params)
        conv_params[0:6] = np.exp(conv_params[0:6])

        return conv_params

class GolishModel(AifModel):
    """
    Class for fitting Golish et al. 2001 input function model
    """

    def __init__(self, aif):
        """
        Create initial model objection

        Parameters
        ----------
        aif: Tac object
            Tac object containing aif
        """

        #Initialize model
        AifModel.__init__(self, aif, name='Golish Aif Model')
        self.par_names = ['alpha', 'beta', 'c_max',
                          'c_zero', 'tau', 't_zero']
        self.par_units = ['NA', 'sec', 'Bq/mL',
                          'Bq/mL', 'sec', 'sec']

    def pred(self, params, kernel=None, t_new=None):
        """
        Generates predictions for Golish model

        Parameters
        ----------
        params: array
            A array contaning natural log of Golish parameters and shift
            (alpha, beta, c_max, c_zero, tau, and t_zero)
        kernel: Tac object
            Tac object containing convolution kernel
        t_new: array
            An array for new timepoints for predictions

        Returns
        -------
        hat: array
            Array of model predictions
        """

        #Extract parameters
        alpha = np.exp(params[0])
        beta = np.exp(params[1])
        c_max = np.exp(params[2])
        c_zero = np.exp(params[3])
        tau = np.exp(params[4])
        t_zero = params[5]

        #Create array to store predictions
        if t_new is None:
            t = self.aif.time
            n = self.aif.n
        else:
            t = t_new
            n = t_new.shape[0]
        samp = t[1] - t[0]
        g_pred = np.zeros(n)

        #Ignore values where t is less than t_zero
        t_mask = t > t_zero
        t_shift = t[t_mask] - t_zero

        if kernel is not None:
            if kernel.samp != samp:
                kernel_cnt = interp.interp1d(
                    kernel.time, kernel.cnt, kind='linear', fill_value='extrapolate'
                )(t)
            else:
                kernel_cnt = kernel.cnt

        #Compute model predictions
        g_one = c_max * np.power(np.exp(1.0) / (alpha*beta) * t_shift, alpha)
        g_two = np.exp(-t_shift/beta)
        g_three = c_zero * (1.0 - np.exp(-t_shift/tau))
        g_pred[t_mask] = g_one * g_two + g_three

        #Convolve with kernel if necessary
        if kernel is not None:
            return np.convolve(g_pred, kernel_cnt)[0:n] * samp
        else:
            return g_pred

    def unit_conv(self, params):
        """
        Converts model units to original scale

        Parameters
        ----------
        params: array
            A array contaning natural log of Golish parameters and shift
            (alpha, beta, c_max, c_zero, tau, and t_zero)

        Returns
        -------
        meas: array
            Golish model parameters in original scale
        """

        #Convert parameters back from log land
        conv_params = np.copy(params)
        conv_params[0:6] = np.exp(conv_params[0:6])

        return conv_params
