#!/usr/bin/python
"""
Main Tac module class for ppg module
"""

#Load those libraries
import numpy as np
import scipy.interpolate as interp

class Tac:
    """
    Class for time activity objects
    """

    def __init__(self, time, cnt, dc=None, h_life=None, 
                 t_unit='seconds', c_unit='Bq/mL'):
        """
        Creates a 1-d time activity boject
        
        Parameters
        ----------
        time: array
            A p length array of sampling times
        cnt: array
            A p length array of counts
        dc: bool
            True indicates data has been decay corrected
        h_life: float
            Half life of tracer
        t_unit: string
            Time unit
        c_unit: string
            Count unit
    
        Returns
        -------
        tac: object
            A tac model object
        """

        #Make sure input dimensions match
        if time.shape[0] != cnt.shape[0]:
            raise ValueError('Dimension of time and cnt must match')

        #Make sure we are only 1d
        if time.ndim != 1 or cnt.ndim != 1:
            raise ValueError('Tac object must be 1-d')

        #Add model parts parts
        self.time = time
        self.cnt = cnt
        self.dc = dc
        self.h_life = h_life
        self.n = time.shape[0]
        self.c_unit = c_unit
        self.t_unit = t_unit

        #Determine if we have uniform sampling
        uniq_samps = np.unique(np.round(np.gradient(self.time)))
        if uniq_samps.shape[0] == 1:
            self.unif = True
            self.samp = uniq_samps[0]
        else:
            self.unif = False
        
    def decay_flip(self):
        """
        Flips the decay status of tac model object
        """

        if self.dc is not None and self.h_life is not None:
        
            #Remove decay correction
            dc_factor = np.exp(np.log(2)/self.h_life*self.time)
            if self.dc is True:
                self.cnt = self.cnt / dc_factor
                self.dc = False

            #Apply decay correction
            else:
                self.cnt = self.cnt * dc_factor
                self.dc = True

