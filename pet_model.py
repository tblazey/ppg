#!/usr/bin/python
"""
Model classes  pet time activity curves in the ppg module
"""

#Load libraries
import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import scipy.optimize as opt
from . import util
from .tac import Tac

class PetModel():
    """
    Defines generic PET model class
    """
    
    def __init__(self, aif, pet, name=None):
        """
        Initialize generic PET model
        
        Parameters
        ----------
        aif: Tac object
            Tac object containing the aif data
        pet: Tac object
            Tac object containing the pet data
        """
        
        #Make sure decay correction status is the same
        if aif.dc != pet.dc:
                raise ValueError('Inputs must have the same decay status')

        #Add input to model objection
        self.aif = aif
        self.pet = pet
        self.name = name

    def cost(self, params):
        """
        Computes cost of model given parameters

        Parameters
        ----------
        params: array
            A n x 1 array containing model parameters

        Returns
        -------
        cost: float
            Sum of sqaures errors given parameters
        """
    
        #Get model prediction
        self.hat = self.pred(params)

        #Compute residuals
        if hasattr(self.pet, 'mask') is True:
            self.resid = self.pet.cnt[self.pet.mask] - self.hat       
        else:
            self.resid = self.pet.cnt - self.hat
         
        #Compute sse
        return np.sum(np.power(self.resid, 2))

class FdgFour(PetModel):
        """
        Defines 4 parameter, 2 compartment fdg model with blood volume
        """

        def __init__(self, aif, pet, plasma=True):
            """
            Initialize model object for two compartment model
        
            Parameters
            ----------
            aif: Tac object
                Tac object containing the aif data
            pet: Tac object
                Tac object containing the pet data
            plasma: boolean
                True converts input function to plasma
            """

            #Add input to model objection
            PetModel.__init__(self, aif, pet, name="FDG with k4")
            self.plasma = plasma

            #Make time range for kernel
            self.kernel_time = np.arange(0, self.aif.n * self.aif.samp, self.aif.samp) 

            #Get plasma version of input function if necessary
            if self.plasma is True:
                self.aif.plasma = (1.071966 - 1.07294E-5 * self.aif.time) * self.aif.cnt
            else:
                self.aif.plasma = self.aif.cnt

        def pred(self, params):
            """
            Generates predictions for 2 compartment model with k4

            Parameters
            ----------
            params: array
                A 5 x 1 array containing K1, vd, k3, k4, and vb

            Returns
            -------
            pred: array
                A vector of model predictions at pet times
            """

            #Rename params
            K1 = params[0]
            k3 = params[2]
            k4 = params[3]
            vb = params[4]
            k2 = K1 / params[1] - k3

            #Compute alpha terms for model
            k_sum = k2 + k3 + k4
            k_sqrt = np.sqrt(np.power(k_sum, 2) - 4.0 * k2 * k4)
            a1 = (k_sum - k_sqrt) / 2.0
            a2 = (k_sum + k_sqrt) / 2.0
        
            #Compute model exponential terms
            exp_1 = np.exp(-a1 * self.kernel_time) * (k3 + k4 - a1)
            exp_2 = np.exp(-a2 * self.kernel_time) * (a2 - k3 - k4)

            #Compute expoential kernel
            kernel = K1 / (a2 - a1) * (exp_1 + exp_2) * self.aif.samp
        
            #Compute model prediction   
            hat = (1.0 - vb) * np.convolve(self.aif.plasma, kernel)[0:self.aif.n]

            #Add in blood volume
            hat += self.aif.cnt * vb
     
            #Interpolate the model prediction at tac sampling time
            return interp.interp1d(self.aif.time, hat, kind='linear')(self.pet.time)

        def comp(self, params):
            """
            Generates predictions for individual model components
            
            Parameters
            ----------
            params: array
                 A 5 x 1 array containing K1, vd, k3, k4, and vb

            Returns
            -------
            comps: list
               A list containing model component vectors
            """

            #Rename params
            K1 = params[0]
            k3 = params[2]
            k4 = params[3]
            vb = params[4]
            k2 = K1 / params[1] - k3
            
            #Compute alpha terms for model
            k_sum = k2 + k3 + k4
            k_sqrt = np.sqrt(np.power(k_sum, 2) - 4.0 * k2 * k4)
            a1 = (k_sum - k_sqrt) / 2.0
            a2 = (k_sum + k_sqrt) / 2.0
        
            #Compute model exponential terms
            exp_1 = np.exp(-a1 * self.kernel_time)
            exp_2 = np.exp(-a2 * self.kernel_time)

            #Compute blood volume piece
            c_p = self.aif.cnt * vb
    
            #Compute kernels for individual compartments
            kern_1 =  K1 / (a2 - a1) * ((k4 - a1) * exp_1 + (a2 - k4) * exp_2)
            kern_2 = K1 * k3 / (a2 - a1) * (exp_1 - exp_2)

            #Calculate free and metabolized compartments
            c_e = np.convolve(self.aif.plasma, kern_1)[0:self.aif.n] * self.aif.samp
            c_m = np.convolve(self.aif.plasma, kern_2)[0:self.aif.n] * self.aif.samp

            #Interpolate all the parts
            c_p_i = interp.interp1d(self.aif.time, c_p, kind='linear')(self.pet.time)
            c_e_i = interp.interp1d(self.aif.time, c_e, kind='linear')(self.pet.time)
            c_m_i = interp.interp1d(self.aif.time, c_m, kind='linear')(self.pet.time)

            #Return list with components
            return np.stack((c_p_i, c_e_i * (1.0 - vb), c_m_i * (1.0 - vb)), axis=1)

        def unit_conv(self, params, glu=None, lc=0.65):
            """
            Generates predictions for 2 compartment model with k4

            Parameters
            ----------
            params: array
                A 5 x 1 array containing K1, vd, k3, k4, and vb
            glu: float
                Plasma glucose level in mg/dL
            lc: float
                Lumped constant for FDG

            Returns
            -------
            meas: array
                A vector of metabolic parameters
            """

            #Convert rato constants and volumes
            K1 = params[0] * 60.0 / 1.05 * 100.0
            k3 = params[2] * 60.0
            k4 = params[3] * 60.0
            vb = params[4] * 100.0
            k2 = K1 / (params[1] / 1.05 * 100) - k3
            ki = (K1 * k3) / (k2 + k3)
            vt = (K1 / k2) * (1 + (k3 / k4))
            
            #Make parameter list for output
            meas = np.array([K1, k2, k3, k4, ki, vt, vb])

            #Parameters that require plasma glucose level
            if glu is not None:

                #Convert plasma glucose level to uMol/ml
                glu_conv = glu / 18.0156

                #See if we need to compute lc
                if lc < 0:
                    lc = 0.39 + (1.48 - 0.39) * (ki / K1)

                #Compute additional parameters
                cmr = ki * glu_conv / lc
                influx = K1 * glu_conv
                conc = K1 / k2 * glu_conv

                #Add terms
                meas = np.append(meas, [cmr, influx, conc])

            return meas

class FdgThree(PetModel):
        """
        Defines 3 parameter, 2 compartment fdg model with blood volume
        """

        def __init__(self, aif, pet, plasma=True):
            """
            Initialize model object for two compartment model
        
            Parameters
            ----------
            aif: Tac object
                Tac object containing the aif data
            pet: Tac object
                Tac object containing the pet data
            plasma: boolean
                True converts input function to plasma
            """

            #Add input to model objection
            PetModel.__init__(self, aif, pet, name="FDG without k4")
            self.plasma = plasma

            #Make time range for kernel
            self.kernel_time = np.arange(0, self.aif.n * self.aif.samp, self.aif.samp) 

            #Get plasma version of input function if necessary
            if self.plasma is True:
                self.aif.plasma = (1.071966 - 1.07294E-5 * self.aif.time) * self.aif.cnt
            else:
                self.aif.plasma = self.aif.cnt

        def pred(self, params):
            """
            Generates predictions for 2 compartment model without k4

            Parameters
            ----------
            params: array
                A 4 x 1 array containing K1, vd, k3, and vb

            Returns
            -------
            pred: array
                A vector of model predictions at pet times
            """

            #Rename params
            K1 = params[0]
            k3 = params[2]
            vb = params[3]
            k2 = K1 / params[1] - k3
        
            #Compute model exponential term
            exp_1 = np.exp(-(k2 + k3) * self.kernel_time)

            #Compute expoential kernel
            kernel = ((K1 * exp_1) + (K1 * k3 / (k2 + k3) * (1.0 - exp_1))) * self.aif.samp
        
            #Compute model prediction   
            hat = (1.0 - vb) * np.convolve(self.aif.plasma, kernel)[0:self.aif.n]

            #Add in blood volume
            hat += self.aif.cnt * vb
     
            #Interpolate the model prediction at tac sampling time
            return interp.interp1d(self.aif.time, hat, kind='linear')(self.pet.time)

        def comp(self, params):
            """
            Generates predictions for individual model components
            
            Parameters
            ----------
            params: array
                 A 4 x 1 array containing K1, vd, k3, and vb

            Returns
            -------
            comps: list
               A list containing model component vectors
            """

            #Rename params
            K1 = params[0]
            k3 = params[2]
            vb = params[3]
            k2 = K1 / params[1] - k3
            
            #Compute model exponential terms
            exp_1 = np.exp(-(k2 + k3) * self.kernel_time)
            exp_2 = 1.0 - exp_1

            #Compute blood volume piece
            c_p = self.aif.cnt * vb

            #Calculate free and metabolized compartments
            c_e = np.convolve(self.aif.plasma, K1 * exp_1)[0:self.aif.n] * self.aif.samp
            c_m = np.convolve(self.aif.plasma, K1 * k3 / (k2 + k3) * exp_2)[0:self.aif.n] * self.aif.samp

            #Interpolate all the parts
            c_p_i = interp.interp1d(self.aif.time, c_p, kind='linear')(self.pet.time)
            c_e_i = interp.interp1d(self.aif.time, c_e, kind='linear')(self.pet.time)
            c_m_i = interp.interp1d(self.aif.time, c_m, kind='linear')(self.pet.time)

            #Return list with components
            return np.stack((c_p_i, c_e_i * (1.0 - vb), c_m_i * (1.0 - vb)), axis=1)

        def unit_conv(self, params, glu=None, lc=0.65):
            """
            Generates predictions for 2 compartment model without k4

            Parameters
            ----------
            params: array
                A 4 x 1 array containing K1, vd, k3, and vb
            glu: float
                Plasma glucose level in mg/dL
            lc: float
                Lumped constant for FDG

            Returns
            -------
            meas: array
                A vector of metabolic parameters
            """

            #Convert rato constants and volumes
            K1 = params[0] * 60.0 / 1.05 * 100.0
            k3 = params[2] * 60.0
            vb = params[3] * 100.0
            k2 = K1 / (params[1] / 1.05 * 100) - k3
            ki = (K1 * k3) / (k2 + k3)
            vt = K1 / k2
            
            #Make parameter list for output
            meas = np.array([K1, k2, k3, ki, vt, vb])

            #Parameters that require plasma glucose level
            if glu is not None:

                #Convert plasma glucose level to uMol/ml
                glu_conv = glu / 18.0156

                #See if we need to compute lc
                if lc < 0:
                    lc = 0.39 + (1.48 - 0.39) * (ki / K1)

                #Compute additional parameters
                cmr = ki * glu_conv / lc
                influx = K1 * glu_conv
                conc = K1 / k2 * glu_conv

                #Add terms
                meas = np.append(meas, [cmr, influx, conc])

            return meas
           
class FlowTwo(PetModel):
        """
        Defines 2 paramter, 1 compartment blood flow model
        """

        def __init__(self, aif, pet):
            """
            Initialize model object for two compartment model
            
            Parameters
            ----------
            aif: Tac object
                Tac object containing the aif data
            pet: Tac object
                Tac object containing the pet data
            """
            
            #Initialize model
            PetModel.__init__(self, aif, pet, name='Flow Model')

            #Make time range for kernel
            self.kernel_time = np.arange(0, self.aif.n * self.aif.samp, self.aif.samp)          

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

            #Rename paramters
            K1 = params[0]
            k2 = params[1]
        
            #Compute model prediction   
            hat = K1 * np.convolve(self.aif.cnt, np.exp(-k2*self.kernel_time))[0:self.aif.n] * self.aif.samp
     
            #Interpolate the model prediction at tac sampling time
            return interp.interp1d(self.aif.time, hat, kind='linear')(self.pet.time)
        
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

            #Compute cbf, k2, and the blood-brain partion coefficient
            cbf = params[0] * 6000.0 / 1.05     #mL/hg/min
            k2 =  params[1] * 60.0              #1/min
            lmbda_w =  cbf / k2 / 100.0         #mL/g

            return np.array([cbf, k2, lmbda_w])

class OhtaTwo(PetModel):
        """
        Defines Ohta two compartment model
        """

        def __init__(self, aif, pet):
            """
            Initialize model object for Ohta model
        
            Parameters
            ----------
            aif: Tac object
                Tac object containing the aif data
            pet: Tac object
                Tac object containing the pet data
            """

            #Initialize model
            PetModel.__init__(self, aif, pet, name='Ohta Two Compartment')

            #Make time range for kernel
            self.kernel_time = np.arange(0, self.aif.n * self.aif.samp, self.aif.samp)          

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

            #Rename paramters
            K1 = params[0]
            k2 = params[1]
            v0 = params[2]

            #Compute model prediction   
            hat = K1 * np.convolve(self.aif.cnt, np.exp(-k2 * self.kernel_time))[0:self.aif.n] * self.aif.samp
            hat += self.aif.cnt * v0
     
            #Interpolate the model prediction at tac sampling time
            return interp.interp1d(self.aif.time, hat, kind='linear')(self.pet.time)

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

            #Compute cbf, k2, blood-brain partion coefficient, and blood volume
            K1= params[0] * 6000.0 / 1.05       #mL/hg/min
            k2 =  params[1] * 60.0              #1/min
            lmbda = K1 / k2 / 100.0             #mL/g
            v0 = params[2] * 105.0              #mL/hg
       
            #Add in cmr if necessary
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

            #Initialize model
            PetModel.__init__(self, aif, pet, name="One Compartment")
            self.vol = vol

            #Integration of aif and pet
            self.aif.int = integ.cumtrapz(self.aif.cnt, self.aif.time, initial=0.0)
            self.pet.int = integ.cumtrapz(self.pet.cnt, self.pet.time, initial=0.0)

            #Create interpolation function for input function and its integral
            self.aif.func_int = interp.interp1d(self.aif.time, self.aif.int, kind='cubic')
            if self.vol is True:
                self.aif.func = interp.interp1d(self.aif.time, self.aif.cnt, kind='cubic')

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

            #Determine times to shift function to
            self.aif.shift_time = self.aif.time - shift
            self.pet.shift_time = self.pet.time + shift

            #Make interpolation masks
            self.pet.shift_mask = np.logical_and(self.pet.shift_time >= self.aif.time[0],
                                                 self.pet.shift_time <= self.aif.time[-1])
            self.pet.mask = np.logical_and(self.pet.time >= self.aif.shift_time[0],
                                           self.pet.time <= self.aif.shift_time[-1])

            #Interpolate the aif and its integral to pet sampling with shift
            self.aif.int_i = self.aif.func_int(self.pet.shift_time[self.pet.shift_mask])

            #Add in blood volume if necessary
            if self.vol is False:

                #Make design matrix without blood volume
                self.x = np.stack((self.aif.int_i, -1.0 * self.pet.int[self.pet.mask]), axis=1)

            else:
       
                #Interpolate input function to pet sampling with shift
                self.aif.cnt_i = self.aif.func(self.pet.shift_time[self.pet.shift_mask])

                #Make design matrix with blood volume
                self.x = np.stack((self.aif.int_i, -1.0 * self.pet.int[self.pet.mask], self.aif.cnt_i), axis=1)

            #Get non-negative least squares solution
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

            #Compute coefficients
            beta = self.coef(shift)
     
            #Compute model prediction
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

            #Compute cbf, k2, and the blood-brain partion coefficient
            K1 = params[0] * 6000.0 / 1.05     #mL/hg/min
            k2 =  params[1] * 60.0             #1/min
            lmbda_w =  K1 / k2 / 100.0         #mL/g

            #Deal with possible blood volume
            if self.vol is True:
                return np.array([K1, k2, lmbda_w, params[2] / 1.05 * 100])
            else:
                return np.array([K1, k2, lmbda_w])

class OxyOne(PetModel):
        """
        Defines 1 paramter, 2 Mintun oxygen consumption model
        """

        def __init__(self, aif_oxy, aif_water, pet, flow, k2,
                     vb):
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
            """

            #Make sure decay correction status is the same
            if aif_oxy.dc != aif_water.dc:
                raise ValueError('Inputs must have the same decay status')

            #Add input to model objection
            PetModel.__init__(self, aif_oxy, pet, name='Oxygen Model')
            self.aif_oxy = aif_oxy
            self.aif_water = aif_water
            self.flow = flow
            self.k2 = k2
            self.vb = vb
            self.ratio = 0.85
            
            #Make kernel object
            kernel_time = np.arange(0, self.aif_oxy.n * self.aif_oxy.samp, self.aif_oxy.samp)
            self.kernel = Tac(kernel_time, self.flow * np.exp(-self.k2 * kernel_time))

            #Generate convolution terms
            conv_water = np.convolve(self.aif_water.cnt,
                                     self.kernel.cnt)[0:self.aif_water.n] * self.aif_water.samp
            conv_oxy = np.convolve(self.aif_oxy.cnt,
                                   self.kernel.cnt)[0:self.aif_oxy.n] * self.aif_oxy.samp

            #Generate blood volume term
            b_vol = self.ratio * self.vb * self.aif_oxy.cnt

            #Interpolate the model terms
            self.conv_water_i = interp.interp1d(self.aif_oxy.time, conv_water, kind='linear')(self.pet.time)
            self.conv_oxy_i = interp.interp1d(self.aif_oxy.time, conv_oxy, kind='linear')(self.pet.time)
            self.b_vol_i = interp.interp1d(self.aif_oxy.time, b_vol, kind='linear')(self.pet.time)

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

            #Compute model prediction   
            hat = self.conv_water_i + self.conv_oxy_i * oef + (1.0 - 0.835 * oef) * self.b_vol_i
     
            #Interpolate the model prediction at tac sampling time
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

            #Compute cmro2 if given oxygen concentration in arterial blood
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

            #Add input to model objection
            PetModel.__init__(self, aif, pet, name="Two Compartment")
            self.fdg = fdg
            self.vol = vol

            #Create scale to convert to plasma
            if self.fdg is True:
                self.aif.scale = 1.071966 - 1.07294E-5 * self.aif.time
            else:
                self.aif.scale = 1.0

            #Integration of aif and pet
            self.aif.int = integ.cumtrapz(self.aif.cnt * self.aif.scale, self.aif.time, initial=0.0)
            self.pet.int = integ.cumtrapz(self.pet.cnt, self.pet.time, initial=0.0)

            #Double integrations
            self.aif.int_dbl = integ.cumtrapz(self.aif.int * self.aif.scale, self.aif.time, initial=0.0)
            self.pet.int_dbl = integ.cumtrapz(self.pet.int, self.pet.time, initial=0.0)

            #Create aif interpolation functions
            self.aif.func_int = interp.interp1d(self.aif.time, self.aif.int, kind='cubic')
            self.aif.func_int_dbl = interp.interp1d(self.aif.time, self.aif.int_dbl, kind='cubic')
            if self.vol is True:
                self.aif.func = interp.interp1d(self.aif.time, self.aif.cnt, kind='cubic')

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

            #Determine times to shift function to
            self.aif.shift_time = self.aif.time - shift
            self.pet.shift_time = self.pet.time + shift

            #Make interpolation masks
            self.pet.shift_mask = np.logical_and(self.pet.shift_time >= self.aif.time[0],
                                                 self.pet.shift_time <= self.aif.time[-1])
            self.pet.mask = np.logical_and(self.pet.time >= self.aif.shift_time[0],
                                           self.pet.time <= self.aif.shift_time[-1])

            #Interpolate the aif integrals
            time_mskd = self.pet.shift_time[self.pet.shift_mask]
            self.aif.int_i = self.aif.func_int(time_mskd)
            self.aif.int_dbl_i = self.aif.func_int_dbl(time_mskd)                                                  

            #Account for blood volume if necessary
            if self.vol is False:

                #Make design matrix without blood
                self.x = np.stack((self.aif.int_i,
                                   self.aif.int_dbl_i,
                                   -self.pet.int[self.pet.mask],
                                   -self.pet.int_dbl[self.pet.mask]), axis=1)

            else:

                #Interpolate input function to pet sampling with shift
                self.aif.cnt_i = self.aif.func(time_mskd)

                #Make design matrix with blood
                self.x = np.stack((self.aif.int_i,
                                   self.aif.int_dbl_i,
                                   -self.pet.int[self.pet.mask],
                                   -self.pet.int_dbl[self.pet.mask],
                                   self.aif.cnt_i), axis=1)

            #Get non-negative least squares solution
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

            #Compute coefficients
            beta = self.coef(shift)
     
            #Compute model prediction
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

            #Compute rate constants
            K1 = params[0] * 6000.0 / 1.05
            k2 = (params[2] - params[1] / params[0])
            k4 = params[3] / k2
            k3 = params[2] - k2 - k4

            #Convert rate constants to minutes
            k2 *= 60.0
            k3 *= 60.0
            k4 *= 60.0

            #Deal with possible blood volume
            if self.vol is True:
                return np.array([K1, k2, k3, k4, params[4] / 1.05 * 100])
            else:
                return np.array([K1, k2, k3, k4])
