# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 15:20:56 2015

@author: Scott
"""

import numpy as np
from neurosrc.spectral.filter import scfilter

import math
import statsmodels.api as sm

def scpac(x, flo_range, fhi_range, rate, pac_method, filt_method, **kwargs):
    '''
    Calculate phase-amplitude coupling between the phase of a low-frequency-
    bandpass filtered signal and the amplitude of a high-frequency-bandpass 
    filtered signal.
    
    Parameters
    ----------
    x : array-like 1d
        Time series of data
    flo_range : 2-element list
        Low and High cutoff frequencies (Hz)
    fhi_range : 2-element list
        Low and High cutoff frequencies (Hz)
    rate : numeric
        Sampling rate of x
    pac_method : string
        Method to calculate PAC.
        See Tort, 2008 for method 'mi' (used in de Hemptinne, 2013)
        See Tort, 2010 for method 'plv'
        See Penny, 2008 for method 'glm'
        See Canolty, 2006 for method 'mi_canolty'
    filt_method : string
        Filtering method
    **kwargs : dictionary
        Filter parameters
        
    Returns
    -------
    pac : numeric
        Phase-amplitude coupling value
    '''
    
    # Bandpass filter raw signal
    xlo = scfilter(x, flo_range, filt_method, rate = rate, **kwargs)
    xhi = scfilter(x, fhi_range, filt_method, rate = rate, **kwargs)
    
    #Calculate PAC
    if pac_method == 'plv':
        ahi = np.abs(fasthilbert(xhi))
        xlo_ahi = scfilter(ahi, flo_range, filt_method, **kwargs)
        pac = pac_plv(xlo, xlo_ahi)
    elif pac_method == 'mi':
        pac = pac_mi(xlo, xhi)
    elif pac_method == 'glm':
        pac = pac_glm(xlo, xhi)
    elif pac_method == 'mi_canolty':
        pac = pac_mi_canolty(xlo, xhi)
    else:
        raise ValueError('Not a valid PAC method')
    
    return pac

def power_two(n):
    '''
    Calculate the next power of 2 from a number
    '''
    return 2**(int(math.log(n, 2))+1)

def fasthilbert(x, axis=-1):
    '''
    Redefinition of scipy.signal.hilbert, which is very slow for some lengths
    of the signal. This version zero-pads the signal to the next power of 2
    for speed.
    '''
    x = np.array(x)
    N = x.shape[axis]
    N2 = power_two(len(x))
    Xf = np.fft.fft(x, N2, axis=axis)
    h = np.zeros(N2)
    h[0] = 1
    h[1:(N2 + 1) // 2] = 2
        
    x = np.fft.ifft(Xf * h, axis=axis)
    return x[:N]

def pac_mi(xlo, xhi):
    '''
    Calculate PAC using the modulation index (MI) method
    
    Parameters
    ----------
    xlo : array-like 1d
        Low-frequency-bandpass filtered signal
    xhi : array-like 1d
        High-frequency-bandpass filtered signal
        
    Returns
    -------
    pac : numeric
        PAC value
    '''
    
    
    # Calculate phase and amplitude
    pha = np.angle(fasthilbert(xlo))
    amp = np.abs(fasthilbert(xhi))
    pha = np.degrees(pha)
    
    # Calculate PAC
    bin_phase_lo = np.arange(-180,180,20)
    binned_meanA = np.zeros(len(bin_phase_lo))
    p_j = np.zeros(len(bin_phase_lo))
    for b in range(len(bin_phase_lo)):
        phaserange = np.logical_and(pha>=bin_phase_lo[b],pha<(bin_phase_lo[b]+20))
        binned_meanA[b] = np.mean(amp[phaserange])
        
    for b in range(len(bin_phase_lo)):
        p_j[b] = binned_meanA[b]/sum(binned_meanA)
        
    H = -sum(np.multiply(p_j,np.log10(p_j)))
    Hmax = np.log10(18)
    pac = (Hmax-H)/Hmax
    
    return pac

def pac_plv(xlo, xlo_ahi):
    '''
    Calculate PAC using the phase-locking value (PLV) method
    
    Parameters
    ----------
    xlo : array-like 1d
        Low-frequency-bandpass filtered signal
    xlo_ahi : array-like 1d
        Low-frequency bandpass filtered signal of the Hilbert amplitude of the
        original high-frequency-bandpass filtered signal
        
    Returns
    -------
    pac : numeric
        PAC value
    '''

     # Calculate phase, and amplitude phase, and PAC
    pha = np.angle(fasthilbert(xlo))
    amp_pha = np.angle(fasthilbert(xlo_ahi))
    pac = np.abs(np.sum(np.exp(1j * (pha - amp_pha)))) / len(xlo)
    
    return pac

def pac_glm(xlo, xhi):
    '''
    Calculate PAC using the generalized linear model (GLM) method
    
    Parameters
    ----------
    xlo : array-like 1d
        Low-frequency-bandpass filtered signal
    xhi : array-like 1d
        High-frequency-bandpass filtered signal
        
    Returns
    -------
    pac : numeric
        PAC value
    '''

    # Calculate phase and amplitude
    pha = np.angle(fasthilbert(xlo))
    amp = np.abs(fasthilbert(xhi))
    
    # Prepare GLM
    y = amp
    X_pre = np.vstack((np.cos(pha), np.sin(pha)))
    X = X_pre.T
    X = sm.add_constant(X, prepend=False)
    
    # Run GLM
    my_glm = sm.GLM(y,X)
    res = my_glm.fit()
    #print(res.summary())
    
    # Calculate R^2 value. Equivalent to mdl.Rsquared.Ordinary in MATLAB
    pac = 1 - np.sum(res.resid_deviance**2) /  np.sum((amp-np.mean(amp))**2)
    
    return pac
    
def pac_mi_canolty(xlo, xhi):
    '''
    Calculate PAC using the modulation index (MI) method defined in Canolty,
    2006
    
    Parameters
    ----------
    xlo : array-like 1d
        Low-frequency-bandpass filtered signal
    xhi : array-like 1d
        High-frequency-bandpass filtered signal
        
    Returns
    -------
    pac : numeric
        PAC value
    '''
    
    # Calculate phase and amplitude
    pha = np.angle(fasthilbert(xlo))
    amp = np.abs(fasthilbert(xhi))
    
    # Calculate PAC
    pac = np.sum(amp * np.exp(1j * pha))
    
    return pac

def pac_palette(x, pac_method, filt_method, rate = 1000,
                dp = 2, da = 4, p_range = None, a_range = None, **kwargs):
    '''
    Calculate PAC for many phase and frequency bands
    '''    
    if p_range == None:
        p_range = (4,50)
    if a_range == None:
        a_range = (10,200)
        
    # Calculate palette frequency parameters
    f_phases = np.arange(p_range[0],p_range[1],dp) 
    f_amps = np.arange(a_range[0],a_range[1],da)
    P = len(f_phases)
    A = len(f_amps)
    
    # Calculate PAC for every combination of P and A
    pac = np.zeros((P,A))
    for p in range(P):
        flo = [f_phases[p],f_phases[p]+dp]
        for a in range(A):
            #print p,a
            fhi = [f_amps[a],f_amps[a]+da]
            pac[p,a] = scpac(x, flo, fhi, rate, pac_method, filt_method, **kwargs)
    
    return pac
