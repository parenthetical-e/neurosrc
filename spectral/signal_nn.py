# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 15:20:56 2015

@author: Scott
"""

import numpy as np
from scipy import signal
import math

def my_nn(x, f_slope, order_fit, order_NN, spec_method, rate=1000, return_psd=0):
    '''
    Calculate phase-amplitude coupling between the phase of a low-frequency-
    bandpass filtered signal and the amplitude of a high-frequency-bandpass 
    filtered signal.
    
    Parameters
    ----------
    x : array-like 1d
        Time series of data
    fslope : 2-element list
        Low and High frequency boundaries for calculating slope of psd
    order_fit : integer
        Order of the polynomial to fit to the psd
    order_NN : integer
        Order of the polynomial fitted to the psd for which to output the
        coefficient
    spec_method : string
        Method for calculating the psd
        'fm': FFT, median filter, resample (Brad did this)
        'wm': Welch's method with 19-point median filter
    rate : numeric
        Sampling rate of the data
    return_psd : binary
        If == 1, return the frequency array and PSD in adition to NN value
        
    Returns
    -------
    nn : numeric
        Slope of the PSD
    '''
    
    # Calculate log power
    if spec_method == 'wm':
        f, psd = spec_wm(x)
    elif spec_method == 'fm':
        f, psd = spec_fm(x)
        win = 19
        psd = signal.medfilt(psd, win)
    else:
        raise ValueError('Not a valid method for calculating the power spectrum')
    
    # Restrict PSD to frequency range of interest
    mask = np.zeros_like(f, dtype=np.bool)
    if f_slope[0] is not None:
        mask = mask | (f <= f_slope[0])
    if f_slope[1] is not None:
        mask = mask | (f >= f_slope[1])
    mask = np.logical_not(mask)
    f_nn = f[mask]
    psd_nn = psd[mask]
    
    # Calculate slope of PSD
    p = np.polyfit(f_nn, psd_nn, deg=order_fit)
    idx = order_fit - order_NN
    nn = p[idx]
    
    if return_psd:
        return nn, f, psd
    else:
        return nn

def power(x, rate=1000, f_low=None, f_high=None, **kwargs):
    """Calculate the power spectrum density within a frequency range

    Parameters
    ----------
    x : array-like 1d
        A neural time-series
    rate : numeric (default: 1000)
        The sampling rate (in Hz)
    f_low : numeric (default: None)
        The bottom end of the estimation window
    f_high : numeric (default: None)
        The top end of the estimation window
    **kwargs
        Parameters passed to scipy.signal.welch(), which
        does the PSD estimation.

    Note
    ----
    Usses signal.welch to do the calculation and
    signal.get_window('hamming', np.floor(rate * 2.)
    for the window.
    
    (from noisy.lfp, By Erik)
    """

    f, psd = signal.welch(
        x, fs=rate,
        window=signal.get_window('hamming', np.floor(rate * 2.)),
        **kwargs
    )

    # Do filtering using f_low and f_high
    mask = np.zeros_like(f, dtype=np.bool)
    if f_low is not None:
        mask = mask | (f <= f_low)
    if f_high is not None:
        mask = mask | (f >= f_high)
    mask = np.logical_not(mask)

    f = f[mask]
    psd = np.log10(psd[mask])

    return f, psd


def spec_wm(x, rate=1000, f_low=None, f_high=None, win=19, **kwargs):
    """A robust version of power.

    The PSD from power is median smoothed.

    Parameters
    ---------
    win : scaler
        The window size
    
    (from noisy.lfp, By Erik)
    """

    f, psd = power(x, rate, f_low, f_high, **kwargs)

    return f, signal.medfilt(psd, win)

def power_two(n):
    '''
    Calculate the next power of 2 from a number
    '''
    return 2**(int(math.log(n, 2))+1)
    
def spec_fm(x, rate=1000):
    '''
    Calculate PSD by first taking the power of the FFT, median filtering it,
    and then resampling at 0.5Hz
    '''
    rawfft = np.fft.fft(x,2**power_two(len(x))) / len(x) 
    psd = 2*np.abs(rawfft)
    f = rate/2*np.linspace(0,1,len(x)/2+1)
    logpsd = 10 * np.log10(psd[:np.floor(len(psd)/2)+1])
    try:
        logpsd_filt = signal.medfilt(logpsd,int(np.floor(len(f)/rate)))
    except ValueError:
        logpsd_filt = signal.medfilt(logpsd,int(np.floor(len(f)/rate)+1))
    spec = signal.resample(logpsd_filt, rate+1)
    f = np.linspace(0, rate/2, len(spec))
    return f, spec
