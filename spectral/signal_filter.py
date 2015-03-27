# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 08:56:34 2015

@author: Scott
"""
import numpy as np
from scipy.signal import butter, filtfilt, firwin, firwin2, kaiserord

def my_filter(x, f_range, method, rate = 1000, **kwargs):
    '''
    Filter a time series with the desired method and parameters
    
    Parameters
    ----------
    x : array-like 1d
        Time series of data
    f_range : 2-element list
        Low and High cutoff frequencies (Hz)
    method : string
        Filtering method
    rate : numeric
        Sampling rate of x
    **kwargs : dictionary
        Filter parameters
        
    Returns
    -------
    x_filt : array-like 1d
        Filtered signal
    '''
    if method == 'butter':
        x_filt = my_butter(x, f_range, rate = rate, **kwargs) #kwargs= deg
    elif method == 'kaiser':
        x_filt = my_kaiser(x, f_range, rate = rate, **kwargs) #kwargs= ripple_db, transition
    elif method == 'eegfilt':
        x_filt = my_eegfilt(x, f_range, rate = rate, **kwargs)
    else:
        raise ValueError("Filter method entered is not valid")
    return x_filt
    
def my_butter(x, f_range, rate = 1000, order = 2):
    '''
    Filter a signal with a butterworth filter
    
    Parameters
    ----------
    x : array-like 1d
        Time series of data
    f_range : 2-element list
        Low and High cutoff frequencies (Hz)
    rate : numeric
        Sampling rate of x
    order : integer
        Order of the butterworth filter
        
    Returns
    -------
    x_filt : array-like 1d
        Filtered signal
    '''
    nyq_rate = rate / 2.0
    Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
    b, a = butter(order, Wn, 'bandpass')
    return filtfilt(b, a, x)

def my_kaiser(x, f_range, rate = 1000, transition = 10.0, ripple = 60.0):
    '''
    Filter a signal with a Kaiser filter
    
    Parameters
    ----------
    x : array-like 1d
        Time series of data
    f_range : 2-element list
        Low and High cutoff frequencies (Hz)
    rate : numeric
        Sampling rate of x
    transition : numeric
        Width of the transition band between passband and stopband (Hz)
    ripple : numeric
        Maximum gain in the stopband (dB)
        
    Returns
    -------
    x_filt : array-like 1d
        Filtered signal
    '''
    nyq_rate = rate / 2.0
    Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
    
    width = transition/nyq_rate
    N, beta = kaiserord(ripple, width)
    
    taps = firwin(N, Wn, window=('kaiser', beta), pass_zero = False)
    x_firk = filtfilt(taps, [1.0], x)
    
    return x_firk

def my_eegfilt(x, f_range, rate = 1000, Ntaps = None, trans = 0.15):
    '''
    Filter a signal with the eegfilt method from EEGLAB in MATLAB
    
    Parameters
    ----------
    x : array-like 1d
        Time series of data
    f_range : 2-element list
        Low and High cutoff frequencies (Hz)
    rate : numeric
        Sampling rate of x
    Ntaps : integer
        Number of samples in the filter
    trans : numeric
        Transition width at cutoff frequencies ()
        
    Returns
    -------
    x_filt : array-like 1d
        Filtered signal
    '''  
    
    if Ntaps == None:
        min_Ntaps = 15
        minfac = 3
        if f_range[0] > 0:
            Ntaps = minfac*np.floor(rate/f_range[0])
        elif f_range[1] > 0:
            Ntaps = minfac*np.floor(rate/f_range[1])
        if Ntaps < min_Ntaps:
            Ntaps = min_Ntaps

    nyq = rate*0.5
    
    if np.logical_and(f_range[0]>0, f_range[1]>0):
        f = [0, (1-trans)*f_range[0]/nyq, f_range[0]/nyq, f_range[1]/nyq, (1+trans)*f_range[1]/nyq, 1]; 
        m = [0,0,1,1,0,0]
    elif f_range[0] > 0:
        f = [0, (1-trans)*f_range[0]/nyq, f_range[0]/nyq, 1] 
        m = [0,0,1,1]
    elif f_range[1] > 0:
        f = [0, f_range[1]/nyq, (1+trans)*f_range[1]/nyq, 1]; 
        m = [1,1,0,0]
    
    taps = firwin2(Ntaps, f, m)
    x_filt = filtfilt(taps,[1],x)
    
    return x_filt