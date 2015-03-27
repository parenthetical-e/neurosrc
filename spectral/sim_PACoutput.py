# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 07:08:30 2015

@author: Scott
"""

import numpy as np
from voytoys.signal_pac import my_pac
from NeuroTools import stgen
import matplotlib.pyplot as plt
import h5py

# Network parameters
finame = 'pac_mc.hdf5'
Nneu = 100 # Number of neurons
spikeHz_baseline1 = np.arange(20) # Firing rate of a neuron (Hz) at baseline
spikeHz_baseline = np.append(spikeHz_baseline1,[19.1,19.2,19.3,19.4,19.5,19.6,19.7,19.8,19.9,19.99,19.999,19.9999])
spikeHz_biased = 20 # Extra firing rate of a neuron (Hz) at favored phase
frac_bias = .2
dur = 3 # Length of simulation (s)
flo_range = (4, 8)
fhi_range = (80, 150)

# Process parameters
E = len(spikeHz_baseline)
t = np.arange(0,dur,.001)
T = len(t)

bias_ratio = spikeHz_biased / spikeHz_baseline
bias_ratio[bias_ratio==np.inf] = 100 #approximation for plotting purposes
log_bias_ratio = np.log10(bias_ratio)

# Define rates for Inhomogeneous Poisson Process (IPP)
# Assign a phase for all points in time
thal_phase0 = 2*np.pi*np.random.rand()
thal_freq = flo_range[0] + (flo_range[1]-flo_range[0])*np.random.rand(1)
thal_phase = t % (1/thal_freq) * 2*np.pi*thal_freq
thal_phase = (thal_phase + thal_phase0) % (2*np.pi)

# Calculate the IPP firing rate at each point in time, dependent on phase
mod_idxs = thal_phase < (frac_bias*2*np.pi)
mod_idxs = mod_idxs + 0
IPP_t = np.arange(dur*1000)

# Simulate IPP
stg = stgen.StGen()
field_raster = np.zeros(E,dtype=object)
for e in xrange(E):
    print e
    IPP_r_dep = (spikeHz_biased-spikeHz_baseline[e])*mod_idxs
    IPP_r_indep = spikeHz_baseline[e]*np.ones(T)
    IPP_rates = IPP_r_dep + IPP_r_indep 
    tspikes = np.zeros(Nneu,dtype=object)
    raster = np.zeros((Nneu,T))
    for neu in range(Nneu):
        tspikes[neu] = stg.inh_poisson_generator(IPP_rates,IPP_t,dur*1000,array=True)
        for tt in IPP_t:
            raster[neu,tt] = sum(np.logical_and(tspikes[neu]>tt,tspikes[neu]<tt+1))
    field_raster[e] = np.sum(raster,0)

# Define alpha function
alpha_dur = .5
alpha_t = np.arange(0,alpha_dur,.001)
alpha_tau = .001
alpha_gmax = .1
alpha = alpha_gmax * (alpha_t/alpha_tau) * np.exp(-(alpha_t-alpha_tau)/alpha_tau)

# Calculate LFP by using convolving spike train with alpha function
lfp = np.zeros(E,dtype=object)
for e in xrange(E):
    lfppre = np.convolve(field_raster[e],alpha,'same')
    # Normalize LFP
    lfppre = lfppre - np.mean(lfppre)
    lfp[e] = lfppre / np.std(lfppre)

# Set PAC parameters and calculate PAC
pac_method = 'plv'
filt_method = 'eegfilt'
rate = 1000
#kwargs = {'order' : 2} # for butter
#kwargs = {'transition' : 12, 'ripple' : 50} # for kaiser
kwargs = {'trans' : .15} # for eegfilt
pac_plv = np.zeros(E)
pac_mi = np.zeros(E)
pac_glm = np.zeros(E)
for e in xrange(E):
    pac_plv[e] = my_pac(lfp[e], flo_range, fhi_range, rate, 'plv', filt_method, **kwargs) # add **kwargs if desired
    pac_mi[e] = my_pac(lfp[e], flo_range, fhi_range, rate, 'mi', filt_method, **kwargs) # add **kwargs if desired
    pac_glm[e] = my_pac(lfp[e], flo_range, fhi_range, rate, 'glm', filt_method, **kwargs) # add **kwargs if desired

# Save data
with h5py.File(finame,'w') as fi:
    fi['Nneu'] = Nneu
    fi['spikeHz_baseline'] = spikeHz_baseline
    fi['spikeHz_biased'] = spikeHz_biased
    fi['frac_bias'] = frac_bias
    fi['flo_range'] = flo_range
    fi['fhi_range'] = fhi_range
    fi['lfp'] = lfp
    fi['pac_plv'] = pac_plv
    fi['pac_glm'] = pac_glm
    fi['pac_mi'] = pac_mi
    
# Visualize relationships between metrics
plt.figure()
plt.subplot(1,3,1)
plt.plot(pac_plv,pac_mi,'.')
plt.xlabel('plv')
plt.ylabel('mi')
plt.subplot(1,3,2)
plt.plot(pac_glm,pac_mi,'.')
plt.xlabel('glm')
plt.ylabel('mi')
plt.subplot(1,3,3)
plt.plot(pac_plv,pac_glm,'.')
plt.xlabel('plv')
plt.ylabel('glm')

# Visualize relationships between metrics
plt.figure()
plt.subplot(1,3,1)
plt.plot(log_bias_ratio,pac_mi,'.')
plt.xlabel('log_bias_ratio')
plt.ylabel('PAC: plv')
plt.subplot(1,3,2)
plt.plot(log_bias_ratio,pac_mi,'.')
plt.xlabel('log_bias_ratioglm')
plt.ylabel('PAC: mi')
plt.subplot(1,3,3)
plt.plot(log_bias_ratio,pac_glm,'.')
plt.xlabel('log_bias_ratio')
plt.ylabel('PAC: glm')