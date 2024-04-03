# -*- coding: utf-8 -*-
"""
test_filter_ssvep_data.py

Load steady-state visual evoked potentials (SSVEPs) data and use filtering and
the Hilbert transform to calculate signal amplitude at frequencies corresponding
to stimuli.
BME6710 BCI Spring 2024 Lab #4

The SSVEP dataset is derived from a tutorial in the MNE-Python package.The dataset
includes electropencephalograpy (EEG) data from 32 channels collected during
a visual checkerboard experiment, where the checkerboard flickered at 12 Hz or
15 Hz. 

This script uses functions from the import_ssvep_data.py module to load the dataset
into variables and epoch the data and functions from filter_ssvep_data.py to
filter and transform the data and calculate signal amplitudes. 

Created on Mar 7 2024

@author: 
    Ardyn Olszko
    Jay Hwasung Jung
"""

#%% 
# Part 1: use functions written in previous labs to load the data from subject 1 into a python dictionary called data.

# import module

import import_ssvep_data
from matplotlib import pyplot as plt
import filter_ssvep_data

# load subject data
subject = 2
data_directory = 'SsvepData/Data' 

data = import_ssvep_data.load_ssvep_data(subject,data_directory)

#%% 
# Part 2: call this function twice to get 2 filters that keep the data around 12Hz and around 15Hz

filter_order = 1000
filter_type = 'hann'
filter_bandwidth = 2
fs = data['fs'] # sampling frequency of EEG data

# Create filter to keep data around 12 Hz
filter_coefficients_12Hz = filter_ssvep_data.make_bandpass_filter(low_cutoff=12-0.5*filter_bandwidth,
                                                high_cutoff=12+0.5*filter_bandwidth,
                                                filter_order=filter_order,
                                                fs=fs,
                                                filter_type=filter_type)
# Create filter to keep data around 15 Hz
filter_coefficients_15Hz = filter_ssvep_data.make_bandpass_filter(15-0.5*filter_bandwidth,
                                                15+0.5*filter_bandwidth,
                                                filter_order,fs,filter_type)

"""
A) How much will 12Hz oscillations be attenuated by the 15Hz filter? How much
    will 15Hz oscillations be attenuated by the 12Hz filter?

B) Experiment with higher and lower order filters. Describe how changing 
    the order changes the frequency and impulse response of the filter.

"""

#%% 
# Part 3: call filter_data() twice to filter the data with each of your two band-pass filters (the ones designed to capture 12Hz and 15Hz oscillations) and store the results in separate arrays.

# Apply filter to keep data around 12 Hz
filtered_data_12Hz = filter_ssvep_data.filter_data(data,filter_coefficients_12Hz)

# Apply filter to keep data around 15 Hz
filtered_data_15Hz = filter_ssvep_data.filter_data(data,filter_coefficients_15Hz)


#%% 
# Part 4: call this function twice to get the 12Hz and 15Hz envelopes. In each case, choose electrode Oz to plot.

channel_to_plot='Oz'

envelope_12Hz = filter_ssvep_data.get_envelope(data,filtered_data_12Hz,ssvep_frequency=12,channel_to_plot=channel_to_plot)
envelope_15Hz = filter_ssvep_data.get_envelope(data,filtered_data_15Hz,ssvep_frequency=15,channel_to_plot=channel_to_plot)
#%% 
# Part 5: Plot the Amplitudes


filter_ssvep_data.plot_ssvep_amplitudes(data,envelope_12Hz,envelope_15Hz,
                                        12,15,subject,
                                        channel_to_plot=channel_to_plot)

"""
Describe what you see. 

What do the two envelopes do when the stimulation frequency changes? 

How large and consistent are those changes? 

Are the brain signals responding to the events in the way you’d expect? 

Check some other electrodes – which electrodes respond in the same way and why?

"""

#%% 
# Part 6: Examine the Spectra

filter_ssvep_data.plot_filtered_spectra(data,filtered_data_15Hz,envelope_15Hz)

"""
Describe how the spectra change at each stage and why. 

Changes you should address include (but are not limited to) the following:
    
1. Why does the overall shape of the spectrum change after filtering?
2. In the filtered data on Oz, why do 15Hz trials appear to have less power 
 12Hz trials at most frequencies?
3. In the envelope on Oz, why do we no longer see any peaks at 15Hz?

"""

# %%
