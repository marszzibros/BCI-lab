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

#%% Part 1
# Use functions written in previous labs to load the data from subject 1 into a python dictionary called data.

# import module

import import_ssvep_data
import filter_ssvep_data

# load subject data
subject = 1
data_directory = 'SsvepData/Data' 

data = import_ssvep_data.load_ssvep_data(subject,data_directory)

#%% Part 2: 
# Call this function twice to get 2 filters that keep the data around 12Hz and around 15Hz

filter_order = 1000
filter_type = 'hann'
filter_bandwidth = 2 # units of Hz
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
    
    The frequency response of the filters show that the 12 Hz filter will
    do little to attentuate the 15 Hz oscillations and the 15 Hz filter will do
    little to attentuate the 12 Hz oscillations. This makes sense given the 
    bandpass filter design with a bandwidth of 2 Hz. In both cases, the output
    reaches about 50% power (-3 dB), within about 1 Hz of the central frequency.
    As the decibels become increasingly negative, the power decreases logarithmically.
    For example, consider the filter at the central frequency of 12 Hz. The power
    remaining at 15 Hz is way less than 1% (~-67 dB).
    
B) Experiment with higher and lower order filters. Describe how changing 
    the order changes the frequency and impulse response of the filter.
    
    Higher-order filters show the sharper frequency cutoffs but show longer 
    impulse responses.

"""

#%% Part 3
# Call filter_data() twice to filter the data with each of your two band-pass filters (the ones designed to capture 12Hz and 15Hz oscillations) and store the results in separate arrays.

# Apply filter to keep data around 12 Hz
filtered_data_12Hz = filter_ssvep_data.filter_data(data,filter_coefficients_12Hz)

# Apply filter to keep data around 15 Hz
filtered_data_15Hz = filter_ssvep_data.filter_data(data,filter_coefficients_15Hz)


#%% Part 4
# Call this function twice to get the 12Hz and 15Hz envelopes. In each case, choose electrode Oz to plot.

channel_to_plot='Oz'

# Plot envelopes for 12Hz and 15Hz filtered data
envelope_12Hz = filter_ssvep_data.get_envelope(data,filtered_data_12Hz,ssvep_frequency=12,channel_to_plot=channel_to_plot)
envelope_15Hz = filter_ssvep_data.get_envelope(data,filtered_data_15Hz,ssvep_frequency=15,channel_to_plot=channel_to_plot)

#%% Part 5
# Plot the Amplitudes

filter_ssvep_data.plot_ssvep_amplitudes(data,envelope_12Hz,envelope_15Hz,
                                        12,15,subject,
                                        channel_to_plot=channel_to_plot)

"""
Describe what you see. 

What do the two envelopes do when the stimulation frequency changes? 

    The envelopes show an increased amplitude at the corresponding stimulation frequency.
    When looking at the signal filtered to capture 15 Hz oscillations, we see
    the amplitude to increase when the subject attends to the 15 Hz stimulus and 
    decrease at other times. This is similar for the signal filtered to capture 12 Hz 
    oscillations (Wolpaw & Wolpaw, 2012, p. 242).
    
How large and consistent are those changes? 

    During the 12 Hz stimuli, the 12 Hz envelope appears to have a larger amplitude
    than the 15 Hz envelope somewhat consistently. During the 15 Hz stimuli, the 
    15 Hz envelope is not as consistentely larger than the 12 Hz envelope. 
    This could be related to the 12 Hz envelope including additional power from
    the alpha band. The 12 Hz envelope does have larger amplitdues during the 12 Hz
    stimuli than during the 15 Hz stimuli; this is pretty consisent. The 15 Hz
    envelope does not have as large or consistent of an amplitude increase during
    the 15 Hz stimuli compared to the 12 Hz stimuli; however, it is still noticeable.

Are the brain signals responding to the events in the way you’d expect? 

    This is expected because attending to the stimulus at a certain frequency 
    should elicit EEG activity at that frequency in the occipital region.
    
Check some other electrodes – which electrodes respond in the same way and why?

    The occipital (O1, O2, and Oz) and parietal (P3, P4, and Pz) electrdoes 
    respond similarly. The SSVEP originates in the occipital lobe, which is 
    resposible for vision and perception, and can be observed throughout other regions of the brain 
    that process vision (e.g., parietal and temporal).
    (Wolpaw & Wolpaw, 2012, p. 241).
    
"""

#%% 
# Part 6: Examine the Spectra

filter_ssvep_data.plot_filtered_spectra(data,filtered_data_15Hz,envelope_15Hz,["Fz","Oz"])

"""
Describe how the spectra change at each stage and why. 

    Raw to Filtered: In the filtered data, the max power is now at the stimuli frequency. 
    Filtering to the specific frequency at 15 Hz decreases the
    power of the spectra at other frequencies. In particular, the very low frequencies
    and the 50 Hz line noise frequencies were reduced. Thus, the spectra now appears
    to have the maximum power at 15 Hz. 
    
    Filtered to Envelope: In the enveloped data, the max power is now at a very low frequency. 
    The envelope rides on top of the 15 Hz signal to show the amplitude of the
    oscillations, so it no longer includes data at 15 Hz. Instead, the envelope
    shows much lower frequency activity, which make it easier to visualize 
    changes between frequency states (e.g., rest to 12 Hz to rest to 15 Hz).

    
Changes you should address include (but are not limited to) the following:
    
1. Why does the overall shape of the spectrum change after filtering?

    The overall shape changes because, at each stage of the analysis, signal
    content is being removed, which reduces the amount of power in the signal.
    The shape is based on the frequency that has the maximum power, which changes
    at each stage.
    
2. In the filtered data on Oz, why do 15Hz trials appear to have less power 
 12Hz trials at most frequencies?
 
     In this part, we are looking at data that were filtered to keep signal
     around 15 Hz. Thus, 12 Hz trials will have less power because the 12 Hz signal
     will have been removed by the filter. However, we are looking at power in
     dB, which normalizes the maximum power to start at 0. Therefore, while it might
     look like the 15 Hz trials have less power, it is actually because the 12 Hz
     trials have a smaller maximum power. If we were to look at the data that 
     were filtered to keep signal around 12 Hz, this trend would be reversed, 
     where 15 Hz trials appear to have MORE power. 
     
3. In the envelope on Oz, why do we no longer see any peaks at 15Hz?

    We no longer see peaks at the stimulus frequency because the envelope rides 
    on top of the actual frequency, representing it at a much lower frequency.

References:
Wolpaw JR, Elizabeth Winter Wolpaw. Brain-computer interfaces : principles and practice. New York: Oxford University Press; 2012.

"""

# %%
