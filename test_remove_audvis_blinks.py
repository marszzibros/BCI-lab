"""
This script uses functions from filter_ssvep_data.py to processes and analyz EEG data. The script imports
necessary functions and loads data for a specific subject, crates bandpass filters centered around 12Hz and 15Hz,
filters the EEG signals using the bandpass filters, extracts envelopes for both the 12Hz and 15Hz frequency bands,
plots the amplitudes of the SSVEP responses, and plots the spectra of the filtered signals.

@author: marszzibros
@author: APC-03

file: test_remove_audvis_blinks.py
BME 6710 - Dr. Jangraw
lab 5: Spatial Components
"""

# Import statement
import remove_audvis_blinks as rmv

# Part 1: Load Data
data_directory = '/Users/aiden/PycharmProjects/BCI_202401/AudVisData.npy'
channels_to_plot = ['Fpz', 'Cz', 'Iz']
rmv.load_data(data_directory, channels_to_plot)
# It appears that a blink may have occurred around 15 seconds into the recording. Whenever a blink occurs,
# there is a large positive deflection in voltage primarily on frontal electrodes. At this time point there is a
# large positive voltage deflection at the Fpz electrode, located on the frontal lobe. There are smaller positive
# voltage deflections in the Cz and Iz electrodes, located in a central and occipital area respectively according to
# the standard 10/10 system.

