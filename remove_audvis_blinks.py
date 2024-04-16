"""
This script uses functions from filter_ssvep_data.py to processes and analyz EEG data. The script imports
necessary functions and loads data for a specific subject, crates bandpass filters centered around 12Hz and 15Hz,
filters the EEG signals using the bandpass filters, extracts envelopes for both the 12Hz and 15Hz frequency bands,
plots the amplitudes of the SSVEP responses, and plots the spectra of the filtered signals.

@author: marszzibros
@author: APC-03

file: remove_audvis_blinks.py
BME 6710 - Dr. Jangraw
lab 5: Spatial Components
"""

# Import Statements
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Load Data


def load_data(data_directory, channels_to_plot = None):

    # already dictionary
    data = np.load(data_directory, allow_pickle=True).item()

    # If channels_to_plot is empty, return the dataset
    if channels_to_plot is not None:

        channel_index = np.where(np.isin(data['channels'], channels_to_plot))[0]

        fig, axes = plt.subplots(len(channels_to_plot), 1, sharex='all', figsize=(10, 5))
        fig.suptitle('Raw AudVis EEG Data')

        for channel_to_plot_index, channel_name in enumerate(channels_to_plot):
            
            axes[channel_to_plot_index].plot(np.arange(data['eeg'].shape[1]) / data['fs'], data['eeg'][channel_index[channel_to_plot_index]])
            axes[channel_to_plot_index].set_ylabel(f'Voltage on {channel_name} (uV)')
            axes[channel_to_plot_index].title(channel_name)
            axes[channel_to_plot_index].grid()

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig("Raw_AudVis_EEG_Data")

    return data





