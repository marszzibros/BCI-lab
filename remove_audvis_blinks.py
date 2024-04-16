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


def load_data(data_directory, channels_to_plot):
    dataset = np.load(data_directory, allow_pickle=True).item()
    # Extract fields from data set
    eeg_data = dataset['eeg']
    channels = dataset['channels']
    fs = dataset['fs']
    event_samples = dataset['event_samples']
    event_types = dataset['event_types']
    unmixing_matrix = dataset['unmixing_matrix']
    mixing_matrix = dataset['mixing_matrix']

    # Create data dict
    data = {
        'eeg': eeg_data,
        'channels': channels,
        'fs': fs,
        'event_samples': event_samples,
        'event_types': event_types,
        'unmixing_matrix': unmixing_matrix,
        'mixing_matrix': mixing_matrix
    }

    # If channels_to_plot is empty, return the dataset
    if not channels_to_plot:
        return data

    # Plot raw data
    num_channels = len(channels_to_plot)
    num_samples = eeg_data.shape[1]

    fig, axes = plt.subplots(num_channels, 1, sharex='all', figsize=(10, 5))
    fig.suptitle('Raw AudVis EEG Data ')
    for i, channel_name in enumerate(channels_to_plot):
        channel_index = np.where(channels == channel_name)[0][0]
        axes[i].plot(np.arange(num_samples) / fs, eeg_data[channel_index])
        axes[i].set_ylabel(f'Voltage on {channel_name} (uV)')
        axes[i].grid()

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    return data





