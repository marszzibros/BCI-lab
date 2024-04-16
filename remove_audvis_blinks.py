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
from plot_topo import plot_topo


# Part 1: Load data


def load_data(data_directory, channels_to_plot):
    """

    Parameters
    ----------
    data_directory: string, the relative filepath to the directory where the audvis data file is located.
    channels_to_plot: list of strings, names of the channels to plot.

    Returns
    -------

    """
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

    return data


def plot_components(mixing_matrix, channels, components_to_plot=np.arange(10)):
    """

    Parameters
    ----------
    mixing_matrix: array of size N x A with A being the number of weights used to combine across sources the
    timpact of a source on all electrodes
    channels: array of size 1 x number of channels, the name of each channel in the same order as th eeg matrix.
    components_to_plot: array of ints, the indices of the components to plot. Default: size 1 x 10
    """
    fig, axs = plt.subplots(int(np.ceil(len(components_to_plot) / 5)), 5,
                            figsize=(20, 5 * int(np.ceil(len(components_to_plot) / 5))))

    for topo_index, channel_index in enumerate(components_to_plot):
        plt.subplot(int(np.ceil(len(components_to_plot) / 5)), 5, topo_index + 1)

        im, cbar = plot_topo(channel_names=list(channels),
                             channel_data=mixing_matrix.T[channel_index],
                             title=f"ICA Component {topo_index}",
                             cbar_label="", montage_name="standard_1005")

        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].imshow(cbar.ax.figure.canvas.renderer.buffer_rgba())
        fig.colorbar(im, ax=axs[int(np.floor(topo_index / 5))][int(topo_index % 5)], fraction=0.05)
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].set_title(f"ICA component {topo_index}")

        # Remove ticks and tick labels from both axes
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].set_xticks([])
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].set_yticks([])

        # Remove the axis lines
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].tick_params(top=False,
                                                                            bottom=False,
                                                                            left=False,
                                                                            right=False,
                                                                            labelleft=False,
                                                                            labelbottom=False)

    plt.tight_layout()
    fig.savefig("ICA_component_topo.png")


def get_sources(eeg, unmixing_matrix, fs, sources_to_plot=[]):
    """

    Parameters
    ----------
    eeg: array of size channels x samples, eeg data in volts.
    unmixing_matrix: array
    fs: float, the sampling frequency in Hz.
    sources_to_plot: list of ints, the indices of the sources to plot.

    Returns
    -------

    """
    source_activations = np.matmul(unmixing_matrix, eeg)
    if len(sources_to_plot) != 0:

        fig, axes = plt.subplots(len(sources_to_plot), 1, sharex='all', figsize=(10, 3 * len(sources_to_plot)))
        fig.suptitle('AudVis EEG Data in ICA source space')

        for source_to_plot_index, source in enumerate(sources_to_plot):
            axes[source_to_plot_index].plot(np.arange(eeg.shape[1]) / fs,
                                            source_activations[sources_to_plot[source_to_plot_index]],
                                            label="reconstructed")
            axes[source_to_plot_index].set_ylabel(f'Source {source} (uV)')
            axes[source_to_plot_index].grid()
            axes[source_to_plot_index].set_xlim([55, 60])
            axes[source_to_plot_index].legend()

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
        plt.savefig("AudVis_EEG_Data_in_ICA_source_space.png")

    return source_activations


def remove_sources(source_activations, mixing_matrix, sources_to_remove):
    """

    Parameters
    ----------
    source_activations:
    mixing_matrix:
    sources_to_remove:

    Returns
    -------

    """
    source_activations[sources_to_remove, :] = 0
    cleaned_eeg = np.matmul(mixing_matrix, source_activations)
    return cleaned_eeg


def compare_reconstructions(eeg, reconstructed_eeg, cleaned_eeg, fs, channels, channels_to_plot):
    """

    Parameters
    ----------
    eeg
    reconstructed_eeg
    cleaned_eeg
    fs
    channels
    channels_to_plot
    """
    if len(channels_to_plot) != 0:

        channel_index = np.where(np.isin(channels, channels_to_plot))[0]

        fig, axes = plt.subplots(len(channels_to_plot), 1, sharex='all', figsize=(10, 3 * len(channels_to_plot)))
        fig.suptitle('AudVis EEG Data reconstructed & cleaned after ICA')

        for channel_to_plot_index, channel_name in enumerate(channels_to_plot):
            axes[channel_to_plot_index].plot(np.arange(eeg.shape[1]) / fs, eeg[channel_index[channel_to_plot_index]],
                                             label="raw")
            axes[channel_to_plot_index].plot(np.arange(reconstructed_eeg.shape[1]) / fs,
                                             reconstructed_eeg[channel_index[channel_to_plot_index]],
                                             label="reconstructed", linestyle='dashed')
            axes[channel_to_plot_index].plot(np.arange(cleaned_eeg.shape[1]) / fs,
                                             cleaned_eeg[channel_index[channel_to_plot_index]], label="cleaned",
                                             linestyle='dotted')

            axes[channel_to_plot_index].set_ylabel(f'Voltage on {channel_name} (uV)')
            axes[channel_to_plot_index].grid()
            axes[channel_to_plot_index].legend()
            axes[channel_to_plot_index].set_xlim([55, 60])
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
        plt.savefig("AudVis_EEG_Data_reconstructed_cleaned_after_ICA.png")
