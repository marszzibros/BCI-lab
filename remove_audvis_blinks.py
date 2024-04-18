"""
This script provides functions for loading EEG data, plotting raw data, plotting ICA components, and performing
source reconstruction and data cleaning.

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

def load_data(data_directory, channels_to_plot = []):
    """
    This function is designed to load AudVisData, and if channels are specified, visualize raw data
    
    Parameters
    ----------
    data_directory: string, 
        the relative filepath to the directory where the audvis data file is located.
    channels_to_plot: list,
        list of strings, names of the channels to plot, default is an empty string

    Returns
    -------
    data: dict,
        contains the data loaded from the data directory
    """

    # load AudVisData
    data = np.load(data_directory, allow_pickle=True).item()

    # check if it is empty list
    if len(channels_to_plot) != 0:

        # find channel indice from channels_to_plot
        channel_index = np.where(np.isin(data['channels'], channels_to_plot))[0]

        # define subplots
        fig, axes = plt.subplots(len(channels_to_plot), 1, sharex='all', figsize=(10, 3 * len(channels_to_plot)))
        fig.suptitle('Raw AudVis EEG Data')

        # plot raw data 
        for channel_to_plot_index, channel_name in enumerate(channels_to_plot):
            
            axes[channel_to_plot_index].plot(np.arange(data['eeg'].shape[1]) / data['fs'], data['eeg'][channel_index[channel_to_plot_index]])
            axes[channel_to_plot_index].set_ylabel(f'Voltage on {channel_name} (uV)')
            axes[channel_to_plot_index].grid()
            # zoom in
            #axes[channel_to_plot_index].set_xlim([0,30])
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig("Raw_AudVis_EEG_Data.png")

    return data

def plot_components(mixing_matrix, channels, components_to_plot=np.arange(10)):
    """
    This function is designed to plot topology of 10 components for ICA components
    
    W: weights used to combine accross sources to get the activity for a single electrode
    S: impact of a saource on all electrodes
    C: channels in raw data dict

    Parameters
    ----------
    mixing_matrix: W X S numpy array,
        ICA components
    channels: C X 1 numpy array,
        channels name from raw data dict, the name of each channel in the same order as th eeg matrix
    components_to_plot: numpy array,
        the indices (Integers) of the components to plot. Default: np aranged array (1-10)
    """
    # just in case it is not a numpy array
    components_to_plot = np.array(components_to_plot)

    # define subplots
    fig, axs = plt.subplots(int(np.ceil(len(components_to_plot) / 5)), 5,
                            figsize=(20, 5 * int(np.ceil(len(components_to_plot) / 5))))

    # get components to plot
    for topo_index, channel_index in enumerate(components_to_plot):

        # get plot topo by passing corresponding mixing matrix
        im, cbar = plot_topo(channel_names=list(channels),
                             channel_data=mixing_matrix.T[channel_index],
                             title=f"ICA Component {topo_index}",
                             cbar_label="", montage_name="standard_1005")

        # get image from buffer to plot image in subplots
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].imshow(cbar.ax.figure.canvas.renderer.buffer_rgba())

        # draw colorbar
        fig.colorbar(im, ax=axs[int(np.floor(topo_index / 5))][int(topo_index % 5)], fraction=0.05)

        # set title for each ICA component
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].set_title(f"ICA component {topo_index}")

        # Remove the axis lines and frame
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].tick_params(top=False,
                                                                            bottom=False,
                                                                            left=False,
                                                                            right=False,
                                                                            labelleft=False,
                                                                            labelbottom=False)
        
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].spines['top'].set_visible(False)
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].spines['right'].set_visible(False)
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].spines['bottom'].set_visible(False)
        axs[int(np.floor(topo_index / 5))][int(topo_index % 5)].spines['left'].set_visible(False)

    # remove the unprocessed parts of the frame
    if len(components_to_plot) % 5 != 0:
        row_index = int(np.floor(len(components_to_plot) / 5))
        print(row_index)
        for col_index in range(len(components_to_plot) % 5, 5):

            # Remove the axis lines
            axs[row_index][col_index].tick_params(top=False,
                                                bottom=False,
                                                left=False,
                                                right=False,
                                                labelleft=False,
                                                labelbottom=False)
            axs[row_index][col_index].spines['top'].set_visible(False)
            axs[row_index][col_index].spines['right'].set_visible(False)
            axs[row_index][col_index].spines['bottom'].set_visible(False)
            axs[row_index][col_index].spines['left'].set_visible(False)

    # save figure
    plt.tight_layout()
    fig.savefig("ICA_component_topo.png")

def get_sources(eeg, unmixing_matrix, fs, sources_to_plot=[]):
    """
    This function is designed to transform into source space with eeg raw data and unmixing_matrix.
    It plots sources if users select. 

    C: number of channels
    S: number of samples
    W: weights used to combine accross electrodes to get the activity for a single source

    Parameters
    ----------
    eeg: C X S numpy array,
        eeg data in microvolts.
    unmixing_matrix: W X C numpy array,
        ICA source transformation
    fs: float, 
        the sampling frequency in Hz.
    sources_to_plot: list,
        the indices (Integer) of the sources to plot.

    Returns
    -------
    source_activations, W x S numpy array,
        source activation timecourses
    """

    # calculate source_activations
    source_activations = np.matmul(unmixing_matrix, eeg)

    # if sources_to_plot is specified
    if len(sources_to_plot) != 0:

        # define subplots
        fig, axes = plt.subplots(len(sources_to_plot), 1, sharex='all', figsize=(10, 3 * len(sources_to_plot)))
        fig.suptitle('AudVis EEG Data in ICA source space')

        # plot subplots
        for source_to_plot_index, source in enumerate(sources_to_plot):
            axes[source_to_plot_index].plot(np.arange(eeg.shape[1]) / fs,
                                            source_activations[sources_to_plot[source_to_plot_index]],
                                            label="reconstructed")
            axes[source_to_plot_index].set_ylabel(f'Source {source} (uV)')
            axes[source_to_plot_index].grid()

            # manual zoom in
            axes[source_to_plot_index].set_xlim([55, 60])
            axes[source_to_plot_index].legend()

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig("AudVis_EEG_Data_in_ICA_source_space.png")

    return source_activations

def remove_sources(source_activations, mixing_matrix, sources_to_remove):
    """
    This function is designed to remove sources by assigning 0 to sources_to_remove in source_activations

    S: number of samples
    W_s: weights used to combine accross electrodes to get the activity for a single source
    W_e: weights used to combine accross sources to get the activity for a single electrode
    S: impact of a saource on all electrodes

    Parameters
    ----------
    source_activations: W_s X S numpy array,
        source activation timecourses
    mixing_matrix: W_e X S numpy array,
        ICA components
    sources_to_remove: list, 
        the indices (Integers) of the sources to remove from the data.

    Returns
    -------
    cleaned_eeg: W_e X S numpy array,
        the matrix product of the mixing matrix and source activations
    """

    # assign 0 to remove
    source_activations[sources_to_remove, :] = 0

    # matmul to convert to eeg
    cleaned_eeg = np.matmul(mixing_matrix, source_activations)
    return cleaned_eeg

def compare_reconstructions(eeg, reconstructed_eeg, cleaned_eeg, fs, channels, channels_to_plot):
    """
    This function is designed to plot raw eeg, reconstructed_eeg, and cleaned_eeg to compare

    C: number of channels
    S: number of samples

    Parameters
    ----------
    eeg: C X S numpy array,
        eeg data in microvolts.
    reconstructed_eeg: C X S numpy array,
        reconstructed eeg data in microvolts.
    cleaned_eeg: C X S numpy array,
        cleaned eeg data in microvolts.
    fs: float, 
        the sampling frequency in Hz.
    channels: C X 1 numpy array,
        channels name from raw data dict, the name of each channel in the same order as th eeg matrix
    channels_to_plot: list, 
        the names (string) of channels to plot.
    """

    # check if channels_to_plot is specified
    if len(channels_to_plot) != 0:
        
        # find indeces
        channel_index = np.where(np.isin(channels, channels_to_plot))[0]

        # define subplots
        fig, axes = plt.subplots(len(channels_to_plot), 1, sharex='all', figsize=(10, 3 * len(channels_to_plot)))
        fig.suptitle('AudVis EEG Data reconstructed & cleaned after ICA')

        # draw plots
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

            # zoomed in
            axes[channel_to_plot_index].set_xlim([55, 60])
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig("AudVis_EEG_Data_reconstructed_cleaned_after_ICA.png")
