# -*- coding: utf-8 -*-
"""
import_ssvep_data.py

Load data and plot frequency specturm of steady-state visual evoked potentials (SSVEPs).
BME6710 BCI Spring 2024 Lab #3

The SSVEP dataset is derived from a tutorial in the MNE-Python package.The dataset
includes electropencephalograpy (EEG) data from 32 channels collected during
a visual checkerboard experiment, where the checkerboard flickered at 12 Hz or
15 Hz. 

The functions in this module can be used to load the dataset into variables, 
plot the raw data, epoch the data, calculate the Fourier Transform (FT), 
and plot the power spectra for each subject. 

Created on Feb 24 2024

@author: 
    Ardyn Olszko
    Yoonki Hong
    Jay Hwasung Jung - modified
"""

# import packages

import numpy as np
from matplotlib import pyplot as plt
import scipy.fft 

# function to load data
def load_ssvep_data(subject, data_directory):
    '''
    Load the SSVEP EEG data for a given subject.
    The format of the data is described in the README.md file with the data.

    Parameters
    ----------
    subject : int
        Subject number, 1 or 2.
    data_directory : str
        Path to the folder where the data files exist.

    Returns
    -------
    data_dict : dict
        Dictionary of data for a subject.

    '''
    
    # Load dictionary
    data_dict = np.load(data_directory + f'/SSVEP_S{subject}.npz',allow_pickle=True)    

    return data_dict


# function to plot the raw data
def plot_raw_data(data,subject,channels_to_plot):
    '''
    Plot events and raw data from specified electrodes.
    Creates a figure with two subplots, where the first is the events and the
    second is the EEG voltage for specified channels. The figure is saved to
    the current directory.

    Parameters
    ----------
    data : dict
        Dictionary of data for a subject.
    subject : int
        Subject number, 1 or 2 (used to annotate plot)
    channels_to_plot : list or array of size n where n is the number of channels
        Channel names of data to plot. Channel name must be in "data['channels']".

    Returns
    -------
    None.

    '''
    
    # extract variables from dictionary
    eeg = data['eeg'] # eeg data in Volts. Each row is a channel and each column is a sample.
    channels = data['channels'] # name of each channel, in the same order as the eeg matrix.
    fs = data['fs'] # sampling frequency in Hz.
    event_samples = data['event_samples'] # sample when each event occurred.
    event_durations = data['event_durations'] # durations of each event in samples.
    event_types = data['event_types'] # frequency of flickering checkerboard for each event.
    
    # calculate time array
    time = np.arange(0,1/fs*eeg.shape[1],1/fs)
    
    # set up figure
    plt.figure(f'raw subject{subject}',clear=True)
    plt.suptitle(f'SSVEP subject {subject} raw data')
    
    # plot the event start and end times and types
    ax1 = plt.subplot(2,1,1)
    start_times = time[event_samples]
    end_times = time[event_samples+event_durations.astype(int)]
    for event_type in np.unique(event_types):
        is_event = event_types == event_type
        plt.plot()
        event_data = np.array([start_times[is_event],end_times[is_event]])
        plt.plot(event_data,
                 np.full_like(event_data,float(event_type[:-2])),
                 marker='o',linestyle='-',color='b',
                 label=event_type)
    plt.xlabel('time (s)')
    plt.ylabel('flash frequency (Hz)')
    plt.grid()
    
    # plot the raw data from the channels spcified
    plt.subplot(2,1,2, sharex=ax1)
    for channel in channels_to_plot:
        is_channel = channels == channel
        plt.plot(time, 10e5*eeg[is_channel,:].transpose(),label=channel) # multiply by 10e5 to convert to uV (confirmed that this matches the original dataset from mne)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (uV)')
    plt.grid()
    plt.legend()
    
    # save the figure
    plt.tight_layout()
    plt.savefig(f'SSVEP_S{subject}_rawdata.png')
    
    return

def epoch_ssvep_data(data_dict, epoch_start_time = 0, epoch_end_time = 20):
    """
    Descriptions
    --------------------------------
    epoch ssvep data based on epoch_start_time and epoch_end_time

    Trials/Events = E, Channels = C, Time = T

    Args
    --------------------------------
    data_dict, dict
        the dictionary variable that stores ssvep data of given subject
    epoch_start_time, float
        the path to the folder where the data afiles sit on your computer
    epoch_end_time, float
        the path to the folder where the data afiles sit on your computer

    Returns
    --------------------------------
    eeg_epochs, E x C x T 3D numpy float array,
        contains the EEG data after
    epoch_times, T x 1 1D numpy float array
        the time in seconds (relative to the event) of each time point in eeg_epochs
    is_trial_15Hz, E x 1 1D numpy bool array
        an array in which is_trial_15Hz[i] is True if the light was flashing at 15Hz during trial/epoch i.

    """
    # change units to micro volt
    eeg_data = data_dict['eeg']

    # Get events index
    epoch_starts = np.array(data_dict['event_samples']) + epoch_start_time * data_dict['fs'] 
    epoch_ends = epoch_starts + data_dict['fs'] * (epoch_end_time - epoch_start_time)

    num_epochs = len(data_dict['event_samples'])
    num_channels = eeg_data.shape[0]
    num_time = int((epoch_end_time - epoch_start_time) * data_dict['fs'])

    # Initialize the eeg_epochs array
    eeg_epochs = np.zeros((num_epochs, num_channels, num_time))

    for epoch_index, (epoch_start_index, epoch_end_index) in enumerate(zip(epoch_starts, epoch_ends)):
        epoch_data = eeg_data[:, int(epoch_start_index):int(epoch_end_index)] 

        # Check if the epoch_data does not fill the entire epoch length
        if epoch_data.shape[1] < num_time:
            # Fill the lacking part with zeros
            epoch_data = np.pad(epoch_data, ((0, 0), (0, num_time - epoch_data.shape[1])), mode='constant')

        eeg_epochs[epoch_index] = epoch_data * 1e+6

    eeg_epochs = eeg_epochs.astype(np.float64)

    # define epoch_times 

    epoch_times = np.arange(epoch_start_time, 
                            epoch_end_time,
                            1 / data_dict['fs'])


    # define is_trial_15Hz
    is_trial_15Hz = np.array(data_dict['event_types'] == "15hz", dtype = bool)

    return eeg_epochs, epoch_times, is_trial_15Hz

def get_frequency_spectrum(eeg_epochs, fs):
    """
    Descriptions
    --------------------------------
    calculate fourier transform using rfft and rfftfreq

    Trials/Events = E, Channels = C, Time = T, Frequency = F

    Args
    --------------------------------
    eeg_epochs, E x C x T 3D numpy float array,
        contains the EEG data after
    fs, int
        sampling frequency

    Returns
    --------------------------------

    eeg_epochs_fft, E x C x F 3D numpy float array
        an array that is the same size as eeg_epochs, except the final dimension now represents frequency 
        instead of time (i.e., size [trials, channels, frequencies]).
    fft_frequencies, F x 1 1D numpy float array
        an array (of length frequencies) that is the frequency corresponding to each
        column in the FFT. That is, eeg_epochs_fft[:,:,i] is the energy at frequency fft_frequencies[i] Hz.

    """
    
    eeg_epochs_fft = np.fft.rfft(eeg_epochs - eeg_epochs.mean(), axis = 2)
    fft_frequencies = np.fft.rfftfreq(eeg_epochs.shape[2], 1 / fs)

    return eeg_epochs_fft, fft_frequencies

# function to plot the mean power spectra for specified channesl
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):
    '''
    Calculate and plot the mean power spectra for specified channels.
    Each channel is plotted on a separate subplot. Event types, 12 Hz and 15 Hz,
    are plotted separately for each channel.

    Parameters
    ----------
    eeg_epochs_fft : array of float, size M x N X F where M is the number of trials,
    N is the number of channels, and F is number of frequencies measured in the epoch
        FT frequency content of each channel in each epoch.
    fft_frequencies : array of float, size F
        Frequencies measured, where the maximum frequency measured is 1/2*fs.
    is_trial_15Hz : array of bool, size M where M is the number of trials
        Event label, where True is 15 Hz event and False is 12 Hz event.
    channels : list of size N where N is the number of channels
        Channel names available in  the original dataset.
    channels_to_plot : list or array of size n where n is the number of channels
        Channel names of data to plot. Channel name must be in "data['channels']".
    subject : int
        Subject number, 1 or 2 (used to annotate plot)
        
    Returns
    -------
    spectrum_db_12Hz : array of float, size n x F where n is the number of channels
    and F is the number of frequencies
        Mean power spectrum of 12 Hz trials. Units in dB.
    spectrum_db_15Hz : array of float, size n x F where n is the number of channels
    and F is the number of frequencies
        Mean power spectrum of 15 Hz trials. Units in dB.

    '''
    
    # calculate mean power spectra for each channel
    signal_power = abs(eeg_epochs_fft)**2 # calculate power by squaring absolute value
    # calculate mean across trials
    power_12Hz = np.mean(signal_power[~is_trial_15Hz],axis=0)
    power_15Hz = np.mean(signal_power[is_trial_15Hz],axis=0)
    
    # normalize (divide by max value)
    norm_power_12Hz = power_12Hz/np.reshape(np.max(power_12Hz, axis=1), (power_12Hz.shape[0],1))
    norm_power_15Hz = power_15Hz/np.reshape(np.max(power_15Hz, axis=1), (power_12Hz.shape[0],1))
    
    # convert to decibel units
    power_db_12Hz = 10*np.log10(norm_power_12Hz)
    power_db_15Hz = 10*np.log10(norm_power_15Hz)
    
    if channels_to_plot:
        # set up figure and arrays for mean power spectra
        channel_count = len(channels_to_plot)
        freq_count = len(fft_frequencies)
        spectrum_db_12Hz = np.full([channel_count,freq_count],np.nan,dtype=float) # set up arrays to store power spectrum
        spectrum_db_15Hz = np.full_like(spectrum_db_12Hz,np.nan)
        row_count = int(np.ceil(np.sqrt(channel_count))) # calculate number of rows of subplots
        if (row_count**2 - channel_count) >= row_count: # calculate number of columns of subplots
            col_count = row_count-1 
        else:
            col_count = row_count
    
        fig = plt.figure(f'spectrum subject{subject}',clear=True,figsize=(6+0.5*channel_count,6+0.5*channel_count))
        plt.suptitle(f'Frequency content for SSVEP subject {subject}')
        axs=[] # set up empty list for subplot axes
        
        # plot and extract data for specified channels
        for channel_index, channel in enumerate(channels_to_plot):
            is_channel = channels == channel
            spectrum_db_12Hz[channel_index,:] = power_db_12Hz[is_channel,:]
            spectrum_db_15Hz[channel_index,:] = power_db_15Hz[is_channel,:]
            
            if channel_index == 0: 
                axs.append(fig.add_subplot(row_count,col_count,channel_index+1))
            else:
                axs.append(fig.add_subplot(row_count,col_count,channel_index+1,
                                           sharex=axs[0],
                                           sharey=axs[0]))
            # plot the mean power spectra
            axs[channel_index].plot(fft_frequencies,spectrum_db_12Hz[channel_index,:],label='12Hz',color='r')
            axs[channel_index].plot(fft_frequencies,spectrum_db_15Hz[channel_index,:],label='15Hz',color='g')
            # plot corresponding frequencies
            axs[channel_index].axvline(12,color='r',linestyle=':')
            axs[channel_index].axvline(15,color='g',linestyle=':')
            # annotate
            axs[channel_index].set_title(channel)
            axs[channel_index].set(xlabel='frequency (Hz)',ylabel='power (db)')
            axs[channel_index].grid()
            axs[channel_index].legend()
               
        plt.tight_layout()
    else:
        spectrum_db_12Hz = power_db_12Hz
        spectrum_db_15Hz = power_db_15Hz
        
    return spectrum_db_12Hz, spectrum_db_15Hz
