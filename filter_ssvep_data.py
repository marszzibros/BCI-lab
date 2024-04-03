# -*- coding: utf-8 -*-
"""
filter_ssvep_data.py

Load steady-state visual evoked potentials (SSVEPs) data and use filtering and
the Hilbert transform to calculate signal amplitude at frequencies corresponding
to stimuli.

BME6710 BCI Spring 2024 Lab #4

The SSVEP dataset is derived from a tutorial in the MNE-Python package.The dataset
includes electropencephalograpy (EEG) data from 32 channels collected during
a visual checkerboard experiment, where the checkerboard flickered at 12 Hz or
15 Hz. 

The functions in this module can be used to filter and transform the data and 
calculate signal amplitudes.

Created on Mar 7 2024

@author: 
    Ardyn Olszko
    Jay Hwasung Jung
"""



from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import math
import import_ssvep_data

def calculate_power_spectra(data):
    """
    Descriptions
    --------------------------------        
    Use functions from previous module to calculate power spectra

    Trials/Events = E, Channels = C, Time = T, Frequency = F

    Args
    --------------------------------
    data, dict
        the raw data dictionary
    
    Returns
    -------
    fft_frequencies, F x 1 1D numpy float array
        an array (of length frequencies) that is the frequency corresponding to each
        column in the FFT. That is, eeg_epochs_fft[:,:,i] is the energy at frequency fft_frequencies[i] Hz.
    spectrum_db_12Hz, size n x F where n is the number of channels and F is the number of frequencies
        the mean power spectrum of 12Hz trials in dB
    spectrum_db_15Hz, size n x F where n is the number of channels and F is the number of frequencies
        the mean power spectrum of 15Hz trials in dB

    """


    # Extract 20-second epochs
    epoch_start_time = 0 # start time of epoch relative to the event (in seconds)
    epoch_end_time = 20 # end time of the epoch relative to the event (in seconds)
    eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data, epoch_start_time, epoch_end_time)

    # Get the frequency specturm for each trial and channel
    fs = data['fs'] # sampling frequency of EEG data
    eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)
    
    # Calculate the spectra for specified channels
    channels = data['channels'] # retrieve names of all channels
    spectrum_db_12Hz, spectrum_db_15Hz = import_ssvep_data.plot_power_spectrum(eeg_epochs_fft, \
                                                                fft_frequencies, is_trial_15Hz, \
                                                                channels, channels_to_plot=None, subject='')
    
    return fft_frequencies, spectrum_db_12Hz, spectrum_db_15Hz

def make_bandpass_filter(low_cutoff,high_cutoff,filter_order=10,fs=1000,filter_type='hann'):
    """
    Descriptions
    --------------------------------
    calculate bandpass filter using scipy.signal firwin and plot filter impulse
    and frequency response
    reference: mark-kramer.github.io/Case-Studies-Python

    Args
    --------------------------------
    low_cutoff, float
        the lower cutoff frequency (in Hz)
    high_cutoff, float
        the higher cutoff frequency (in Hz)
    filter_order, int
        the filter order (10 by default)
    fs, float
        the sampling frequency in Hz (1000 by default)
    filter_type, str
        the filter type (hann by default)

    Returns
    --------------------------------
    filter_coefficients, 1 X (filter_order + 1) 1D numpy float array
        calculated coefficients using scipy.signal firwin
    """    
    
    # Create filter coefficients
    filter_coefficients = signal.firwin(numtaps=filter_order + 1, 
                                        cutoff=[low_cutoff, high_cutoff],
                                        window=filter_type,
                                        pass_zero='bandpass',
                                        fs=fs)
     
    # Calculate and plot frequency response
    frequency, frequency_response = signal.freqz(b=filter_coefficients, fs=fs)

    # Convert to decibels
    # the formula is 20*np.log10()
    # https://www.mathworks.com/help/signal/ref/mag2db.html
    frequency_response_db = 20 * np.log10(np.abs(frequency_response))

    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6),
                            num=f'{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}',
                            clear=True)

    # Plot impulse response
    filter_times = np.arange(0, len(filter_coefficients)) / fs

    axs[0].plot(filter_times, filter_coefficients, label='Impulse Response')
    axs[0].set_title('Impulse Response')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Gain')

    max_coefficient = round(max(filter_coefficients), 3)
    axs[0].set_ylim(-(max_coefficient + 0.0005), (max_coefficient + 0.0005))
    axs[0].set_yticks([-(max_coefficient), 0, (max_coefficient)])
    axs[0].grid()

    # plot frequency response in amplitude db
    axs[1].plot(frequency, frequency_response_db, label='Frequency Response')
    min_rounded = 100 * math.floor(min(frequency_response_db) / 100)

    axs[1].set_title('Frequency Response')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude (dB)')
    axs[1].set_xlim(0, 40)
    axs[1].set_ylim(min_rounded, 10)
    axs[1].set_yticks([y_value for y_value in range(min_rounded + 100, 1, 100)])
    axs[1].grid()

    fig.suptitle(f"bandpass {filter_type} filter with fc=[{low_cutoff},{high_cutoff}], order={filter_order}")
    
    plt.tight_layout()
    plt.savefig(f'{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}.png')
    plt.show()

    return filter_coefficients

def filter_data(data,b):
    """
    Descriptions
    --------------------------------
    Apply FIR filter both forwards and backwards using scipy.signal filtfilt
    reference: https://mark-kramer.github.io/Case-Studies-Python/06.html

    Channels = C, Samples = T, filter order = O

    Args
    --------------------------------
    data, dict
        the raw data dictionary (with EEG data size C X T)
    b, 1 X O+1 1D array of floats
        filter coefficients for the raw data 
        
    Returns
    --------------------------------
    filtered_data, C X T 2D numpy array of floats
        Filter-coefficients-b-applied data across each channel. Units of uV.
    """    
    eeg = data['eeg']*1e6 # data converted to uV
    filtered_data = np.array([signal.filtfilt(b=b,a=1,x=eeg[channel_index]) for channel_index in range(len(eeg))])

    return filtered_data


def get_envelope(data,filtered_data,ssvep_frequency=None,channel_to_plot=None):
    """
    Descriptions
    --------------------------------
    Calculate envelope using scipy.signal hilbert and plot one provided channels

    Channels = C, Samples = T 

    Args
    --------------------------------
    data, dict
        the raw data dictionary
    filtered_data, C X T 2D numpy array of floats
        Filter-coefficients-b-applied data across each channel. Units of uV.
    ssvep_frequency, int
        the SSVEP frequency being isolated. The default is None.
    channel_to_plot, str
        an optional string indicating which channel you’d like to plot. The default is None.
        
    Returns
    --------------------------------
    envelope, C X T 2D numpy array of floats
        calculated envelope using scipy.signal hilbert for each channel. Units of uV.
    """    

    # extract data from dict
    eeg = data['eeg']
    channels = data['channels'] # name of each channel, in the same order as the eeg matrix.
    fs = data['fs'] # sampling frequency in Hz.
    
    # calculate time and envelope
    time = np.arange(0,1/fs*eeg.shape[1],1/fs) # time array
    envelope = np.abs(signal.hilbert(filtered_data))
    
    # Plot the envelope
    if channel_to_plot is not None:
        
        # determine frequency being isolated (for plot title)
        if ssvep_frequency is None:
            ssvep_frequency='unknown '
        
        # find channel to plot
        channel_index = np.where(channels==channel_to_plot)[0] 
        
        plt.figure(f'{channel_to_plot} envelope {ssvep_frequency}Hz',clear=True, figsize=(10, 4))
        plt.plot(time,filtered_data[channel_index].transpose(),label='Filtered Signal')
        plt.plot(time,envelope[channel_index].transpose(),label='Envelope')
        plt.title(f'{ssvep_frequency}Hz Bandpass Filtered Data')
        plt.ylabel('voltage (uV)')
        plt.xlabel('time (s)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{channel_to_plot}_envelope_{ssvep_frequency}Hz.png')
        plt.show()
        
    return envelope

def plot_ssvep_amplitudes(data,envelope_a,envelope_b,ssvep_freq_a,ssvep_freq_b,
                          subject,channel_to_plot):
    """
    Descriptions
    --------------------------------
    plot SSVEP amplitudes and compare envelops of two groups

    Channels = C, Samples = T 

    Args
    --------------------------------
    data, dict
        the raw data dictionary
    envelope_a, C X T 2D numpy array of float
        the envelope of oscillations at the first SSVEP frequency. Units of uV.
    envelope_b, C X T 2D numpy array of float
        the envelope of oscillations at the second SSVEP frequency. Units of uV.
    ssvep_frequency_a, int
        the SSVEP frequency (in Hz) being isolated in the first envelope
    ssvep_frequency_b, int
        the SSVEP frequency (in Hz) being isolated in the second envelope
    subject, int
        the subject number
    channel_to_plot, str
         string indicating which channel you’d like to plot
    """    
    
    # Convert event times to seconds
    event_start_times_sec = np.array(data['event_samples']) / data['fs']
    event_end_times_sec = (np.array(data['event_samples']) + np.array(data['event_durations'])) / data['fs']

    channel_index = np.where(data['channels'] == channel_to_plot)[0][0]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), 
                                   num=f'{channel_to_plot} amplitudes subject{subject}', 
                                   clear=True)

    # Top subplot: Event start and end times/types for 12Hz and 15Hz
    for start_sec, end_sec, event_type in zip(event_start_times_sec, event_end_times_sec, data['event_types']):
        frequency = int(event_type[:-2]) # Extract frequency from string
        ax1.plot([start_sec, end_sec], [frequency, frequency], color='blue', marker='o', linestyle='-')
        ax1.scatter([start_sec, end_sec], [frequency, frequency], color='blue', s=30) 

    # Set y-axis ticks for 12Hz and 15Hz
    ax1.set_yticks([12, 15])
    ax1.set_yticklabels(['12Hz', '15Hz'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flash Frequency')
    ax1.set_title(f"{channel_to_plot} Subject {subject} SSVEP Event Data for Amplitude Comparison")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Bottom subplot: envelope data
    time_axis = np.arange(0,len(envelope_a[channel_index]))/data['fs']
    
    ax2.plot(time_axis, envelope_a[channel_index], label=f"{ssvep_freq_a}Hz Envelope")
    ax2.plot(time_axis, envelope_b[channel_index], label=f"{ssvep_freq_b}Hz Envelope")

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (uV)')
    ax2.set_title("Envelope Comparison")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yticks([0 ,5, 10 ,15])
    
    plt.tight_layout()
    fig.savefig(f'SSVEP_subject{subject}_{channel_to_plot}_envelopes.png')
    
    # Show the plot
    plt.show()

def plot_filtered_spectra(data, filtered_data, envelope, channels = ["Fz", "Oz"]):
    """
    Descriptions
    --------------------------------
    plot filtered spectra and examine

    Channels = C, Samples = T, Number of channels to be plotted = N

    Args
    --------------------------------
    data, dict
        the raw data dictionary
    envelope, C X T 2D numpy array of floats
        calculated envelope using scipy.signal hilbert for each channel. Units of uV.
    filtered_data, C X T 2D numpy array of floats
        Filter-coefficients-b-applied data across each channel
    channels, 1 x N 1D array or list of str
        a list/array of the names of the channels you want to plot (default is Fz and Oz)
    """    
   
    # Prepare the data for plotting
    # modify the "data" variable in order to use existing functions for plotting
    data_dict = dict(data) # convert npzfile object to dict to allow modification
    filtered_dict = data_dict.copy() # create dict for filtered data
    filtered_dict['eeg'] = filtered_data # add filtered data
    envelope_dict = data_dict.copy() # create dict for envelope data
    envelope_dict['eeg'] = envelope # add envelope data
    
    # Plot the spectra for raw, filtered, envelope
    channels_from_data = data['channels'] # retrieve names of all channels
    channels_to_plot = channels # specify channels to plot
    fig,axs=plt.subplots(len(channels_to_plot),3,sharex=True,sharey=True,num='spectra',clear=True, figsize=(12, 3 * len(channels_to_plot))) # set up figure
    fig.suptitle('Spectra at 3 stages of analysis')
    fig.supxlabel('frequency (Hz)')
    fig.supylabel('power (dB)')
    axs=axs.flatten()
    subplot_index = 0
    for channel in channels_to_plot:
        
        channel_index = np.where(channels_from_data==channel)[0]
        
        # raw data
        fft_frequencies, spectrum_db_12Hz, spectrum_db_15Hz = calculate_power_spectra(data_dict)
        axs[subplot_index].plot(fft_frequencies,spectrum_db_15Hz[channel_index,:].transpose(),label='15Hz trials',color='r')
        axs[subplot_index].plot(fft_frequencies,spectrum_db_12Hz[channel_index,:].transpose(),label='12Hz trials',color='g')
        axs[subplot_index].set_title(f'{channel} Raw')
        axs[subplot_index].legend()
        axs[subplot_index].grid()
        subplot_index+=1 
        
        # filtered data
        fft_frequencies, spectrum_db_12Hz, spectrum_db_15Hz = calculate_power_spectra(filtered_dict)
        axs[subplot_index].plot(fft_frequencies,spectrum_db_15Hz[channel_index,:].transpose(),label='15Hz trials',color='r')
        axs[subplot_index].plot(fft_frequencies,spectrum_db_12Hz[channel_index,:].transpose(),label='12Hz trials',color='g')
        axs[subplot_index].set_title(f'{channel} Filtered')
        axs[subplot_index].legend()
        axs[subplot_index].grid()
        subplot_index+=1 
        
        # envelope
        fft_frequencies, spectrum_db_12Hz, spectrum_db_15Hz = calculate_power_spectra(envelope_dict)
        axs[subplot_index].plot(fft_frequencies,spectrum_db_15Hz[channel_index,:].transpose(),label='15Hz trials',color='r')   
        axs[subplot_index].plot(fft_frequencies,spectrum_db_12Hz[channel_index,:].transpose(),label='12Hz trials',color='g')
        axs[subplot_index].set_title(f'{channel} Envelope')
        axs[subplot_index].legend()
        axs[subplot_index].grid()
        subplot_index+=1 
    
    axs[subplot_index-1].set_xlim((0,80))  # zoom in to relevant frequencies 
    plt.tight_layout()   
    plt.savefig("Spectra_" + "_".join(channels_to_plot) + ".png")
    plt.show()




