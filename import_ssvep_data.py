import numpy as np
import matplotlib.pyplot as plt

def load_ssvep_data (subject = 1, data_directory = "SsvepData/"):
    """
    Descriptions
    --------------------------------
    load ssvep data into numpy array

    Args
    --------------------------------
    subject, int
        the subject number
    data_directory, string
        the path to the folder where the data afiles sit on your computer

    Returns
    --------------------------------
    data_dict, dict
        loaded data of subject # 'subject' from 'data_directory'

    """

    # define defaulted file info 
    file_name = "SSVEP_S"
    file_extension = ".npz"

    # load data dict from npz file
    data_dict = np.load(f"{data_directory}{file_name}{subject}{file_extension}", allow_pickle=True)

    return data_dict


def plot_raw_data(data, subject = 1, channels_to_plot = ["Fz", "Oz"]):
    """
    Descriptions
    --------------------------------
    plot ssvep raw data, with given subject and channels

    Args
    --------------------------------
    data, dict
        the dictionary variable that stores ssvep data of given subject
    subject, int
        the subject number
    channels_to_plot, 1D list/array
        a list/array of channel names you want to plot in different subplots

    """
    # Convert event times to seconds
    event_start_times_sec = np.array(data['event_samples'][:4]) / data['fs']
    event_end_times_sec = (np.array(data['event_samples'][:4]) + np.array(data['event_durations'][:4])) / data['fs']

    channel_index = []
    for channel in channels_to_plot:
        channel_index.append(np.where(data['channels'] == channel)[0][0])

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Top subplot: Event start and end times/types for 12Hz and 15Hz
    for start_sec, end_sec, event_type in zip(event_start_times_sec, event_end_times_sec, data['event_types'][:4]):
        frequency = int(event_type[:-2]) # Extract frequency from string
        ax1.plot([start_sec, end_sec], [frequency, frequency], color='blue', marker='o', linestyle='-')
        ax1.scatter([start_sec, end_sec], [frequency, frequency], color='blue', s=30) 

    ax1.set_yticks([12, 15])  # Set y-axis ticks for 12Hz and 15Hz
    ax1.set_yticklabels(['12Hz', '15Hz'])
    ax1.set_xlabel('Time (s)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Bottom subplot: Raw electrode data
    for channel_index, channel in zip(channel_index, channels_to_plot):
        time_axis = np.arange(0, len(data['eeg'][channel_index][:int(event_end_times_sec[3] * data['fs'])])) / data['fs']

        # convert volts to micro-volts
        ax2.plot(time_axis, data['eeg'][channel_index][:int(event_end_times_sec[3] * data['fs'])] * 1e6, label=channel)


    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (uV)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(f'SSVEP subject  {subject} Raw Data', fontsize=16)
    fig.savefig(f'SSVEP_subject_{subject}_Raw_Data.png')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

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
    eeg_data = data_dict['eeg']

    # Get events index
    epoch_starts = np.array(data_dict['event_samples']) + epoch_start_time * data_dict['fs'] 
    epoch_ends = epoch_starts + data_dict['fs'] * epoch_end_time

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

        eeg_epochs[epoch_index] = epoch_data

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
    
    eeg_epochs_fft = np.fft.rfft(eeg_epochs - eeg_epochs.mean(), axis = -1)
    fft_frequencies = np.fft.rfftfreq(eeg_epochs.shape[2], fs)

    return eeg_epochs_fft, fft_frequencies

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):
    """
    Descriptions
    --------------------------------
    calculate fourier transform using rfft and rfftfreq

    Trials/Events = E, Channels = C, Time = T, Frequency = F

    Args
    --------------------------------
    eeg_epochs_fft, E x C x F 3D numpy float array
        an array that is the same size as eeg_epochs, except the final dimension now represents frequency 
        instead of time (i.e., size [trials, channels, frequencies]).
    fft_frequencies, F x 1 1D numpy float array
        an array (of length frequencies) that is the frequency corresponding to each
        column in the FFT. That is, eeg_epochs_fft[:,:,i] is the energy at frequency fft_frequencies[i] Hz.
    is_trial_15Hz, E x 1 1D numpy bool array
        an array in which is_trial_15Hz[i] is True if the light was flashing at 15Hz during trial/epoch i.
    channels, C x 1 1D array
        the list of the names of each channel found in the original dataset
    channels_to_plot, 1D list/array
        a list/array of channel names you want to plot in different subplots
    subject, int
        the subject number

    Returns
    --------------------------------

    spectrum_db_12Hz,
        the mean power spectrum of 12Hz trials in dB
    spectrum_db_15Hz
        the mean power spectrum of 15Hz trials in dB

    """
    
    abs_spectrum_12Hz = np.empty((len(is_trial_15Hz) - is_trial_15Hz.sum(), eeg_epochs_fft.shape[1], eeg_epochs_fft.shape[2]),dtype = np.float64)
    abs_spectrum_15Hz = np.empty((is_trial_15Hz.sum(), eeg_epochs_fft.shape[1], eeg_epochs_fft.shape[2]),dtype = np.float64)

    count_12Hz = 0
    count_15Hz = 0

    for epoch_index in range(0, eeg_epochs_fft.shape[0]):
        if is_trial_15Hz[epoch_index]:
            for channel_index in range(0, eeg_epochs_fft.shape[1]):
                abs_spectrum_15Hz[count_15Hz][channel_index] = np.abs(eeg_epochs_fft[epoch_index][channel_index])
            count_15Hz += 1
        else:
            for channel_index in range(0, eeg_epochs_fft.shape[1]):
                abs_spectrum_12Hz[count_12Hz][channel_index] = np.abs(eeg_epochs_fft[epoch_index][channel_index])
            count_12Hz += 1


    abs_spectrum_12Hz = abs_spectrum_12Hz ** 2
    abs_spectrum_15Hz = abs_spectrum_15Hz ** 2

    normalized_trials_12Hz = abs_spectrum_12Hz / np.amax(abs_spectrum_12Hz)
    normalized_trials_15Hz = abs_spectrum_15Hz / np.amax(abs_spectrum_15Hz)

    trials_12Hz_in_db = 10 * np.log10(normalized_trials_12Hz)
    trials_15Hz_in_db = 10 * np.log10(normalized_trials_15Hz)

    trials_12Hz_in_db = trials_12Hz_in_db.transpose(1,2,0)
    trials_15Hz_in_db = trials_15Hz_in_db.transpose(1,2,0)

    mean_trials_12Hz = np.mean(trials_12Hz_in_db, axis = -1)
    mean_trials_15Hz = np.mean(trials_15Hz_in_db, axis = -1)




    channel_indexs = []

    for channel in channels_to_plot:
        channel_indexs.append(np.where(channels == channel)[0][0])
    print(channel_indexs)

    # Create figure and subplots
    fig, ax = plt.subplots(len(channels_to_plot), 1, sharex=True, figsize=(5 * len(channels_to_plot), 8))

    for fig_index, (channel_name, channel_index) in enumerate(zip(channels_to_plot, channel_indexs)):
        ax[fig_index].plot(fft_frequencies, mean_trials_12Hz[channel_index], color='red', linestyle='-')
        ax[fig_index].plot(fft_frequencies, mean_trials_15Hz[channel_index], color='green', linestyle='-')

    plt.tight_layout()

    # Show the plot
    plt.show()