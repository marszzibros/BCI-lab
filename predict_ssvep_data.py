"""
This module provides functions for analyzing SSVEP (Steady-State Visually Evoked Potential) EEG (Electroencephalography) data.

Functions:
- generate_predictions: Generates predictions for SSVEP events based on EEG epochs' Fast Fourier Transform (FFT) data.
- calculate_accuracy_and_ITR: Calculates accuracy and Information Transfer Rate (ITR) for SSVEP predictions.
- plot_accuracy_and_ITR: Plots heatmaps of accuracy and ITR for SSVEP data based on different epoch start and end times.
- plot_predictor_histogram: Plots a histogram of predictor variable calculated from EEG epochs FFT, indicating the presence or absence of SSVEP events.

@author: 
    Jay Hwasung Jung
    Tynan Gacy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import import_ssvep_data
import filter_ssvep_data
import predict_ssvep_data


def generate_predictions(eeg_epochs_fft, fft_frequencies, event_frequency, threshold=0, predictor=15):
    """
    Generates predictions based on EEG epochs' Fast Fourier Transform (FFT) data.

    Parameters:
    - eeg_epochs_fft (numpy.ndarray): EEG epochs data after performing FFT, with shape (n_epochs, 1, n_frequencies).
    - fft_frequencies (numpy.ndarray): Array of frequencies corresponding to the FFT data. (n_frequencies)
    - event_frequency (tuple): Tuple containing two frequencies representing the events of interest. (2,)
    - threshold (float, optional): Threshold for amplitude difference between event frequencies. Defaults to 0.
    - predictor (int, optional): frequency we are looking for

    Returns:
    - predicted_labels (bool,numpy.ndarray): Array of predicted labels corresponding to each EEG epoch.
                                        The labels are selected from event_frequency based on amplitude difference and threshold.
                                        The array has shape (n_epochs,) and dtype 'bool'.
    """

    # Find indices corresponding to event frequencies in the FFT frequencies array
    # Find the one that is closest
    frequency_index_1 = np.argmin(np.abs(fft_frequencies - event_frequency[0]))
    frequency_index_2 = np.argmin(np.abs(fft_frequencies - event_frequency[1]))

    # Extract amplitudes for the two event frequencies
    amplitudes_1 = np.abs(eeg_epochs_fft[:, frequency_index_1])
    amplitudes_2 = np.abs(eeg_epochs_fft[:, frequency_index_2])

    # Initialize array to store predicted labels
    predicted_labels = np.empty(eeg_epochs_fft.shape[0], dtype=int)

    # Iterate over EEG epochs to predict labels based on amplitude difference
    for trial_index in range(0, eeg_epochs_fft.shape[0]):
        amplitude_difference = (amplitudes_1[trial_index] - amplitudes_2[trial_index])

        # Predict label based on amplitude difference and threshold
        if amplitude_difference > threshold:
            predicted_labels[trial_index] = event_frequency[0]
        elif amplitude_difference <= threshold:
            predicted_labels[trial_index] = event_frequency[1]

    predicted_labels = np.array(predicted_labels == predictor)
    return predicted_labels

def calculate_accuracy_and_ITR(true_labels, predicted_labels, trials, duration):
    """
    Calculate accuracy and Information Transfer Rate (ITR).

    Parameters:
    - true_labels (numpy.ndarray): Array of true labels. (num_epochs, )
    - predicted_labels (numpy.ndarray): Array of predicted labels. (num_epochs, )
    - trials (int): Total number of trials.
    - duration (float): Total duration in seconds.

    Returns:
    - accuracy (float): Accuracy of the predictions.
    - ITR_time (float): Information Transfer Rate (ITR) in bits per second.
    """
    # Calculate accuracy
    num_trials = true_labels.shape[0]
    correct_labels = np.sum(true_labels == predicted_labels)
    accuracy = correct_labels / num_trials

    # Calculate ITR
    N = 2
    
    if accuracy == 1:
        ITR_trial = 1
    else:
        ITR_trial = np.log2(N) + accuracy * np.log2(accuracy) + (1 - accuracy) * np.log2((1 - accuracy) / (N - 1))

    tps = trials / (duration)
    ITR_time = ITR_trial * tps


    return accuracy, ITR_time

def plot_accuracy_and_ITR(data, channel, subject, start_time_array=np.linspace(start=0, stop=20, num=21), end_time_array=np.linspace(start=0, stop=20, num=21)):
    """
    Plot accuracy and Information Transfer Rate (ITR) heatmaps.

    Parameters:
    - data (dictionary): raw data dictionary
    - channel (int): Channel number.
    - subject (int): Subject number.
    - start_time_array (numpy.ndarray, optional): start time arrays Defaults to np.linspace(start=0, stop=20, num=21)
    - end_time_array (numpy.ndarray, optional): end time arrays Defaults to np.linspace(start=0, stop=20, num=21)

    Returns:
    - None
    """
    # find channel index
    channel_index = np.where(data['channels'] == channel)[0]

    # initialize arrays
    accuracy_array = np.ones((21,21), dtype=float) * -1
    ITR_array = np.ones((21,21), dtype=float) * -1

    # set values to find minimum value
    min_accuracy = 1
    min_ITR = 100

    fs = data['fs']

    # find event frequency in our case 12 and 15
    event_frequency = np.array([event[:-2] for event in set(data['event_types'])], dtype=int)
    event_frequency= np.sort(event_frequency)[::-1] 

    for start_index, start_time in enumerate(start_time_array):
        for end_index, end_time in enumerate(end_time_array):
            if start_time < end_time:

                # get epochs and fft
                eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data, epoch_start_time=start_time, epoch_end_time=end_time)

                # calculate fft filters
                eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)
                channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()

                # get predictions
                predictions = predict_ssvep_data.generate_predictions(channel_eeg_epochs_fft, fft_frequencies, event_frequency)

                # get accuracy and ITR
                accuracy, ITR_time = predict_ssvep_data.calculate_accuracy_and_ITR(is_trial_15Hz, predictions, eeg_epochs_fft.shape[1], duration=(end_time-start_time) * fs * (1/10))

                # find min value to pad
                if min_accuracy > accuracy:
                    min_accuracy = accuracy
                if min_ITR > ITR_time:
                    min_ITR = ITR_time
                
                accuracy_array[start_index, end_index] = accuracy
                ITR_array[start_index, end_index] = ITR_time

    # pad to the min value 
    for start_time in range(0, 21):
        for end_time in range(0, 21):
            if accuracy_array[start_time, end_time] == -1:
                accuracy_array[start_time, end_time] = min_accuracy
            if ITR_array[start_time, end_time] == -1:
                ITR_array[start_time, end_time] = min_ITR

    # change floatign to percent scale
    accuracy_array = accuracy_array * 100

    # Set up the figure and axes using Seaborn
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first heatmap using Seaborn with 'viridis' colormap
    sns.heatmap(accuracy_array, cmap='viridis', vmin=accuracy_array.min(), vmax=accuracy_array.max(),
                ax=axs[0], cbar=True, cbar_kws={'label': '% correct'}, xticklabels=end_time_array, yticklabels=start_time_array)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch End Time (s)')
    axs[0].set_ylabel('Epoch Start Time (s)')
    axs[0].invert_yaxis()

    # Plot the second heatmap using Seaborn with 'viridis' colormap
    sns.heatmap(ITR_array, cmap='viridis', vmin=ITR_array.min(), vmax=ITR_array.max(),
                ax=axs[1], cbar=True, cbar_kws={'label': 'ITR (bits/sec)'}, xticklabels=end_time_array, yticklabels=start_time_array)
    axs[1].set_title('Information Transfer Rate')
    axs[1].set_xlabel('Epoch End Time (s)')
    axs[1].set_ylabel('Epoch Start Time (s)')
    axs[1].invert_yaxis()
    fig.suptitle(f"SSVEP Subject {subject}, Channel {channel}", fontsize="x-large")
    plt.tight_layout()
    plt.savefig(f"subject{subject}/{channel}_{subject}_heatmap.png")

def plot_predictor_histogram(eeg_epochs_fft, fft_frequencies, event_frequency,true_label, channel,subject, condition):
    """
    Plot histogram of predictor variable calculated from EEG epochs FFT.

    Parameters:
    - eeg_epochs_fft (numpy.ndarray): FFT of EEG epochs, shape (num_epochs, num_channels, num_frequencies).
    - fft_frequencies (numpy.ndarray): Frequencies corresponding to FFT, shape (num_frequencies,).
    - event_frequency (tuple): Tuple containing two event frequencies of interest.
    - true_label (bool): True if the event is present in the epoch, False otherwise.
    - channel (str): channel to plot
    - subject (int): subject number
    - condition (array): containing start time and end time of epoch

    Returns:
    - None
    """

    # Find indices corresponding to event frequencies in the FFT frequencies array
    # Find the one that is closest
    frequency_index_1 = np.argmin(np.abs(fft_frequencies - event_frequency[0]))
    frequency_index_2 = np.argmin(np.abs(fft_frequencies - event_frequency[1]))

    # Extract amplitudes for the two event frequencies
    present_amplitudes = eeg_epochs_fft[true_label, frequency_index_1] - eeg_epochs_fft[true_label, frequency_index_2]
    absent_amplitudes = eeg_epochs_fft[~true_label, frequency_index_1] - eeg_epochs_fft[~true_label, frequency_index_2]

    # Plot KDE graph
    sns.kdeplot(np.real(present_amplitudes), color='skyblue', label='Present', fill=True)
    sns.kdeplot(np.real(absent_amplitudes), color='orange', label='Absent', fill=True)

    plt.title(f'KDE of Predictor Variable for SSVEP subject {subject} {channel} epoch time: [{condition[0]}, {condition[1]}]')
    plt.xlabel('Predictor Variable')
    plt.ylabel('Density')

    plt.legend()
    plt.grid(True)

    plt.show()