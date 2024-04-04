"""

@author: 
    Jay Hwasung Jung
    Tynan Gacy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_predictions(eeg_epochs_fft, fft_frequencies, event_frequency, threshold=0):
    """
    Generates predictions based on EEG epochs' Fast Fourier Transform (FFT) data.

    Parameters:
    - eeg_epochs_fft (numpy.ndarray): EEG epochs data after performing FFT, with shape (n_epochs, 1, n_frequencies).
    - fft_frequencies (numpy.ndarray): Array of frequencies corresponding to the FFT data. (n_frequencies)
    - event_frequency (tuple): Tuple containing two frequencies representing the events of interest. (2,)
    - threshold (float, optional): Threshold for amplitude difference between event frequencies. Defaults to 0.

    Returns:
    - predicted_labels (numpy.ndarray): Array of predicted labels corresponding to each EEG epoch.
                                        The labels are selected from event_frequency based on amplitude difference and threshold.
                                        The array has shape (n_epochs,) and dtype 'object'.
    """

    # Find indices corresponding to event frequencies in the FFT frequencies array
    # Find the one that is closest
    frequency_index_1 = np.argmin(np.abs(fft_frequencies - event_frequency[0]))
    frequency_index_2 = np.argmin(np.abs(fft_frequencies - event_frequency[1]))

    # Extract amplitudes for the two event frequencies
    amplitudes_1 = np.abs(eeg_epochs_fft[:, :, frequency_index_1])
    amplitudes_2 = np.abs(eeg_epochs_fft[:, :, frequency_index_2])

    # Initialize array to store predicted labels
    predicted_labels = np.empty(eeg_epochs_fft.shape[0], dtype=int)

    # Iterate over EEG epochs to predict labels based on amplitude difference
    for trial_index in range(0, eeg_epochs_fft.shape[0]):
        amplitude_difference = (amplitudes_1[trial_index][0] - amplitudes_2[trial_index][0])

        # Predict label based on amplitude difference and threshold
        if amplitude_difference > threshold:
            predicted_labels[trial_index] = event_frequency[0]
        elif amplitude_difference <= threshold:
            predicted_labels[trial_index] = event_frequency[1]

    return predicted_labels

def calculate_accuracy_and_ITR(true_labels, predicted_labels, epoch_start_time=0, epoch_end_time=20):
    """
    Calculate accuracy and Information Transfer Rate (ITR).

    Parameters:
    - true_labels (numpy.ndarray): Array of true labels.
    - predicted_labels (numpy.ndarray): Array of predicted labels.
    - epoch_start_time (float): Start time of epoch. Default is 0.
    - epoch_end_time (float): End time of epoch. Default is 20.

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

    ITR_time = ITR_trial * (num_trials / (epoch_end_time - epoch_start_time))

    return accuracy, ITR_time

def plot_accuracy_and_ITR (accuracy_array, ITR_array):

    accuracy_array = accuracy_array * 100

    # Set up the figure and axes using Seaborn
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first heatmap using Seaborn with 'viridis' colormap
    sns.heatmap(accuracy_array, cmap='viridis', vmin=0, vmax=accuracy_array.max(), ax=axs[0], cbar=True, cbar_kws={'label': '% correct'})
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Start Time (s)')
    axs[0].set_ylabel('End Time (s)')
    axs[0].invert_yaxis()

    # Plot the second heatmap using Seaborn with 'viridis' colormap
    sns.heatmap(ITR_array, cmap='viridis', vmin=0, vmax=ITR_array.max(), ax=axs[1], cbar=True, cbar_kws={'label': 'ITR (bits/sec)'})
    axs[1].set_title('Information Transfer Rate')
    axs[1].set_xlabel('Epoch Start Time (s)')
    axs[1].set_ylabel('Epoch End Time (s)')
    axs[1].invert_yaxis()
    plt.tight_layout()
    plt.show()