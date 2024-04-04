"""

@author: 
    Jay Hwasung Jung
    Tynan Gacy
"""

import numpy as np

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
    predicted_labels = np.empty(eeg_epochs_fft.shape[0], dtype=bool)

    # Iterate over EEG epochs to predict labels based on amplitude difference
    for trial_index in range(0, eeg_epochs_fft.shape[0]):
        amplitude_difference = (amplitudes_1[trial_index][0] - amplitudes_2[trial_index][0])

        # Predict label based on amplitude difference and threshold
        if amplitude_difference > threshold:
            predicted_labels[trial_index] = event_frequency[0]
        elif amplitude_difference <= threshold:
            predicted_labels[trial_index] = event_frequency[1]


    return predicted_labels
