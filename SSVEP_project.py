# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:17:10 2024

@author: tgsha
"""


"""
Overarching goal: create code to translate SSVEPs into steering wheelchair left
or right

1.) Use FFT method on epochs
2.) Quantify performance of BCI for different epoch start/end times
    a.) Accuracy (correctly classified trials/total trials)
    b.) ITR (info transfer rate) ITRtrial = log2n+P+log2P + (1-P)*log2(1-P/N-1))
            P=accuracy N=# of classes
3.) Generate pseudocolor plots graphing start/end times for acc and ITR
    a.) What start/end times should be used?
4.) Generate predictor histogram for distribution of predictor values 
    a.) What threshold should be used for predictions?
"""
import numpy as np
import matplotlib.pyplot as plt
import import_ssvep_data
import seaborn as sns
import SSVEP_project


#%% Part A: Generate Predictions

"""
 - load data
 - extract epochs with custom start/end times
 - take FFT for each epoch

Main function: 
    inputs: data, channel, start_time, end_time, trial_num, frequencies=('12Hz','15Hz' - closest?)

    find elements of FFT representing amplitude of oscillations for 12Hz and 15Hz
    identify higher amplitude between 12Hz and 15Hz and set as predicted frequency
    generate array of predicted labels (compare to truth labels)
    
    outputs: predicted_labels
"""

def generate_predictions(subject,data_directory,channel,start_time,end_time):

    # Load raw data
    data = import_ssvep_data.load_ssvep_data(subject, data_directory)
    fs = data['fs']
    
    # Load epochs
    eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data,start_time,end_time)
    
    true_labels = is_trial_15Hz
    
    # Apply FFT to epochs
    eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs,fs)

    # Limit data to channel
    channel_data = np.where(data['channels'] == channel)[0]
    
    # Get epochs only from channel
    channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_data,:]
    
    # Sort frequency type
    event_frequency = np.array([event[:-2] for event in set(data['event_types'])], dtype=int)    

    # Find elements of FFT representing amplitude of oscillations for 12Hz and 15Hz
    index_15Hz = np.argmin(np.abs(fft_frequencies - event_frequency[0]))
    index_12Hz = np.argmin(np.abs(fft_frequencies - event_frequency[1]))

    # Extract amplitudes for the two event frequencies
    amplitudes_15Hz = np.abs(channel_eeg_epochs_fft[:, :, index_15Hz])
    amplitudes_12Hz = np.abs(channel_eeg_epochs_fft[:, :, index_12Hz])
    
    # Create empty array for predictions
    predictions = np.empty(eeg_epochs_fft.shape[0], dtype=int)
    
    # Iterate over EEG epochs to predict labels based on amplitude difference
    for i in range(0, channel_eeg_epochs_fft.shape[0]):
        amplitude_difference = (amplitudes_15Hz[i][0] - amplitudes_12Hz[i][0])

        # Predict label based on amplitude difference 
        if amplitude_difference > 0:
            predictions[i] = event_frequency[0]
        elif amplitude_difference <= 0:
            predictions[i] = event_frequency[1]
            
    # as predictions is 12 or 15 based on which amplitude is higher, convert it to bool
    predicted_labels = np.array(predictions == 15)
    
    return predicted_labels, true_labels, fft_frequencies, event_frequency, eeg_epochs_fft, fs

#%% Part B: Calculate Accuracy and ITR
"""
Main function:
    inputs: truth_labels, predicted_labels, start_time, end_time
    
    calculate accuracy: (correctly classified trials/total trials)
    calculate ITR: ITRtrial = log2n+P+log2P + (1-P)*log2(1-P/N-1))
            P=accuracy N=# of classes
    
    outputs: accuracy, ITR
"""
def calculate_accuracy_and_ITR(true_labels,predicted_labels,start_time,end_time,fs):

    """
    Calculate accuracy and Information Transfer Rate (ITR).

    Parameters:
    - true_labels (numpy.ndarray): Array of true labels.
    - predicted_labels (numpy.ndarray): Array of predicted labels.
    - start_time (float): Start time of epoch. Default is 0.
    - end_time (float): End time of epoch. Default is 20.

    Returns:
    - accuracy (float): Accuracy of the predictions.
    - ITR (float): Information Transfer Rate (ITR) in bits per second.
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


#    duration=(end_time-start_time) * fs * (1/10)
#    tps = num_trials / (duration)
#    ITR_time = ITR_trial * tps


    ITR_time = ITR_trial * (num_trials / ((end_time-start_time) * 100))

    return accuracy, ITR_time

#%% Part C/D: Loop Through Epoch Limits and Plot Results

"""
Main function:
    inputs: data, channel, trial_num, frequencies, epoch_times 
    
    calculate FFT (part A1)
    calculate predictions (part A2)
    calculate merit_figs (part B)
    pair accuracy and ITR to epoch_times (validated_epochs)
    
    outputs: validated_epochs
"""
def plot_accuracy_and_ITR(accuracy, ITR_time, subject, data_directory, channel):
    """
    Plot accuracy and Information Transfer Rate (ITR) heatmaps.

    Parameters:
    - accuracy_array (numpy.ndarray): Array containing accuracy values.
    - ITR_array (numpy.ndarray): Array containing ITR values.

    Returns:
    - None
    """
        
    accuracy_array = np.ones((21,21), dtype=float) * -1
    ITR_array = np.ones((21,21), dtype=float) * -1
    
    min_accuracy = 1
    min_ITR = 1
    
    for start_time in range(0, 21):
        for end_time in range(0, 21):
            if start_time < end_time: 

                # generate predictions
                predicted_labels, true_labels, fs, *rest = SSVEP_project.generate_predictions(subject,data_directory,channel,start_time,end_time)
    
                accuracy, ITR_time = SSVEP_project.calculate_accuracy_and_ITR(true_labels, predicted_labels,start_time,end_time,fs)
    
                if min_accuracy > accuracy:
                    min_accuracy = accuracy
                if min_ITR > ITR_time:
                    min_ITR = ITR_time
                
                accuracy_array[start_time, end_time] = accuracy
                ITR_array[start_time, end_time] = ITR_time
    
    for start_time in range(0, 21):
        for end_time in range(0, 21):
            if accuracy_array[start_time, end_time] == -1:
                accuracy_array[start_time, end_time] = min_accuracy
            if ITR_array[start_time, end_time] == -1:
                ITR_array[start_time, end_time] = min_ITR    
    accuracy_array = accuracy_array * 100

    # Set up the figure and axes using Seaborn
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first heatmap using Seaborn with 'viridis' colormap
    sns.heatmap(accuracy_array, cmap='viridis', vmin=accuracy_array.min(), vmax=accuracy_array.max(),
                ax=axs[0], cbar=True, cbar_kws={'label': '% correct'})
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch End Time (s)')
    axs[0].set_ylabel('Epoch Start Time (s)')
    axs[0].invert_yaxis()

    # Plot the second heatmap using Seaborn with 'viridis' colormap
    sns.heatmap(ITR_array, cmap='viridis', vmin=ITR_array.min(), vmax=ITR_array.max(),
                ax=axs[1], cbar=True, cbar_kws={'label': 'ITR (bits/sec)'})
    axs[1].set_title('Information Transfer Rate')
    axs[1].set_xlabel('Epoch End Time (s)')
    axs[1].set_ylabel('Epoch Start Time (s)')
    axs[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

#%% Part E: Create a Predictor Histogram
"""
Main function:
    inputs: start_time, end_time, validated_epochs
    
    calculate predictor variable for optimal validated_epoch
    plot predictor histogram for selected epoch
    
    outputs: None (creates plots)
"""

def plot_predictor_histogram(start_time, end_time, subject, data_directory, channel):

    data = import_ssvep_data.load_ssvep_data(subject, data_directory)    

    predicted_labels, true_labels, fft_frequencies, event_frequency, eeg_epochs_fft, \
        *rest = SSVEP_project.generate_predictions(subject,data_directory,channel,start_time,end_time)

    # Find indices corresponding to event frequencies in the FFT frequencies array
    # Find the one that is closest
    frequency_index_1 = np.argmin(np.abs(fft_frequencies - event_frequency[0]))
    frequency_index_2 = np.argmin(np.abs(fft_frequencies - event_frequency[1]))
    
    channel_index = np.where(data['channels'] == channel)[0]
    
    channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()
    
    # Extract amplitudes for the two event frequencies
    present_amplitudes = channel_eeg_epochs_fft[true_labels, frequency_index_1] - channel_eeg_epochs_fft[true_labels, frequency_index_2]
    absent_amplitudes = channel_eeg_epochs_fft[~true_labels, frequency_index_1] - channel_eeg_epochs_fft[~true_labels, frequency_index_2]

    plt.figure()

    # Plot KDE graph
    sns.kdeplot(np.real(present_amplitudes), color='skyblue', label='Present', fill=True)
    sns.kdeplot(np.real(absent_amplitudes), color='orange', label='Absent', fill=True)
    
    plt.title(f'Kernel Density Estimate (KDE) of Predictor Variable in channel {channel}')
    plt.xlabel('Predictor Variable')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.show()