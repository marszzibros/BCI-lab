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
    
    # Apply FFT to epochs
    eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs,fs)

    # Limit data to channel
    channel_data = np.where(data['channels'] == channel)[0]
    
    # Get epochs only from channel
    channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_data,:]

    # Sort frequency type
    event_frequency = np.array([event[:-2] for event in set(data['event_types'])], dtype=int)

    # Find elements of FFT representing amplitude of oscillations for 12Hz and 15Hz
    index_12Hz = np.argmin(np.abs(fft_frequencies - event_frequency[0]))
    index_15Hz = np.argmin(np.abs(fft_frequencies - event_frequency[1]))

    # Extract amplitudes for the two event frequencies
    amplitudes_12Hz = np.abs(channel_eeg_epochs_fft[:, :, index_12Hz])
    amplitudes_15Hz = np.abs(channel_eeg_epochs_fft[:, :, index_15Hz])
    
    # Create empty array for predicted labels
    predicted_labels = np.empty(eeg_epochs_fft.shape[0], dtype=int)
    
    # Iterate over EEG epochs to predict labels based on amplitude difference
    for i in range(0, channel_eeg_epochs_fft.shape[0]):
        amplitude_difference = (amplitudes_12Hz[i][0] - amplitudes_15Hz[i][0])

        # Predict label based on amplitude difference 
        if amplitude_difference > 0:
            predicted_labels[i] = event_frequency[0]
        elif amplitude_difference <= 0:
            predicted_labels[i] = event_frequency[1]
    
    return predicted_labels

#%% Part B: Calculate Accuracy and ITR
"""
Main function:
    inputs: truth_labels, predicted_labels, start_time, end_time
    
    calculate accuracy: (correctly classified trials/total trials)
    calculate ITR: ITRtrial = log2n+P+log2P + (1-P)*log2(1-P/N-1))
            P=accuracy N=# of classes
    
    outputs: accuracy, ITR
"""
def calculate_accuracy_ITR(truth_labels,predicted_labels,start_time,end_time):
    
    return accuracy, ITR

#%% Part C: Loop Through Epoch Limits

"""
Main function:
    inputs: data, channel, trial_num, frequencies, epoch_times 
    
    calculate FFT (part A1)
    calculate predictions (part A2)
    calculate merit_figs (part B)
    pair accuracy and ITR to epoch_times (validated_epochs)
    
    outputs: validated_epochs
"""


#%% Part D: Plot Results

"""
Main function:
    inputs: validated_epochs, channel, start_time, end_time
    
    plot pseudocolor plot (both subjects)
    
    outputs: None (creates plots)
"""

#%% Part E: Create a Predictor Histogram
"""
Main function:
    inputs: start_time, end_time, validated_epochs
    
    calculate predictor variable for optimal validated_epoch
    plot predictor histogram for selected epoch
    
    outputs: None (creates plots)
"""