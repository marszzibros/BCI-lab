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

#%% Part B: Calculate Accuracy and ITR


"""
Main function:
    inputs: truth_labels, predicted_labels, start_time, end_time
    
    calculate accuracy: (correctly classified trials/total trials)
    calculate ITR: ITRtrial = log2n+P+log2P + (1-P)*log2(1-P/N-1))
            P=accuracy N=# of classes
    
    outputs: accuracy, ITR
"""


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