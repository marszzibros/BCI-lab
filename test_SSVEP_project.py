# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:17:10 2024

@author: tgsha
"""


#%% Part A: Generate Predictions

"""
 - load data
 - extract epochs with custom start/end times
 - take FFT for each epoch

Main function: 
    inputs: data, channel, start_time, end_time, trial_num, frequencies=('12Hz','15Hz' - closest?)

    outputs: predicted_labels
"""

#%% Part B: Calculate Accuracy and ITR


"""
Main function:
    inputs: truth_labels, predicted_labels, start_time, end_time
    
    outputs: accuracy, ITR
"""


#%% Part C: Loop Through Epoch Limits

"""
Main function:
    inputs: data, channel, trial_num, frequencies, epoch_times 
    
    outputs: validated_epochs
"""


#%% Part D: Plot Results

"""
Main function:
    inputs: validated_epochs, channel, start_time, end_time
    
    outputs: None (creates plots)
"""

#%% Part E: Create a Predictor Histogram
"""
Main function:
    inputs: start_time, end_time, validated_epochs
    
    outputs: None (creates plots)
"""


#%% Part F: Write-up
"""
Introduction:
    Summarize purpose of SSVEP BCI, function of code, usefulness of findings
    
Methods:
    Summarize use of functions
    Explain how to use functions (without changing code)
    Diagrams/graphs for clarity
    
Results:
    Representative accuracy, ITR, and predictor histogram graphs for each subject
    BCI-relevant takeaways for each graph
    Different start/end times and channels with rationale
    
Discussion:
    Final conclusions
    Optimal epoch start/end times for each subject with rationale
    Threshold for predictions
        What to value for high accuracy, sensitivity, specificity 
        When to value high accuracy, sensitivity, specificity


"""