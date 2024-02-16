#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_p300_erps.py
loads and plots p300 data and extracts epochs

Created on Thu Jan 18 15:25:31 2024
@author: marszzibros
"""
import numpy as np
from matplotlib import pyplot as plt

def get_events(rowcol_id, is_target):
    """Args:
        rowcol_id: 1d array of ints, 0 for no rowcol and 1-12 for rowcols on the P300 matrix 
        is_target: 1d array, boolean, True if rowcol being flashed is target     
       Returns:
        event_sample: 1d array of ints (indices) where a rowcol is being flashed
        is_target_event: 1d array, boolean mask of length equal to event_sample. True if the event is a target event. 
    """

    # get where rowcol_id is present +1 for indexing
    event_sample = np.array(np.where(np.diff(rowcol_id) > 0)) + 1
    
    # get if the event is target
    is_target_event = np.array([True if is_target[ind] else False for ind in event_sample[0]])
    return event_sample, is_target_event

def epoch_data (eeg_time, eeg_data, event_sample, epoch_start_time = -0.5, epoch_end_time=1):
    """
     Args: 
          eeg_time: 1d array (float), time axis in millseconds 
          eeg_data: 2d array of dimensions channels x samples 
          event_sample: 1d array (float) of indices where events occurred 
          epoch_start_time: float, fraction of second before event to start epoch
          epoch_end_time: float, fraction of second after event to end epoch
     Returns:
          eeg_epochs: 3d array of size (event_samples.shape[1], samples_per_epoch, eeg_data.shape[0]) that contains 
          all eeg_epochs (both target and non-target). 
          --event_samples.shape[1] is the number of events (both target and non-target). 
          --eeg_data.shape[0] is number of EEG channels. 
          --samples_per_epoch is the number of EEG samples (data points) per epoch
          erp_times: 1d array of size (seconds_per_epoch * samples_per_second), axis of time relative to the event onset 

    """

    # calculate samples_per_second
    samples_per_second = 0
    seconds_per_epoch = epoch_end_time - epoch_start_time

    while eeg_time[samples_per_second] <= 1:
        samples_per_second += 1

    # calculate samples_per_epoch
    samples_per_epoch = (samples_per_second - 1)* seconds_per_epoch

    # calculate 0.5 second before and 1 second after the event
    event_before = int(round((abs(epoch_start_time) / seconds_per_epoch) * samples_per_epoch))
    event_after = int(round((abs(epoch_end_time) / seconds_per_epoch) * samples_per_epoch))

    # define the array based on the dimensions
    # 0 d: pages (epochs) 
    # 1 d: rows (samples)
    # 2 d: columns (channels)
    eeg_epochs = np.empty((event_sample.shape[1], int(samples_per_epoch), eeg_data.shape[0]), dtype = object)
    erp_times = np.linspace(epoch_start_time,epoch_end_time,int(samples_per_epoch))

    for evnet_ind, event in enumerate(event_sample[0]):
        epoch = []
        
        # get eeg data for each event
        for eeg_time_ind in range(event - event_before, event + event_after):

            epoch.append(eeg_data.T[eeg_time_ind])
        eeg_epochs[evnet_ind] = np.array(epoch)

    return eeg_epochs, erp_times

def get_erps (eeg_epochs, is_target_event):
    """
    Args:
        eeg_epochs: 3d array of size (event_samples.shape[1], samples_per_epoch, eeg_data.shape[0]) that contains 
        all eeg_epochs (both target and non-target). 
        is_target_event: 1d array of size (event_samples.shape[1]), boolean mask for each event in event_sample, True if target event

    Returns:
        target_erp: 2d array of size (samples_per_epoch, eeg_data.shape[0]) where eeg_data.shape[0] refers to number of EEG channels. Mean 
        ERP of all epochs that are labelled as target epochs 
        nontarget_erp: 2d array of size (samples_per_epoch, eeg_data.shape[0]). Mean ERP of all epochs that are labelled as non-target epochs. 
        
    """
    target_erp = np.mean(eeg_epochs[is_target_event], axis = 0)
    nontarget_erp = np.mean(eeg_epochs[~is_target_event], axis = 0)
    return target_erp, nontarget_erp


def plot_erps(target_erp, nontarget_erp, erp_times):
    """
    Args:
        target_erp: 2d array of size (samples_per_epoch, eeg_data.shape[0]) where eeg_data.shape[0] refers to number of EEG channels. Mean 
        ERP of all epochs that are labelled as target epochs 

        nontarget_erp: 2d array of size (samples_per_epoch, eeg_data.shape[0]) where eeg_data.shape[0] refers to number of EEG channels. Mean 
        ERP of all epochs that are labelled as nontarget epochs 

        erp_times: 1d array of erp time axis, relative to onset of event
    Returns:
        fig: matplotlib class, fig information processing
        axes: 3x3 array, axes information after processing
    """
    # Reshape the data for subplot arrangement
    target_erp = target_erp.T  # Transpose to have shape (8, 150)
    nontarget_erp = nontarget_erp.T  # Transpose to have shape (8, 150)

    # Create a 3x3 subplot grid for 8 plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Plot each subplot
    for row_ind in range(3):
        for col_ind in range(3):
            plot_index = row_ind * 3 + col_ind
            ax = axes[row_ind, col_ind]
            # Check if there are fewer than 8 plots
            if plot_index < 8:
                yticks = 0
                if np.amax(abs(target_erp[plot_index])) > np.amax(abs(nontarget_erp[plot_index])):
                    yticks = np.amax(abs(target_erp[plot_index]))//1
                else:
                    yticks = np.amax(abs(nontarget_erp[plot_index]))//1
                # Plot target ERP
                ax.plot(erp_times, target_erp[plot_index], label='Target')
                
                # Plot nontarget ERP
                ax.plot(erp_times, nontarget_erp[plot_index], label='Nontarget')
                
                # Add a dotted line at x=0 and y=0
                ax.axvline(x=0, linestyle='--', color='black', linewidth=1)
                ax.axhline(y=0, linestyle='--', color='black', linewidth=1)

                # Set y-axis ticks
                ax.set_yticks([-yticks, 0, yticks])
                
                # Set x-axis ticks
                ax.set_xticks(np.linspace(erp_times[0], erp_times[-1], 4))                

                # Set subplot title and labels
                ax.set_title(f'Channel {plot_index}', fontsize=14)
                ax.set_xlabel('time from flash onset( s)')
                ax.set_ylabel('Voltage (uV)')
                
                # Add legend to the last subplot
                if plot_index == 7:
                    ax.legend()

            # Remove empty subplots
            else:
                fig.delaxes(ax)

    return fig, axes