
import plot_p300_erps
import load_p300_data
from plot_topo import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,ttest_rel
from mne.stats import fdr_correction
from pathlib import Path

def bootstrap_erp(eeg, size=None): 
    """
    bootstrap erp from eeg data

    Parameters
    ----------
    eeg : float 3d array, required
        eeg data, target/nontarget combined under null hyposthesis (no difference)
    size : int, optional
        size to be sampled

    Returns
    -------
    bootstrapped_erp : 2d array
        averaged through trials (0 dimension)
    """
    # get number of trials
    ntrials = len(eeg)           

    # if size is not specified; set it to ntrials
    if size == None:        
        size = ntrials        

    # samples with replacement
    rand_index = np.random.choice(np.arange(0, ntrials), size, replace=True)

    # calculate erp
    bootstrapped_erp = eeg[rand_index].mean(0)     

    return bootstrapped_erp       

def bootstrap_stat(eeg_epochs, is_target_event):
    """
    bootstrap erp from eeg data

    Parameters
    ----------
    eeg_epochs : float 3d array, required
        eeg data, target/nontarget combined under null hyposthesis (no difference)
    is_target_event : boolean 1d array, required
        is_target event 
    Returns
    -------
    absolute_difference : 2d array
        averaged through trials (dimension 0)
    """

    # get size of each samples
    target_trials = np.count_nonzero(is_target_event)
    nontarget_trials = len(is_target_event) - target_trials
    
    # get bootstrap of each type of trials (target, and nontarget)
    target_bootstrap = bootstrap_erp(eeg_epochs, target_trials)
    nontarget_bootstrap = bootstrap_erp(eeg_epochs, nontarget_trials)

    # calculate the absolute value of erp difference
    absolute_difference = np.abs(target_bootstrap.T - nontarget_bootstrap.T)

    return absolute_difference


def calculate_bootstrap_p_values(iterations = 3000, subject = 3, data_directory = "P300Data/"):
    """
    calculate bootsraped p values

    Parameters
    ----------
    iterations : int, optional
        Bootstrap iterations
    subject : int, optional
        Subject number to be plotted
    data_directory : string, optional
        Directory where the data is stored 

    Returns
    -------
    p_values : 2d array (channel, time)
        calculated p_values

    """
    # call load and plot functions from load_p300_data module
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                                subject = subject)
    
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)

    eeg_epochs = np.array(eeg_epochs, dtype=np.float64)
    erp_times = np.array(erp_times, dtype=np.float64)

    # calculate the difference in real datad
    observed_absolute_difference = abs(target_erp.T - nontarget_erp.T)
    
    # get bootstrap erp
    bootstrap_erp_iterations = np.array([bootstrap_stat(eeg_epochs, is_target_event) for _ in range(iterations)], dtype=np.float64)
    
    # Calculate the p-values
    p_values = np.zeros_like(observed_absolute_difference)

    for channel_idx in range(0, observed_absolute_difference.shape[0]):
        for time_point_idx in range(0, observed_absolute_difference.shape[1]):
            
            # Obtain the bootstrapped absolute differences for the current channel and time point
            bootstrapped_differences = bootstrap_erp_iterations[:,channel_idx, time_point_idx]
            
            # Compare the observed absolute difference with the bootstrapped distribution
            chances_of_randonly_difference = np.mean(bootstrapped_differences >= observed_absolute_difference[channel_idx, time_point_idx])
            
            # Store the p-value
            p_values[channel_idx, time_point_idx] = chances_of_randonly_difference

    return p_values


def fdr_correction_check (p_values, p_value = 0.05):
    """
    Make a correction based on the FDR (False Discovery Rate)

    Parameters
    ----------
    p_values : int, required
        p_values to be checked by fdr correction
    p_value : int, optional
        statistically significant p_value

    Returns
    -------
    fdr_corrected_values : 3d array (2, channel, time)
        calculated p_values

    """
    # call mne.stats fdr_correction
    fdr_corrected_values = fdr_correction(p_values, p_value)

    return fdr_corrected_values

def plot_confidence_intervals (confident_intervals = 0.95, subject = 3, data_directory = "P300Data/"):
    """
    plot a confidence intervals specified by users

    Parameters
    ----------
    confident_intervals : float, optional
        Confident intervals to be plotted
    subject : int, optional
        Subject number to be plotted
    data_directory : string, optional
        Directory where the data is stored 

    Returns
    -------
    fig : matplotlib object
        For future usage (i.e. adding scatter plots, etc...)
    axes : 3x3 array
        axes information after plotting confidence intervals
    erp_times : 1d array
        Plotted points (x-axis)

    """

    # calcualte z_value
    z_value = norm.ppf((1 + (confident_intervals)) / 2)

    # call load and plot functions from load_p300_data module
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                                subject = subject)
    
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)

    # get plotted erp graph
    fig, axes = plot_p300_erps.plot_erps(target_erp, nontarget_erp, erp_times)

    # Transpose target_erp/nontarget_erp to the desired dimensions
    target_erp = target_erp.T 
    nontarget_erp = nontarget_erp.T  

    # set dtype to np.float64
    eeg_epochs = np.array(eeg_epochs, dtype=np.float64)
    erp_times = np.array(erp_times, dtype=np.float64)

    # Calculate mean ERP and standard error of the mean (SEM) for each time point and channel
    sem_target_erps = np.std(eeg_epochs[is_target_event].transpose((2,1,0)), axis=2) / np.sqrt(eeg_epochs[is_target_event].shape[0])
    confidence_interval_target = z_value * sem_target_erps

    sem_nontarget_erps = np.std(eeg_epochs[~is_target_event].transpose((2,1,0)), axis=2) / np.sqrt(eeg_epochs[~is_target_event].shape[0])
    confidence_interval_nontarget = z_value * sem_nontarget_erps


    for row_ind in range(3):
        for col_ind in range(3):
            plot_index = row_ind * 3 + col_ind
            ax = axes[row_ind, col_ind]

            # Check if there are fewer than 8 plots
            if plot_index < 8:                
                
                # perform fill_between based on the calculated interval target
                ax.fill_between(erp_times, 
                                np.array(target_erp[plot_index] - confidence_interval_target[plot_index],dtype=np.float64), 
                                np.array(target_erp[plot_index] + confidence_interval_target[plot_index],dtype=np.float64), 
                                alpha=0.3,
                                color="C0",
                                label=f'Target +/- {confident_intervals}% CI')
                
                ax.fill_between(erp_times, 
                                np.array(nontarget_erp[plot_index] - confidence_interval_nontarget[plot_index],dtype=np.float64), 
                                np.array(nontarget_erp[plot_index] + confidence_interval_nontarget[plot_index],dtype=np.float64), 
                                alpha=0.3,
                                color="C1",
                                label=f'Nontarget +/- {confident_intervals}% CI')
                
                # Add legend to the last subplot
                if plot_index == 7:
                    handles, labels = ax.get_legend_handles_labels()
            else:
                for side in ['top','right','bottom','left']:
                    ax.spines[side].set_visible(False)
                ax.tick_params(axis='both',which='both',labelbottom=False,bottom=False,left=False)
                ax.set_yticks([])
                ax.legend(handles, labels)
    fig.tight_layout()
    return fig, axes, erp_times

def plot_statistically_significant(reject_fdr, subject = 3, data_directory = "P300Data/", confident_intervals = 0.95, p_value = 0.05):
    """
    plot a scatterplot of statistically significant point by channel

    Parameters
    ----------
    reject_fdr: 2d array, required
        Rejected FDR, True if below p < p_value
    confident_intervals : float, optional
        Confident intervals to be plotted
    subject : int, optional
        Subject number to be plotted
    data_directory : string, optional
        Directory where the data is stored 
    p_value : int, optional
        statistically significant p_value

    Returns
    -------
    fig : matplotlib object
        For future usage (i.e. adding scatter plots, etc...)
    axes : 3x3 array
        axes information after plotting confidence intervals
    erp_times : 1d array
        Plotted points (x-axis)

    """

    # call plot_confidence_intervals as a base
    fig, axes, erp_times = plot_confidence_intervals(subject=subject, confident_intervals = confident_intervals, data_directory = data_directory)
    
    # indice the index where fdr_result is less than
    true_indices = np.where(reject_fdr)

    # make ax_list pair so when plotting a dot, we do not need to access to the channel where we do not need
    ax_list = []
    for row_ind in range(3):
        for col_ind in range(3):
            ax_list.append((row_ind, col_ind))

    # plot scatter plot
    handles, labels = [], []
    for plot_index, (channel_index, time_point) in enumerate(zip(*true_indices)):
        ax = axes[ax_list[channel_index][0], ax_list[channel_index][1]]
        ax.scatter(erp_times[time_point], 0, color='black',s=10)
        if plot_index == len(true_indices):
            ax.scatter(erp_times[time_point], 0, color='black',s=10, label= f"p < {p_value}")
            handles, labels = ax.get_legend_handles_labels()

    axes[2,2].legend(handles, labels)

    fig.tight_layout()
    return fig, axes, erp_times

def plot_statistically_significant_by_subjects (rejected_fdr, erp_times):
    """
    plot a scatterplot of statistically significant point by channel

    Parameters
    ----------
    reject_fdr: 2d array, required
        Rejected FDR, True if below p < p_value
    erp_times : 1d array
        Plotted points (x-axis)

    Returns
    -------
    fig : matplotlib object
        For future usage (i.e. adding scatter plots, etc...)
    axes : 3x3 array
        axes information after plotting confidence intervals

    """
    
    rejected_fdr = rejected_fdr.transpose(1,2,0).sum(axis=2)

    # Create a 3x3 subplot grid for 8 plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Plot each subplot
    for row_ind in range(3):
        for col_ind in range(3):
            plot_index = row_ind * 3 + col_ind
            ax = axes[row_ind, col_ind]
            # Check if there are fewer than 8 plots
            if plot_index < 8:
 
                # Plot target ERP
                ax.plot(erp_times, rejected_fdr[plot_index], label='Target', color='C0')
                 
 
                # Set x-axis ticks
                ax.set_xticks(np.linspace(erp_times[0], erp_times[-1], 4))   
                ax.set_yticks([0,2,4])             
 
                # Set subplot title and labels
                ax.set_title(f'Channel {plot_index}', fontsize=14)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('# subjects significant')
                
                # Add legend to the last subplot
                if plot_index == 7:
                    handles, labels = ax.get_legend_handles_labels()
            else:
                for side in ['top','right','bottom','left']:
                    ax.spines[side].set_visible(False)
                ax.tick_params(axis='both',which='both',labelbottom=False,bottom=False,left=False)
                ax.set_yticks([])
                ax.legend(handles, labels)
 

    fig.tight_layout()

    return fig, axes

def plot_save_graph (filename = "sample.png", directory_path = "output/", fig=None):
    """
    save graph

    Parameters
    ----------
    filename: string, optional
        Rejected FDR, True if below p < p_value
    directory_path : string, optional
        Plotted points (x-axis)
    fig : matplotlib object
        For future usage (i.e. adding scatter plots, etc...)
    """

    # make directory
    new_directory = Path(directory_path)
    new_directory.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    # save figure
    fig.savefig(directory_path + filename)

