
#%%
import plot_p300_erps
import load_p300_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap, norm, sem


def bootstrapERP(EEGdata, size=None):  # Steps 1-2
    """ Calculate bootstrap ERP from data (array type)"""
    ntrials = len(EEGdata)             # Get the number of trials
    if size == None:                   # Unless the size is specified,
        size = ntrials                 # ... choose ntrials
    i = np.random.randint(ntrials, size=size)    # ... draw random trials,
    EEG0 = EEGdata[i]                  # ... create resampled EEG,
    return EEG0.mean(0)                # ... return resampled ERP.
                                       # Step 3: Repeat 3000 times 


def plot_confidence_intervals (confident_intervals = 95, subject = 3, data_directory = "P300Data/"):
    """
    Args:
        confident_intervals: float, define the interval
        subject: int, subject number to be plotted
    """
    z_value = norm.ppf((1 + (confident_intervals / 100)) / 2)

    lower_bound = ((100 - confident_intervals) / 2) / 100
    higher_bound = 1 - lower_bound

    # call load and plot functions from load_p300_data module
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                                subject = 3)
    
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)


    fig, axes = plot_p300_erps.plot_erps(target_erp, nontarget_erp, erp_times)

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
                ax.fill_between(erp_times, 
                                np.array(target_erp.T[plot_index] - confidence_interval_target[plot_index],dtype=np.float64), 
                                np.array(target_erp.T[plot_index] + confidence_interval_target[plot_index],dtype=np.float64), 
                                color='C0', 
                                alpha=0.3,
                                label='Target 95%')
                
                ax.fill_between(erp_times, 
                                np.array(nontarget_erp.T[plot_index] - confidence_interval_nontarget[plot_index],dtype=np.float64), 
                                np.array(nontarget_erp.T[plot_index] + confidence_interval_nontarget[plot_index],dtype=np.float64), 
                                color='C1', 
                                alpha=0.3,
                                label='Non-Target 95%')
                # Add legend to the last subplot
                if plot_index == 7:
                    ax.legend()

    fig.tight_layout()
    fig.show()

#%%
plot_confidence_intervals()
# %%
