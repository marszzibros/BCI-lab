
#%%
import plot_p300_erps
import load_p300_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,ttest_rel
from mne.stats import fdr_correction

def plot_confidence_intervals (confident_intervals = 95, subject = 3, data_directory = "P300Data/"):
    """
    Args:
        confident_intervals: float, define the interval
        subject: int, subject number to be plotted
    """
    z_value = norm.ppf((1 + (confident_intervals / 100)) / 2)
    # call load and plot functions from load_p300_data module
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                                subject = subject)
    
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)

    target_erp = target_erp.T 
    nontarget_erp = nontarget_erp.T  

    eeg_epochs = np.array(eeg_epochs, dtype=np.float64)
    erp_times = np.array(erp_times, dtype=np.float64)

    # Calculate mean ERP and standard error of the mean (SEM) for each time point and channel
    sem_target_erps = np.std(eeg_epochs[is_target_event].transpose((2,1,0)), axis=2) / np.sqrt(eeg_epochs[is_target_event].shape[0])
    confidence_interval_target = z_value * sem_target_erps

    sem_nontarget_erps = np.std(eeg_epochs[~is_target_event].transpose((2,1,0)), axis=2) / np.sqrt(eeg_epochs[~is_target_event].shape[0])
    confidence_interval_nontarget = z_value * sem_nontarget_erps

    # Create a 3x3 subplot grid for 8 plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
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
                ax.plot(erp_times, target_erp[plot_index], label='Target', color='C0')
                 
                # Plot nontarget ERP
                ax.plot(erp_times, nontarget_erp[plot_index], label='Nontarget', color='C1')
                
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
                ax.fill_between(erp_times, 
                                np.array(target_erp[plot_index] - confidence_interval_target[plot_index],dtype=np.float64), 
                                np.array(target_erp[plot_index] + confidence_interval_target[plot_index],dtype=np.float64), 
                                alpha=0.3,
                                color="C0",
                                label='Target 95%')
                
                ax.fill_between(erp_times, 
                                np.array(nontarget_erp[plot_index] - confidence_interval_nontarget[plot_index],dtype=np.float64), 
                                np.array(nontarget_erp[plot_index] + confidence_interval_nontarget[plot_index],dtype=np.float64), 
                                alpha=0.3,
                                color="C1",
                                label='Non-Target 95%')
                # Add legend to the last subplot
                if plot_index == 7:
                    ax.legend()

                # Remove empty subplots
            else:
                fig.delaxes(ax)

    return fig, axes, erp_times


def bootstrap_erp(eeg, size=None): 
    """ Calculate bootstrap ERP from data (array type)"""
    ntrials = len(eeg)           

    if size == None:        
        size = ntrials        

    rand_index = np.random.choice(np.arange(0, ntrials), size, replace=False)

    bootstrapped_eeg = eeg[rand_index]     

    return bootstrapped_eeg.mean(0)          

def bootstrap_stat(eeg_epochs, is_target_event, target_size = None, nontarget_size = None):

    target_trials = len(eeg_epochs[is_target_event])
    nontarget_trials = len(eeg_epochs[~is_target_event])

    if target_size == None:                
        target_size = target_trials           
    if nontarget_size == None:                 
        nontarget_size = nontarget_trials        

    target_bootstrap = bootstrap_erp(eeg_epochs, target_trials)
    nontarget_bootstrap = bootstrap_erp(eeg_epochs, nontarget_trials)

    return np.abs(target_bootstrap.T - nontarget_bootstrap.T)


def calculate_bootstrap_p_values(iterations = 3000, subject = 3, data_directory = "P300Data/"):
    # call load and plot functions from load_p300_data module
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                                subject = subject)
    
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)

    eeg_epochs = np.array(eeg_epochs, dtype=np.float64)
    erp_times = np.array(erp_times, dtype=np.float64)

    observed_absolute_difference = abs(target_erp.T - nontarget_erp.T)
    
    bootstrap_erp_iterations = np.array([bootstrap_stat(eeg_epochs, is_target_event) for _ in range(iterations)], dtype=np.float64)
    
    # Calculate the p-values
    p_values = np.zeros_like(observed_absolute_difference)



    for channel_idx in range(0, observed_absolute_difference.shape[0]):
        for time_point_idx in range(0, observed_absolute_difference.shape[1]):
            
            # Obtain the bootstrapped absolute differences for the current channel and time point
            bootstrapped_differences = bootstrap_erp_iterations[:,channel_idx, time_point_idx]
            
            # Compare the observed absolute difference with the bootstrapped distribution
            chances_of_randonly_difference = np.sum(bootstrapped_differences >= observed_absolute_difference[channel_idx, time_point_idx]) / (iterations + 1)
            
            # Store the p-value
            p_values[channel_idx, time_point_idx] = chances_of_randonly_difference

    return p_values


def fdr_correction_check (p_values, p_value = 0.05):
    return fdr_correction(p_values, p_value)

def plot_statistically_significant(fdr_result, subject = 3, data_directory = "P300Data/", confident_intervals = 95):
    """
    Args:
        confident_intervals: float, define the interval
        subject: int, subject number to be plotted
    """
    fig, axes, erp_times = plot_confidence_intervals(subject=subject, confident_intervals = confident_intervals, data_directory = data_directory)
    fdr_result = np.array(fdr_result, dtype=np.float64)
    true_indices = np.where(fdr_result < 0.05)

    ax_list = []
    for row_ind in range(3):
        for col_ind in range(3):
            ax_list.append((row_ind, col_ind))
    for channel_index, time_point in zip(*true_indices):
        ax = axes[ax_list[channel_index][0], ax_list[channel_index][1]]
        ax.scatter(erp_times[time_point], 0, color='black',s=10)

    
    fig.tight_layout()
    fig.show()
    return fig, axes

def plot_subjects (is_significant, erp_times):
 
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
                ax.plot(erp_times, is_significant[plot_index], label='Target', color='C0')
                 
 
                # Set x-axis ticks
                ax.set_xticks(np.linspace(erp_times[0], erp_times[-1], 4))                
 
                # Set subplot title and labels
                ax.set_title(f'Channel {plot_index}', fontsize=14)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('# subjects significant')
                
                # Add legend to the last subplot
                if plot_index == 7:
                    ax.legend()
 
            # Remove empty subplots
            else:
                fig.delaxes(ax)
 
 
    fig.tight_layout()
    fig.show()
#%%
p_values = []
for i in range(3, 11):
    p_values.append(calculate_bootstrap_p_values(subject = i))

#%%
p_values = np.array(p_values,dtype = np.float64)
print(p_values.shape)
is_significant = p_values < 0.05

reject_fdr, p_values_fdr = fdr_correction_check(p_values, 0.05)


# print((p_values_fdr< 0.05).sum())
#%%
plot_confidence_intervals(subject=3)
# p_values = np.array(p_values,dtype=np.float64)

# %%
is_significant = p_values < 0.05
print(p_values)
#%%
for i in range(0, 8):
    plot_statistically_significant(fdr_result=p_values_fdr[i], subject=i + 3)

# # %%

# #%%


# # %%

# %%
    # call load and plot functions from load_p300_data module
eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg()

event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
print(reject_fdr.shape)
sum_fdr_result = reject_fdr.transpose(1,2,0).sum(axis=2)
plot_subjects(sum_fdr_result,erp_times)
# %%
plot_subjects
# %%
