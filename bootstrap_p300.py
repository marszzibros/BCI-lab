
#%%
import plot_p300_erps
import load_p300_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap


def plot_confidence_intervals (confident_intervals = 95, subject = 3, data_directory = "P300Data/"):
    """
    Args:
        confident_intervals: float, define the interval
        subject: int, subject number to be plotted
    """
    lower_bound = (100 - confident_intervals) / 2
    higher_bound = 100 - lower_bound
    # call load and plot functions from load_p300_data module
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                                subject = 3)
    
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)

    fig, axes = plot_p300_erps.plot_erps(target_erp, nontarget_erp, erp_times)

    for row_ind in range(3):
            for col_ind in range(3):
                plot_index = row_ind * 3 + col_ind
                ax = axes[row_ind, col_ind]
                # Check if there are fewer than 8 plots
                if plot_index < 8:
                    # Bootstrap for target ERP
                    target_bootstrap = bootstrap(target_erp.T[plot_index], n_resamples=1000, confidence_level = confident_intervals / 100, statistic=np.std)

                    target_ci = np.percentile(target_bootstrap, [lower_bound, higher_bound], axis=0)
                    ax.fill_between(erp_times, target_ci[0], target_ci[1], color='blue', alpha=0.3)

                    # Bootstrap for nontarget ERP
                    nontarget_bootstrap = bootstrap(nontarget_erp.T[plot_index], n_resamples=1000,statistic=np.mean)
                    nontarget_ci = np.percentile(nontarget_bootstrap, [lower_bound, higher_bound], axis=0)
                    ax.fill_between(erp_times, nontarget_ci[0], nontarget_ci[1], color='orange', alpha=0.3)

    fig.tight_layout()
    fig.show()
#%%
plot_confidence_intervals()
# %%
