#%%
# import necessary modules
from bootstrap_p300 import *
from plot_topo import *
#%%
# Part B Calculate & Plot Parametric Confidence Intervals
#
# IMPORTANT
#
# plot_save_graph function will save plots in current directory with a directory name called "output"
# in interactive mode, it might not show perfectly, but in saving mode, it will work the best.
# name will start with "confident_intervals_subject_[subject].png"

for subject_index in range(0, 8):
    fig, axes, erp_times = plot_confidence_intervals(subject= subject_index + 3)
    plot_save_graph(filename=f"confident_intervals_subject_{subject_index}.png", fig = fig)

#%%
# Part C Bootstrap P values
p_values = []
for subject_index in range(3, 11):
    p_values.append(calculate_bootstrap_p_values(subject = subject_index))

#%%
# Part D plot FDR-corrected P values
p_values = np.array(p_values, dtype = np.float64)
reject_fdr, p_values_fdr = fdr_correction_check(p_values, 0.05)

# IMPORTANT
#
# plot_save_graph function will save plots in current directory with a directory name called "output"
# in interactive mode, it might not show perfectly, but in saving mode, it will work the best.
# name will start with "statistically_significant_subject_[subject].png"

for subject_index in range(0, 8):
    fig, axes, erp_times = plot_statistically_significant(reject_fdr=reject_fdr[subject_index], subject=subject_index + 3)
    plot_save_graph(filename=f"statistically_significant_subject_{subject_index}.png", fig = fig)

# %%
# Part E Evaluate Across Subjects
    
eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg()
event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)

eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)

# IMPORTANT
#
# plot_save_graph function will save plots in current directory with a directory name called "output"
# in interactive mode, it might not show perfectly, but in saving mode, it will work the best.
# name will be "subject_stats.png"

fig, axes = plot_statistically_significant_by_subjects(reject_fdr,erp_times)
plot_save_graph(filename=f"subject_stats.png", fig = fig)


#%%
# Part F: Plot a Spatial Map

eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg()
event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)

channel_names = ['PO7', 'PO8', 'Fz', 'P3', 'P4', 'Oz', 'P6', 'Cz']
 
# N2 and P3b time ranges in erp_times
n2_time_range_stamps  = [0.2, 0.35]
p3b_time_range_stamps = [0.25, 0.5]
 
# get N2 and p3b epoch erps and time ranges

n2_eeg_epoch, n2_time_range     = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=n2_time_range_stamps[0], epoch_end_time=n2_time_range_stamps[1])
target_erp_n2, nontarget_erp_n2 = get_median_erps(n2_eeg_epoch, is_target_event)
# calculate median_erps
p3b_eeg_epoch, p3b_time_range = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=p3b_time_range_stamps[0], epoch_end_time=p3b_time_range_stamps[1])
target_erp_p3b, nontarget_erp_p3b = get_median_erps(p3b_eeg_epoch, is_target_event)

# IMPORTANT
#
# plot_save_graph function will save plots in current directory with a directory name called "output"
# in interactive mode, it might not show perfectly, but in saving mode, it will work the best.
# name will be "topo_target/nontarget_[N2/P3b].png

# plot topology
im, colorbar = plot_topo(channel_names, target_erp_n2.T, title="N2")
cbar_fig = colorbar.ax.figure
cbar_fig.savefig("output/topo_target_N2.png")

im, colorbar = plot_topo(channel_names, target_erp_p3b.T, title="P3b")
cbar_fig = colorbar.ax.figure
cbar_fig.savefig("output/topo_target_P3b")

im, colorbar = plot_topo(channel_names, nontarget_erp_n2.T, title="N2")
cbar_fig = colorbar.ax.figure
cbar_fig.savefig("output/topo_nontarget_N2.png")
im, colorbar = plot_topo(channel_names, nontarget_erp_p3b.T, title="P3b")
cbar_fig = colorbar.ax.figure
cbar_fig.savefig("output/topo_nontarget_P3b")


# %%
