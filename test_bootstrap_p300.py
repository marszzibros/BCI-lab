#%%
from bootstrap_p300 import *

#%%
for i in range(0, 8):
    fig, axes, erp_times = plot_confidence_intervals(subject=i + 3)
    plot_save_graph(filename=f"confident_intervals_subject_{i}.png", fig = fig)

#%%
p_values = []
for i in range(3, 11):
    p_values.append(calculate_bootstrap_p_values(subject = i))

#%%
p_values = np.array(p_values, dtype = np.float64)
reject_fdr, p_values_fdr = fdr_correction_check(p_values, 0.05)


#%%
for i in range(0, 8):
    fig, axes, erp_times = plot_statistically_significant(reject_fdr=reject_fdr[i], subject=i + 3)
    plot_save_graph(filename=f"statistically_significant_subject_{i}.png", fig = fig)

# %%
# call load and plot functions from load_p300_data module
eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg()
event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)

fig, axes = plot_statistically_significant_by_subjects(reject_fdr,erp_times)
plot_save_graph(filename=f"subject_stats.png", fig = fig)

# %%
