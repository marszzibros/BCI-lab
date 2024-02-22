#%%

# import necessary modules
from bootstrap_p300 import *
from plot_topo import *

eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(subject=3)
event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)

channel_names = ['Fz', 'Cz','P3','Pz','P4','PO7', 'Oz', 'PO8'] 

# N2 and P3b time ranges in erp_times
n2_time_range_stamps  = [0.2, 0.35]
p3b_time_range_stamps = [0.25, 0.5]

n2_eeg_epoch, n2_time_range     = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=n2_time_range_stamps[0], epoch_end_time=n2_time_range_stamps[1])

n2_eeg_epoch = np.mean(n2_eeg_epoch[is_target_event], axis = 0).T

p3b_eeg_epoch, p3b_time_range     = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=p3b_time_range_stamps[0], epoch_end_time=p3b_time_range_stamps[1])
p3b_eeg_epoch = np.mean(p3b_eeg_epoch[is_target_event], axis = 0).T

# Create a single plot
plt.figure(figsize=(12, 8))

# Plot each channel on the same plot
for plot_index in range(8):
    # Plot target ERP
    plt.plot(n2_time_range, n2_eeg_epoch[plot_index], label=f'Channel {plot_index}', alpha=0.7)

# Set x-axis ticks
plt.xticks(np.linspace(n2_time_range[0], n2_time_range[-1], 4))

# Set plot title and labels
plt.title('ERP for All Channels', fontsize=16)
plt.xlabel('Time from Flash Onset (s)')
plt.ylabel('Voltage (uV)')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Create a single plot
plt.figure(figsize=(12, 8))

# Plot each channel on the same plot
for plot_index in range(8):
    # Plot target ERP
    plt.plot(p3b_time_range, p3b_eeg_epoch[plot_index], label=f'Channel {plot_index}', alpha=0.7)

# Set x-axis ticks
plt.xticks(np.linspace(p3b_time_range[0], p3b_time_range[-1], 4))

# Set plot title and labels
plt.title('ERP for All Channels', fontsize=16)
plt.xlabel('Time from Flash Onset (s)')
plt.ylabel('Voltage (uV)')

# Add legend
plt.legend()

# Show the plot
plt.show()