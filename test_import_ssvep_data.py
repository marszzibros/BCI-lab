#%%
# Part 1: call the function to load the data from subject 1

from import_ssvep_data import *

data_dict = load_ssvep_data(subject=2)

# %%
# Part 2: call the function to plot the raw data from electrodes Fz and Oz for the current subject
plot_raw_data(data_dict, subject= 2, channels_to_plot=["Fz", "Oz"])

# %%
# Part 3: call the function to extract 20-second epochs from your data
eeg_epochs, eeg_times, is_trial_15Hz = epoch_ssvep_data(data_dict)

# %%
# Part 4: call the function to get the frequency spectrum for each trial and channel
eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs=eeg_epochs, fs=data_dict['fs'])

# %%
plot_power_spectrum(eeg_epochs_fft=eeg_epochs_fft, 
                    fft_frequencies=fft_frequencies, 
                    is_trial_15Hz=is_trial_15Hz, 
                    channels=data_dict["channels"], 
                    channels_to_plot=["Fz", "Oz"], 
                    subject=2)
# %%
print(fft_frequencies * data_dict['fs'])
print(fft_frequencies[1] - fft_frequencies[0])
print(fft_frequencies[2] - fft_frequencies[1])
print(fft_frequencies[3] - fft_frequencies[2])
print(eeg_epochs_fft.shape)
print(fft_frequencies)
print(is_trial_15Hz.shape)
# %%
print(data_dict["channels"])
