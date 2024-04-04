


#%%
# Part 1
import import_ssvep_data
import filter_ssvep_data
import predict_ssvep_data
import numpy as np

# initialize import parameters
directory_path = "SsvepData/"
subject = 2

# load ssvep data and prepare for the epochs
data = import_ssvep_data.load_ssvep_data(subject, directory_path)
fs = data['fs']

# get epochs and fft
eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data)
eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs,fs)

# initialize predict parameters
channel = 'Oz'
channel_index = np.where(data['channels'] == channel)[0]
channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:]
event_frequency = np.array([event[:-2] for event in set(data['event_types'])], dtype=int)

predictions = predict_ssvep_data.generate_predictions(channel_eeg_epochs_fft, fft_frequencies, event_frequency)

# as predictions is 12 or 15 based on which amplitude is higher, convert it to bool
predicted_labels = np.array(predictions == 15)

# %%
# Part 2
accuracy, ITR_time = predict_ssvep_data.calculate_accuracy_and_ITR(is_trial_15Hz, predicted_labels)

# %%
print(accuracy)
print(ITR_time)
# %%
print(predictions)
# %%
