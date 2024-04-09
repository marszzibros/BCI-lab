


#%%

# Part 1
import import_ssvep_data
import filter_ssvep_data
import predict_ssvep_data
import numpy as np

# initialize import parameters
directory_path = "SsvepData/"
subject = 1

# load ssvep data and prepare for the epochs
data = import_ssvep_data.load_ssvep_data(subject, directory_path)

fs = data['fs']
# get epochs and fft

eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data)
eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs,fs)

# initialize predict parameters
channel = 'O2'

channel_index = np.where(data['channels'] == channel)[0]
channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()

event_frequency = np.array([event[:-2] for event in set(data['event_types'])], dtype=int)
event_frequency= np.sort(event_frequency)[::-1] 

# get predictions based on epoched data
predictions = predict_ssvep_data.generate_predictions(channel_eeg_epochs_fft, fft_frequencies, event_frequency)

# %%
# Part 2
start_time = 0
end_time = 20

duration=(end_time-start_time) * fs * (1/10)

accuracy, ITR_time = predict_ssvep_data.calculate_accuracy_and_ITR(is_trial_15Hz, predictions, eeg_epochs_fft.shape[1], duration=duration)

# %%
# Part 3 & 4 

# initialize predict parameters
# we will use Oz electrode
# initialize predict parameters
channel = 'O2'

start_time_array = np.linspace(start=0, stop=20, num=21)
end_time_array = np.linspace(start=0, stop=20, num=21)

predict_ssvep_data.plot_accuracy_and_ITR(data, channel=channel, subject=subject, end_time_array=end_time_array, start_time_array=start_time_array)

# %%
# Part 5
condition = [11,12]

# initialize predict parameters
channel = 'O2'
channel_index = np.where(data['channels'] == channel)[0]
channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()

eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data, epoch_start_time= condition[0], epoch_end_time=condition[1])

eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)
channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()

predict_ssvep_data.plot_predictor_histogram(eeg_epochs_fft=channel_eeg_epochs_fft, fft_frequencies=fft_frequencies, event_frequency=event_frequency,true_label=is_trial_15Hz, channel=channel, subject=subject, condition=condition)

# %%
