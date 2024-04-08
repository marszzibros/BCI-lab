


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
channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()
event_frequency = np.array([event[:-2] for event in set(data['event_types'])], dtype=int)
event_frequency= np.sort(event_frequency)[::-1]

# get predictions based on epoched data
predictions = predict_ssvep_data.generate_predictions(channel_eeg_epochs_fft, fft_frequencies, event_frequency)

# as predictions is 12 or 15 based on which amplitude is higher, convert it to bool
predicted_labels = np.array(predictions == 15)

# %%
# Part 2
accuracy, ITR_time = predict_ssvep_data.calculate_accuracy_and_ITR(is_trial_15Hz, predicted_labels)

# %%
# Part 3 & 4 

# initialize predict parameters
# we will use Oz electrode



for channel_index, channel in enumerate(data['channels']):
    accuracy_array = np.ones((21,21), dtype=float) * -1
    ITR_array = np.ones((21,21), dtype=float) * -1

    min_accuracy = 1
    min_ITR = 100
    for start_time in range(0, 21):
        for end_time in range(0, 21):
            if start_time < end_time:

                # get epochs and fft
                eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data, epoch_start_time=start_time, epoch_end_time=end_time)

                eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)
                channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()

                predictions = predict_ssvep_data.generate_predictions(channel_eeg_epochs_fft, fft_frequencies, event_frequency)

                # as predictions is 12 or 15 based on which amplitude is higher, convert it to bool
                predicted_labels = np.array(predictions == 15)
                accuracy, ITR_time = predict_ssvep_data.calculate_accuracy_and_ITR(is_trial_15Hz, predicted_labels)

                if min_accuracy > accuracy:
                    min_accuracy = accuracy
                if min_ITR > ITR_time:
                    min_ITR = ITR_time
                
                accuracy_array[start_time, end_time] = accuracy
                ITR_array[start_time, end_time] = ITR_time

    for start_time in range(0, 21):
        for end_time in range(0, 21):
            if accuracy_array[start_time, end_time] == -1:
                accuracy_array[start_time, end_time] = min_accuracy
            if ITR_array[start_time, end_time] == -1:
                ITR_array[start_time, end_time] = min_ITR

    predict_ssvep_data.plot_accuracy_and_ITR(accuracy_array=accuracy_array, ITR_array=ITR_array, channel=channel, subject=subject)

# %%
# Part 5
condition = [7,16]

eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data, epoch_start_time= condition[0], epoch_end_time=condition[1])

eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)
channel_eeg_epochs_fft = eeg_epochs_fft[:,channel_index,:].squeeze()

predict_ssvep_data.plot_predictor_histogram(eeg_epochs_fft=channel_eeg_epochs_fft, fft_frequencies=fft_frequencies, event_frequency=event_frequency, true_label=is_trial_15Hz)

# False Positive - place near 0 to minimize the chances of false positive; it will increase the False Negative
# False Negative - place near -5000 to minimize the chances of False negative; it will increase the False Positive
# Therefore, if we want to minimize False Positive while minimizing the False Negative, we should choose where both are 
# %%
