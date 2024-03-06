"""
test_import_ssvep_data.py

tester file to import_ssvep_data.py
Created on Mar 6, 2024

Jay Hwasung Jung, Varshney Gentela
"""


#%%
# Part 1: call the function to load the data from subject 1

from import_ssvep_data import *

subject = 2
data_dict = load_ssvep_data(subject=subject)

# %%
# Part 2: call the function to plot the raw data from electrodes Fz and Oz for the current subject
plot_raw_data(data_dict, subject= subject, channels_to_plot=["Fz", "Oz"])

# %%
# Part 3: call the function to extract 20-second epochs from your data
eeg_epochs, eeg_times, is_trial_15Hz = epoch_ssvep_data(data_dict)

# %%
# Part 4: call the function to get the frequency spectrum for each trial and channel
eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs=eeg_epochs, fs=data_dict['fs'])

# %%
# Part 5: call the function to plot the spectra for channels Oz and Fz
spectrum_db_12Hz, spectrum_db_15Hz = plot_power_spectrum(eeg_epochs_fft=eeg_epochs_fft, 
                    fft_frequencies=fft_frequencies, 
                    is_trial_15Hz=is_trial_15Hz, 
                    channels=data_dict["channels"], 
                    channels_to_plot=["Fz", "Oz"], 
                    subject=subject)

# %%
# Part 6: Reflect
"""
1. The peaks in the spectra at specific frequencies, such as 12Hz and 15Hz, are 
indicative of oscillatory neural activity known as neural oscillations. The peaks at
12 Hz and 15 Hz are associated with Beta oscillations. Beta oscillations are originated from 
the motor cortex, and it plays a crucial role in motor planning and executing voluntary movements.

2. SSVEP harmonics emerge in our data due to the non-linear nature of neuronal responses. 
Neurons don't perceive inputs linearly, and this inherent non-linearity can result in 
some neurons responding at frequencies that are multiples of the input frequency, such as 2x or 3x. 

3. It is likely that line noise is interfering the data. In SSVEP (Steady-State Visually Evoked Potentials), 
line noise at 50Hz can introduce unwanted artifacts in the EEG signal, potentially affecting the accuracy of 
frequency-based analysis. 

4. The little bump around 10 Hz in the EEG probably comes from the alpha brain rhythm, which is usually  
observed between 8-13 Hz. You'll likely notice this more in occipital channels like O1, O2, and Oz 
since the alpha rhythm is associated with this region. This is all happening in the occipital cortex, 
where the main visual processing action goes down.
"""


# %%
print(eeg_epochs)
print(eeg_epochs_fft)
# %%
