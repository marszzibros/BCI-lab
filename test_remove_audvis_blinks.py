"""
This script uses functions from remove_audvis_blinks.py to load a dataset, create a plot of ICA component spatial
topographies, plot two EOG components identified by ICA, removes the two EOG artifact components,transforms the
source data back into electrode space without removing any sources, and plots raw/cleaned/reconstructed EEG data.  

@author: marszzibros
@author: APC-03

file: test_remove_audvis_blinks.py
BME 6710 - Dr. Jangraw
lab 5: Spatial Components
"""

# Import statement
from remove_audvis_blinks import load_data, plot_components, get_sources, remove_sources, compare_reconstructions

# Part 1: Load Data
data_directory = '/Users/aiden/PycharmProjects/BCI_202401/AudVisData.npy'
channels_to_plot = ['Fpz', 'Cz', 'Iz']
data = load_data(data_directory, channels_to_plot)

# It appears that a blink may have occurred around 15 seconds into the recording. Whenever a blink occurs,
# there is a large positive deflection in voltage primarily on frontal electrodes. At this time point there is a
# large positive voltage deflection at the Fpz electrode, located on the frontal lobe. There are smaller positive
# voltage deflections in the Cz and Iz electrodes, located in a central and occipital area respectively according to
# the standard 10/10 system.

# Part 2: Plot the Components
mixing_matrix = data['mixing_matrix']
channels = data['channels']
mixing_matrix.shape

plot_components(mixing_matrix=mixing_matrix, channels=channels)

# Part 3: Transform into Source Space
eeg = data['eeg']
unmixing_matrix = data['unmixing_matrix']
fs = data['fs']

sources_to_plot = [0, 3, 9]

source_activations = get_sources(eeg=eeg, unmixing_matrix=unmixing_matrix, fs=fs, sources_to_plot=sources_to_plot)

# Part 4: Remove Artifact Components
sources_to_remove = [0, 3, 9]
cleaned_eeg = remove_sources(source_activations=source_activations.copy(), mixing_matrix=mixing_matrix,
                             sources_to_remove=sources_to_remove)
reconstructed_eeg = remove_sources(source_activations=source_activations.copy(), mixing_matrix=mixing_matrix,
                                   sources_to_remove=[])

# Part 5: Transform Back into Electrode Space
channels_to_plot = ["Fpz", "Cz", "Iz"]
compare_reconstructions(eeg=eeg.copy(), reconstructed_eeg=reconstructed_eeg.copy(), cleaned_eeg=cleaned_eeg.copy(),
                        fs=fs, channels=channels, channels_to_plot=channels_to_plot)
