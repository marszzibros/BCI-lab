import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

def plot_erps_with_confidence_intervals(data, time_points, channel_names, event_type='target'):
    """
    Plot ERPs with 95% confidence intervals based on the standard error of the mean.

    Parameters:
    - data: A 3D numpy array with dimensions (channels, time points, trials).
    - time_points: 1D array representing the time points.
    - channel_names: List of channel names.
    - event_type: String specifying the event type ('target' or 'nontarget').

    Returns:
    - None (plots the ERPs with confidence intervals).
    """
    # Assuming data has dimensions (channels, time points, trials)
    num_channels, num_time_points, num_trials = data.shape
    print(data.shape)
    # Calculate mean ERP and standard error of the mean (SEM) for each time point and channel
    mean_erp = np.mean(data, axis=-1)
    sem_erp = sem(data, axis=-1)
    print(sem_erp.shape)
    print(mean_erp.shape)
    # Calculate 95% confidence interval
    confidence_interval = 1.96 * sem_erp  # Assuming normal distribution, 1.96 is the z-value for 95% confidence

    # Plot ERPs
    for channel_idx in range(num_channels):
        plt.plot(time_points, mean_erp[channel_idx], label=f'{channel_names[channel_idx]} - {event_type.capitalize()} ERP')

        # Plot confidence intervals using fill_between
        plt.fill_between(time_points,
                         mean_erp[channel_idx] - confidence_interval[channel_idx],
                         mean_erp[channel_idx] + confidence_interval[channel_idx],
                         alpha=0.2, label=f'{channel_names[channel_idx]} - 95% CI')

    # Add labels, title, legend, etc.
    plt.xlabel('Time Points')
    plt.ylabel('Voltage')
    plt.title(f'ERPs with 95% Confidence Intervals ({event_type.capitalize()} Events)')
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Example data generation (replace this with your actual data)
np.random.seed(42)
num_channels = 5
num_time_points = 100
num_trials = 20

data_target = np.random.normal(loc=0, scale=1, size=(num_channels, num_time_points, num_trials))
data_nontarget = np.random.normal(loc=1, scale=1, size=(num_channels, num_time_points, num_trials))

time_points = np.arange(num_time_points)
channel_names = [f'Channel_{i+1}' for i in range(num_channels)]

# Using the provided function to plot ERPs with confidence intervals for target events
plot_erps_with_confidence_intervals(data_target, time_points, channel_names, event_type='target')

# Using the same function to plot ERPs with confidence intervals for non-target events
plot_erps_with_confidence_intervals(data_nontarget, time_points, channel_names, event_type='nontarget')
