import numpy as np
import matplotlib.pyplot as plt

# Generating sample data for accuracies (replace with your actual data)
num_epochs = 400
num_time_points = 20
accuracies_subject1 = np.random.rand(num_epochs, num_time_points, num_time_points) * 100
accuracies_subject2 = np.random.rand(num_epochs, num_time_points, num_time_points) * 100

# Plotting
plt.figure(figsize=(16, 8))

# Pseudocolor plot for accuracies of Subject 1
plt.subplot(2, 1, 1)
plt.imshow(accuracies_subject1.mean(axis=0), cmap='viridis', aspect='auto')
plt.colorbar(label='Accuracy (%)')
plt.title('Subject 1 Mean Accuracies at Various Epoch Limits')
plt.xlabel('End Time')
plt.ylabel('Start Time')

# Pseudocolor plot for accuracies of Subject 2
plt.subplot(2, 1, 2)
plt.imshow(accuracies_subject2.mean(axis=0), cmap='viridis', aspect='auto')
plt.colorbar(label='Accuracy (%)')
plt.title('Subject 2 Mean Accuracies at Various Epoch Limits')
plt.xlabel('End Time')
plt.ylabel('Start Time')

plt.tight_layout()
plt.show()
