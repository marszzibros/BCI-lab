import numpy as np
import matplotlib.pyplot as plt

# Assuming you have three arrays: arr1, arr2, arr3 with lengths 20, 32, and 20000, respectively
arr1 = np.random.rand(20)
arr2 = np.random.rand(32)
arr3 = np.random.rand(20000)

# Compute the FFT for each array
fft_arr1 = np.fft.rfft(arr1)
fft_arr2 = np.fft.rfft(arr2)
fft_arr3 = np.fft.rfft(arr3)

# Plot the results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(np.abs(fft_arr1))
plt.title('FFT of arr1 (length 20)')

plt.subplot(132)
plt.plot(np.abs(fft_arr2))
plt.title('FFT of arr2 (length 32)')

plt.subplot(133)
plt.plot(np.abs(fft_arr3))
plt.title('FFT of arr3 (length 20000)')
print(np.abs(fft_arr3)[0])
plt.tight_layout()
plt.show()