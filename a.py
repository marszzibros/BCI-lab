import numpy as np

# Assuming you have fdr_result as a boolean array
fdr_result = np.array([[True, False, True],
                       [False, True, False],
                       [True, True, False]])

# Find indices where fdr_result is True
significant_indices = np.where(fdr_result < 0.05)

# Print the result
print("Indices where fdr_result < 0.05:")
print(significant_indices)
