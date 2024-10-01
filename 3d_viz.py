import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load("E:/Universidade/Dados_Sísmicos/Impedance/impedance_synth.npy")

# Check if the data has three dimensions
if data.ndim != 3:
    raise ValueError("The loaded data does not have three dimensions")

# Plotting the 3D data
fig = plt.figure(figsize=(15, 5))

# First section (slice along the first axis)
ax1 = fig.add_subplot(131)
ax1.imshow(data[:, :, data.shape[2] // 2], cmap="viridis", aspect="auto")
ax1.set_xlabel("X Label")
ax1.set_ylabel("Y Label")
ax1.set_title("Section 1")

# Second section (slice along the second axis)
ax2 = fig.add_subplot(132)
ax2.imshow(data[:, data.shape[1] // 2, :], cmap="viridis", aspect="auto")
ax2.set_xlabel("X Label")
ax2.set_ylabel("Z Label")
ax2.set_title("Section 2")

# Third section (slice along the third axis)
ax3 = fig.add_subplot(133)
ax3.imshow(data[data.shape[0] // 2, :, :], cmap="viridis", aspect="auto")
ax3.set_xlabel("Y Label")
ax3.set_ylabel("Z Label")
ax3.set_title("Section 3")

plt.tight_layout()
plt.show()
