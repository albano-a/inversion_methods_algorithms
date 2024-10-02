import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the .npy file
data = np.load("E:/Universidade/Dados_Sísmicos/Impedance/impedance_synth.npy")

# Check if the data has three dimensions
if data.ndim != 3:
    raise ValueError("The loaded data does not have three dimensions")

# Extract the specific sections
section1 = data[:, :,128]
section2 = data[:, 128, :]
section3 = data[128, :, :]

# Create grids for the sections
X1, Y1 = np.meshgrid(np.arange(section1.shape[1]), np.arange(section1.shape[0]))
X2, Z2 = np.meshgrid(np.arange(section2.shape[1]), np.arange(section2.shape[0]))
Y3, Z3 = np.meshgrid(np.arange(section3.shape[1]), np.arange(section3.shape[0]))

# Plotting the 3D data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the first section
ax.plot_surface(X1, Y1, section1, cmap="viridis", alpha=0.7)

# Plot the second section
ax.plot_surface(X2, section2, Z2, cmap="plasma", alpha=0.7)

# Plot the third section
ax.plot_surface(section3, Y3, Z3, cmap="inferno", alpha=0.7)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
ax.set_title("3D Intersecting Sections")

plt.show()
