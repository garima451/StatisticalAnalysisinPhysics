import numpy as np
import matplotlib.pyplot as plt
 # Function to compute joint probability for continuous variables
def joint_probability_continuous(xi, yi, bins=20):
    histogram, x_edges, y_edges = np.histogram2d(xi, yi, bins=bins, density=True)
    return x_edges, y_edges, histogram
# Generate continuous data for two independent variables
num_samples = 10000
xi_continuous = np.random.normal(0, 1, num_samples) # Standard normal distribution for X
yi_continuous = np.random.normal(0, 1, num_samples) # Standard normal distribution for Y
 # Compute joint probability for continuous variables
x_edges, y_edges, joint_prob_continuous = joint_probability_continuous(xi_continuous, yi_continuous, bins=20)
 # Visualize joint probability as a heatmap

plt.figure(figsize=(8, 6))
plt.imshow(joint_prob_continuous.T, origin="lower", aspect="auto", cmap="viridis",
extent=[y_edges[0], y_edges[-1], x_edges[0], x_edges[-1]])
plt.colorbar(label="Probability Density")

plt.title("Joint Probability Distribution (Continuous)")
plt.xlabel("Y")
plt.ylabel("X")
plt.grid(False)
plt.show()

#Display joint probabilities (sample values for bins)
print("Sample Joint Probability Density Matrix:")
print(joint_prob_continuous)
