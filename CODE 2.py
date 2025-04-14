import numpy as np
import matplotlib.pyplot as plt
# Function to compute joint probability for discrete variables
def joint_probability_discrete(xi, yi):
    joint_counts = {}
    # Count occurrences of each pair (xi, yi)
    for x, y in zip(xi, yi):
        if (x, y) in joint_counts:
            joint_counts[(x, y)] += 1
        else:
            joint_counts[(x, y)] = 1
    # Calculate joint probabilities
    total_samples = len(xi)
    joint_prob = {key: value / total_samples for key, value in joint_counts.items()}
    return joint_prob
num_samples = 10000
x_discrete = np.random.randint(1, 5, size=num_samples)  # Random integers between 1 and 4
y_discrete = np.random.randint(1, 5, size=num_samples)  # Random integers between 1 and 4
joint_prob_discrete = joint_probability_discrete(x_discrete, y_discrete)
print("Joint Probability Distribution (Discrete):")
for (x, y), prob in sorted(joint_prob_discrete.items()):
    print(f"P(X={x}, Y={y}) = {prob:.4f}")
unique_x = np.unique(x_discrete)
unique_y = np.unique(y_discrete)
joint_matrix = np.zeros((len(unique_x), len(unique_y)))
for (x, y), prob in joint_prob_discrete.items():
    joint_matrix[x - unique_x[0], y - unique_y[0]] = prob
plt.figure(figsize=(8, 6))
plt.imshow(
    joint_matrix, origin="lower", aspect="auto", cmap="viridis",
    extent=[unique_y[0] - 0.5, unique_y[-1] + 0.5, unique_x[0] - 0.5, unique_x[-1] + 0.5]
)
plt.colorbar(label="Probability")
plt.title("Joint Probability Distribution (Discrete)")
plt.xlabel("Y")
plt.ylabel("X")
plt.xticks(unique_y)
plt.yticks(unique_x)
plt.grid(False)
plt.show()
