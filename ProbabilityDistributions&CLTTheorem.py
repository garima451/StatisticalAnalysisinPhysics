import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def binomial(n, p, size):
    return np.sum(np.random.rand(size, n) < p, axis=1)
def poisson(lam, size):
    result = np.zeros(size)
    for i in range(size):
        L = np.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= np.random.rand()
        result[i] = k - 1
    return result
def normal(mu, sigma, size):
    u1 = np.random.rand(size)
    u2 = np.random.rand(size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mu + sigma * z0
def cauchy_lorentz(x0, gamma, size):
    u = np.random.rand(size)
    return x0 + gamma * np.tan(np.pi * (u - 0.5))
def plot_all_distributions(N_values, color_map):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    distributions = [
        ('Binomial', binomial, (10, 0.5)),
        ('Poisson', poisson, (5,)),
        ('Normal', normal, (0, 1)),
        ('Cauchy-Lorentz', cauchy_lorentz, (0, 1))
    ]
    for ax, (title, dist_func, params) in zip(axes.flat, distributions):
        for i, N in enumerate(N_values):
            means = np.array([np.mean(dist_func(*params, size=N)) for _ in range(10000)])
            ax.hist(means, bins=50, alpha=0.6, color=color_map[i], label=f'N={N}')
        ax.set_title(f'{title} Distribution - CLT Verification')
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()
def animate_clt(N_values, dist_func, params, title, color):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def update(frame):
        ax.clear()
        N = N_values[frame]
        means = np.array([np.mean(dist_func(*params, size=N)) for _ in range(10000)])
        ax.hist(means, bins=50, alpha=0.6, color=color, label=f'N={N}')
        ax.set_title(f'{title} Distribution - N={N}')
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
    
    ani = animation.FuncAnimation(fig, update, frames=len(N_values), repeat=False)
    plt.show()
# Parameters
N_values = [1, 5, 10, 30, 50, 100, 500, 1000]
color_map = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
# Plot all distributions in a single figure
plot_all_distributions(N_values, color_map)
# Animations
animate_clt(N_values, binomial, (10, 0.5), 'Binomial', 'blue')
animate_clt(N_values, poisson, (5,), 'Poisson', 'green')
animate_clt(N_values, normal, (0, 1), 'Normal', 'orange')
animate_clt(N_values, cauchy_lorentz, (0, 1), 'Cauchy-Lorentz', 'red')
