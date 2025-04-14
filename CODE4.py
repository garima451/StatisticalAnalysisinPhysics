import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import gamma

# Parameters
N = 100  # Total flips
M = 60   # Number of heads
a, b = 2, 2  # Beta prior parameters
mu, sigma = 0.5, np.sqrt(0.01)  # Gaussian prior parameters

# Grid of f values
f = np.linspace(0, 1, 1000)

# Likelihood (unnormalized)
likelihood = f**M * (1-f)**(N-M)

# Part (a.i): Beta Prior
prior_beta = beta_dist.pdf(f, a, b)
posterior_beta = beta_dist.pdf(f, M + a, N - M + b)

# Part (a.ii): Gaussian Prior
def gaussian_prior(f, mu, sigma):
    # Truncated Gaussian (approximate, ignoring normalization for simplicity)
    return np.exp(-0.5 * ((f - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

prior_gaussian = gaussian_prior(f, mu, sigma)
posterior_gaussian = likelihood * prior_gaussian
# Normalize posterior numerically
posterior_gaussian /= np.trapz(posterior_gaussian, f)

# Print shapes for verification
print("Beta Posterior: Beta({}, {})".format(M + a, N - M + b))

# Part (b): MLE
f_mle = M / N
print("MLE for f: ", f_mle)

import matplotlib.pyplot as plt

# Normalize likelihood
norm_const = gamma(M+1) * gamma(N-M+1) / gamma(N+2)
likelihood_normalized = likelihood / norm_const

# Plotting
plt.figure(figsize=(12, 8))

# Beta Prior Case
plt.subplot(2, 1, 1)
plt.plot(f, likelihood_normalized, label='Normalized Likelihood', color='black')
plt.plot(f, prior_beta, label='Beta Prior (a={}, b={})'.format(a, b), color='blue')
plt.plot(f, posterior_beta, label='Beta Posterior', color='red')
plt.axvline(f_mle, color='green', linestyle='--', label='MLE (f={})'.format(f_mle))
plt.title('Bayesian Inference with Beta Prior')
plt.xlabel('f (Probability of Heads)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# Gaussian Prior Case
plt.subplot(2, 1, 2)
plt.plot(f, likelihood_normalized, label='Normalized Likelihood', color='black')
plt.plot(f, prior_gaussian / np.trapz(prior_gaussian, f), label='Gaussian Prior (μ={}, σ²={})'.format(mu, sigma**2), color='blue')
plt.plot(f, posterior_gaussian, label='Gaussian Posterior', color='red')
plt.axvline(f_mle, color='green', linestyle='--', label='MLE (f={})'.format(f_mle))
plt.title('Bayesian Inference with Gaussian Prior')
plt.xlabel('f (Probability of Heads)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
