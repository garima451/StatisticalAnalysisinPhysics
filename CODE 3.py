import numpy as np
import matplotlib.pyplot as plt
from math import comb
def toss_coin(n, q):
    return np.random.binomial(1, q, n)
def binomial_cdf(k, n, p):
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))
def binomial_test(n, q_observed, q_expected=0.5, alpha=0.05):
    observed_heads = np.sum(q_observed)
    p_lower = binomial_cdf(observed_heads, n, q_expected)
    p_upper = 1 - binomial_cdf(observed_heads - 1, n, q_expected)
    p_value = 2 * min(p_lower, p_upper)
    reject_null = p_value < alpha
    return p_value, reject_null, observed_heads
# Parameters
n = 100  # Number of coin tosses
q = 0.55  # Probability of heads
alpha = 0.05  # Significance level
observed_tosses = toss_coin(n, q)
p_value, reject_null, observed_heads = binomial_test(n, observed_tosses, q_expected=0.5, alpha=alpha)
print(f"Number of tosses: {n}")
print(f"Number of heads observed: {observed_heads}")
print(f"P-value: {p_value}")
if reject_null:
    print("Null hypothesis (q = 0.5) is rejected: The observed result is significantly different from 0.5.")
else:
    print("Fail to reject the null hypothesis (q = 0.5). The result is not significantly different from 0.5.")
x = np.arange(0, n + 1)
pmf = [comb(n, k) * (0.5 ** k) * (0.5 ** (n - k)) for k in x]  # PMF for H0 (q = 0.5)
plt.plot(x, pmf, label="Expected Distribution (q = 0.5)", color='blue')
plt.vlines(observed_heads, 0, pmf[observed_heads], colors='red', label="Observed Heads")
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.legend()
plt.title('Binomial Test for Coin Tosses')
plt.show()


