# Problem 2 Solution
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# Define the unnormalized target density
def p_tilde(x):
    return np.exp(-x**4 / 4 - x**2 / 2)

# Define the proposal distribution q(x) - Standard Normal
q_dist = stats.norm(0, 1)
def q(x):
    return q_dist.pdf(x)

# Function to find the optimal M for rejection sampling
# We need to maximize p_tilde(x) / q(x)
def objective_for_M(x):
    # Avoid division by zero or very small numbers if q(x) is near zero far from origin
    qx_val = q(x)
    if qx_val < 1e-100:
        # If q(x) is tiny, p_tilde(x) should also be tiny or zero
        # If p_tilde(x) is non-zero here, M would need to be huge, indicates issue
        # However, for our functions, p_tilde decays faster than q, so this shouldn't be the max
        return 0 # Return 0 or a small number, as this won't be the maximum
    return -p_tilde(x) / qx_val # Minimize the negative ratio

# Find the x that maximizes p_tilde(x) / q(x)
# The ratio is p_tilde(x) / q(x) = exp(-x^4/4 - x^2/2) / ( (1/sqrt(2*pi)) * exp(-x^2/2) )
# = sqrt(2*pi) * exp(-x^4/4)
# This is maximized when x=0.
max_ratio_at_x0 = np.sqrt(2 * np.pi) * np.exp(0) # M = p_tilde(0) / q(0)
M = max_ratio_at_x0
print(f"Calculated M = sqrt(2*pi) = {M:.4f}")

# Alternatively, use numerical optimization to verify (though analytical is better here)
result_opt = optimize.minimize_scalar(objective_for_M)
x_max = result_opt.x
M_opt = -result_opt.fun # Maximum value of the ratio
print(f"Numerically optimized M: {M_opt:.4f} at x = {x_max:.4f}")
# Use the analytical M for better accuracy

# Rejection Sampling Implementation
def rejection_sampling(n_samples, M):
    samples = []
    proposals_tried = 0
    while len(samples) < n_samples:
        proposals_tried += 1
        # Sample x_star from proposal q(x)
        x_star = q_dist.rvs()
        # Sample u from Uniform(0, 1)
        u = np.random.uniform(0, 1)
        # Acceptance condition
        acceptance_ratio = p_tilde(x_star) / (M * q(x_star))
        if u <= acceptance_ratio:
            samples.append(x_star)
    acceptance_rate = n_samples / proposals_tried
    return np.array(samples), acceptance_rate

# Generate samples
num_target_samples = 5000
generated_samples, acceptance_rate = rejection_sampling(num_target_samples, M)

print(f"Generated {len(generated_samples)} samples.")
print(f"Acceptance rate: {acceptance_rate:.4f}")

# Estimate the normalization constant Z
# Z is estimated as M times the acceptance rate (averaged over many trials)
# More directly, Z = integral(p_tilde(x) dx). The acceptance rate = integral(p_tilde(x) dx) / (M * integral(q(x) dx))
# Since integral(q(x) dx) = 1, acceptance_rate = Z / M. So, Z_estimated = M * acceptance_rate.
Z_estimated = M * acceptance_rate
print(f"Estimated normalization constant Z: {Z_estimated:.4f}")

# Compare with numerical integration
Z_numerical, abs_error = integrate.quad(p_tilde, -np.inf, np.inf)
print(f"Numerical integration result for Z: {Z_numerical:.4f} (Error estimate: {abs_error:.2e})")

# Plotting the results
x_range = np.linspace(-4, 4, 500)
p_normalized_numerical = p_tilde(x_range) / Z_numerical

plt.figure(figsize=(10, 6))
plt.hist(generated_samples, bins=50, density=True, alpha=0.6, label=f'Rejection Samples (N={num_target_samples})')
plt.plot(x_range, p_normalized_numerical, 'r-', lw=2, label='Target p(x) (Normalized Numerically)')
plt.plot(x_range, q(x_range), 'g--', lw=2, label='Proposal q(x) (Standard Normal)')
# Plot M*q(x) to show the envelope
plt.plot(x_range, M * q(x_range), 'k:', lw=1, label=f'M * q(x) (Envelope, M={M:.2f})')
plt.title('Rejection Sampling Results')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('/home/ubuntu/homework_solution/problem2_plot.png')
print("Plot saved to /home/ubuntu/homework_solution/problem2_plot.png")


