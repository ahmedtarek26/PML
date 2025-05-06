# Problem 3 Solution
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define the unnormalized target densities
def p_tilde_1(x):
    return stats.norm.pdf(x, loc=-5, scale=1) + stats.norm.pdf(x, loc=5, scale=1)

def p_tilde_2(x):
    return stats.norm.pdf(x, loc=-2, scale=1) + stats.norm.pdf(x, loc=2, scale=1)

# Metropolis-Hastings Algorithm
def metropolis_hastings(target_pdf, proposal_std, n_samples, initial_value, burn_in=1000):
    samples = np.zeros(n_samples + burn_in)
    current_x = initial_value
    samples[0] = current_x
    accepted_count = 0

    for i in range(1, n_samples + burn_in):
        # Propose a new sample
        proposed_x = np.random.normal(loc=current_x, scale=proposal_std)

        # Calculate acceptance probability
        p_current = target_pdf(current_x)
        p_proposed = target_pdf(proposed_x)

        # Proposal density q(x_prop | x_curr) is symmetric (Normal), so it cancels out
        acceptance_prob = min(1, p_proposed / p_current) if p_current > 0 else 1

        # Accept or reject
        if np.random.uniform(0, 1) < acceptance_prob:
            current_x = proposed_x
            accepted_count += 1

        samples[i] = current_x

    acceptance_rate = accepted_count / (n_samples + burn_in)
    return samples[burn_in:], acceptance_rate

# Convergence Diagnostics Functions
def calculate_r_hat(chains):
    M = len(chains) # Number of chains
    N = len(chains[0]) # Length of each chain

    # Calculate within-chain variance W
    chain_vars = [np.var(chain, ddof=1) for chain in chains]
    W = np.mean(chain_vars)

    # Calculate between-chain variance B
    chain_means = [np.mean(chain) for chain in chains]
    overall_mean = np.mean(chain_means)
    B = (N / (M - 1)) * np.sum((chain_means - overall_mean)**2)

    # Estimate marginal posterior variance
    var_plus = ((N - 1) / N) * W + (1 / N) * B

    # Calculate R-hat
    R_hat = np.sqrt(var_plus / W) if W > 0 else 1 # Handle W=0 case
    return R_hat, W, B

def calculate_autocorrelation(chain, max_lag):
    N = len(chain)
    mean = np.mean(chain)
    var = np.var(chain)
    autocorr = np.zeros(max_lag + 1)

    if var == 0:
        return autocorr # Return zeros if variance is zero

    autocorr[0] = 1.0
    for k in range(1, max_lag + 1):
        cov = np.sum((chain[:N-k] - mean) * (chain[k:] - mean))
        autocorr[k] = cov / ((N - k) * var) # Use N-k for unbiased estimate, though N*var is common too
        # Alternative often used: autocorr[k] = np.corrcoef(chain[:N-k], chain[k:])[0, 1]

    return autocorr

def calculate_effective_sample_size(chain):
    N = len(chain)
    autocorr = calculate_autocorrelation(chain, N // 2) # Calculate up to lag N/2

    # Sum positive autocorrelations (Geyer's initial positive sequence)
    rho_sum = 0
    for k in range(1, len(autocorr)):
        # Sum pairs of autocorrelations rho_{2t} + rho_{2t+1}
        if k % 2 == 1: # Odd lag (start of pair)
            if k + 1 < len(autocorr):
                pair_sum = autocorr[k] + autocorr[k+1]
                if pair_sum > 0:
                    rho_sum += pair_sum
                else:
                    break # Stop when the sum becomes non-positive
            else: # Handle last odd lag if N/2 is odd
                 if autocorr[k] > 0:
                     rho_sum += autocorr[k]
                 break

    # Calculate n_eff
    n_eff = N / (1 + 2 * rho_sum)
    return n_eff

# --- Analysis Setup ---
N_samples_per_chain = 5000
num_chains = 4
initial_values = [-10, -2, 2, 10]
max_lag_plot = 20
proposal_stds = [0.1, 2.0]
target_pdfs = {
    "Mixture 1 (modes at -5, 5)": p_tilde_1,
    "Mixture 2 (modes at -2, 2)": p_tilde_2
}

results = {}

# --- Run Analysis ---
for name, target_pdf in target_pdfs.items():
    print(f"\n--- Analyzing Target Distribution: {name} ---")
    results[name] = {}
    for prop_std in proposal_stds:
        print(f"\nProposal Standard Deviation: {prop_std}")
        results[name][prop_std] = {}
        chains = []
        acceptance_rates = []
        for i in range(num_chains):
            chain, acc_rate = metropolis_hastings(target_pdf, prop_std, N_samples_per_chain, initial_values[i])
            chains.append(chain)
            acceptance_rates.append(acc_rate)
            print(f"  Chain {i+1}: Initial value={initial_values[i]}, Acceptance rate={acc_rate:.3f}")

        # Calculate R-hat
        R_hat, W, B = calculate_r_hat(chains)
        print(f"  R-hat: {R_hat:.4f}")
        print(f"  Within-chain variance (W): {W:.4f}")
        print(f"  Between-chain variance (B): {B:.4f}")
        results[name][prop_std]["R_hat"] = R_hat
        results[name][prop_std]["W"] = W
        results[name][prop_std]["B"] = B

        # Calculate Autocorrelation (using the first chain for illustration)
        autocorr = calculate_autocorrelation(chains[0], max_lag_plot)
        results[name][prop_std]["autocorrelation"] = autocorr

        # Calculate Effective Sample Size (ESS) for each chain and average
        ess_values = [calculate_effective_sample_size(chain) for chain in chains]
        avg_ess = np.mean(ess_values)
        print(f"  Effective Sample Size (ESS) per chain (avg): {avg_ess:.2f}")
        results[name][prop_std]["avg_ess"] = avg_ess
        results[name][prop_std]["chains"] = chains # Store chains for plotting

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")

for name, prop_results in results.items():
    fig, axes = plt.subplots(3, len(proposal_stds), figsize=(12, 10), sharex='col')
    fig.suptitle(f"MCMC Diagnostics for {name}", fontsize=16)

    for j, (prop_std, data) in enumerate(prop_results.items()):
        chains = data["chains"]
        autocorr = data["autocorrelation"]
        R_hat = data["R_hat"]
        avg_ess = data["avg_ess"]

        # Plot Trace Plots
        ax = axes[0, j]
        for i, chain in enumerate(chains):
            ax.plot(chain, alpha=0.7, label=f"Chain {i+1} (Start={initial_values[i]})")
        ax.set_title(f"Proposal Std = {prop_std}\nTrace Plots")
        ax.set_ylabel("Sample Value")
        if j == 0:
            ax.legend(fontsize='small')

        # Plot Autocorrelation
        ax = axes[1, j]
        lags = np.arange(max_lag_plot + 1)
        ax.stem(lags, autocorr, basefmt=" ")
        ax.set_title(f"Autocorrelation (Chain 1)\nR-hat={R_hat:.3f}, Avg ESS={avg_ess:.1f}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_ylim(-0.2, 1.05)

        # Plot Histograms
        ax = axes[2, j]
        all_samples = np.concatenate(chains)
        ax.hist(all_samples, bins=50, density=True, alpha=0.7, label='All Chains')
        # Overlay true density (scaled)
        x_range = np.linspace(all_samples.min() - 1, all_samples.max() + 1, 300)
        true_pdf_vals = target_pdfs[name](x_range)
        # Normalize true PDF for comparison
        integral_true_pdf, _ = integrate.quad(target_pdfs[name], -np.inf, np.inf)
        ax.plot(x_range, true_pdf_vals / integral_true_pdf, \'r-\' , lw=2, label='True Density')
        ax.set_title("Posterior Histogram")
        ax.set_xlabel("Sample Value")
        ax.set_ylabel("Density")
        if j == 0:
            ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_filename = f"/home/ubuntu/homework_solution/problem3_{name.replace(\' 
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

print("\nProblem 3 Analysis Complete.")

# --- Discussion Placeholder ---
# Discussion points based on the results:
# - Compare R-hat values: Values close to 1 indicate convergence.
# - Compare autocorrelation plots: Faster decay to 0 indicates better mixing.
# - Compare ESS: Higher ESS means more independent samples and better efficiency.
# - Effect of proposal_std: Too small leads to slow exploration (high autocorrelation, low ESS, potentially poor R-hat if chains get stuck). Too large leads to low acceptance rates and inefficient sampling (though might explore modes better initially).
# - Effect of target distribution modality: Wider separation of modes (Mixture 1) makes it harder for chains to jump between modes, especially with small proposal_std. This can lead to poor R-hat, high autocorrelation, and low ESS.

