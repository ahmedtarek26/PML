# Problem 4 Solution
import numpy as np
import pandas as pd
import ast
import torch
import torch.optim as optim
from torch.distributions import constraints, Distribution, Normal, MultivariateNormal
from torch.autograd.functional import hessian
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import scipy.special as sc_special
import math
import matplotlib.pyplot as plt

# --- 0. Data Preprocessing ---
def preprocessing_dataset(dataset_path=\"/home/ubuntu/homework_solution/data/ADSAI_football.csv\"):
    try:
        football = pd.read_csv(dataset_path)
        football[\"Team A\"] = football[\"Team A\"].apply(ast.literal_eval)
        football[\"Team B\"] = football[\"Team B\"].apply(ast.literal_eval)

        max_player_id = max(
            max(p for team in football[\"Team A\"] for p in team),
            max(p for team in football[\"Team B\"] for p in team)
        )
        # Player IDs seem to be 1-based, adjust for 0-based indexing
        num_players = max_player_id

        goal_diff = torch.tensor((football[\"Goal A\"] - football[\"Goal B\"]).values, dtype=torch.float) # Use float for gradients

        # Adjust player IDs to be 0-based
        teams_A = [torch.tensor([p - 1 for p in team]) for team in football[\"Team A\"]]
        teams_B = [torch.tensor([p - 1 for p in team]) for team in football[\"Team B\"]]

        print(f"Data loaded: {len(goal_diff)} matches, {num_players} players.")
        return teams_A, teams_B, goal_diff, num_players
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None, None, None, 0
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None, None, None, 0

teams_A, teams_B, goal_diff, num_players = preprocessing_dataset()

if num_players > 0:
    # --- 1. MAP Estimation and Laplace Approximation ---

    # Log Modified Bessel function approximation (for large z)
    # log(I_k(z)) approx z - 0.5 * log(2*pi*z)
    # Using torch.special.i0e and i1e might be better but requires checking for general k
    # Let\"s try the asymptotic expansion as hinted.
    def log_bessel_i_approx(k, z):
        # Ensure z is positive and non-zero for the log
        z_safe = torch.clamp(z, min=1e-9)
        # The approximation is better for large z. For small z, I_k(z) ~ (z/2)^k / Gamma(k+1)
        # Let\"s use the asymptotic form for simplicity as hinted.
        log_approx = z_safe - 0.5 * torch.log(2 * torch.pi * z_safe)
        # Handle k=0 case where I_0(0) = 1, log(I_0(0)) = 0
        # Handle small z where approx might be poor? For now, stick to the formula.
        return log_approx

    # Skellam log probability using approximation
    def log_skellam_approx(k, lambdaA, lambdaB):
        # k: goal difference (value)
        # lambdaA, lambdaB: exp(strengthA), exp(strengthB)
        term1 = -(lambdaA + lambdaB)
        # term2 = (k / 2) * (torch.log(lambdaA) - torch.log(lambdaB))
        # Avoid log(0) if lambda can be zero, use log(lambdaA / lambdaB)
        lambdaA_safe = torch.clamp(lambdaA, min=1e-9)
        lambdaB_safe = torch.clamp(lambdaB, min=1e-9)
        term2 = (k / 2) * (torch.log(lambdaA_safe) - torch.log(lambdaB_safe))

        z = 2 * torch.sqrt(lambdaA_safe * lambdaB_safe)
        # Use torch.abs(k) for the order of the Bessel function
        # Need to handle k potentially being a tensor
        log_bessel_term = log_bessel_i_approx(torch.abs(k), z)

        return term1 + term2 + log_bessel_term

    def log_likelihood(theta, teams_A, teams_B, goal_diff):
        log_lik = 0.0
        for i in range(len(goal_diff)):
            sA = theta[teams_A[i]].sum()
            sB = theta[teams_B[i]].sum()
            lambdaA = torch.exp(sA)
            lambdaB = torch.exp(sB)
            # Clamp lambdas to avoid numerical issues (exp can grow large)
            lambdaA = torch.clamp(lambdaA, max=1e6)
            lambdaB = torch.clamp(lambdaB, max=1e6)

            log_lik += log_skellam_approx(goal_diff[i], lambdaA, lambdaB)
        return log_lik

    def log_prior(theta):
        # Standard Normal prior N(0, 1) for each theta_j
        prior_dist = Normal(0, 1)
        return prior_dist.log_prob(theta).sum()

    def neg_log_posterior(theta, teams_A, teams_B, goal_diff):
        # Loss = - (log_likelihood + log_prior)
        return -(log_likelihood(theta, teams_A, teams_B, goal_diff) + log_prior(theta))

    # Gradient Descent Optimization
    def gradient_descent_optimization(loss_fn, initial_guess, lr, n_iter, teams_A, teams_B, goal_diff):
        theta = initial_guess.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=lr) # Adam often works well
        losses = []
        print("Starting MAP optimization...")
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = loss_fn(theta, teams_A, teams_B, goal_diff)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Iteration {i}: Loss is NaN or Inf. Stopping optimization.")
                # Try reducing LR or check calculations
                # For now, just stop and return the last valid theta
                return theta.detach(), losses # Return the last non-nan theta
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{n_iter}, Loss: {loss.item():.4f}")
        print("MAP optimization finished.")
        return theta.detach(), losses

    # Compute Hessian for Laplace Approximation
    def compute_hessian(f, w):
        print("Computing Hessian...")
        # Ensure w requires grad for hessian calculation
        w_req_grad = w.clone().detach().requires_grad_(True)
        hess = hessian(f, w_req_grad)
        print("Hessian computation finished.")
        return hess.detach().numpy()

    # Run MAP Optimization
    initial_theta = torch.zeros(num_players, dtype=torch.float)
    learning_rate = 0.01
    num_iterations = 1000 # Adjust as needed

    theta_MAP, map_losses = gradient_descent_optimization(
        neg_log_posterior,
        initial_theta,
        learning_rate,
        num_iterations,
        teams_A, teams_B, goal_diff
    )

    print(f"\nMAP Estimate (first 10 players): {theta_MAP[:10]}")

    # Plot MAP optimization loss
    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.title("MAP Optimization Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log Posterior")
    plt.grid(True)
    plt.savefig("/home/ubuntu/homework_solution/problem4_map_loss.png")
    print("MAP loss plot saved.")

    # Laplace Approximation
    # Need Hessian of the negative log posterior at the MAP estimate
    hessian_neg_log_post = compute_hessian(
        lambda t: neg_log_posterior(t, teams_A, teams_B, goal_diff),
        theta_MAP
    )

    # Covariance matrix is the inverse of the Hessian
    try:
        posterior_cov_laplace = np.linalg.inv(hessian_neg_log_post)
        # Check for positive definiteness (optional but good practice)
        try:
            np.linalg.cholesky(posterior_cov_laplace)
            print("Laplace covariance matrix is positive definite.")
        except np.linalg.LinAlgError:
            print("Warning: Laplace covariance matrix is not positive definite. Using pseudo-inverse.")
            posterior_cov_laplace = np.linalg.pinv(hessian_neg_log_post)

        print(f"Laplace Posterior Covariance matrix shape: {posterior_cov_laplace.shape}")
        # Convert to torch tensor for consistency
        posterior_cov_laplace_torch = torch.from_numpy(posterior_cov_laplace).float()

    except np.linalg.LinAlgError:
        print("Error: Hessian matrix is singular. Cannot compute inverse for Laplace approximation.")
        posterior_cov_laplace_torch = None # Indicate failure

    # Visualize marginals from Laplace Approximation
    if posterior_cov_laplace_torch is not None:
        try:
            mvn_laplace = MultivariateNormal(loc=theta_MAP, covariance_matrix=posterior_cov_laplace_torch)
            posterior_samples_laplace = mvn_laplace.sample((1000,))

            selected_indices = [0, 1, 2, 20] # Example indices
            plt.figure(figsize=(10, 6))
            for i, idx in enumerate(selected_indices):
                if idx < num_players:
                    plt.subplot(2, 2, i + 1)
                    plt.hist(posterior_samples_laplace[:, idx].numpy(), bins=40, density=True, alpha=0.7)
                    plt.title(f"Laplace Posterior of $\\theta_{{{idx+1}}}$") # Use 1-based index for title
                    plt.xlabel("Skill Value")
                    plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig("/home/ubuntu/homework_solution/problem4_laplace_marginals.png")
            print("Laplace marginal plots saved.")
        except Exception as e:
            print(f"Error during Laplace visualization: {e}")
            # This can happen if covariance is not positive definite

    # --- 2. Skellam Distribution Class ---
    # Using torch.distributions.Poisson difference for sampling
    # Using the approximation for log_prob to match MAP
    class SkellamApprox(Distribution):
        arg_constraints = {\"lambdaA\": constraints.positive, \"lambdaB\": constraints.positive}
        support = constraints.integer
        has_rsample = False # Sample method uses Poisson which might not be reparameterizable directly

        def __init__(self, lambdaA, lambdaB, validate_args=None):
            self.lambdaA = lambdaA
            self.lambdaB = lambdaB
            batch_shape = torch.broadcast_shapes(lambdaA.shape, lambdaB.shape)
            super().__init__(batch_shape, validate_args=validate_args)

        def sample(self, sample_shape=torch.Size()):
            shape = self._extended_shape(sample_shape)
            # Sample from two Poisson distributions and subtract
            poissonA = dist.Poisson(self.lambdaA.expand(shape))
            poissonB = dist.Poisson(self.lambdaB.expand(shape))
            sampleA = poissonA.sample()
            sampleB = poissonB.sample()
            return sampleA - sampleB

        def log_prob(self, value):
            # Use the same approximation as in MAP for consistency
            if self._validate_args:
                self._validate_sample(value)
            # Ensure value is float for calculations involving gradients
            value_float = value.float()
            return log_skellam_approx(value_float, self.lambdaA, self.lambdaB)

    # --- 3. Pyro Model ---
    def pyro_model(teams_A, teams_B, goal_diff, num_players):
        # Prior for player skills (theta)
        with pyro.plate(\"players\", num_players):
            theta = pyro.sample(\"theta\", dist.Normal(0., 1.))

        # Likelihood for each match
        log_lik_total = 0.0
        for i in pyro.plate(\"matches\", len(goal_diff)):
            sA = theta[teams_A[i]].sum()
            sB = theta[teams_B[i]].sum()
            lambdaA = torch.exp(sA)
            lambdaB = torch.exp(sB)
            # Clamp lambdas for stability
            lambdaA = torch.clamp(lambdaA, min=1e-9, max=1e6)
            lambdaB = torch.clamp(lambdaB, min=1e-9, max=1e6)

            # Observe goal difference using the custom Skellam distribution
            pyro.sample(f\"obs_{i}\", SkellamApprox(lambdaA, lambdaB), obs=goal_diff[i])

    # --- 4. MCMC Inference ---
    print("\nStarting MCMC inference...")
    nuts_kernel = NUTS(pyro_model)
    # num_samples is number of samples *after* warmup
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=1) # Use 1 chain for speed, ideally >1
    # Pass data to the model
    mcmc.run(teams_A, teams_B, goal_diff, num_players)
    print("MCMC inference finished.")

    # Get posterior samples
    mcmc_samples = mcmc.get_samples()
    theta_MCMC_samples = mcmc_samples[\"theta\"] # Shape: (num_samples, num_players)
    theta_MCMC_mean = theta_MCMC_samples.mean(dim=0)

    print(f"MCMC Posterior Mean (first 10 players): {theta_MCMC_mean[:10]}")
    mcmc.summary()

    # Visualize marginals from MCMC
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(selected_indices):
        if idx < num_players:
            plt.subplot(2, 2, i + 1)
            plt.hist(theta_MCMC_samples[:, idx].numpy(), bins=40, density=True, alpha=0.7)
            plt.title(f"MCMC Posterior of $\\theta_{{{idx+1}}}$") # Use 1-based index
            plt.xlabel("Skill Value")
            plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("/home/ubuntu/homework_solution/problem4_mcmc_marginals.png")
    print("MCMC marginal plots saved.")

    # --- 5. Comparison ---
    print("\n--- Comparison: MAP/Laplace vs MCMC ---")

    # Compare means
    print(f"MAP Estimate (mean): {theta_MAP[:10]}")
    print(f"MCMC Posterior Mean: {theta_MCMC_mean[:10]}")

    # Compare marginal variances (diagonal of covariance matrix)
    if posterior_cov_laplace_torch is not None:
        laplace_vars = torch.diag(posterior_cov_laplace_torch)
        print(f"Laplace Posterior Variance (first 10): {laplace_vars[:10]}")
    else:
        print("Laplace variance not available.")

    mcmc_vars = theta_MCMC_samples.var(dim=0)
    print(f"MCMC Posterior Variance (first 10): {mcmc_vars[:10]}")

    # Plot comparison of means
    plt.figure(figsize=(10, 5))
    player_indices = np.arange(num_players)
    plt.plot(player_indices, theta_MAP.numpy(), \"bo\", markersize=4, alpha=0.7, label=\"MAP Estimate\")
    plt.plot(player_indices, theta_MCMC_mean.numpy(), \"rx\", markersize=4, alpha=0.7, label=\"MCMC Mean\")
    plt.title(\"Comparison of Player Skill Estimates (MAP vs MCMC Mean)\")
    plt.xlabel(\"Player Index (0-based)\")
    plt.ylabel(\"Estimated Skill (theta)\")
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/homework_solution/problem4_map_vs_mcmc_mean.png")
    print("MAP vs MCMC mean comparison plot saved.")

    # Plot comparison of marginal std deviations
    if posterior_cov_laplace_torch is not None:
        laplace_stds = torch.sqrt(torch.diag(posterior_cov_laplace_torch))
        mcmc_stds = torch.sqrt(theta_MCMC_samples.var(dim=0))

        plt.figure(figsize=(10, 5))
        plt.plot(player_indices, laplace_stds.numpy(), \"bo\", markersize=4, alpha=0.7, label=\"Laplace Std Dev\")
        plt.plot(player_indices, mcmc_stds.numpy(), \"rx\", markersize=4, alpha=0.7, label=\"MCMC Std Dev\")
        plt.title(\"Comparison of Posterior Standard Deviations (Laplace vs MCMC)\")
        plt.xlabel(\"Player Index (0-based)\")
        plt.ylabel(\"Estimated Std Dev\")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0) # Std dev cannot be negative
        plt.savefig("/home/ubuntu/homework_solution/problem4_laplace_vs_mcmc_std.png")
        print("Laplace vs MCMC std dev comparison plot saved.")

    # Prediction function (as provided in notebook description, slightly adapted)
    def predict_goal_diff_skellam(teamA_ids, teamB_ids, theta_samples):
        # theta_samples can be MAP (single sample) or MCMC samples (multiple)
        if theta_samples.ndim == 1:
            theta_samples = theta_samples.unsqueeze(0) # Add batch dim for single sample

        n_sim = theta_samples.shape[0]
        predicted_diffs = []

        for i in range(n_sim):
            theta = theta_samples[i]
            # Adjust IDs to be 0-based if they aren\"t already
            teamA_ids_0based = torch.tensor([p - 1 for p in teamA_ids])
            teamB_ids_0based = torch.tensor([p - 1 for p in teamB_ids])

            sA = theta[teamA_ids_0based].sum()
            sB = theta[teamB_ids_0based].sum()
            lam_A = torch.exp(sA) # Expected goal rate A
            lam_B = torch.exp(sB) # Expected goal rate B

            # Mean of Skellam(lam_A, lam_B) is lam_A - lam_B
            mean_diff = lam_A - lam_B
            predicted_diffs.append(mean_diff.item())

        return np.mean(predicted_diffs), np.std(predicted_diffs)

    # Example prediction for a hypothetical match (Players 1, 2, 3 vs 4, 5, 6)
    # Use 1-based IDs as input to the function
    teamA_example = [1, 2, 3]
    teamB_example = [4, 5, 6]

    # Prediction using MAP estimate
    mean_diff_map, _ = predict_goal_diff_skellam(teamA_example, teamB_example, theta_MAP)
    print(f"\nExample Match ({teamA_example} vs {teamB_example}):")
    print(f"  Predicted Mean Goal Diff (MAP): {mean_diff_map:.2f}")

    # Prediction using MCMC samples (captures uncertainty)
    mean_diff_mcmc, std_diff_mcmc = predict_goal_diff_skellam(teamA_example, teamB_example, theta_MCMC_samples)
    print(f"  Predicted Mean Goal Diff (MCMC): {mean_diff_mcmc:.2f} +/- {std_diff_mcmc:.2f}")

else:
    print("Skipping Problem 4 due to data loading failure.")

print("\nProblem 4 Analysis Complete.")


