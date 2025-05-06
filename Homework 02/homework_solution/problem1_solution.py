# Problem 1 Solution
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy.stats import multivariate_normal

# --- Data Loading Placeholder ---
# NOTE: The Szeged weather dataset is required here.
# Please ensure 'weatherHistory.csv' is in the '/home/ubuntu/homework_solution/data/' directory.
# If you have downloaded it elsewhere, please adjust the path.
try:
    weather_data = pd.read_csv('/home/ubuntu/homework_solution/data/weatherHistory.csv')
    # Preprocessing
    weather_data = weather_data.dropna(subset=['Apparent Temperature (C)', 'Humidity'])
    X = weather_data[['Humidity']].values
    y = weather_data['Apparent Temperature (C)'].values
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add bias term
    N, D = X.shape
    data_loaded = True
except FileNotFoundError:
    print("Weather data file not found. Skipping Problem 1 execution.")
    print("Please download 'weatherHistory.csv' from https://www.kaggle.com/datasets/budincsevity/szeged-weather and place it in /home/ubuntu/homework_solution/data/")
    # Use dummy data for function definition checks if needed, but optimization won't run meaningfully
    N, D = 100, 2
    X = np.random.rand(N, D)
    y = np.random.rand(N)
    data_loaded = False

# Define the negative log marginal likelihood function
def neg_log_marginal_likelihood(params, X, y, N, D):
    alpha, beta = np.exp(params) # Ensure alpha and beta are positive
    if alpha <= 0 or beta <= 0:
        return np.inf # Return infinity for invalid parameters

    try:
        # Calculate covariance matrix Sigma_N
        Sigma_N = (1/beta) * np.eye(N) + (1/alpha) * (X @ X.T)
        # Calculate the log determinant and inverse (or solve)
        sign, logdet = np.linalg.slogdet(Sigma_N)
        if sign <= 0:
             # If determinant is not positive, return infinity
             return np.inf
        log_det_Sigma_N = logdet

        # Solve Sigma_N * inv_Sigma_N_y = y for inv_Sigma_N_y
        inv_Sigma_N_y = np.linalg.solve(Sigma_N, y)

        # Calculate the log marginal likelihood components
        term1 = -0.5 * log_det_Sigma_N
        term2 = -0.5 * (y.T @ inv_Sigma_N_y)
        term3 = - (N / 2) * np.log(2 * np.pi)

        log_ml = term1 + term2 + term3

        # Return the negative log marginal likelihood
        return -log_ml
    except np.linalg.LinAlgError:
        # Handle cases where Sigma_N is singular
        return np.inf
    except ValueError:
        # Handle potential numerical issues
        return np.inf

# Initial guess for log(alpha) and log(beta)
initial_params = np.log([1.0, 1.0])

optimal_alpha = None
optimal_beta = None

if data_loaded:
    # Optimize the negative log marginal likelihood
    # Using L-BFGS-B which handles bounds implicitly via optimization process
    # We optimize log(alpha) and log(beta) to avoid explicit constraints
    result = optimize.minimize(
        neg_log_marginal_likelihood,
        initial_params,
        args=(X, y, N, D),
        method='L-BFGS-B',
        options={'disp': True}
    )

    if result.success:
        optimal_log_alpha, optimal_log_beta = result.x
        optimal_alpha = np.exp(optimal_log_alpha)
        optimal_beta = np.exp(optimal_log_beta)
        print(f"Optimization successful.")
        print(f"Optimal alpha: {optimal_alpha}")
        print(f"Optimal beta: {optimal_beta}")
        print(f"Minimum negative log marginal likelihood: {result.fun}")
    else:
        print(f"Optimization failed: {result.message}")
else:
    print("Skipping optimization due to missing data.")

# Final optimal values (to be displayed)
print(f"\nFinal Optimal Alpha: {optimal_alpha}")
print(f"Final Optimal Beta: {optimal_beta}")

