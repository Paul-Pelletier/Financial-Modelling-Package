import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Define the SVI implied volatility function
def svi_implied_vol(params, k, T):
    a, b, rho, m, sigma = params
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    return np.sqrt(w / T)

# Define the objective function to minimize
def objective_function(params, k, T, sigma_obs, weights):
    sigma_imp = svi_implied_vol(params, k, T)
    error = weights * (sigma_imp - sigma_obs)**2
    return np.sum(error)

# Store parameter evolution
param_history = []

# Verbose callback function for each iteration
def verbose_callback(xk, convergence):
    param_history.append(xk)  # Store current parameters
    print(f"Current parameters: {xk}")
    print(f"Current convergence: {convergence}")
    print("-" * 50)

# Bounds for SVI parameters: (a, b, rho, m, sigma)
bounds = [(0, 1),  # a: offset
          (0, 10), # b: slope
          (-1, 1), # rho: skewness
          (-1, 1), # m: log-moneyness of minimum
          (0.01, 1)] # sigma: curvature

# Example data: Increase the number of points
k = np.linspace(-0.5, 0.5, 15)  # log-moneyness values
T = 10.0  # time to maturity

# Hypothetical observed implied volatilities
sigma_obs = np.array([0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25, 0.30])
weights = np.ones_like(sigma_obs)  # equal weights for simplicity

# Optimize using Differential Evolution with verbosity
result = differential_evolution(
    objective_function, 
    bounds, 
    args=(k, T, sigma_obs, weights), 
    strategy='best1bin', 
    maxiter=1000, 
    callback=verbose_callback, 
    disp=True  # Enables detailed output during optimization
)

# Print the optimal parameters
print("Optimal SVI Parameters:", result.x)

# Function to plot training data and predictions
def plot_svi_fit_and_params(k, sigma_obs, params, T, param_history):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot observed and predicted implied volatilities
    k_fine = np.linspace(k.min() - 0.1, k.max() + 0.1, 100)
    sigma_pred = svi_implied_vol(params, k_fine, T)
    axes[0].scatter(k, sigma_obs, color='blue', label='Observed Implied Volatility', zorder=5)
    axes[0].plot(k_fine, sigma_pred, color='red', linestyle='--', label='Fitted SVI Implied Volatility', zorder=4)
    axes[0].set_title("SVI Fit to Implied Volatility Data")
    axes[0].set_xlabel("Log-Moneyness (k)")
    axes[0].set_ylabel("Implied Volatility")
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    # Plot parameter evolution
    param_history = np.array(param_history)
    param_labels = ['a', 'b', 'rho', 'm', 'sigma']
    for i, label in enumerate(param_labels):
        axes[1].plot(param_history[:, i], label=f'{label}', marker='o')
    axes[1].set_title("Parameter Evolution During Optimization")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Parameter Value")
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    plt.tight_layout()
    plt.show()

# Plot the results
plot_svi_fit_and_params(k, sigma_obs, result.x, T, param_history)
