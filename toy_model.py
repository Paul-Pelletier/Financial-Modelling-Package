import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the double exponential model
def double_exponential_model(x, a, b, d, e, c):
    return a * np.exp(b * x) + d * np.exp(e * x) + c

# Define the single exponential model for comparison
def single_exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

# Load the cleaned data
data = pd.read_csv("cleaned_data.csv", sep=";")

# Extract the independent and dependent variables
x = data["YTE"].values
y = data["Forward"].values

# Adjust scaling if necessary (optional)
if x.max() > 10 or x.max() < 1:
    x = x / x.max()  # Scale YTE values to make them numerically stable

# Fit the two-exponential terms model
initial_guess_2exp = [y.max(), -1, y.max()/2, -0.5, y.min()]
params_2exp, covariance_2exp = curve_fit(double_exponential_model, x, y, p0=initial_guess_2exp, maxfev=10000)

# Fit the single exponential model for comparison
initial_guess_1exp = [y.max(), -1, y.min()]
params_1exp, covariance_1exp = curve_fit(single_exponential_model, x, y, p0=initial_guess_1exp, maxfev=10000)

# Extract parameters
a, b, d, e, c = params_2exp
a1, b1, c1 = params_1exp

# Sensitivity Analysis
def sensitivity_analysis(x, y, params, model, param_names):
    """
    Performs sensitivity analysis by varying each parameter by ±10%.
    """
    sensitivities = {}
    baseline = np.mean((y - model(x, *params))**2)  # Baseline RMSE

    for i, param in enumerate(params):
        param_up = params.copy()
        param_down = params.copy()

        param_up[i] += 0.1 * param  # Increase by 10%
        param_down[i] -= 0.1 * param  # Decrease by 10%

        # Compute RMSE for perturbed parameters
        rmse_up = np.mean((y - model(x, *param_up))**2)
        rmse_down = np.mean((y - model(x, *param_down))**2)

        sensitivities[param_names[i]] = {
            "baseline": baseline,
            "rmse_up": rmse_up,
            "rmse_down": rmse_down,
            "sensitivity_up": (rmse_up - baseline) / baseline * 100,
            "sensitivity_down": (rmse_down - baseline) / baseline * 100,
        }

    return sensitivities

# Perform sensitivity analysis for the two-exponential terms model
param_names_2exp = ["a", "b", "d", "e", "c"]
sensitivity_2exp = sensitivity_analysis(x, y, params_2exp, double_exponential_model, param_names_2exp)

# Pertinence Assessment
def calculate_metrics(x, y, model, params):
    """
    Calculates R², RMSE, and AIC for a given model and parameters.
    """
    y_pred = model(x, *params)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)  # R²
    rmse = np.sqrt(np.mean(residuals**2))  # RMSE
    n = len(y)
    k = len(params)
    aic = n * np.log(ss_res / n) + 2 * k  # AIC
    return {"R²": r2, "RMSE": rmse, "AIC": aic}

# Calculate metrics for both models
metrics_2exp = calculate_metrics(x, y, double_exponential_model, params_2exp)
metrics_1exp = calculate_metrics(x, y, single_exponential_model, params_1exp)

# Print results
print("Two-Exponential Terms Model Parameters:")
print(f"  a = {a:.4f}, b = {b:.4f}, d = {d:.4f}, e = {e:.4f}, c = {c:.4f}")
print("\nSensitivity Analysis for Two-Exponential Terms Model:")
for param, sensitivity in sensitivity_2exp.items():
    print(f"  {param}: Sensitivity Up = {sensitivity['sensitivity_up']:.2f}%, Sensitivity Down = {sensitivity['sensitivity_down']:.2f}%")

print("\nModel Metrics:")
print("  Two-Exponential Terms Model:", metrics_2exp)
print("  Single-Exponential Term Model:", metrics_1exp)

# Plot the models
x_fit = np.linspace(min(x), max(x), 500)
y_fit_2exp = double_exponential_model(x_fit, *params_2exp)
y_fit_1exp = single_exponential_model(x_fit, *params_1exp)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Cleaned Data", color="blue", alpha=0.7)
plt.plot(x_fit, y_fit_2exp, label="Two-Exponential Terms Model", color="red")
plt.plot(x_fit, y_fit_1exp, label="Single-Exponential Term Model", color="green")
plt.xlabel("YTE")
plt.ylabel("calc")
plt.title("Comparison of Exponential Models")
plt.legend()
plt.grid()
plt.show()
