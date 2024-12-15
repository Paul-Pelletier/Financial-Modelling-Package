import numpy as np
import matplotlib.pyplot as plt

import os
import sys
#Allows for importing neighbouring packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelling.SVI_model import SVI_model

# Assuming SVI_model is already defined and imported
# from svi_model import SVI_model

# Generate realistic synthetic market data
def generate_market_data():
    # Log-moneyness values
    log_moneyness = np.linspace(-0.4, 0.4, 20)  # 20 points from deep ITM to deep OTM
    
    # True volatility smile
    def true_volatility(log_m):
        atm_vol = 0.2  # ATM implied volatility (20%)
        skew = -0.1    # Skewness
        curvature = 0.4  # Smile curvature
        return atm_vol + skew * log_m + curvature * log_m**2

    # Generate market volatilities with noise
    np.random.seed(42)  # For reproducibility
    market_vols = true_volatility(log_moneyness) + np.random.normal(0, 0.005, size=log_moneyness.shape)

    # Convert to total implied variances
    maturity = 1.0  # Time to maturity (1 year)
    market_variances = market_vols**2 * maturity

    return log_moneyness, market_variances, market_vols

# Generate synthetic market data
log_moneyness, market_variances, market_vols = generate_market_data()

# Initialize and calibrate the SVI model
svi = SVI_model()
svi.fit(log_moneyness, market_variances)

# Predict implied variances using the calibrated model
predicted_variances = svi.predict(log_moneyness)

# Convert to implied volatilities
maturity = 1.0  # Time to maturity (1 year)
predicted_vols = np.sqrt(predicted_variances / maturity)

# Plot market data vs. SVI fit
plt.figure(figsize=(10, 6))
plt.plot(log_moneyness, market_vols, 'o', label="Market Implied Volatility", markersize=8)
plt.plot(log_moneyness, predicted_vols, '-', label="SVI Model Fit", linewidth=2)
plt.xlabel("Log-Moneyness (log(K/F))")
plt.ylabel("Implied Volatility")
plt.title("SVI Calibration: Market vs Model")
plt.legend()
plt.grid()
plt.show()

# Print the calibrated parameters
print("Calibrated Parameters:", svi.params)
