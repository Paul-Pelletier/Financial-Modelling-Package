import numpy as np
import matplotlib.pyplot as plt

# Import the SVI_model class (assuming it's saved in svi_model.py)
# from svi_model import SVI_model

from modelling.SVI_model import SVI_model

# Market Data
log_moneyness = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])  # log(K/F)
market_variances = np.array([0.06, 0.052, 0.05, 0.055, 0.06])  # Total implied variances (sigma^2 * T)

# Initialize the SVI model
svi = SVI_model()

# Step 1: Calibrate the Model
print("Calibrating the SVI model...")
svi.fit(log_moneyness, market_variances)
print("Calibrated Parameters:", svi.params)

# Step 2: Predict Variances
predicted_variances = svi.predict(log_moneyness)
print("Predicted Variances:", predicted_variances)

# Step 3: Predict Implied Volatilities
maturity = 1.0  # Time to maturity in years
new_moneyness = np.linspace(-0.2,0.2,100)
implied_vols = svi.implied_volatility(new_moneyness, maturity)
print("Implied Volatilities:", implied_vols)

# Step 4: Plot Market vs. Model
plt.figure(figsize=(8, 6))
plt.plot(log_moneyness, np.sqrt(market_variances / maturity), 'o', label="Market Implied Volatility", markersize=8)
plt.plot(new_moneyness, implied_vols, '-', label="SVI Model Fit", linewidth=2)
plt.xlabel("Log-Moneyness (log(K/F))")
plt.ylabel("Implied Volatility")
plt.title("SVI Calibration: Market vs Model")
plt.legend()
plt.grid()
plt.show()
