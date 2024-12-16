# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelling.SABR_model import SABRModelGPU  # Save the class above as SABR_model_gpu.py

# Example data
F = 100.0  # Forward price
T = 1.0    # Time to maturity
strikes = np.linspace(80, 120, 10).astype(np.float32)  # Strike prices
market_vols = np.array([0.25, 0.24, 0.23, 0.22, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], dtype=np.float32)

# Initialize SABR model
sabr_gpu = SABRModelGPU(alpha=0.04, beta=0.5, rho=-0.2, nu=0.3)

# Calibrate the SABR model
calibrated_params = sabr_gpu.fit(strikes, market_vols, F, T, learning_rate=0.2, epochs=100)
print("Calibrated Parameters:", calibrated_params)

# Predict implied volatilities
predicted_vols = sabr_gpu.predict(strikes, F, T).numpy()

# Plot market vs. predicted volatilities
plt.figure(figsize=(10, 6))
plt.plot(strikes, market_vols, 'o', label="Market Volatilities")
plt.plot(strikes, predicted_vols, 'x-', label="SABR Model Volatilities")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.legend()
plt.grid()
plt.show()
