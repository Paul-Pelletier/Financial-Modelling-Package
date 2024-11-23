import numpy as np
from modelling.NSS_model import ExtendedNSSModel
import matplotlib.pyplot as plt

# Time to maturity
maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 20], dtype=np.float32)

# Observed forward rates with more curvature (hump and dip)
forward_rates = np.array([0.012, 0.018, 0.028, 0.025, 0.032, 0.030, 0.029, 0.027], dtype=np.float32)

# Initialize the NSS model
nss = ExtendedNSSModel()

# Fit the model
fitted_params = nss.fit(maturities, forward_rates)
print("Fitted Parameters:", fitted_params)

# Predict forward rates
predicted_rates = nss.predict(maturities)
print("Predicted Forward Rates:", predicted_rates.numpy())

# Plot observed and predicted rates
plt.figure(figsize=(10, 6))
plt.plot(maturities, forward_rates, 'o-', label="Observed Forward Rates", linewidth=2)
plt.plot(maturities, predicted_rates.numpy(), 'x--', label="Predicted Forward Rates", linewidth=2)
plt.title("NSS Model: Observed vs. Predicted Forward Rates")
plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Forward Rate")
plt.legend()
plt.grid()
plt.show()
