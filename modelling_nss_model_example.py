import numpy as np
from modelling.gaussian_process_model import GaussianProcessModel
import matplotlib.pyplot as plt

# Maturities (in years) and forward rates
maturities = np.array([
    0, 0.006219178, 0.014438356, 0.019917808, 0.02539726,
    0.033616438, 0.03909589, 0.044575342, 0.055534247, 0.058273973,
    0.063753425, 0.071972603, 0.077452055, 0.080191781, 0.082931507,
    0.102109589, 0.121287671, 0.15690411, 0.197890411, 0.236246575,
    0.291041096, 0.323917808, 0.408849315, 0.466383562, 0.485561644,
    0.71569863, 0.74309589, 0.965123288, 0.995260274, 1.041835616,
    1.214328767, 1.463643836, 1.962383562, 2.959643836
], dtype=np.float32)

forward_rates = np.array([
    2469.3845, 2469.5146, 2469.424727, 2469.165166, 2469.934033,
    2469.777096, 2469.329042, 2470.717691, 2469.57942, 2469.498867,
    2471.441107, 2470.19101, 2470.385109, 2471.549498, 2470.607102,
    2469.939569, 2470.990339, 2469.934152, 2469.93663, 2470.21689,
    2470.60913, 2473.286687, 2472.594037, 2474.449141, 2471.79222,
    2477.673476, 2479.040776, 2471.669135, 2452.578169, 2484.253973,
    2472.120148, 2482.352876, 2489.121562, 2491.752884
], dtype=np.float32)

# Initialize and fit the Gaussian Process model
model = GaussianProcessModel(length_scale=0.1, variance=1000.0, noise_variance=1e-2)
fitted_params = model.fit(maturities, forward_rates)
print("Fitted Parameters:", fitted_params)

# Predict forward rates
predicted_rates = model.predict(maturities)
print("Predicted Forward Rates:", predicted_rates.numpy())

# Plot observed vs. predicted rates
plt.figure(figsize=(10, 6))
plt.plot(maturities, forward_rates, 'o-', label="Observed Forward Rates", linewidth=2)
plt.plot(maturities, predicted_rates.numpy(), 'x--', label="Predicted Forward Rates", linewidth=2)
plt.title("Gaussian Process Model: Observed vs. Predicted Forward Rates")
plt.xlabel("Time to Maturity (Years)")
plt.ylabel("Forward Rate")
plt.legend()
plt.grid()
plt.show()