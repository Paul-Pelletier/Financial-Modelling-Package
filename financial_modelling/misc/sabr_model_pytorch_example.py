import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np


class SABRModelGPU:
    def __init__(self, alpha: float, beta: float, rho: float, nu: float, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Parameters as PyTorch tensors with gradients enabled
        self.alpha = torch.tensor([alpha], requires_grad=True, device=self.device)
        self.beta = torch.tensor([beta], requires_grad=True, device=self.device)
        self.rho = torch.tensor([rho], requires_grad=True, device=self.device)
        self.nu = torch.tensor([nu], requires_grad=True, device=self.device)

    def sabr_volatility(self, strikes, F, T):
        """Calculate SABR implied volatilities."""
        K = torch.tensor(strikes, device=self.device)
        F = torch.tensor(F, device=self.device)
        T = torch.tensor(T, device=self.device)
        
        epsilon = 1e-8  # Prevent division by zero
        beta_term = (F * K).pow((1 - self.beta) / 2)
        log_moneyness = torch.log(F / K + epsilon)
        z = (self.nu / self.alpha) * beta_term * log_moneyness
        x_z = torch.log((torch.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))
        atm_vol = self.alpha * beta_term / (1 + epsilon)
        vol = atm_vol * z / (x_z + epsilon)
        return vol

    def loss_fn(self, predicted_vols, market_vols):
        """Mean Squared Error loss."""
        return torch.mean((predicted_vols - market_vols) ** 2)

    def fit(self, strikes, market_vols, F, T, learning_rate=0.1, epochs=100):
        """Calibrate the SABR model parameters."""
        optimizer = optim.Adam([self.alpha, self.beta, self.rho, self.nu], lr=learning_rate)
        strikes_tensor = torch.tensor(strikes, dtype=torch.float32, device=self.device)
        market_vols_tensor = torch.tensor(market_vols, dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted_vols = self.sabr_volatility(strikes_tensor, F, T)
            loss = self.loss_fn(predicted_vols, market_vols_tensor)
            loss.backward()
            optimizer.step()

            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
        
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "rho": self.rho.item(),
            "nu": self.nu.item(),
        }

    def predict(self, strikes, F, T):
        """Predict volatilities using calibrated parameters."""
        strikes_tensor = torch.tensor(strikes, dtype=torch.float32, device=self.device)
        return self.sabr_volatility(strikes_tensor, F, T).detach().cpu()


# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch is using CUDA: {torch.cuda.is_available()}")
    # Example data
    F = 100.0  # Forward price
    T = 1.0    # Time to maturity
    strikes = np.linspace(80, 120, 10).astype(np.float32)  # Strike prices
    market_vols = np.array([0.25, 0.24, 0.23, 0.22, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], dtype=np.float32)

    # Initialize SABR model on GPU
    sabr_gpu = SABRModelGPU(alpha=0.04, beta=0.5, rho=-0.2, nu=0.3)

    # Calibrate the SABR model
    print("Calibrating SABR model...")
    calibrated_params = sabr_gpu.fit(strikes, market_vols, F, T, learning_rate=0.01, epochs=500)
    print("Calibrated Parameters:", calibrated_params)

    # Predict implied volatilities
    print("Predicting implied volatilities...")
    predicted_vols = sabr_gpu.predict(strikes, F, T).numpy()

    # Plot market vs. predicted volatilities
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, market_vols, 'o', label="Market Volatilities")
    plt.plot(strikes, predicted_vols, 'x-', label="SABR Model Volatilities")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid()
    plt.show()
