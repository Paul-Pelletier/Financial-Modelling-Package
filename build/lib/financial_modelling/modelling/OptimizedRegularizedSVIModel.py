import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class OptimizedRegularizedSVIModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Initialize parameters with torch tensors on the device
        self.a = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, device=self.device))
        self.b = nn.Parameter(torch.tensor(0.2, dtype=torch.float32, device=self.device))
        self.rho = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.m = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.sigma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))

    def forward(self, log_moneyness, params):
        """
        Compute total variance using the SVI formula.
        """
        # Unpack parameters
        a, b, rho, m, sigma = params.T

        # Clamp parameters to enforce constraints
        b = torch.clamp(b, min=1e-8)
        sigma = torch.clamp(sigma, min=1e-8)
        rho = torch.clamp(rho, min=-1.0, max=1.0)

        # Compute SVI formula
        log_moneyness_diff = log_moneyness - m
        term1 = rho * log_moneyness_diff
        term2 = torch.sqrt(log_moneyness_diff ** 2 + sigma ** 2)
        return a + b * (term1 + term2)

    def fit(self, log_moneyness, total_variance, residual_maturity, lr=1e-3, epochs=1000,
            regularization_strength=1e-3, lambda_decay=5.0):
        """
        Fit the SVI model to the given data, emphasizing short-term maturities with exponential decay.
        """
        # Unique maturities
        unique_maturities = np.unique(residual_maturity)
        n_maturities = len(unique_maturities)

        # Initialize parameters for all maturities
        params = torch.nn.Parameter(torch.rand((n_maturities, 5), dtype=torch.float32, device=self.device, requires_grad=True))
        optimizer = optim.Adam([params], lr=lr)

        # Convert inputs to tensors
        log_moneyness = torch.tensor(log_moneyness, dtype=torch.float32, device=self.device)
        total_variance = torch.tensor(total_variance, dtype=torch.float32, device=self.device)
        residual_maturity = torch.tensor(residual_maturity, dtype=torch.float32, device=self.device)

        # Exponential decay weights
        decay_weights = torch.exp(-lambda_decay * residual_maturity)

        # Mask creation with tolerance for floating-point alignment
        tolerance = 1e-4
        masks = torch.abs(residual_maturity[:, None] - torch.tensor(unique_maturities, device=self.device)) < tolerance
        masks = masks.float()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Broadcast log_moneyness for all maturities
            log_moneyness_broadcasted = log_moneyness[:, None].repeat(1, n_maturities)

            # Compute model variance
            model_variance = self.forward(log_moneyness_broadcasted, params)

            # Variance loss with exponential decay
            variance_loss = torch.sum(decay_weights[:, None] * masks * (model_variance - total_variance[:, None]) ** 2)

            # Regularization loss
            reg_loss = torch.sum((params[1:] - params[:-1]) ** 2)

            # Total loss
            total_loss = variance_loss + regularization_strength * reg_loss

            # Backward and optimization step
            total_loss.backward()

            # Apply parameter clamping after gradient update
            with torch.no_grad():
                params[:, 1] = torch.clamp(params[:, 1], min=1e-6)  # Clamp b
                params[:, 4] = torch.clamp(params[:, 4], min=1e-6)  # Clamp sigma
                params[:, 2] = torch.clamp(params[:, 2], min=-1.0, max=1.0)  # Clamp rho

            optimizer.step()

            # Debugging output
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Variance Loss: {variance_loss.item()}, "
                      f"Reg Loss: {reg_loss.item()}, Total Loss: {total_loss.item()}")

        # Convert parameters to a dictionary
        param_dict = {
            float(maturity): {key: value.item() for key, value in zip(['a', 'b', 'rho', 'm', 'sigma'], params[i])}
            for i, maturity in enumerate(unique_maturities)
        }

        return param_dict
