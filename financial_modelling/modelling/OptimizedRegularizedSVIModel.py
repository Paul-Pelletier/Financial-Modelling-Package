import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class OptimizedRegularizedSVIModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # Initialize Nelson-Siegel parameters for SVI parameters (a, b, rho, m, sigma)
        self.beta_a = nn.Parameter(torch.tensor([0.05, 0.1, 0.01], dtype=torch.float32, device=self.device))
        self.beta_b = nn.Parameter(torch.tensor([0.2, -0.05, 0.01], dtype=torch.float32, device=self.device))
        self.beta_rho = nn.Parameter(torch.tensor([0.0, 0.01, -0.01], dtype=torch.float32, device=self.device))
        self.beta_m = nn.Parameter(torch.tensor([0.0, 0.05, 0.01], dtype=torch.float32, device=self.device))
        self.beta_sigma = nn.Parameter(torch.tensor([0.1, 0.02, -0.01], dtype=torch.float32, device=self.device))

        # Initialize lambda for Nelson-Siegel term structure
        self.lambda_a = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))
        self.lambda_b = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))
        self.lambda_rho = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))
        self.lambda_m = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))
        self.lambda_sigma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))

    def nelson_siegel(self, maturity, beta, lambd):
        """Nelson-Siegel model for parameterizing term structure."""
        decay = torch.exp(-lambd * maturity)
        return beta[0] + beta[1] * (1 - decay) / (lambd * maturity + 1e-6) + beta[2] * ((1 - decay) / (lambd * maturity + 1e-6) - decay)

    def compute_svi_params(self, maturities):
        """Compute SVI parameters using the Nelson-Siegel model."""
        a = self.nelson_siegel(maturities, self.beta_a, self.lambda_a)
        b = self.nelson_siegel(maturities, self.beta_b, self.lambda_b)
        rho = self.nelson_siegel(maturities, self.beta_rho, self.lambda_rho)
        m = self.nelson_siegel(maturities, self.beta_m, self.lambda_m)
        sigma = self.nelson_siegel(maturities, self.beta_sigma, self.lambda_sigma)
        return torch.stack([a, b, rho, m, sigma], dim=1)

    def forward(self, log_moneyness, params):
        """Compute total variance using the SVI formula."""
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

    def fit(self, log_moneyness, total_variance, residual_maturity, lr=1e-3, epochs=1000, lambda_decay=5.0):
        """
        Fit the SVI model to the given data, using Nelson-Siegel parameterization.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Convert inputs to tensors
        log_moneyness = torch.tensor(log_moneyness, dtype=torch.float32, device=self.device)
        total_variance = torch.tensor(total_variance, dtype=torch.float32, device=self.device)
        residual_maturity = torch.tensor(residual_maturity, dtype=torch.float32, device=self.device)

        # Exponential decay weights
        decay_weights = torch.exp(-lambda_decay * residual_maturity)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute Nelson-Siegel parameterized SVI parameters
            params = self.compute_svi_params(residual_maturity)

            # Broadcast log_moneyness for all maturities
            log_moneyness_broadcasted = log_moneyness[:, None].repeat(1, len(residual_maturity))

            # Compute model variance
            model_variance = self.forward(log_moneyness_broadcasted, params)

            # Variance loss with exponential decay
            variance_loss = torch.sum(decay_weights[:, None] * (model_variance - total_variance[:, None]) ** 2)

            # Total loss
            total_loss = variance_loss

            # Backward and optimization step
            total_loss.backward()
            optimizer.step()

            # Debugging output
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Variance Loss: {variance_loss.item()}, Total Loss: {total_loss.item()}")

        # Convert parameters to a dictionary for plotting
        params = self.compute_svi_params(residual_maturity).detach().cpu().numpy()
        param_dict = {
            float(maturity): {key: value for key, value in zip(['a', 'b', 'rho', 'm', 'sigma'], params[i])}
            for i, maturity in enumerate(residual_maturity.cpu().numpy())
        }

        return param_dict
