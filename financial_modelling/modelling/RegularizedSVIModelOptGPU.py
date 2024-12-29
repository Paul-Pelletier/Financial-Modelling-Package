import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.cuda.amp import autocast, GradScaler

class RegularizedSVIModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Initialize global parameters on the GPU
        self.global_params = nn.Parameter(torch.tensor([0.05, 0.2, 0.0, 0.0, 0.1], dtype=torch.float32, device=self.device))

    def forward(self, log_moneyness, params):
        """
        Vectorized forward pass to compute total variance for all maturities at once.
        """
        a, b, rho, m, sigma = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
        term1 = rho[:, None] * (log_moneyness[None, :] - m[:, None])
        term2 = torch.sqrt((log_moneyness[None, :] - m[:, None]) ** 2 + sigma[:, None] ** 2)
        return a[:, None] + b[:, None] * (term1 + term2)

    def fit(self, log_moneyness, total_variance, residual_maturity, quote_unixtime, expire_date,
        lr=1e-3, epochs=1000, regularization_strength=1e-3, lambda_decay=50, min_atm_volatility=0.1):
        """
        Optimized fitting method using vectorized computations and batching.

        Args:
            log_moneyness (torch.Tensor): Log-moneyness values.
            total_variance (torch.Tensor): Observed total variance.
            residual_maturity (torch.Tensor): Residual maturities.
            quote_unixtime (torch.Tensor): Unix timestamps of quotes.
            expire_date (torch.Tensor): Expiry dates.
            lr (float): Learning rate for optimization.
            epochs (int): Number of training epochs.
            regularization_strength (float): Regularization coefficient.
            lambda_decay (float): Decay factor for regularization.
            min_atm_volatility (float): Minimum ATM volatility.

        Returns:
            dict: Fitted parameters per residual maturity.
        """
        # Unique maturities on CPU for indexing
        unique_maturities = np.unique(residual_maturity.detach().cpu().numpy())

        # Initialize parameters for each maturity
        params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([0.05, 0.2, 0.0, 0.0, 0.1], dtype=torch.float32, device=self.device, requires_grad=True))
            for _ in unique_maturities
        ])

        optimizer = optim.Adam(params, lr=lr)
        scaler = GradScaler()  # Gradient scaler for precision control

        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0

            for i, maturity in enumerate(unique_maturities):
                a, b, rho, m, sigma = params[i]
                mask = residual_maturity == maturity

                log_m_subset = log_moneyness[mask]
                variance_subset = total_variance[mask]

                if log_m_subset.numel() == 0:
                    continue

                # Compute variance with the forward pass
                term1 = rho * (log_m_subset - m)
                term2 = torch.sqrt((log_m_subset - m) ** 2 + sigma ** 2)
                model_variance = a + b * (term1 + term2)

                # Loss computation
                loss = torch.mean((model_variance - variance_subset) ** 2)
                total_loss += loss

                # Regularization
                if i > 0:
                    prev_params = params[i - 1]
                    reg_penalty = torch.sum((params[i] - prev_params) ** 2)
                    total_loss += regularization_strength * reg_penalty

            # Backpropagate the loss
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                logging.info(f"Epoch {epoch}/{epochs}, Total Loss: {total_loss.item()}")

        # Generate parameter dictionary
        param_dict = {
            (quote_unixtime[mask][0].item(), expire_date[mask][0].item(), maturity): {
                "a": params[i][0].item(),
                "b": params[i][1].item(),
                "rho": params[i][2].item(),
                "m": params[i][3].item(),
                "sigma": params[i][4].item(),
            }
            for i, maturity in enumerate(unique_maturities)
        }

        logging.info(f"Final fitted parameters: {param_dict}")
        return param_dict
