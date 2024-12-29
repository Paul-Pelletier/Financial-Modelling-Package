import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler


class RegularizedSVIModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.params = None  # Placeholder for all-maturities parameters

    def forward(self, log_moneyness, params):
        """
        Compute the total variance using SVI formula with constraints for all maturities at once.

        Args:
        - log_moneyness (torch.Tensor): Log-moneyness values (batch_size, max_num_points).
        - params (torch.Tensor): SVI parameters [a, b, rho, m, sigma] (batch_size, 5).

        Returns:
        - torch.Tensor: Total variance (batch_size, max_num_points).
        """
        a, b, rho, m, sigma = params.unbind(dim=1)

        # Enforce constraints
        b = torch.clamp(b, min=1e-6)  # b > 0
        rho = torch.clamp(rho, min=-1 + 1e-6, max=1 - 1e-6)  # -1 < rho < 1
        sigma = torch.clamp(sigma, min=1e-6)  # sigma > 0

        term1 = rho.unsqueeze(1) * (log_moneyness - m.unsqueeze(1))
        term2 = torch.sqrt((log_moneyness - m.unsqueeze(1)) ** 2 + sigma.unsqueeze(1) ** 2)
        return a.unsqueeze(1) + b.unsqueeze(1) * (term1 + term2)

    def fit(self, log_moneyness, total_variance, residual_maturity, quote_unixtime, expire_date,
            lr=1e-3, epochs=1000, regularization_strength=1e-3, lambda_decay=50, min_atm_volatility=0.1):
        """
        Fit the model to the data in a batched manner for all maturities using mixed precision.

        Args:
        - log_moneyness (torch.Tensor): Log-moneyness values (num_points,).
        - total_variance (torch.Tensor): Observed total variance (num_points,).
        - residual_maturity (torch.Tensor): Residual maturities (num_points,).
        - quote_unixtime (torch.Tensor): Unix timestamps of quotes (num_points,).
        - expire_date (torch.Tensor): Expiry dates (num_points,).
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - regularization_strength (float): Regularization strength for parameter smoothness.
        - lambda_decay (float): Decay factor for weighting loss by maturity.
        - min_atm_volatility (float): Minimum ATM volatility.

        Returns:
        - dict: Fitted parameters keyed by (QUOTE_UNIXTIME, EXPIRE_DATE, MATURITY).
        """
        # Get unique maturities and indices for batching
        unique_maturities, batch_indices = torch.unique(residual_maturity, return_inverse=True)

        # Create per-maturity parameters
        self.params = nn.Parameter(torch.tensor(
            [[0.05, 0.2, 0.0, 0.0, 0.1] for _ in unique_maturities],
            device=self.device,
            requires_grad=True
        ))

        optimizer = optim.Adam([self.params], lr=lr)
        scaler = GradScaler()  # Gradient scaler for mixed precision

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Group log-moneyness and total variance by maturity
            grouped_log_moneyness = [log_moneyness[batch_indices == i] for i in range(len(unique_maturities))]
            grouped_total_variance = [total_variance[batch_indices == i] for i in range(len(unique_maturities))]

            # Pad sequences to ensure uniform batch dimensions
            padded_log_moneyness = pad_sequence(grouped_log_moneyness, batch_first=True).to(self.device)
            padded_total_variance = pad_sequence(grouped_total_variance, batch_first=True).to(self.device)

            # Mask to ignore padded values during loss computation
            lengths = torch.tensor([len(seq) for seq in grouped_log_moneyness], device=self.device)
            mask = torch.arange(padded_log_moneyness.size(1), device=self.device).unsqueeze(0) < lengths.unsqueeze(1)

            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=torch.float16):
                model_variance = self.forward(padded_log_moneyness, self.params)

                # Compute loss
                loss = torch.tensor(0.0, device=self.device)
                for i, maturity in enumerate(unique_maturities):
                    decay_weight = torch.exp(-lambda_decay * maturity)

                    # Apply mask to exclude padded values
                    valid_model_variance = model_variance[i][mask[i]]
                    valid_total_variance = padded_total_variance[i][mask[i]]

                    loss += decay_weight * torch.mean((valid_model_variance - valid_total_variance) ** 2)

                    # Add regularization for parameter smoothness
                    if i > 0:
                        reg_penalty = torch.sum((self.params[i] - self.params[i - 1]) ** 2)
                        loss += regularization_strength * reg_penalty

                    # Add penalty for short-term maturities
                    if maturity < 0.05:
                        atm_variance = self.params[i][0] + self.params[i][1] * self.params[i][4]
                        short_term_penalty = torch.clamp(min_atm_volatility**2 - atm_variance, min=0)
                        loss += 1e8 * short_term_penalty

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch % 100 == 0:
                logging.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

        # Generate parameter dictionary
        param_dict = {}
        for i, maturity in enumerate(unique_maturities):
            mask = residual_maturity == maturity
            unique_quote_unixtime = quote_unixtime[mask][0].item()
            unique_expire_date = expire_date[mask][0].item()

            param_dict[(unique_quote_unixtime, unique_expire_date, maturity.item())] = {
                key: value.item()
                for key, value in zip(['a', 'b', 'rho', 'm', 'sigma'], self.params[i])
            }

        logging.info(f"Final fitted parameters: {param_dict}")
        return param_dict
