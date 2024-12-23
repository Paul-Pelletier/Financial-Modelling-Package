import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegularizedSVIModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Initialize parameters with torch tensors on the device
        self.a = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, device=self.device))
        self.b = nn.Parameter(torch.tensor(0.2, dtype=torch.float32, device=self.device))
        self.rho = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.m = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.sigma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))

    def forward(self, log_moneyness):
        term1 = self.rho * (log_moneyness - self.m)
        term2 = torch.sqrt((log_moneyness - self.m) ** 2 + self.sigma ** 2)
        return self.a + self.b * (term1 + term2)

    def fit(self, log_moneyness, total_variance, residual_maturity, lr=1e-3, epochs=1000, regularization_strength=1e-3, lambda_decay=50, min_atm_volatility=0.1):
        unique_maturities = np.unique(residual_maturity)
        param_dict = {}

        params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([0.05, 0.2, 0.0, 0.0, 0.1], dtype=torch.float32, device=self.device, requires_grad=True))
            for _ in unique_maturities
        ])

        optimizer = optim.Adam(params, lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0

            for i, maturity in enumerate(unique_maturities):
                mask = residual_maturity == maturity
                log_moneyness_subset = torch.tensor(log_moneyness[mask], dtype=torch.float32, device=self.device)
                total_variance_subset = torch.tensor(total_variance[mask], dtype=torch.float32, device=self.device)

                a, b, rho, m, sigma = params[i]
                term1 = rho * (log_moneyness_subset - m)
                term2 = torch.sqrt((log_moneyness_subset - m) ** 2 + sigma ** 2)
                model_variance = a + b * (term1 + term2)

                decay_weight = torch.exp(-lambda_decay * torch.tensor(maturity, dtype=torch.float32, device=self.device))
                loss = decay_weight * torch.mean((model_variance - total_variance_subset) ** 2)
                total_loss += loss

                if i > 0:
                    prev_params = params[i - 1]
                    reg_penalty = torch.sum((params[i] - prev_params) ** 2)
                    total_loss += regularization_strength * reg_penalty

                if maturity < 0.05:
                    atm_variance = a + b * sigma
                    short_term_penalty = torch.clamp(min_atm_volatility**2 - atm_variance, min=0)
                    total_loss += 1e8 * short_term_penalty

            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Total Loss: {total_loss.item()}")

        for i, maturity in enumerate(unique_maturities):
            param_dict[np.float32(maturity)] = {key: value.item() for key, value in zip(['a', 'b', 'rho', 'm', 'sigma'], params[i])}

        return param_dict
