import torch
import torch.nn as nn

class VolatilityNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=200):
        """
        Neural Network for Implied Volatility Surface Fitting
        """
        super(VolatilityNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
            )


    def forward(self, x):
        return self.model(x)
