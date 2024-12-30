import numpy as np
from scipy.optimize import minimize

class SVIModel:
    def __init__(self, initial_params=None):
        """
        Initialize the SVI model with optional initial parameters.
        """
        self.params = None  # Model parameters: a, b, rho, m, sigma
        if initial_params is None:
            self.initial_guess = [0.05, 0.2, 0.0, 0.0, 0.1]
        else:
            self.initial_guess = initial_params

    @staticmethod
    def svi(log_moneyness, a, b, rho, m, sigma):
        """
        SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        """
        log_moneyness = np.array(log_moneyness)
        term1 = rho * (log_moneyness - m*np.ones(len(log_moneyness)))
        term2 = np.sqrt((log_moneyness - m) ** 2 + sigma ** 2)
        return a + b * (term1 + term2)

    def fit(self, train_data, volume_weights=None, verbose=True):
        """
        Fit SVI parameters to training data.

        Parameters:
        - train_data: DataFrame with ['Log_Moneyness', 'Implied_Volatility']
        - volume_weights: Optional array-like weights for each data point
        - verbose: Print intermediate results during optimization
        """
        log_moneyness = train_data['Log_Moneyness'].values
        implied_volatility = train_data['Implied_Volatility'].values

        if volume_weights is None:
            volume_weights = np.ones_like(implied_volatility)

        def objective(params):
            """Objective function with constraints and weights"""
            a, b, rho, m, sigma = params
            if not (0 < b and -1 <= rho <= 1 and sigma > 0):
                return np.inf  # Invalid parameter regions
            
            model_variance = self.svi(log_moneyness, a, b, rho, m, sigma)
            model_volatility = np.sqrt(np.maximum(model_variance, 1e-8))  # Avoid negative sqrt
            errors = (model_volatility - implied_volatility) ** 2
            weighted_errors = errors * volume_weights
            return np.sum(weighted_errors)

        # Parameter bounds
        bounds = [
            (1e-6, None),  # a > 0
            (1e-6, None),  # b > 0
            (-0.99, 0.99),  # -1 <= rho <= 1
            (-1.0, 1.0),    # m bounded
            (1e-6, None)    # sigma > 0
        ]

        # Optimize
        result = minimize(objective, self.initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            self.params = result.x
            if verbose:
                print("SVI Calibration Successful!")
                print("Parameters:", dict(zip(["a", "b", "rho", "m", "sigma"], self.params)))
        else:
            raise ValueError(f"SVI fit failed: {result.message}")

    def predict(self, log_moneyness):
        """
        Predict implied volatilities using the fitted SVI model.
        """
        if self.params is None:
            raise ValueError("Model parameters are not fitted yet.")

        a, b, rho, m, sigma = self.params
        model_variance = self.svi(log_moneyness, a, b, rho, m, sigma)
        return np.sqrt(np.maximum(model_variance, 1e-8))  # Safe square root

    def get_params(self):
        """Return the fitted parameters."""
        if self.params is None:
            raise ValueError("Model parameters are not fitted yet.")
        return dict(zip(["a", "b", "rho", "m", "sigma"], self.params))
