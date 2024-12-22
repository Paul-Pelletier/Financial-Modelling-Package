import numpy as np
from scipy.optimize import minimize

class GlobalSVIFit:
    def __init__(self, initial_params=None):
        """
        Initialize the global SVI fit model with optional initial parameters.
        """
        self.global_params = None  # Global parameters: a, b, rho, m, sigma
        if initial_params is None:
            self.initial_guess = [0.05, 0.2, 0.0, 0.0, 0.1]
        else:
            self.initial_guess = initial_params

    @staticmethod
    def svi_total_variance(log_moneyness, T, a, b, rho, m, sigma):
        """
        SVI formula for total variance w(k, T).
        """
        term1 = rho * (log_moneyness - m)
        term2 = np.sqrt((log_moneyness - m) ** 2 + sigma ** 2)
        return a + b * (term1 + term2)

    def fit(self, train_data, verbose=True):
        """
        Fit the global SVI parameters to training data.

        Parameters:
        - train_data: DataFrame with ['Log_Moneyness', 'Implied_Volatility', 'Maturity']
        - verbose: Print intermediate results during optimization
        """
        log_moneyness = train_data["Log_Moneyness"].values
        implied_volatility = train_data["Implied_Volatility"].values
        maturity = train_data["Maturity"].values

        # Total variance: w = T * (implied_volatility)^2
        total_variance = maturity * (implied_volatility ** 2)

        def objective(params):
            """Objective function to minimize across all expiries."""
            a, b, rho, m, sigma = params
            if not (0 < b and -1 <= rho <= 1 and sigma > 0):
                return np.inf  # Invalid parameter regions

            model_variance = self.svi_total_variance(log_moneyness, maturity, a, b, rho, m, sigma)
            errors = (model_variance - total_variance) ** 2
            return np.sum(errors)

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
            self.global_params = result.x
            if verbose:
                print("Global SVI Calibration Successful!")
                print("Parameters:", dict(zip(["a", "b", "rho", "m", "sigma"], self.global_params)))
        else:
            raise ValueError(f"Global SVI fit failed: {result.message}")

    def predict(self, log_moneyness, T):
        """
        Predict implied volatilities for given log-moneyness and maturity.

        Parameters:
        - log_moneyness: array-like, log-moneyness values
        - T: scalar or array-like, residual maturity (T)
        """
        if self.global_params is None:
            raise ValueError("Global parameters are not fitted yet.")

        a, b, rho, m, sigma = self.global_params
        model_variance = self.svi_total_variance(log_moneyness, T, a, b, rho, m, sigma)
        implied_variance = model_variance / T  # Implied variance = total variance / T
        return np.sqrt(np.maximum(implied_variance, 1e-8))  # Safe square root

    def get_params(self):
        """
        Return the fitted global parameters.
        """
        if self.global_params is None:
            raise ValueError("Global parameters are not fitted yet.")
        return dict(zip(["a", "b", "rho", "m", "sigma"], self.global_params))
