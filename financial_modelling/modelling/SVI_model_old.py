import numpy as np
from scipy.optimize import minimize

class SVI_model:
    def __init__(self, initial_params=None):
        """
        Initialize the SVI model with optional initial parameters.
        
        Parameters:
        - initial_params: dict, initial guesses for a, b, rho, m, sigma
        """
        if initial_params is None:
            # Default initial guesses for SVI parameters
            self.params = {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.0, "sigma": 0.2}
        else:
            self.params = initial_params

    def svi_formula(self, k, a, b, rho, m, sigma):
        """
        SVI formula to calculate total implied variance for a given log-moneyness k.
        
        Parameters:
        - k: Log-moneyness (log(K / Spot))
        - a, b, rho, m, sigma: SVI parameters
        
        Returns:
        - Total implied variance w(k)
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def fit(self, log_moneyness, market_variances):
        """
        Calibrate the SVI model to market data.
        
        Parameters:
        - log_moneyness: array-like, log-moneyness values (log(K / Spot))
        - market_variances: array-like, market total implied variances
        
        Updates:
        - self.params with calibrated parameters
        """
        def objective(params):
            a, b, rho, m, sigma = params
            # Compute SVI variances
            model_variances = self.svi_formula(log_moneyness, a, b, rho, m, sigma)
            # Minimize squared error
            return np.sum((model_variances - market_variances) ** 2)
        
        # Initial guesses
        initial_guess = list(self.params.values())
        # Bounds for parameters
        bounds = [
            (0, None),   # a > 0
            (0, None),   # b > 0
            (-1, 1),     # -1 <= rho <= 1
            (None, None), # m unbounded
            (1e-3, None)  # sigma > 0
        ]
        
        # Optimize parameters
        result = minimize(objective, initial_guess, bounds=bounds)
        
        if result.success:
            self.params = dict(zip(["a", "b", "rho", "m", "sigma"], result.x))
        else:
            raise ValueError("SVI calibration failed. Check your data and initial parameters.")

    def predict(self, log_moneyness):
        """
        Predict total implied variance for given log-moneyness using calibrated parameters.
        
        Parameters:
        - log_moneyness: array-like, log-moneyness values (log(K / Spot))
        
        Returns:
        - Array of predicted total implied variances
        """
        a, b, rho, m, sigma = self.params.values()
        return self.svi_formula(log_moneyness, a, b, rho, m, sigma)

    def implied_volatility(self, log_moneyness, maturity):
        """
        Calculate implied volatilities from total implied variances.
        
        Parameters:
        - log_moneyness: array-like, log-moneyness values (log(K / Spot))
        - maturity: float, time to maturity
        
        Returns:
        - Array of implied volatilities
        """
        variances = self.predict(log_moneyness)
        return np.sqrt(variances / maturity)
