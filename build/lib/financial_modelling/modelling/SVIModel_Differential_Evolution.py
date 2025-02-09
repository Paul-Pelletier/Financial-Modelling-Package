import numpy as np
from scipy.optimize import differential_evolution

class SVIDifferentialEvolutionModel:
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
        term1 = rho * (log_moneyness - m)
        term2 = np.sqrt((log_moneyness - m) ** 2 + sigma ** 2)
        return a + b * (term1 + term2)

    def fit(self, train_data, verbose=True):
        """
        Fit SVI parameters to training data using differential evolution.

        Parameters:
        - train_data: DataFrame with ['Log_Moneyness', 'Implied_Volatility', 'Maturity']
        - verbose: Print intermediate results during optimization
        """
        log_moneyness = train_data['Log_Moneyness'].values
        implied_volatility = train_data['Implied_Volatility'].values
        maturity = train_data['Maturity'].values

        # Total variance: w = T * (implied_volatility)^2
        total_variance = maturity * (implied_volatility ** 2)

        def objective(params):
            """Objective function with constraints and regularization for parameter bounds"""
            a, b, rho, m, sigma = params

            # Penalize invalid parameter regions
            if not (0 < b and -1 <= rho <= 1 and sigma > 0):
                return np.inf

            # SVI model variance
            model_variance = self.svi(log_moneyness, a, b, rho, m, sigma)
            errors = (model_variance - total_variance) ** 2

            # Penalize extreme values of 'a'
            penalty_a = 100 * max(a - 0.05, 0) ** 2  # Penalize a > 0.05
            penalty_a += 100 * max(0 - a, 0) ** 2    # Penalize a < 0

            return np.sum(errors) + penalty_a

        # Define bounds
        try:
            k_min, k_max = np.min(log_moneyness), np.max(log_moneyness)
            if np.isnan(k_min) or np.isnan(k_max):
                raise ValueError("k_min or k_max is NaN.")
        except Exception as e:
            print(f"Error computing bounds for log-moneyness: {e}")
            k_min, k_max = -1, 1  # Fallback values

        bounds = [
            (1e-6, 0.05),       # a: small positive values with tighter bounds
            (1e-6, 5.0),        # b: moderate positive values
            (-0.99, 0.99),      # rho: correlation between -1 and 1
            (k_min - 0.5, k_max + 0.5),  # m: bounded near observed log-moneyness
            (1e-6, 1.0)         # sigma: small positive values
        ]

        # Default parameters to use if calibration fails
        default_params = [0, 0, 0, 0, 0]

        # Optimize using differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds,
                strategy='best1bin',
                maxiter=1000,  # Increased iterations for better convergence
                popsize=5,    # Increased population size for diversity
                tol=1e-6,
                seed=None
            )
            if result.success:
                self.params = result.x
                if verbose:
                    print("SVI Calibration Successful!")
                    print("Parameters:", dict(zip(["a", "b", "rho", "m", "sigma"], self.params)))
            else:
                print(f"SVI Calibration failed: {result.message}. Using default parameters.")
                self.params = default_params
        except Exception as e:
            print(f"Error during optimization: {e}. Using default parameters.")
            self.params = default_params

    def predict(self, log_moneyness, maturity):
        """
        Predict implied volatilities using the fitted SVI model.

        Parameters:
        - log_moneyness: array-like, log-moneyness values
        - maturity: scalar or array-like, residual maturity (T)
        """
        if self.params is None:
            raise ValueError("Model parameters are not fitted yet.")

        a, b, rho, m, sigma = self.params
        model_variance = self.svi(log_moneyness, a, b, rho, m, sigma)
        implied_variance = model_variance / maturity  # Implied variance = total variance / T
        return np.sqrt(np.maximum(implied_variance, 1e-8))  # Safe square root

    def get_params(self):
        """Return the fitted parameters."""
        if self.params is None:
            raise ValueError("Model parameters are not fitted yet.")
        return dict(zip(["a", "b", "rho", "m", "sigma"], self.params))
