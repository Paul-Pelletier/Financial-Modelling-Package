import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import CubicSpline
from sklearn.neighbors import KernelDensity

class GlobalSVIFit:
    def __init__(self, initial_params=None):
        """
        Initialize the global SVI model with optional initial parameters.
        """
        self.params = None  # Model parameters: a(T), b(T), rho(T), m(T), sigma(T)
        if initial_params is None:
            self.initial_guess = [1e-4, 0.05, -0.5, 0.0, 0.1,  # Initial params for maturity-independent base values
                                  0.01, 0.01, 0.01, 0.01, 0.01]  # Coefficients for linear/maturity-dependence
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

    def parameterize(self, maturity, params):
        """
        Parameterize SVI parameters as functions of maturity T.
        Using a linear model for simplicity, but this can be replaced with splines or other smooth functions.

        Parameters:
        - maturity: Array of maturities.
        - params: Model coefficients for global fit.

        Returns:
        - Parameterized a, b, rho, m, sigma.
        """
        a_base, b_base, rho_base, m_base, sigma_base = params[:5]
        a_slope, b_slope, rho_slope, m_slope, sigma_slope = params[5:]

        a = a_base + a_slope * maturity
        b = b_base + b_slope * maturity
        rho = rho_base + rho_slope * np.log(1 + maturity)  # Log-decaying skew
        m = m_base + m_slope * maturity
        sigma = sigma_base + sigma_slope * np.exp(-maturity)  # Exponentially decaying curvature

        return a, b, rho, m, sigma

    def fit(self, train_data, verbose=True):
        """
        Fit global SVI parameters to training data using Differential Evolution.

        Parameters:
        - train_data: DataFrame with ['Log_Moneyness', 'Implied_Volatility', 'Maturity'].
        - verbose: Print intermediate results during optimization.
        """
        log_moneyness = train_data['Log_Moneyness'].values
        implied_volatility = train_data['Implied_Volatility'].values
        maturity = train_data['Maturity'].values

        # Total variance: w = T * (implied_volatility)^2
        total_variance = maturity * (implied_volatility ** 2)

        # Apply Gaussian kernel to smooth the training data
        kernel = KernelDensity(kernel='gaussian', bandwidth=0.1)
        kernel.fit(np.column_stack([log_moneyness, maturity]), sample_weight=total_variance)
        
        # Generate synthetic data from the fitted kernel
        n_samples = len(train_data) * 10
        synthetic_samples = kernel.sample(n_samples)
        synthetic_log_moneyness, synthetic_maturity = synthetic_samples[:, 0], synthetic_samples[:, 1]
        synthetic_variance = kernel.score_samples(synthetic_samples)  # Log-likelihood as a proxy for variance
        synthetic_variance = np.exp(synthetic_variance)  # Convert back from log-likelihood

        def objective(params):
            """Objective function for global SVI calibration."""
            a, b, rho, m, sigma = self.parameterize(synthetic_maturity, params)
            model_variance = self.svi(synthetic_log_moneyness, a, b, rho, m, sigma)
            errors = (model_variance - synthetic_variance) ** 2
            return np.sum(errors)

        # Parameter bounds (to enforce stylized facts)
        bounds = [
            (1e-6, 10.0),    # a_base > 0
            (1e-6, 10.0),    # b_base > 0
            (-0.99, 0.99),   # -1 <= rho_base <= 1
            (-1.0, 1.0),     # m_base bounded
            (1e-6, 2.0),     # sigma_base > 0
            (-0.1, 0.1),     # Slope for a
            (-0.1, 0.1),     # Slope for b
            (-0.1, 0.1),     # Slope for rho
            (-0.1, 0.1),     # Slope for m
            (-0.1, 0.1),     # Slope for sigma
        ]

        # Optimize using Differential Evolution
        result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, tol=1e-6, seed=None)

        if result.success:
            self.params = result.x
            if verbose:
                print("Global SVI Calibration Successful!")
                print("Parameters:", self.params)
        else:
            raise ValueError(f"Global SVI fit failed: {result.message}")

    def predict(self, log_moneyness, maturity):
        """
        Predict implied volatilities using the fitted global SVI model.

        Parameters:
        - log_moneyness: Array of log-moneyness values.
        - maturity: Array of maturities.

        Returns:
        - Implied volatilities.
        """
        if self.params is None:
            raise ValueError("Global SVI model parameters are not fitted yet.")

        a, b, rho, m, sigma = self.parameterize(maturity, self.params)
        model_variance = self.svi(log_moneyness, a, b, rho, m, sigma)
        implied_volatility = np.sqrt(np.maximum(model_variance / maturity, 1e-8))  # Safe division
        return implied_volatility

    def get_params(self):
        """Return the fitted parameters."""
        if self.params is None:
            raise ValueError("Global SVI model parameters are not fitted yet.")
        return self.params
