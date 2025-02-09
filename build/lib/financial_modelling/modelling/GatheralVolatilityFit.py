import numpy as np
from scipy.optimize import differential_evolution

class GatheralQuadraticModel:
    def __init__(self, initial_bounds=None):
        """
        Initialize the Gatheral quadratic model for implied volatility with Differential Evolution.
        """
        self.params = None  # Model parameters: σ_0(T), σ_1(T), σ_2(T)
        if initial_bounds is None:
            self.bounds = [
                (1e-6, 1.0),  # σ_0(T) > 0 (positive volatility)
                (-10, 10),  # σ_1(T) (skew coefficient can be negative or positive)
                (1e-6, 1.0)   # σ_2(T) > 0 (curvature coefficient)
            ]
        else:
            self.bounds = initial_bounds

    @staticmethod
    def gatheral_quadratic_formula(strike_distance, sigma_0, sigma_1, sigma_2):
        """
        Gatheral quadratic model formula:
        σ(K, T) = σ_0 + σ_1 * strike_distance + σ_2 * strike_distance^2
        """
        return sigma_0 + sigma_1 * strike_distance + sigma_2 * strike_distance**2

    def fit(self, train_data, verbose=True):
        """
        Fit the Gatheral quadratic model to training data using Differential Evolution.

        Parameters:
        - train_data: DataFrame with ['strike_distance', 'Implied_Volatility'].
        - verbose: Print intermediate results during optimization.
        """
        train_data = train_data.dropna(axis=0, how='any')
        strike_distance = train_data['STRIKE_DISTANCE'].values
        implied_volatility = train_data['Implied_Volatility'].values

        def objective(params):
            """Objective function for Gatheral quadratic calibration."""
            sigma_0, sigma_1, sigma_2 = params
            model_volatility = self.gatheral_quadratic_formula(strike_distance, sigma_0, sigma_1, sigma_2)
            errors = (model_volatility - implied_volatility) ** 2
            return np.sum(errors)

        # Optimize using Differential Evolution

        result = differential_evolution(objective, self.bounds, strategy='randtobest1bin', maxiter=1000, tol=1e-6, seed=None)

        if result.success:
            self.params = result.x
            if verbose:
                print("Gatheral Quadratic Model Calibration Successful!")
                print("Parameters:", self.params)
        else:
            raise ValueError(f"Gatheral model fit failed: {result.message}")

    def predict(self, strike_distance):
        """
        Predict implied volatilities using the fitted Gatheral quadratic model.

        Parameters:
        - strike_distance: Array of log-moneyness values.

        Returns:
        - Implied volatilities.
        """
        if self.params is None:
            raise ValueError("Gatheral model parameters are not fitted yet.")

        sigma_0, sigma_1, sigma_2 = self.params
        return self.gatheral_quadratic_formula(strike_distance, sigma_0, sigma_1, sigma_2)

    def get_params(self):
        """
        Return the fitted parameters.
        """
        if self.params is None:
            raise ValueError("Gatheral model parameters are not fitted yet.")
        return dict(zip(["sigma_0", "sigma_1", "sigma_2"], self.params))
