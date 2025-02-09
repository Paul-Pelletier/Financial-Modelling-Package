import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from scipy.stats import rankdata

class NonLinearModel:
    def __init__(self, initial_params=None):
        """
        Initialize the model with optional default parameters.
        """
        self.initial_params = initial_params if initial_params is not None else [0, 0, 0, 0, 0]
        self.calibrated_params = None

    @staticmethod
    def total_variance_form(x, params):
        """
        Compute the total variance form for a single maturity group.

        Args:
            x: Array of log-moneyness values.
            params: Array of parameters (5,).

        Returns:
            Array of total variance values.
        """
        a, b, rho, m, sigma = params
        delta = x - m
        return a + b * (rho * delta + np.sqrt(delta**2 + sigma**2))

    def fit(self, x_train_list, y_train_list, maturities):
        """
        Fit the SVI model to all maturities simultaneously using masking for varying data sizes,
        rank-based weighting, and regularity penalty.

        Args:
            x_train_list: List of 1D NumPy arrays (log-moneyness values for each maturity).
            y_train_list: List of 1D NumPy arrays (implied volatility values for each maturity).
            maturities: List or 1D NumPy array of maturities (one per subset).
        """
        maturities = np.array(maturities)  # Ensure maturities is a NumPy array
        num_maturities = len(maturities)
        num_params = 5  # a, b, rho, m, sigma

        # Prepare a mask to align data for vectorized processing
        max_length = max(len(x) for x in x_train_list)
        x_padded = np.full((num_maturities, max_length), np.nan)
        y_padded = np.full((num_maturities, max_length), np.nan)
        mask = np.zeros((num_maturities, max_length), dtype=bool)

        for i, (x_train, y_train) in enumerate(zip(x_train_list, y_train_list)):
            x_padded[i, :len(x_train)] = x_train
            y_padded[i, :len(y_train)] = y_train
            mask[i, :len(x_train)] = True

        # Vectorized loss function with rank-based weighting and regularity penalty
        def loss_function(flat_params):
            """
            Vectorized loss function for all maturities, including rank-based weighting and regularity penalty.
            """
            params = flat_params.reshape(num_maturities, num_params)
            total_variance = np.zeros_like(x_padded)
            
            # Compute total variance for all maturities
            for i in range(num_maturities):
                total_variance[i, :] = self.total_variance_form(x_padded[i, :], params[i])

            # Residuals computation
            actual_total_variance = (y_padded**2) * maturities[:, None]
            residuals = total_variance - actual_total_variance
            residuals[~mask] = 0  # Ignore residuals for padded elements
            
            # Rank-based weighting
            residuals_squared = residuals**2
            ranks = np.apply_along_axis(rankdata, axis=1, arr=residuals_squared)
            weighted_residuals = ranks * residuals_squared * maturities[:, None]

            # Regularity penalty
            lambda_reg = 1e-4  # Regularization strength
            param_differences = np.diff(params, axis=0)  # Differences between neighboring maturities
            regularity_penalty = lambda_reg * np.sum(param_differences**2)

            # Total loss
            total_loss = np.nansum(weighted_residuals) + regularity_penalty
            return total_loss

        # Optimize parameters for all maturities
        initial_params = np.tile(self.initial_params, num_maturities)
        bounds = [(0, None), (0, None), (-1, 1), (-np.inf, np.inf), (0, None)] * num_maturities
        result = minimize(
            loss_function,
            x0=initial_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={
                'ftol': 1e-10,  # Function tolerance
                'gtol': 1e-10,  # Gradient tolerance
                'maxiter': 200  # Maximum iterations
            }
        )

        # Store calibrated parameters for each maturity
        self.calibrated_params = {
            maturities[i]: result.x[i * num_params: (i + 1) * num_params]
            for i in range(num_maturities)
        }
        for maturity, params in self.calibrated_params.items():
            print(f"Maturity {maturity}: Optimized parameters: {params}")


    @staticmethod
    def functional_form(x, params, maturity):
        """
        Compute implied volatility from total variance for a single maturity group.

        Args:
            x: Array of log-moneyness values.
            params: Array of parameters (5,).
            maturity: Scalar maturity.

        Returns:
            Array of implied volatilities.
        """
        total_variance = NonLinearModel.total_variance_form(x, params)
        return np.sqrt(total_variance / maturity)

    @staticmethod
    def generate_synthetic_data(params_list, x_train_list, maturities):
        """
        Generate synthetic y_train data based on a given set of parameters.

        Args:
            params_list: List of parameter arrays (one for each maturity).
            x_train_list: List of 1D NumPy arrays (log-moneyness values for each maturity).
            maturities: List of maturities (one for each subset).

        Returns:
            List of 1D NumPy arrays (synthetic implied volatilities).
        """
        y_train_list = []
        for params, x_train, maturity in zip(params_list, x_train_list, maturities):
            y_train = NonLinearModel.functional_form(x_train, params, maturity)
            y_train = y_train + np.random.normal(0, 0.0002, size = np.size(y_train))
            y_train_list.append(y_train)
        return y_train_list

    def plot(self, x_train_list, y_train_list, maturities):
        """
        Plot the training data and the fitted model for each maturity.

        Args:
            x_train_list: List of 1D NumPy arrays (log-moneyness values for each maturity).
            y_train_list: List of 1D NumPy arrays (implied volatility values for each maturity).
            maturities: List of maturities (one per subset).
        """
        if self.calibrated_params is None:
            print("Error: Model parameters are not calibrated. Run 'fit' first.")
            return

        # Create a plot for each maturity
        for i, (x_train, y_train, maturity) in enumerate(zip(x_train_list, y_train_list, maturities)):
            plt.figure(figsize=(8, 6))

            # Scatter plot of training data
            plt.scatter(x_train, y_train, color='blue', label='Training Data', alpha=0.6)

            # Generate fitted implied volatility
            params = self.calibrated_params[maturity]
            x_fitted = np.linspace(np.min(x_train), np.max(x_train), 100)
            y_fitted = self.functional_form(x_fitted, params, maturity)

            # Plot fitted curve
            plt.plot(x_fitted, y_fitted, color='red', label='Fitted SVI Model')

            # Add labels and legend
            plt.title(f"Maturity {maturity} Years")
            plt.xlabel("Log-Moneyness")
            plt.ylabel("Implied Volatility")
            plt.legend()
            plt.grid(True)

            # Show plot
            plt.show()


def main_test():
    # Example data with varying subset sizes
    number_of_sets = 5
    x_train_list = [np.linspace(-0.20, 0.20, 200) for i in range(number_of_sets)]
    
    maturities = [1.5 * (i + 1) for i in range(number_of_sets)]  # Maturities in days
    np.random.seed(23)
    # Ground-truth parameters for generating synthetic data
    true_params_list = [[0.2 + i / 1000, 
                         0.1 + i / 1000,
                         -0.5 + i / 1000,
                         0.0 + i / 1000,
                         0.15 + i / 1000] for i in range(number_of_sets)]
    
    for param, maturity in zip(true_params_list, maturities):
        print(f"Maturity:{maturity}", param)

    # Generate synthetic y_train data
    model = NonLinearModel()
    y_train_list = model.generate_synthetic_data(true_params_list, x_train_list, maturities)

    # Fit the model to the synthetic data
    model.fit(x_train_list, y_train_list, maturities)

    # Plot the results
    model.plot(x_train_list, y_train_list, maturities)


if __name__ == "__main__":
    main_test()
