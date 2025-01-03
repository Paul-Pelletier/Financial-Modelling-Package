import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

class NonLinearModel:
    def __init__(self, initial_params=None):
        """
        Initialize the model with optional default parameters.
        """
        self.initial_params = initial_params if initial_params is not None else [0, 0, 0, 0, 0]
        self.calibrated_params = None

    def fit(self, x_train_list, y_train_list, maturities):
        """
        Fit the SVI model with weighted loss for each maturity.

        Args:
            x_train_list: List of 1D NumPy arrays (log-moneyness values for each maturity).
            y_train_list: List of 1D NumPy arrays (implied volatility values for each maturity).
            maturities: 1D NumPy array of maturities (one per subset).
        """
        num_maturities = len(maturities)
        num_params = 5  # a, b, rho, m, sigma

        # Prepare to store calibrated parameters for each maturity
        self.calibrated_params = {}

        def loss_function(params, x, y, maturity):
            """
            Loss function for a single maturity group with weighting by inverse maturity.
            """
            predicted_total_variance = self.total_variance_form(x, params)
            actual_total_variance = (y ** 2) * maturity
            residuals = predicted_total_variance - actual_total_variance
            weighted_residuals = (residuals**2)*maturity
            return np.sum(weighted_residuals)

        # Optimize parameters for each subset individually
        for i, (x_train, y_train, maturity) in enumerate(zip(x_train_list, y_train_list, maturities)):
            result = minimize(
                loss_function,
                x0=self.initial_params,
                args=(x_train, y_train, maturity),
                bounds=[(0, None), (0, None), (-1, 1), (-np.inf, np.inf), (0, None)],
                method="L-BFGS-B",
                options={
                    'ftol': 1e-10,   # Function tolerance
                    'gtol': 1e-10,   # Gradient tolerance
                    'maxiter': 200  # Maximum iterations
                }
            )

            # Store calibrated parameters
            self.calibrated_params[maturity] = result.x
            print(f"Maturity {maturity}: Optimized parameters: {result.x}")

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
    number_of_sets = 30
    x_train_list = [np.linspace(-0.20, 0.20, 500) for i in range(number_of_sets)]
    
    maturities = [1.5 * (i + 1) for i in range(number_of_sets)]  # Maturities in days
    np.random.seed(23)
    # Ground-truth parameters for generating synthetic data
    true_params_list = [[0.2 + np.random.rand() / 10, 
                         0.1 + np.random.rand() / 10,
                         -0.5 + np.random.rand() / 10,
                         0.0 + np.random.rand() / 10,
                         0.15 + np.random.rand() / 10] for i in range(number_of_sets)]
    
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
