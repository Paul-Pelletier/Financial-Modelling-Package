from scipy.optimize import least_squares
import numpy as np

class NonLinearModel:
    def __init__(self, initial_params = None):
        self.initial_params = initial_params
    
    def fit(self, x_train_list, y_train_list, maturities, epochs = 100, learning_rate = 0.01, log_intervals = 50):
        optimizer = least_squares(x)
        pass

    @staticmethod
    def functional_form(x, params, maturity):
        """
        Compute the functional form for a single parameter set.
        Args:
            x: Tensor of shape (num_points,).
            params: Tensor of shape (num_params,).
            maturity: Scalar tensor.

        Returns:
            Tensor of shape (num_points,).
        """
        a, b, rho, m, sigma = params[0], params[1], params[2], params[3], params[4]
        delta = x - m
        total_variance = a + b * (rho * delta + np.sqrt(delta**2 + sigma**2))
        return np.sqrt(total_variance) / np.sqrt(maturity)


from financial_modelling.data_acquisition.file_fetcher import FileFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor


raw_data = FileFetcher().fetch("raw_data.csv", separator = ";")
processed_data = IVPreprocessor(raw_data).preprocess()
print("processed data")
print(processed_data)
NonLinearModel_instance = NonLinearModel()
