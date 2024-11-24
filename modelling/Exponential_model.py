
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class ExponentialModel:
    """
    Implements a single exponential model and fits it to data using scipy.
    """
    def __init__(self, a=1.0, b=-1.0, c=0.0):
        """
        Initialize the single exponential model with optional initial guesses for parameters.
        :param a: Coefficient of the exponential term.
        :param b: Exponent coefficient.
        :param c: Constant offset.
        """
        self.a = a
        self.b = b
        self.c = c

    def model(self, x, a, b, c):
        """
        Single exponential model: y = a * exp(b * x) + c
        :param x: Input values.
        :param a: Coefficient of the exponential term.
        :param b: Exponent coefficient.
        :param c: Constant offset.
        :return: Predicted values.
        """
        return a * np.exp(b * x) + c

    def fit(self, x_data, y_data):
        """
        Fits the exponential model to the data using scipy's curve_fit.
        :param x_data: Array of independent variable values.
        :param y_data: Array of dependent variable values.
        :return: Optimized parameters (a, b, c).
        """
        # Fit the model
        params, covariance = curve_fit(self.model, x_data, y_data, p0=[self.a, self.b, self.c], maxfev=10000)
        
        # Update the parameters with the optimized values
        self.a, self.b, self.c = params
        return params

    def predict(self, x):
        """
        Predict values using the fitted model.
        :param x: Input values.
        :return: Predicted values.
        """
        return self.model(x, self.a, self.b, self.c)

    def plot(self, x_data, y_data):
        """
        Plot the data and the fitted model.
        :param x_data: Independent variable values (x).
        :param y_data: Dependent variable values (y).
        """
        # Generate predictions
        y_pred = self.predict(x_data)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, label="Data", color="blue", alpha=0.7)
        plt.plot(x_data, y_pred, label=f"Fitted Model: y = {self.a:.4f} * exp({self.b:.4f} * x) + {self.c:.4f}", color="red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Single Exponential Model Fit")
        plt.legend()
        plt.grid()
        plt.show()