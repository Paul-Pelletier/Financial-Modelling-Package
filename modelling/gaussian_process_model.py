import tensorflow as tf
import numpy as np

class GaussianProcessModel:
    """
    Implements Gaussian Process Regression using TensorFlow.
    """
    def __init__(self, length_scale=1.0, variance=1.0, noise_variance=1e-2):
        """
        Initializes the Gaussian Process Regression model.
        :param length_scale: Length scale for the RBF kernel.
        :param variance: Variance for the RBF kernel.
        :param noise_variance: Noise variance for the GP model.
        """
        self.length_scale = length_scale
        self.variance = variance
        self.noise_variance = noise_variance
        self.alpha = None  # Precision-weighted target values
        self.L = None  # Cholesky decomposition of the kernel matrix
        self.X_train = None  # Training data

    def rbf_kernel(self, X1, X2):
        """
        Squared Exponential (RBF) Kernel function.
        :param X1: First set of input points (N x 1).
        :param X2: Second set of input points (M x 1).
        :return: Covariance matrix (N x M).
        """
        X1 = tf.expand_dims(X1, axis=1)
        X2 = tf.expand_dims(X2, axis=1)
        sqdist = tf.square(X1 - tf.transpose(X2))
        return self.variance * tf.exp(-0.5 * sqdist / self.length_scale**2)

    def fit(self, maturities, forward_rates):
        """
        Fit the Gaussian Process model to the training data.
        :param maturities: Training input data (maturities).
        :param forward_rates: Training output data (forward rates).
        """
        # Save training data
        self.X_train = tf.convert_to_tensor(maturities, dtype=tf.float32)
        y_train = tf.convert_to_tensor(forward_rates, dtype=tf.float32)

        # Compute the kernel matrix K(X_train, X_train)
        K = self.rbf_kernel(self.X_train, self.X_train)
        K += self.noise_variance * tf.eye(len(self.X_train), dtype=tf.float32)

        # Perform Cholesky decomposition
        self.L = tf.linalg.cholesky(K)

        # Solve for alpha: K^-1 * y_train using the Cholesky factorization
        self.alpha = tf.linalg.cholesky_solve(self.L, tf.expand_dims(y_train, axis=1))

        return {"length_scale": self.length_scale, "variance": self.variance, "noise_variance": self.noise_variance}

    def predict(self, maturities):
        """
        Predict forward rates using the fitted model.
        :param maturities: Test input data (maturities).
        :return: Predicted forward rates.
        """
        if self.alpha is None or self.L is None or self.X_train is None:
            raise ValueError("Model parameters not fitted yet.")

        # Convert input to tensor
        X_test = tf.convert_to_tensor(maturities, dtype=tf.float32)

        # Compute the kernel K(X_test, X_train)
        K_star = self.rbf_kernel(X_test, self.X_train)

        # Compute the predictive mean
        mean = tf.linalg.matmul(K_star, self.alpha)
        return tf.squeeze(mean)

    def get_parameters(self):
        """
        Get the fitted model parameters.
        :return: A dictionary containing the parameters of the model.
        """
        return {
            "length_scale": self.length_scale,
            "variance": self.variance,
            "noise_variance": self.noise_variance
        }

    def predict_with_uncertainty(self, maturities):
        """
        Predict forward rates with uncertainty using the fitted model.
        :param maturities: Test input data (maturities).
        :return: Predicted forward rates and uncertainties (standard deviation).
        """
        if self.alpha is None or self.L is None or self.X_train is None:
            raise ValueError("Model parameters not fitted yet.")

        # Convert input to tensor
        X_test = tf.convert_to_tensor(maturities, dtype=tf.float32)

        # Compute the kernel K(X_test, X_train)
        K_star = self.rbf_kernel(X_test, self.X_train)

        # Compute the predictive mean
        mean = tf.linalg.matmul(K_star, self.alpha)

        # Compute the uncertainty
        v = tf.linalg.triangular_solve(self.L, tf.transpose(K_star))
        K_star_star = self.rbf_kernel(X_test, X_test)
        variance = tf.linalg.diag_part(K_star_star) - tf.reduce_sum(tf.square(v), axis=0)
        std_dev = tf.sqrt(tf.maximum(variance, 0.0))

        return tf.squeeze(mean), tf.squeeze(std_dev)

