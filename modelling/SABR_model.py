import tensorflow as tf
import numpy as np

class SABRModelGPU:
    """
    Implements the SABR model and computes implied volatility using TensorFlow for GPU acceleration.
    """
    def __init__(self, alpha=0.04, beta=0.5, rho=-0.2, nu=0.3):
        """
        Initialize the SABR model parameters.
        :param alpha: Initial volatility (σ_0).
        :param beta: Elasticity parameter (β).
        :param rho: Correlation coefficient (ρ).
        :param nu: Volatility of volatility (ν).
        """
        self.alpha = tf.Variable(alpha, dtype=tf.float32, trainable=True)
        self.beta = tf.Variable(beta, dtype=tf.float32, trainable=True)
        self.rho = tf.Variable(rho, dtype=tf.float32, trainable=True)
        self.nu = tf.Variable(nu, dtype=tf.float32, trainable=True)

    @tf.function
    def implied_volatility(self, F, K, T):
        """
        Computes the implied volatility using Hagan's SABR approximation.
        :param F: Forward price (tensor).
        :param K: Strike price (tensor).
        :param T: Time to maturity (scalar tensor).
        :return: Implied volatility (tensor).
        """
        epsilon = 1e-12  # To avoid division by zero
        F = tf.maximum(F, epsilon)
        K = tf.maximum(K, epsilon)
        
        # Handle ATM case (F == K)
        ATM_case = tf.equal(F, K)

        # Common terms
        FK = F * K
        z = (self.nu / self.alpha) * (FK ** ((1 - self.beta) / 2)) * tf.math.log(F / K + epsilon)
        x_z = tf.where(
            tf.abs(z) < epsilon,
            tf.ones_like(z),
            tf.math.log((tf.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))
        )

        # SABR volatility approximation
        A = self.alpha / (FK ** ((1 - self.beta) / 2))
        B = 1 + ((1 - self.beta)**2 / 24) * (tf.math.log(F / K)**2) + \
            ((1 - self.beta)**4 / 1920) * (tf.math.log(F / K)**4)
        C = 1 + ((2 - 3 * self.rho**2) / 24) * (self.nu**2 * T)

        vol = tf.where(ATM_case, 
                       self.alpha * F ** (self.beta - 1), 
                       (A * z / x_z) * B * C)
        return vol

    def fit(self, strikes, market_vols, F, T, learning_rate=0.1, epochs=1000):
        """
        Calibrate the SABR model parameters to market implied volatilities using GPU-accelerated TensorFlow.
        :param strikes: Tensor of strike prices.
        :param market_vols: Tensor of market implied volatilities.
        :param F: Forward price (scalar tensor).
        :param T: Time to maturity (scalar tensor).
        :param learning_rate: Learning rate for the optimizer.
        :param epochs: Number of optimization steps.
        :return: Optimized parameters (alpha, beta, rho, nu).
        """
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float32)
        market_vols = tf.convert_to_tensor(market_vols, dtype=tf.float32)
        F = tf.convert_to_tensor(F, dtype=tf.float32)
        T = tf.convert_to_tensor(T, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def loss_fn():
            model_vols = self.implied_volatility(F, strikes, T)
            return tf.reduce_mean((model_vols - market_vols)**2)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = loss_fn()
            gradients = tape.gradient(loss, [self.alpha, self.beta, self.rho, self.nu])
            optimizer.apply_gradients(zip(gradients, [self.alpha, self.beta, self.rho, self.nu]))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.numpy():.6f}")

        return {
            "alpha": self.alpha.numpy(),
            "beta": self.beta.numpy(),
            "rho": self.rho.numpy(),
            "nu": self.nu.numpy()
        }

    @tf.function
    def predict(self, strikes, F, T):
        """
        Predict implied volatilities for a set of strike prices.
        :param strikes: Tensor of strike prices.
        :param F: Forward price (scalar tensor).
        :param T: Time to maturity (scalar tensor).
        :return: Predicted implied volatilities.
        """
        return self.implied_volatility(F, strikes, T)
