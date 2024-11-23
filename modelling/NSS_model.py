import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PolynomialDegree3Model:
    """
    Implements a classic polynomial regression model of degree 3 using TensorFlow.
    """
    def __init__(self):
        self.alpha = None  # Intercept
        self.beta = None   # Linear coefficient
        self.gamma = None  # Quadratic coefficient
        self.delta = None  # Cubic coefficient

    def forward_rate(self, t, params):
        """
        Computes the forward rate based on the degree 3 polynomial.
        """
        alpha, beta, gamma, delta = tf.split(params, 4, axis=1)
        return alpha + beta * t + gamma * t**2 + delta * t**3

    def fit(self, maturities, forward_rates, num_restarts=50):
        """
        Fits the polynomial model using TensorFlow's Adam optimizer.
        """
        maturities = tf.convert_to_tensor(maturities, dtype=tf.float32)
        forward_rates = tf.convert_to_tensor(forward_rates, dtype=tf.float32)

        # Initialize parameters: [alpha, beta, gamma, delta]
        params = tf.Variable(tf.random.uniform([num_restarts, 4], 
                                               minval=[2000.0, -100.0, -10.0, -1.0], 
                                               maxval=[3000.0, 100.0, 10.0, 1.0]), 
                              dtype=tf.float32)
        
        # Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        def loss_fn(params):
            model_rates = self.forward_rate(maturities, params)
            return tf.reduce_mean(tf.square(model_rates - forward_rates[tf.newaxis, :]), axis=1)

        # Optimize parameters
        for step in range(50):
            with tf.GradientTape() as tape:
                losses = loss_fn(params)
            
            gradients = tape.gradient(losses, [params])
            gradients = [tf.clip_by_value(g, -10.0, 10.0) for g in gradients]
            optimizer.apply_gradients(zip(gradients, [params]))
            
            if step % 100 == 0:
                print(f"Step {step}: Losses = {losses.numpy()}")

            # Early stopping
            if step > 100 and abs(losses.numpy().min() - losses.numpy().max()) < 1e-6:
                print("Convergence achieved.")
                break

        # Select the best parameters
        best_idx = tf.argmin(losses)
        best_params = {
            "alpha": params[best_idx, 0].numpy(),
            "beta": params[best_idx, 1].numpy(),
            "gamma": params[best_idx, 2].numpy(),
            "delta": params[best_idx, 3].numpy(),
        }

        # Save the best parameters
        self.alpha = best_params["alpha"]
        self.beta = best_params["beta"]
        self.gamma = best_params["gamma"]
        self.delta = best_params["delta"]

        return best_params

    def predict(self, maturities):
        """
        Predicts forward rates using the fitted parameters.
        """
        if any(param is None for param in [self.alpha, self.beta, self.gamma, self.delta]):
            raise ValueError("Model parameters not fitted yet.")

        maturities = tf.convert_to_tensor(maturities, dtype=tf.float32)
        params = tf.convert_to_tensor([self.alpha, self.beta, self.gamma, self.delta], dtype=tf.float32)
        return self.forward_rate(maturities, params[tf.newaxis, :])[0]
