import tensorflow as tf
import numpy as np

class ExtendedNSSModel:
    """
    Implements the extended Nelson-Siegel-Svensson (NSS) model with global optimization on GPU.
    """
    def __init__(self):
        self.beta_0 = None
        self.beta_1 = None
        self.beta_2 = None
        self.beta_3 = None
        self.beta_4 = None
        self.tau_1 = None
        self.tau_2 = None
        self.tau_3 = None

    def forward_rate(self, t, beta_0, beta_1, beta_2, beta_3, beta_4, tau_1, tau_2, tau_3):
        """
        Computes the forward rate at time t using the extended NSS formula on GPU.
        """
        term1 = beta_0
        term2 = beta_1 * tf.exp(-t / tau_1)
        term3 = beta_2 * (t / tau_1) * tf.exp(-t / tau_1)
        term4 = beta_3 * (t / tau_2) * tf.exp(-t / tau_2)
        term5 = beta_4 * (t / tau_3) * tf.exp(-t / tau_3)
        return term1 + term2 + term3 + term4 + term5

    def fit(self, maturities, forward_rates, num_restarts=3):
        """
        Fits the model using random restarts for global optimization.

        Parameters:
        ----------
        maturities : array-like
            Maturities (times to maturity).
        forward_rates : array-like
            Observed forward rates corresponding to the maturities.
        num_restarts : int
            Number of random restarts for global optimization.

        Returns:
        -------
        dict
            Best fitted parameters.
        """
        # Convert inputs to tensors
        maturities = tf.convert_to_tensor(maturities, dtype=tf.float32)
        forward_rates = tf.convert_to_tensor(forward_rates, dtype=tf.float32)

        # Define the loss function
        def loss_fn(beta_0, beta_1, beta_2, beta_3, beta_4, tau_1, tau_2, tau_3):
            model_rates = self.forward_rate(maturities, beta_0, beta_1, beta_2, beta_3, beta_4, tau_1, tau_2, tau_3)
            return tf.reduce_mean((model_rates - forward_rates) ** 2)

        best_loss = float('inf')
        best_params = None

        # Random restarts
        for restart in range(num_restarts):
            print(f"Restart {restart + 1}/{num_restarts}")

            # Random initialization of parameters
            beta_0 = tf.Variable(tf.random.uniform([1], 0, 0.1), dtype=tf.float32)
            beta_1 = tf.Variable(tf.random.uniform([1], -0.1, 0.1), dtype=tf.float32)
            beta_2 = tf.Variable(tf.random.uniform([1], -0.1, 0.1), dtype=tf.float32)
            beta_3 = tf.Variable(tf.random.uniform([1], -0.1, 0.1), dtype=tf.float32)
            beta_4 = tf.Variable(tf.random.uniform([1], -0.1, 0.1), dtype=tf.float32)
            tau_1 = tf.Variable(tf.random.uniform([1], 0.5, 2.0), dtype=tf.float32)
            tau_2 = tf.Variable(tf.random.uniform([1], 1.0, 3.0), dtype=tf.float32)
            tau_3 = tf.Variable(tf.random.uniform([1], 2.0, 4.0), dtype=tf.float32)

            # Define the optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            # Optimization loop
            for step in range(200):  # Maximum 500 iterations
                with tf.GradientTape() as tape:
                    loss = loss_fn(beta_0, beta_1, beta_2, beta_3, beta_4, tau_1, tau_2, tau_3)
                gradients = tape.gradient(loss, [beta_0, beta_1, beta_2, beta_3, beta_4, tau_1, tau_2, tau_3])
                optimizer.apply_gradients(zip(gradients, [beta_0, beta_1, beta_2, beta_3, beta_4, tau_1, tau_2, tau_3]))

                if step % 100 == 0:
                    print(f"Step {step}: Loss = {loss.numpy()}")

            # Check if this is the best result so far
            if loss.numpy() < best_loss:
                best_loss = loss.numpy()
                best_params = {
                    "beta_0": beta_0.numpy()[0],
                    "beta_1": beta_1.numpy()[0],
                    "beta_2": beta_2.numpy()[0],
                    "beta_3": beta_3.numpy()[0],
                    "beta_4": beta_4.numpy()[0],
                    "tau_1": tau_1.numpy()[0],
                    "tau_2": tau_2.numpy()[0],
                    "tau_3": tau_3.numpy()[0],
                }

        # Store the best parameters
        self.beta_0 = best_params["beta_0"]
        self.beta_1 = best_params["beta_1"]
        self.beta_2 = best_params["beta_2"]
        self.beta_3 = best_params["beta_3"]
        self.beta_4 = best_params["beta_4"]
        self.tau_1 = best_params["tau_1"]
        self.tau_2 = best_params["tau_2"]
        self.tau_3 = best_params["tau_3"]

        return best_params

    def predict(self, maturities):
        """
        Predict forward rates for given maturities using the fitted model.

        Parameters:
        ----------
        maturities : array-like
            Maturities (times to maturity).

        Returns:
        -------
        tf.Tensor
            Predicted forward rates.
        """
        if any(param is None for param in [self.beta_0, self.beta_1, self.beta_2, self.beta_3, self.beta_4, self.tau_1, self.tau_2, self.tau_3]):
            raise ValueError("Model parameters not fitted yet")

        maturities = tf.convert_to_tensor(maturities, dtype=tf.float32)
        return self.forward_rate(
            maturities, self.beta_0, self.beta_1, self.beta_2, self.beta_3, self.beta_4, self.tau_1, self.tau_2, self.tau_3
        )
