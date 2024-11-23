import tensorflow as tf

class NSSModel:
    """
    Implements the Nelson-Siegel-Svensson (NSS) model for yield curve fitting using TensorFlow.
    """
    def __init__(self):
        self.beta_0 = None
        self.beta_1 = None
        self.beta_2 = None
        self.beta_3 = None
        self.tau_1 = None
        self.tau_2 = None

    def forward_rate(self, t, beta_0, beta_1, beta_2, beta_3, tau_1, tau_2):
        """
        Computes the forward rate at time t using the NSS formula on GPU.

        Parameters:
        ----------
        t : tf.Tensor
            Time to maturity (TensorFlow tensor).
        beta_0, beta_1, beta_2, beta_3 : float
            NSS parameters for the yield curve.
        tau_1, tau_2 : float
            Decay factors controlling the curve's shape.

        Returns:
        -------
        tf.Tensor
            The forward rate at time t.
        """
        term1 = beta_0
        term2 = beta_1 * tf.exp(-t / tau_1)
        term3 = beta_2 * (t / tau_1) * tf.exp(-t / tau_1)
        term4 = beta_3 * (t / tau_2) * tf.exp(-t / tau_2)
        return term1 + term2 + term3 + term4

    def fit(self, maturities, forward_rates):
        """
        Fits the NSS model to the given data using TensorFlow optimizers.

        Parameters:
        ----------
        maturities : array-like
            Maturities (times to maturity).
        forward_rates : array-like
            Observed forward rates corresponding to the maturities.

        Returns:
        -------
        dict
            Fitted parameters: beta_0, beta_1, beta_2, beta_3, tau_1, tau_2.
        """
        # Convert inputs to tensors on GPU
        maturities = tf.convert_to_tensor(maturities, dtype=tf.float32)
        forward_rates = tf.convert_to_tensor(forward_rates, dtype=tf.float32)

        # Define the trainable variables
        beta_0 = tf.Variable(0.03, dtype=tf.float32)
        beta_1 = tf.Variable(-0.02, dtype=tf.float32)
        beta_2 = tf.Variable(0.02, dtype=tf.float32)
        beta_3 = tf.Variable(0.01, dtype=tf.float32)
        tau_1 = tf.Variable(1.0, dtype=tf.float32)
        tau_2 = tf.Variable(1.5, dtype=tf.float32)

        # Define the loss function
        def loss_fn():
            model_rates = self.forward_rate(
                maturities, beta_0, beta_1, beta_2, beta_3, tau_1, tau_2
            )
            return tf.reduce_mean((model_rates - forward_rates) ** 2)

        # Set up the optimizer
        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        # Optimize the parameters
        for step in range(1000):  # Maximum 1000 iterations
            optimizer.minimize(loss_fn, var_list=[beta_0, beta_1, beta_2, beta_3, tau_1, tau_2])
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss_fn().numpy()}")

        # Store the fitted parameters
        self.beta_0 = beta_0.numpy()
        self.beta_1 = beta_1.numpy()
        self.beta_2 = beta_2.numpy()
        self.beta_3 = beta_3.numpy()
        self.tau_1 = tau_1.numpy()
        self.tau_2 = tau_2.numpy()

        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "tau_1": self.tau_1,
            "tau_2": self.tau_2,
        }

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
        if any(param is None for param in [self.beta_0, self.beta_1, self.beta_2, self.beta_3, self.tau_1, self.tau_2]):
            raise ValueError("Model parameters not fitted yet")

        maturities = tf.convert_to_tensor(maturities, dtype=tf.float32)
        return self.forward_rate(
            maturities, self.beta_0, self.beta_1, self.beta_2, self.beta_3, self.tau_1, self.tau_2
        )