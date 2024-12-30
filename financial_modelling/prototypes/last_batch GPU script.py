import logging
from torch import device, tensor, float32, optim, no_grad, cuda, sqrt
import numpy as np
import matplotlib.pyplot as plt

class NonLinearModel:
    def __init__(self, initial_params, device_index=0):
        if cuda.is_available():
            self.device = device(f"cuda:{device_index}")
            logging.info(f"Using GPU: {cuda.get_device_name(device_index)}")
        else:
            self.device = device("cpu")
            logging.info("GPU not available. Using CPU.")
        
        # Convert initial_params to a 2D tensor for multiple sets
        self.initial_params = tensor(initial_params, requires_grad=True, dtype=float32, device=self.device)
        self.num_sets = self.initial_params.shape[0]  # Number of parameter sets
        self.fitted_params = None
        self.param_evolution = []  # List to store parameter evolution

    def fit(self, x_train_list, y_train_list, maturities, epochs=1000, learning_rate=0.01, log_interval=50):
        """
        Fit the model to the given data using Adam optimizer with a weighted MSE loss.

        Args:
            x_train_list (list of Tensors): List of input data for each parameter set.
            y_train_list (list of Tensors): List of target data for each parameter set.
            maturities (Tensor): Maturities for each parameter set.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate for optimization.
            log_interval (int): Epoch interval for logging.
        """
        maturities = maturities.to(self.device, dtype=float32)
        
        # Adam optimizer
        optimizer = optim.Adam([self.initial_params], lr=learning_rate)
        
        # Calculate dataset weights based on sizes
        dataset_sizes = tensor([x_train.shape[0] for x_train in x_train_list], dtype=float32, device=self.device)
        weights = dataset_sizes / dataset_sizes.sum()  # Normalize to create weights

        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()

            # Shuffle the dataset order
            indices = np.random.permutation(self.num_sets)

            for i in indices:
                x_train = x_train_list[i].to(self.device, dtype=float32)
                y_train = y_train_list[i].to(self.device, dtype=float32)

                # Compute predictions and loss
                y_pred = self.functional_form(x_train, self.initial_params[i], maturities[i])
                mse_loss = ((y_pred - y_train) ** 2).mean()  # Compute MSE
                weighted_loss = weights[i] * mse_loss  # Apply weight based on dataset size

                # Accumulate loss
                weighted_loss.backward(retain_graph=True)
                total_loss += weighted_loss.item()

            optimizer.step()

            if (epoch + 1) % log_interval == 0:
                self.param_evolution.append(self.initial_params.detach().cpu().numpy().copy())
                logging.info(f"Epoch {epoch + 1}/{epochs}, Total Weighted Loss: {total_loss:.4f}")


        self.fitted_params = self.initial_params.detach().cpu().numpy()
        logging.info("Model fitting complete.")
        return self.fitted_params


    def plot_param_evolution(self):
        """
        Plot the evolution of parameters during training.
        """
        num_epochs = len(self.param_evolution)
        param_evolution_array = np.array(self.param_evolution)  # Shape: (num_epochs, num_sets, num_params)

        fig, axes = plt.subplots(self.num_sets, 1, figsize=(8, 4 * self.num_sets))
        if self.num_sets == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one parameter set

        for i in range(self.num_sets):
            for param_idx in range(self.initial_params.shape[1]):  # Iterate over parameters
                axes[i].plot(
                    range(1, num_epochs + 1),
                    param_evolution_array[:, i, param_idx],
                    label=f"Param {param_idx + 1}",
                )
            axes[i].set_title(f"Parameter Set {i + 1}")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Parameter Value")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def predict(self, x_test_list, maturities):
        """
        Predict output for the given input data.
        """
        maturities = maturities.to(self.device, dtype=float32)
        predictions = []
        with no_grad():
            for i in range(self.num_sets):
                x_test = x_test_list[i].to(self.device, dtype=float32)
                y_pred = self.functional_form(x_test, self.initial_params[i], maturities[i])
                predictions.append(y_pred.cpu())
        return predictions

    def plot_results(self, x_train_list, y_train_list, y_pred_list, num_sets_to_plot=5):
        """
        Plot the training data vs. fitted model predictions.

        Args:
            x_train_list (list of Tensors): List of input data for each parameter set.
            y_train_list (list of Tensors): List of target data for each parameter set.
            y_pred_list (list of Tensors): List of predicted data for each parameter set.
            num_sets_to_plot (int): Number of parameter sets to plot.
        """
        num_sets_to_plot = min(num_sets_to_plot, self.num_sets)
        fig, axes = plt.subplots(num_sets_to_plot, 1, figsize=(8, 4 * num_sets_to_plot))
        if num_sets_to_plot == 1:
            axes = [axes]  # Ensure axes is iterable when num_sets_to_plot is 1

        for i in range(num_sets_to_plot):
            x_train = x_train_list[i].cpu().numpy()
            y_train = y_train_list[i].cpu().numpy()
            y_pred = y_pred_list[i].cpu().numpy()

            axes[i].plot(x_train, y_train, label="Training Data", marker="o", linestyle="None")
            axes[i].plot(x_train, y_pred, label="Fitted Model", linestyle="-")
            axes[i].set_title(f"Parameter Set {i + 1}")
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

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
        total_variance = a + b * (rho * delta + sqrt(delta**2 + sigma**2))
        return sqrt(total_variance) / sqrt(maturity)

# Example Usage with Parameter Evolution Tracking
amount_of_data = 3  # Number of parameter sets
x_train_list = [tensor(np.linspace(-0.2, 0.2, 50*(i+1)), dtype=float32) for i in range(amount_of_data)]
maturities = tensor([i+1 for i in range(amount_of_data)], dtype=float32)
y_train_list = [
    NonLinearModel.functional_form(x, tensor([0.04+np.random.random()/10,
                                              0.1+np.random.random()/10,
                                               -0.3+np.random.random()/10,
                                                0.0+np.random.random()/10,
                                                0.2+np.random.random()/10], dtype=float32), maturities[i])
    for i,x in enumerate(x_train_list)
    ]
initial_params = np.array([[0.04, 0.1, -0.3, 0.0, 0.2] for _ in range(amount_of_data)])
initial_params = tensor(initial_params, dtype=float32)

# Initialize Model
model = NonLinearModel(initial_params)

# Fit Model
fitted_params = model.fit(x_train_list, y_train_list, maturities, epochs=250, log_interval=1)

# Predict and Plot Results
y_pred_list = model.predict(x_train_list, maturities)
model.plot_results(x_train_list, y_train_list, y_pred_list, num_sets_to_plot=amount_of_data)

# Plot Parameter Evolution
#model.plot_param_evolution()
