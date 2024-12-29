import numpy as np
from torch import tensor, sqrt, split, float32, optim, nn, no_grad, cuda, device, is_tensor, autograd
import logging
import matplotlib.pyplot as plt
from itertools import chain
from multiprocessing import Pool
from pprint import pprint
import sys

class NonLinearModel:
    def __init__(self, initial_params=[0, 0, 0, 0, 0], device_index=0):
        if cuda.is_available():
            self.device = device(f"cuda:{device_index}")
            logging.info(f"Using GPU: {cuda.get_device_name(device_index)}")
        else:
            self.device = device("cpu")
            logging.info("GPU not available. Using CPU.")
        
        # Properly initialize initial_params as a leaf tensor
        self.initial_params = tensor([0.04, 0.1, -0.3, 0.0, 0.2], requires_grad=True, dtype=float32, device = self.device)
        self.fitted_params = None

    def fit(self, x_train, y_train, maturity, epochs=500, learning_rate=0.01):
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        optimizer = optim.Adam([self.initial_params], lr=learning_rate)
        loss_fn = nn.MSELoss()
        maturity = tensor(maturity, dtype=float32, device=self.device)
        
        # Enable anomaly detection
        autograd.set_detect_anomaly(True)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.functional_form(x_train, self.initial_params, maturity)
            loss = loss_fn(y_pred, y_train)
            try:
                loss.backward()  # Compute gradients
            except RuntimeError as e:
                print(f"Error during backward pass: {e}")
                raise
            
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        self.fitted_params = self.initial_params.detach().cpu().numpy()
        logging.info("Model fitting complete.")
        return self.fitted_params


    def predict(self, x_test, maturity):
        """
        Predict output for the given input data.
        """
        x_test = x_test.to(self.device)
        with no_grad():
            y_pred = self.functional_form(x_test, maturity, params=self.initial_params)
        return y_pred.cpu()
    
    @staticmethod
    def functional_form(x, params, maturity):
        # Ensure that `params` is a PyTorch tensor and does not detach
        if is_tensor(x):
            if not is_tensor(params):
                raise TypeError("If x is a tensor, params must also be a tensor.")
            # Perform PyTorch-based computation
            total_variance = params[0] + params[1] * (params[2] * (x - params[3]) + sqrt((x - params[3]) ** 2 + params[4] ** 2))
            return sqrt(total_variance) / sqrt(maturity)
        else:
            # Assume x is a NumPy array or scalar, and params is a compatible array
            if not isinstance(params, (list, tuple, np.ndarray)):
                raise TypeError("If x is not a tensor, params must be a list, tuple, or NumPy array.")

            # Extract parameters
            a, b, rho, m, sigma = params

            # Perform NumPy-based computation
            return np.sqrt(a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma ** 2))) / np.sqrt(maturity)

def tweak_params(params, tweak):
    result_params = []
    for i,v in enumerate(params.keys()):
        result_params.append(params[v] + np.random.normal(0,tweak))
    return result_params

def generate_params(usual_params, number_of_maturities = 10):
    #generate artifical data
    maturities = np.linspace(0.0001,10,number_of_maturities)*(1-np.exp(-0.1*np.linspace(0,10,number_of_maturities)))+0.1
    generated_params = [[*tweak_params(usual_params, 0.00001),v] for i,v in enumerate(maturities)] #generate a list of parameters for each maturity
    return generated_params

def generate_data(generated_params):
    x = [np.linspace(-0.20,0.20,10) for i in range(len(generated_params))]
    y = [NonLinearModel.functional_form(x[i], generated_params[i][:-1], generated_params[i][-1]) for i in range(len(generated_params))]

    x_flat = list(chain.from_iterable(x))
    y_flat = list(chain.from_iterable(y))
    return tensor(x_flat, dtype = float32), tensor(y_flat, dtype = float32)

def fit_model(x_train, y_train, maturity):
    model = NonLinearModel()
    model.fit(x_train, y_train, maturity)
    return model.fitted_params

def split_list(a_list, number):
    return [a_list[i:i + number] for i in range(0, len(a_list), number)]

def fit_model_worker(args):
    x, y, params = args
    maturity = params[-1]
    sys.stdout.flush()
    model = NonLinearModel()
    print("Fitting model...")
    result = model.fit(x, y, maturity)
    sys.stdout.flush()  # Force flushing again
    return result

def compare_params(generated_params, fitted_params):
    """
    Compare the generated parameters with the fitted ones.
    Args:
        generated_params (list): List of generated parameter sets.
        fitted_params (list): List of fitted parameter sets.

    Returns:
        list: A list of dictionaries containing the differences for each parameter set.
    """
    comparisons = []
    for gen, fit in zip(generated_params, fitted_params):
        comparison = {
            "a_diff": abs(gen[0] - fit[0]),
            "b_diff": abs(gen[1] - fit[1]),
            "rho_diff": abs(gen[2] - fit[2]),
            "m_diff": abs(gen[3] - fit[3]),
            "sigma_diff": abs(gen[4] - fit[4]),
        }
        comparisons.append(comparison)
    return comparisons

def main():
    svi_params = {
        "a": 0.04,  # Vertical offset (minimum implied variance)
        "b": 0.1,   # Slope (controls curvature, must be >= 0)
        "rho": -0.3, # Skew/asymmetry (-1 <= rho <= 1)
        "m": 0.0,   # Log-moneyness shift (horizontal position of the smile)
        "sigma": 0.2 # Width/curvature (must be > 0)
    }
    number_of_workers = 10
    generated_params = generate_params(svi_params, number_of_workers)
    generated_data = generate_data(generated_params)
    split_tensors = [split(generated_data[0], 10), split(generated_data[1], 10)]
    x, y = split_tensors[0], split_tensors[1]
    args = [(x[i], y[i], generated_params[i]) for i in range(len(generated_params))]

    # Parallel processing
    with Pool() as p:
        results = p.map(fit_model_worker, args)

    # Compare generated and fitted parameters
    comparisons = compare_params(generated_params, results)

    # Print or save the comparisons
    for i, comp in enumerate(comparisons):
        print(f"Comparison for worker {i}: {comp}")

    # Optionally save comparisons to a file
    with open("param_comparisons.txt", "w") as f:
        for i, comp in enumerate(comparisons):
            f.write(f"Comparison for worker {i}: {comp}\n")

if __name__ == "__main__":
    main()
