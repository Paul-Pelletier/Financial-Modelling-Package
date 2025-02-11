import itertools
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from financial_modelling.modelling.RegularizedSVIModel import RegularizedSVIModel
from financial_modelling.big_data_pipelines.RegularizedSVIPipeline import RegularizedSVICalibrationPipeline
import torch

class HyperparameterGridSearchPipeline(RegularizedSVICalibrationPipeline):
    def __init__(self, data_fetcher, preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/OutputFiles"):
        super().__init__(data_fetcher, preprocessor, date, output_folder)
        self.hyperparameter_results = []

    def fit_model_with_hyperparams(self, preprocessed_data, lr, epochs, regularization_strength, lambda_decay):
        """
        Fit the SVI model with specified hyper-parameters.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data.
        - lr (float): Learning rate.
        - epochs (int): Number of epochs for optimization.
        - regularization_strength (float): Regularization strength for parameter differences.
        - lambda_decay (float): Weight decay factor for short-term prioritization.

        Returns:
        - dict: Fitted model parameters for each residual maturity.
        - float: Evaluation metric (e.g., MSE).
        """
        # Select the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pass the device to the RegularizedSVIModel
        model = RegularizedSVIModel(device=device)

        train_data = {
            'log_moneyness': preprocessed_data["Log_Moneyness"].values,
            'total_variance': (preprocessed_data["Implied_Volatility"].values ** 2) * preprocessed_data["Residual_Maturity"].values,
            'residual_maturity': preprocessed_data["Residual_Maturity"].values
        }

        logging.info(f"Fitting model with lr={lr}, epochs={epochs}, regularization_strength={regularization_strength}, lambda_decay={lambda_decay}")

        # Fit the model
        model_params = model.fit(
            train_data['log_moneyness'],
            train_data['total_variance'],
            train_data['residual_maturity'],
            lr=lr,
            epochs=epochs,
            regularization_strength=regularization_strength,
            lambda_decay=lambda_decay
        )

        # Calculate evaluation metric (MSE)
        residuals = []
        try:
            # Convert maturities in model_params to a NumPy array for robust matching
            model_maturities = np.array(list(model_params.keys()), dtype=np.float32)

            for maturity in train_data['residual_maturity']:
                # Find the closest match in model_params keys
                closest_match_idx = np.argmin(np.abs(model_maturities - maturity))
                closest_match = model_maturities[closest_match_idx]

                # Check if the closest match is within a reasonable tolerance
                if not np.isclose(maturity, closest_match, atol=1e-6):
                    logging.warning(f"No matching parameters found for maturity {maturity}. Skipping.")
                    continue

                # Retrieve the exact key from the dictionary
                closest_key = list(model_params.keys())[closest_match_idx]

                # Validate the key exists
                if closest_key not in model_params:
                    logging.warning(f"No matching parameters found for maturity {maturity}. Skipping.")
                    continue

                # Retrieve matched parameters
                params = model_params[closest_key]
                a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']

                # Filter data for this maturity
                mask = train_data['residual_maturity'] == maturity
                log_moneyness_subset = train_data['log_moneyness'][mask]
                total_variance_subset = train_data['total_variance'][mask]

                # Compute SVI variance
                term1 = rho * (log_moneyness_subset - m)
                term2 = np.sqrt((log_moneyness_subset - m) ** 2 + sigma ** 2)
                model_variance = a + b * (term1 + term2)

                residuals.extend((model_variance - total_variance_subset) ** 2)

            mse = np.nanmean(residuals)  # Safely handle NaN values
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            mse = float('inf')

        logging.info(f"MSE for lr={lr}, epochs={epochs}, regularization_strength={regularization_strength}, lambda_decay={lambda_decay}: {mse}")

        # Include fitted parameters in hyperparameter results
        result = {
            'lr': lr,
            'epochs': epochs,
            'regularization_strength': regularization_strength,
            'lambda_decay': lambda_decay,
            'mse': mse,
            'fitted_params': model_params  # Add fitted params here
        }

        return result


    def run_grid_search(self, preprocessed_data, hyperparameter_grid):
        """
        Run the grid search over the hyper-parameter grid.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data.
        - hyperparameter_grid (dict): Dictionary defining hyper-parameter ranges.

        Returns:
        - dict: Best hyper-parameter configuration and its evaluation metric.
        """
        best_params = None
        best_mse = float('inf')

        # Create combinations of hyper-parameters
        hyperparameter_combinations = list(itertools.product(
            hyperparameter_grid['lr'],
            hyperparameter_grid['epochs'],
            hyperparameter_grid['regularization_strength'],
            hyperparameter_grid['lambda_decay']
        ))

        for (lr, epochs, regularization_strength, lambda_decay) in hyperparameter_combinations:
            result = self.fit_model_with_hyperparams(preprocessed_data, lr, epochs, regularization_strength, lambda_decay)

            # Save hyperparameters, MSE, and detailed parameters
            self.hyperparameter_results.append(result)

            # Check if this is the best configuration
            if result['mse'] < best_mse:
                best_mse = result['mse']
                best_params = {
                    'lr': result['lr'],
                    'epochs': result['epochs'],
                    'regularization_strength': result['regularization_strength'],
                    'lambda_decay': result['lambda_decay']
                }

        logging.info(f"Best hyper-parameters: {best_params} with MSE: {best_mse}")
        return best_params, best_mse

    def run(self, hyperparameter_grid, output_folder=None):
        """
        Run the pipeline with grid search.

        Args:
        - hyperparameter_grid (dict): Hyper-parameter grid for searching.
        - output_folder (str): Output folder to save results.

        Returns:
        - None
        """
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        preprocessed_data = self.process_data(fetched_data)
        best_params, best_mse = self.run_grid_search(preprocessed_data, hyperparameter_grid)

        if output_folder is None:
            output_folder = self.output_folder

        # Save the grid search results
        results = []
        for result in self.hyperparameter_results:
            for maturity, params in result['fitted_params'].items():
                results.append({
                    'lr': result['lr'],
                    'epochs': result['epochs'],
                    'regularization_strength': result['regularization_strength'],
                    'lambda_decay': result['lambda_decay'],
                    'mse': result['mse'],  # Include the overall MSE for this hyperparameter set
                    'maturity': maturity,
                    **params  # Unpack the fitted parameters (a, b, rho, m, sigma)
                })

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_folder, f"grid_search_results_{self.date}.csv")
        results_df.to_csv(results_file, index=False)
        logging.info(f"Grid search results saved to {results_file}")

        # Save the best parameters
        best_params_file = os.path.join(output_folder, f"best_params_{self.date}.json")
        pd.Series(best_params).to_json(best_params_file)
        logging.info(f"Best hyper-parameters saved to {best_params_file}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    # Define hyper-parameter grid
    hyperparameter_grid = {
        'lr': [1e-3, 1e-2],
        'epochs': [500, 1000],
        'regularization_strength': [1e-5, 1e-4],
        'lambda_decay': [1, 10]
    }

    pipeline = HyperparameterGridSearchPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run(hyperparameter_grid, "D://")
