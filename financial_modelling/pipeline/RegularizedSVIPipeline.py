import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from financial_modelling.modelling.RegularizedSVIModel import RegularizedSVIModel
import torch

class RegularizedSVICalibrationPipeline:
    def __init__(self, data_fetcher, preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/OutputFiles"):
        """
        Initialize the pipeline.

        Args:
        - data_fetcher: Instance of a data fetcher class.
        - preprocessor: Preprocessor class for data processing.
        - date (str): Unix timestamp in string format.
        - output_folder (str): Path to save output files.
        """
        self.date = date
        self.db_config = {
            'server': 'DESKTOP-DK79R4I',
            'database': 'DataMining',
        }
        self.connection_string = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={self.db_config['server']};"
            f"DATABASE={self.db_config['database']};"
            f"Trusted_Connection=yes;"
        )
        self.fetcher = data_fetcher(self.connection_string, use_sqlalchemy=False)
        self.preprocessor_class = preprocessor
        self.output_folder = output_folder

        # Internal attributes for pipeline flow
        self.data = None
        self.preprocessed_data = None
        self.model_params = None
        self.model = None

    def fetch_data(self):
        """
        Fetch data for the given date from the database.

        Returns:
        - pd.DataFrame: Fetched data.
        """
        query = f"""
        SELECT *
        FROM [DataMining].[dbo].[OptionData]
        WHERE [QUOTE_UNIXTIME] = '{self.date}'
        """
        us_eastern = pytz.timezone("US/Eastern")
        readable_time = datetime.fromtimestamp(int(self.date), us_eastern).strftime('%d-%m-%Y %H:%M')
        logging.info(f"Fetching data for date: {readable_time}")

        try:
            data = self.fetcher.fetch(query)
            if data.empty:
                logging.warning("No data fetched.")
                self.data = pd.DataFrame()
            else:
                self.data = data
                logging.info(f"Data fetched successfully for: {readable_time}")
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")
            self.data = pd.DataFrame()

        return self.data

    def process_data(self, data, call_limits=(0.8, 1.1), put_limits=(0.9, 1.2)):
        """
        Process the fetched data.

        Args:
        - data (pd.DataFrame): Raw data fetched from the database.
        - call_limits (tuple): Moneyness limits for calls.
        - put_limits (tuple): Moneyness limits for puts.

        Returns:
        - pd.DataFrame: Processed data.
        """
        preprocessor = self.preprocessor_class(data)
        self.preprocessed_data = preprocessor.preprocess(call_limits, put_limits, volume_limits=1, mode="split")
        return self.preprocessed_data

    def fit_model(self, preprocessed_data, model=None):
        """
        Fit the processed data to a given model, ensuring per-maturity parameterization.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data.
        - model (RegularizedSVIModel): SVI model instance for fitting.

        Returns:
        - dict: Fitted model parameters for each residual maturity.
        """
        if model is None:
            self.model = RegularizedSVIModel()

        train_data = {
            'log_moneyness': preprocessed_data["Log_Moneyness"].values,
            'total_variance': (preprocessed_data["Implied_Volatility"].values ** 2) * preprocessed_data["Residual_Maturity"].values,
            'residual_maturity': preprocessed_data["Residual_Maturity"].values
        }

        logging.info("Starting model fitting...")
        logging.info(f"Training data sample: {preprocessed_data.head()}")

        # Fit the model
        fitted_params = model.fit(
            train_data['log_moneyness'],
            train_data['total_variance'],
            train_data['residual_maturity'],
            lr=1e-2,
            epochs=200,
            regularization_strength=1e-4,
            lambda_decay = 0.5
        )

        # Store parameters using stringified maturities
        self.model_params = {
            f"{maturity:.6f}": params for maturity, params in fitted_params.items()
        }

        logging.info("Model fitting completed.")
        logging.info(f"Fitted model parameters: {self.model_params}")

        return self.model_params


    def plot_fitted_model(self, preprocessed_data):
        """
        Plot the fitted SVI model across residual maturity.

        Args:
        - preprocessed_data (pd.DataFrame): Training data used for fitting.
        - model (RegularizedSVIModel): Fitted SVI model.
        """
        log_moneyness = preprocessed_data["Log_Moneyness"].values
        residual_maturity = preprocessed_data["Residual_Maturity"].values
        implied_volatility = preprocessed_data["Implied_Volatility"].values

        # Generate grid for predictions
        log_moneyness_grid = np.linspace(log_moneyness.min(), log_moneyness.max(), 100)
        maturity_grid = np.linspace(residual_maturity.min(), residual_maturity.max(), 100)
        log_moneyness_mesh, maturity_mesh = np.meshgrid(log_moneyness_grid, maturity_grid)

        # Prepare fitted volatility grid
        fitted_volatility = np.zeros_like(log_moneyness_mesh)
        for i, maturity in enumerate(np.unique(residual_maturity)):
            key = f"{maturity:.6f}"  # Use stringified maturity as key
            if key in self.model_params:
                params = self.model_params[key]
                a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']

                term1 = rho * (log_moneyness_mesh[i, :] - m)
                term2 = np.sqrt((log_moneyness_mesh[i, :] - m) ** 2 + sigma ** 2)
                total_variance = a + b * (term1 + term2)
                fitted_volatility[i, :] = np.sqrt(total_variance / maturity)

        # Plot results
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(residual_maturity, log_moneyness, implied_volatility, color="red", label="Observed")
        ax.plot_surface(maturity_mesh, log_moneyness_mesh, fitted_volatility, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Residual Maturity")
        ax.set_ylabel("Log-Moneyness")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("Fitted SVI Model")
        plt.legend()
        plt.show()

    def plot_individual_expiries(self, preprocessed_data):
        """
        Plot the fitted SVI models for individual expiries.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data with log-moneyness and implied volatilities.
        - model (RegularizedSVIModel): Fitted SVI model.
        """
        unique_maturities = np.unique(preprocessed_data["Residual_Maturity"].values)

        # Convert keys of model_params to a NumPy array for robust matching
        model_maturities = np.array(list(self.model_params.keys()), dtype=np.float32)

        for maturity in unique_maturities:
            # Find the closest match in model_params keys
            closest_match_idx = np.argmin(np.abs(model_maturities - maturity))
            closest_match = model_maturities[closest_match_idx]

            # Retrieve the exact key from the dictionary
            closest_key = list(self.model_params.keys())[closest_match_idx]

            # Check if the closest match is within a reasonable tolerance
            if not np.isclose(maturity, closest_match, atol=1e-6):
                logging.warning(f"No matching parameters found for maturity {maturity:.6f}. Skipping.")
                continue

            subset = preprocessed_data[preprocessed_data["Residual_Maturity"] == maturity]
            log_moneyness = subset["Log_Moneyness"].values
            implied_volatility = subset["Implied_Volatility"].values

            # Retrieve fitted parameters for this maturity
            params = self.model_params[closest_key]
            a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']

            # Validate parameters
            if b <= 0 or sigma <= 0 or not (-1 <= rho <= 1):
                logging.warning(f"Invalid SVI parameters for maturity {maturity:.6f}. Skipping.")
                continue

            # Generate a smooth curve
            log_moneyness_grid = np.linspace(log_moneyness.min() - 0.1, log_moneyness.max() + 0.1, 500)
            term1 = rho * (log_moneyness_grid - m)
            term2 = np.sqrt((log_moneyness_grid - m) ** 2 + sigma ** 2)
            total_variance = a + b * (term1 + term2)
            fitted_volatility = np.sqrt(np.maximum(total_variance / maturity, 0))  # Ensure non-negative variance

            # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(log_moneyness, implied_volatility, color="blue", label="Observed")
            plt.plot(log_moneyness_grid, fitted_volatility, color="red", label="Fitted", linewidth=2)
            plt.title(f"Expiry with Residual Maturity: {maturity:.6f}")
            plt.xlabel("Log-Moneyness")
            plt.ylabel("Implied Volatility")
            plt.legend()
            plt.grid(True)
            plt.show()

    def run(self, output_folder=None):
        """
        Run the entire pipeline.

        Args:
        - output_folder (str): Output folder to save results.

        Returns:
        - None
        """
        # Check for GPU availability and set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Step 1: Fetch data from the database
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        # Step 2: Process the fetched data
        preprocessed_data = self.process_data(fetched_data)

        # Step 3: Initialize the RegularizedSVIModel with the selected device
        model = RegularizedSVIModel(device=device)

        # Step 4: Fit the model to the preprocessed data
        fitted_params = self.fit_model(preprocessed_data, model)

        # Step 5: Save results
        if output_folder is None:
            output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)

        # Build a DataFrame with one row per maturity
        records = [
            {
                "Maturity": float(maturity_str),
                "a": params["a"],
                "b": params["b"],
                "rho": params["rho"],
                "m": params["m"],
                "sigma": params["sigma"],
            }
            for maturity_str, params in fitted_params.items()
        ]

        # Convert to DataFrame
        df_output = pd.DataFrame(records).sort_values("Maturity")

        # Define the output file path
        output_file = os.path.join(output_folder, f"output_{self.date}.csv")

        # Save the DataFrame to a CSV file
        df_output.to_csv(output_file, index=False)

        logging.info(f"Results saved to {output_file}")

        # (Optional) Step 6: Visualize the results with plots
        #self.plot_fitted_model(preprocessed_data)
        #self.plot_individual_expiries(preprocessed_data)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = RegularizedSVICalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run()
