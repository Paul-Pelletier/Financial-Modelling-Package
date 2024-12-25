import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from financial_modelling.modelling.OptimizedRegularizedSVIModel import OptimizedRegularizedSVIModel
import torch


class OptimizedRegularizedSVICalibrationPipeline:
    def __init__(self, data_fetcher, preprocessor, date="1546440960", output_folder="E:/OutputParamsFiles/OutputFiles"):
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
        Fit the processed data to the SVI model.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data.
        - model (OptimizedRegularizedSVIModel, optional): Instance of the SVI model.

        Returns:
        - dict: Fitted parameters.
        """
        if model is None:
            logging.info("Model not provided. Initializing a new OptimizedRegularizedSVIModel instance.")
            self.model = OptimizedRegularizedSVIModel(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.model = model

        if not hasattr(self.model, 'fit'):
            raise AttributeError("Model is not properly initialized or does not have a 'fit' method.")

        train_data = {
            'log_moneyness': preprocessed_data["Log_Moneyness"].values,
            'total_variance': (preprocessed_data["Implied_Volatility"].values ** 2) * preprocessed_data["Residual_Maturity"].values,
            'residual_maturity': preprocessed_data["Residual_Maturity"].values
        }

        logging.info("Starting model fitting...")
        try:
            fitted_params = self.model.fit(
                train_data['log_moneyness'],
                train_data['total_variance'],
                train_data['residual_maturity'],
                lr=1e-3,
                epochs=10000,
                lambda_decay=0
            )
        except Exception as e:
            logging.error(f"Error during model fitting: {e}")
            raise

        self.model_params = fitted_params
        logging.info("Model fitting completed.")
        return self.model_params

    def plot_individual_expiries(self, preprocessed_data):
        """
        Plot the fitted SVI models for individual expiries.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data with log-moneyness and implied volatilities.
        """
        unique_maturities = np.unique(preprocessed_data["Residual_Maturity"].values)

        for maturity in unique_maturities:
            subset = preprocessed_data[preprocessed_data["Residual_Maturity"] == maturity]
            log_moneyness = subset["Log_Moneyness"].values
            implied_volatility = subset["Implied_Volatility"].values

            maturity_key = f"{maturity:.6f}"
            if maturity_key not in self.model_params:
                logging.warning(f"No fitted parameters found for maturity {maturity:.6f}. Skipping.")
                continue

            params = self.model_params[maturity_key]
            a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]

            log_moneyness_grid = np.linspace(log_moneyness.min() - 0.1, log_moneyness.max() + 0.1, 500)
            term1 = rho * (log_moneyness_grid - m)
            term2 = np.sqrt((log_moneyness_grid - m) ** 2 + sigma ** 2)
            total_variance = a + b * (term1 + term2)
            fitted_volatility = np.sqrt(np.maximum(total_variance / maturity, 0))

            plt.figure(figsize=(8, 6))
            plt.scatter(log_moneyness, implied_volatility, color="blue", label="Observed")
            plt.plot(log_moneyness_grid, fitted_volatility, color="red", label="Fitted", linewidth=2)
            plt.title(f"Expiry with Residual Maturity: {maturity:.6f}")
            plt.xlabel("Log-Moneyness")
            plt.ylabel("Implied Volatility")
            plt.legend()
            plt.grid(True)
            plt.show()

    def save_results(self, output_folder):
        """
        Save the model parameters to a CSV file.

        Args:
        - output_folder (str): Output folder to save results.

        Returns:
        - str: Path to the output file.
        """
        os.makedirs(output_folder, exist_ok=True)

        records = [
            {
                "Maturity": float(maturity_str),
                **params,
            }
            for maturity_str, params in self.model_params.items()
        ]

        df_output = pd.DataFrame(records).sort_values("Maturity")
        output_file = os.path.join(output_folder, f"output_{self.date}.csv")
        df_output.to_csv(output_file, index=False)

        logging.info(f"Results saved to {output_file}")
        return output_file

    def run(self, output_folder=None):
        """
        Run the entire pipeline.

        Args:
        - output_folder (str): Output folder to save results.
        """
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        preprocessed_data = self.process_data(fetched_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OptimizedRegularizedSVIModel(device=device)

        try:
            self.fit_model(preprocessed_data, model)
        except Exception as e:
            logging.error(f"Model fitting failed: {e}")
            return

        output_folder = output_folder or self.output_folder
        self.save_results(output_folder)

        try:
            self.plot_individual_expiries(preprocessed_data)
        except Exception as e:
            logging.error(f"Visualization failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = OptimizedRegularizedSVICalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run()
