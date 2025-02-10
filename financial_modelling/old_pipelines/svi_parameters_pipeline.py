import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from financial_modelling.data_pre_processing.Preprocessor import Preprocessor
from financial_modelling.data_acquisition.database_fetcher import DataFetcher
from financial_modelling.modelling.SVIModel import SVIModel
#region

class SVICalibrationPipeline:
    def __init__(self, data_fetcher: DataFetcher, preprocessor=Preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/OutputFiles"):
        """
        Initialize the pipeline.

        Args:
        - data_fetcher (DataFetcher): Instance of a data fetcher class.
        - preprocessor (class): Preprocessor class for data processing.
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
        self.train_data = None
        self.model_params = None

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

    def process_data(self, data, expiry, call_limits=(0.9, 1.0), put_limits=(1.0, 1.1)):
        """
        Process the fetched data for a specific expiry.

        Args:
        - data (pd.DataFrame): Raw data fetched from the database.
        - expiry (int): Expiry timestamp to filter the data.
        - call_limits (tuple): Moneyness limits for calls.
        - put_limits (tuple): Moneyness limits for puts.

        Returns:
        - pd.DataFrame: Processed data.
        """
        expiry_specific_data = data[data["EXPIRE_UNIX"] == expiry]
        preprocessor = self.preprocessor_class(expiry_specific_data)
        self.preprocessed_data = preprocessor.preprocess(call_limits, put_limits, volume_limits=1, mode="split")
        return self.preprocessed_data

    def fit_model(self, preprocessed_data, model=None):
        """
        Fit the processed data to a given model.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data.
        - model (SVIModel): SVI model instance for fitting.

        Returns:
        - dict: Fitted model parameters.
        """
        if model is None:
            model = SVIModel()

        train_data = pd.DataFrame({
            'Log_Moneyness': preprocessed_data["Log_Moneyness"],
            'Implied_Volatility': preprocessed_data["Implied_Volatility"],
            'Volume': preprocessed_data["Volume"]
        })
        volume_weights = train_data['Volume'].values / train_data['Volume'].sum()
        model.fit(train_data, volume_weights=volume_weights)
        self.model_params = model.get_params()
        return self.model_params

    def plot_fitted_models(self, results):
        """
        Plot the fitted SVI models for all expiries.

        Args:
        - results (list): List of tuples (expiry, train_data, model_params).
        """
        if not results:
            logging.warning("No results to plot.")
            return

        for expiry, train_data, model_params in results:
            if train_data.empty:
                logging.warning(f"No data for expiry {expiry}. Skipping plot.")
                continue

            log_moneyness = train_data["Log_Moneyness"].values
            implied_volatility = train_data["Implied_Volatility"].values

            def svi_formula(k, a, b, rho, m, sigma):
                return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

            a, b, rho, m, sigma = model_params["a"], model_params["b"], model_params["rho"], model_params["m"], model_params["sigma"]
            log_moneyness_range = np.linspace(log_moneyness.min() - 0.2, log_moneyness.max() + 0.2, 500)
            fitted_volatility = np.sqrt(np.maximum(svi_formula(log_moneyness_range, a, b, rho, m, sigma), 0))

            plt.scatter(log_moneyness, implied_volatility, label="Observed")
            plt.plot(log_moneyness_range, fitted_volatility, label="Fitted", color="red")
            plt.legend()
            plt.show()

    def run(self, output_folder = None):
        """
        Run the entire pipeline.

        Returns:
        - None: Outputs results to a CSV file.
        """
        output_data = pd.DataFrame(columns=["QUOTE_UNIXTIME", "EXPIRE_UNIX", "a", "b", "rho", "m", "sigma"])
        fetched_data = self.fetch_data()

        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        expiries = fetched_data["EXPIRE_UNIX"].unique()
        results = []

        for expiry in expiries:
            preprocessed_data = self.process_data(fetched_data, expiry)
            model_params = self.fit_model(preprocessed_data)
            results.append((expiry, preprocessed_data, model_params))

            new_row = {
                "QUOTE_UNIXTIME": self.date,
                "EXPIRE_UNIX": expiry,
                "a": model_params["a"],
                "b": model_params["b"],
                "rho": model_params["rho"],
                "m": model_params["m"],
                "sigma": model_params["sigma"]
            }
            output_data = pd.concat([output_data, pd.DataFrame([new_row])], ignore_index=True)
        if output_folder is None:
            output_folder = self.output_folder

        output_file = os.path.join(output_folder, f"output_{self.date}.csv")
        output_data.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

        self.plot_fitted_models(results)

#endregion

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,  # Adjust level (e.g., DEBUG for detailed logs, INFO for less verbosity)
                        format="%(asctime)s - %(levelname)s - %(message)s"
                        )
    # Run the pipeline
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = SVICalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run("D://")