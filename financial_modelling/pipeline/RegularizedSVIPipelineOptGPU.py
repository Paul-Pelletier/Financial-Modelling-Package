import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from financial_modelling.modelling.RegularizedSVIModelOptGPU import RegularizedSVIModel
import torch
from torch.cuda.amp import autocast, GradScaler

class RegularizedSVICalibrationPipeline:
    def __init__(self, data_fetcher, preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/OutputFiles"):
        """
        Initialize the pipeline with GPU support.

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()

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
        SELECT TOP(6302) *
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

    def process_data(self, data, call_limits=(0.9, 1.1), put_limits=(0.9, 1.1)):
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
        self.model = model
        if self.model is None:
            self.model = RegularizedSVIModel(device=self.device)

        train_data = {
            'log_moneyness': torch.tensor(preprocessed_data["Log_Moneyness"].values, dtype=torch.float32, device=self.device),
            'total_variance': torch.tensor((preprocessed_data["Implied_Volatility"].values ** 2) * preprocessed_data["Residual_Maturity"].values, dtype=torch.float32, device=self.device),
            'residual_maturity': torch.tensor(preprocessed_data["Residual_Maturity"].values, dtype=torch.float32, device=self.device),
            'quote_unixtime': torch.tensor(preprocessed_data["QUOTE_UNIXTIME"].values, dtype=torch.float32, device=self.device),
            'expire_date': torch.tensor(preprocessed_data["EXPIRE_UNIX"].values, dtype=torch.float32, device=self.device),
        }

        logging.info("Starting model fitting...")
        logging.info(f"Training data sample: {preprocessed_data.head()}")

        # Use mixed precision training with the correct device_type
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            self.model_params = self.model.fit(
                log_moneyness=train_data['log_moneyness'],
                total_variance=train_data['total_variance'],
                residual_maturity=train_data['residual_maturity'],
                quote_unixtime=train_data['quote_unixtime'],
                expire_date=train_data['expire_date'],
                lr=1e-2,
                epochs=200,
                regularization_strength=1e-4,
                lambda_decay=0.5
            )

        if not self.model_params:
            logging.error("Model fitting failed; no parameters returned.")
        else:
            logging.info("Model fitting completed.")
            logging.info(f"Fitted model parameters: {self.model_params}")

        return self.model_params

    def run(self, output_folder=None):
        """
        Run the entire pipeline.

        Args:
        - output_folder (str): Output folder to save results.

        Returns:
        - None
        """
        output_folder = output_folder or self.output_folder

        logging.info(f"Using device: {self.device}")

        # Step 1: Fetch data
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        # Step 2: Process data
        preprocessed_data = self.process_data(fetched_data)

        # Step 3: Initialize model
        model = RegularizedSVIModel(device=self.device)

        # Step 4: Fit the model
        fitted_params = self.fit_model(preprocessed_data, model)
        if not fitted_params:
            logging.error("No fitted parameters returned. Exiting pipeline.")
            return

        logging.info(f"Fitted parameters: {fitted_params}")

        # Step 5: Save results
        records = [
            {
                "QUOTE_UNIXTIME": quote_unixtime,
                "EXPIRE_DATE": expire_date,
                "Maturity": maturity,
                "a": params["a"],
                "b": params["b"],
                "rho": params["rho"],
                "m": params["m"],
                "sigma": params["sigma"],
            }
            for (quote_unixtime, expire_date, maturity), params in fitted_params.items()
        ]

        if not records:
            logging.error("No records to save. Skipping output.")
            return

        df_output = pd.DataFrame(records).sort_values("Maturity")
        output_file = os.path.join(output_folder, f"output_{self.date}.csv")
        df_output.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = RegularizedSVICalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    from datetime import datetime
    start = datetime.now()
    pipeline.run("D://")
    print("Elapsed time: ", datetime.now() - start)