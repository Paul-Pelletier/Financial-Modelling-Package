import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
from financial_modelling.data_pre_processing.Preprocessor import Preprocessor
from financial_modelling.data_acquisition.database_fetcher import DataFetcher
from financial_modelling.modelling.GlobalSVIFit import GlobalSVIFit

class GlobalSVICalibrationPipeline:
    def __init__(self, data_fetcher: DataFetcher, preprocessor=Preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/GlobalOutputFiles"):
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
        self.global_model_params = None

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

    def preprocess_data(self, data):
        """
        Preprocess the fetched data for all expiries.

        Args:
        - data (pd.DataFrame): Raw data fetched from the database.

        Returns:
        - pd.DataFrame: Processed data.
        """
        preprocessor = self.preprocessor_class(data)
        displacement = 0.2
        self.preprocessed_data = preprocessor.preprocess(call_limits = (1-displacement,1+displacement), put_limits = (1-displacement,1+displacement), volume_limits=1, mode="split")
        return self.preprocessed_data

    def fit_global_model(self, preprocessed_data, model=None):
        """
        Fit the processed data to a global SVI model.

        Args:
        - preprocessed_data (pd.DataFrame): Processed data.
        - model (GlobalSVIFit): Global SVI model instance for fitting.

        Returns:
        - dict: Fitted global model parameters.
        """
        if model is None:
            model = GlobalSVIFit()

        train_data = pd.DataFrame({
            'Log_Moneyness': preprocessed_data["Log_Moneyness"],
            'Implied_Volatility': preprocessed_data["Implied_Volatility"],
            'Maturity': preprocessed_data["Residual_Maturity"]
        })
        model.fit(train_data)
        self.global_model_params = model.get_params()
        return self.global_model_params

    def run(self, output_folder=None):
        """
        Run the entire pipeline.

        Returns:
        - None: Outputs results to a CSV file.
        """
        fetched_data = self.fetch_data()

        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        preprocessed_data = self.preprocess_data(fetched_data)
        global_model_params = self.fit_global_model(preprocessed_data)

        output_file = os.path.join(output_folder or self.output_folder, f"global_output_{self.date}.csv")
        output_data = pd.DataFrame([global_model_params])
        output_data.to_csv(output_file, index=False)
        logging.info(f"Global SVI fit parameters saved to {output_file}")
        print("Global SVI Fit Parameters:")
        print(global_model_params)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    # Run the pipeline
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = GlobalSVICalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run("D://")
