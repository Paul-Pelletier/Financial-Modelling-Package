import os
import logging
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from financial_modelling.data_pre_processing.Preprocessor import Preprocessor
from financial_modelling.data_acquisition.database_fetcher import DataFetcher
from financial_modelling.modelling.GatheralVolatilityFit import GatheralQuadraticModel

class GatheralCalibrationPipeline:
    def __init__(self, data_fetcher: DataFetcher, preprocessor=Preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/GatheralOutputFiles"):
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
        self.data = None
        self.preprocessed_data = None
        self.fitted_models = {}

    def fetch_data(self):
        query = f"""
        SELECT *
        FROM [DataMining].[dbo].[OptionData]
        WHERE [QUOTE_UNIXTIME] = '{self.date}'
        """
        try:
            data = self.fetcher.fetch(query)
            self.data = data if not data.empty else pd.DataFrame()
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")
            self.data = pd.DataFrame()
        return self.data

    def process_data(self, data, call_limits=(0.8, 1.2), put_limits=(0.8, 1.2)):
        preprocessor = self.preprocessor_class(data)
        self.preprocessed_data = preprocessor.preprocess(call_limits, put_limits, volume_limits=1, mode="split")
        return self.preprocessed_data

    def fit_gatheral_model(self, preprocessed_data):
        maturities = preprocessed_data["Residual_Maturity"].unique()
        for maturity in maturities:
            data_for_maturity = preprocessed_data[preprocessed_data["Residual_Maturity"] == maturity]
            model = GatheralQuadraticModel()
            train_data = pd.DataFrame({
                'STRIKE_DISTANCE': data_for_maturity["STRIKE_DISTANCE"],
                'Implied_Volatility': data_for_maturity["Implied_Volatility"]
            })
            model.fit(train_data)
            self.fitted_models[maturity] = model

    def plot_fitted_smiles(self, preprocessed_data):
        maturities = preprocessed_data["Residual_Maturity"].unique()
        num_maturities = len(maturities)
        num_cols = 7  # Maximum number of columns
        num_rows = math.ceil(num_maturities / num_cols)  # Number of rows needed

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axs = axs.flatten()  # Flatten to handle subplots as a 1D list

        for ax, maturity in zip(axs, maturities):
            data_for_maturity = preprocessed_data[preprocessed_data["Residual_Maturity"] == maturity]
            log_moneyness = data_for_maturity["STRIKE_DISTANCE"]
            implied_volatility = data_for_maturity["Implied_Volatility"]

            # Predict the smile using the fitted model
            fitted_model = self.fitted_models[maturity]
            fitted_volatility = fitted_model.predict(log_moneyness)

            # Plot observed and fitted smiles
            ax.scatter(log_moneyness, implied_volatility, label="Observed", color="blue", alpha=0.6)
            ax.plot(log_moneyness, fitted_volatility, label="Fitted Smile", color="red")
            ax.set_title(f"Maturity: {maturity:.2f}")
            ax.set_xlabel("STRIKE_DISTANCE")
            ax.set_ylabel("Implied Volatility")
            ax.legend()

        # Hide unused subplots if num_maturities < num_rows * num_cols
        for unused_ax in axs[num_maturities:]:
            unused_ax.axis('off')

        plt.tight_layout()
        output_file = os.path.join(self.output_folder, f"fitted_smiles_{self.date}.png")
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Fitted smiles plot saved to {output_file}")

    def run(self, output_folder=None):
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        preprocessed_data = self.process_data(fetched_data)
        self.fit_gatheral_model(preprocessed_data)

        if output_folder is None:
            output_folder = self.output_folder

        # Plot all fitted smiles
        self.plot_fitted_smiles(preprocessed_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = GatheralCalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run("D://")
