import os
import logging
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from financial_modelling.data_pre_processing.Preprocessor import Preprocessor
from financial_modelling.data_acquisition.database_fetcher import DataFetcher
from financial_modelling.modelling.GlobalSVIFit import GlobalSVIFit


class GlobalSVICalibrationPipeline:
    def __init__(self, data_fetcher: DataFetcher, preprocessor=Preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/GlobalOutputFiles"):
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
        self.global_model_params = None

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

    def process_data(self, data, call_limits=(0.9, 1.0), put_limits=(1.0, 1.1)):
        preprocessor = self.preprocessor_class(data)
        self.preprocessed_data = preprocessor.preprocess(call_limits, put_limits, volume_limits=1, mode="overlap")
        return self.preprocessed_data

    def fit_global_model(self, preprocessed_data, model=None):
        if model is None:
            model = GlobalSVIFit()
        train_data = pd.DataFrame({
            'Log_Moneyness': preprocessed_data["Log_Moneyness"],
            'Implied_Volatility': preprocessed_data["Implied_Volatility"],
            'Maturity': preprocessed_data["Residual_Maturity"]
        })
        model.fit(train_data)
        self.global_model_params = model.get_params()
        return model

    def plot_3d_animation(self, preprocessed_data, model):
        preprocessed_data = preprocessed_data[preprocessed_data["Residual_Maturity"]>0]
        log_moneyness = preprocessed_data["Log_Moneyness"].values
        implied_volatility = preprocessed_data["Implied_Volatility"].values
        maturity = preprocessed_data["Residual_Maturity"].values

        # Generate grid data for visualization
        log_moneyness_grid = np.linspace(log_moneyness.min(), log_moneyness.max(), 100)
        maturity_grid = np.linspace(maturity.min(), maturity.max(), 100)
        log_moneyness_mesh, maturity_mesh = np.meshgrid(log_moneyness_grid, maturity_grid)

        # Predict fitted volatilities
        fitted_volatility = model.predict(log_moneyness_mesh.flatten(), maturity_mesh.flatten())
        fitted_volatility = fitted_volatility.reshape(log_moneyness_mesh.shape)

        # Plot setup
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            maturity_mesh, log_moneyness_mesh, fitted_volatility,
            cmap='viridis', edgecolor='none', alpha=0.8
        )
        scatter = ax.scatter(maturity, log_moneyness, implied_volatility, color='red', label="Observed Data")
        ax.set_xlabel('Maturity')
        ax.set_ylabel('Log-Moneyness')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Global SVI Model Fit (Rotating)')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Animation
        def update(frame):
            ax.view_init(elev=30, azim=frame)

        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
        output_file = os.path.join(self.output_folder, f"global_fit_{self.date}.gif")
        anim.save(output_file, writer='pillow', fps=20)
        logging.info(f"3D Animation saved to {output_file}")

    def run(self, output_folder=None):
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        preprocessed_data = self.process_data(fetched_data)
        fitted_model = self.fit_global_model(preprocessed_data)

        if output_folder is None:
            output_folder = self.output_folder

        output_file = os.path.join(output_folder, f"global_output_{self.date}.csv")
        pd.DataFrame([fitted_model.get_params()]).to_csv(output_file, index=False)
        logging.info(f"Global model results saved to {output_file}")

        # Create the 3D animation
        self.plot_3d_animation(preprocessed_data, fitted_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = GlobalSVICalibrationPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run("D://")
