import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.animation import FuncAnimation
from financial_modelling.data_pre_processing.Preprocessor import Preprocessor
from financial_modelling.data_acquisition.database_fetcher import DataFetcher
from financial_modelling.modelling.VolatilityNN import VolatilityNN

class VolatilityNNPipeline:
    def __init__(self, data_fetcher: DataFetcher, preprocessor=Preprocessor, date="1546439410", output_folder="E:/OutputParamsFiles/NNOutputFiles"):
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
        self.fitted_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def process_data(self, data, call_limits=(0.8, 1.2), put_limits=(0.95, 1.05)):
        preprocessor = self.preprocessor_class(data)
        self.preprocessed_data = preprocessor.preprocess(call_limits, put_limits, volume_limits=1, mode="split")
        
        # Normalize STRIKE_DISTANCE and Residual_Maturity
        self.preprocessed_data['STRIKE_DISTANCE_NORM'] = (
            (self.preprocessed_data['STRIKE_DISTANCE'] - self.preprocessed_data['STRIKE_DISTANCE'].min()) /
            (self.preprocessed_data['STRIKE_DISTANCE'].max() - self.preprocessed_data['STRIKE_DISTANCE'].min())
        )
        self.preprocessed_data['RESIDUAL_MATURITY_NORM'] = (
            (self.preprocessed_data['Residual_Maturity'] - self.preprocessed_data['Residual_Maturity'].min()) /
            (self.preprocessed_data['Residual_Maturity'].max() - self.preprocessed_data['Residual_Maturity'].min())
        )

        # Add Polynomial Features
        self.preprocessed_data['STRIKE_DISTANCE_SQR'] = self.preprocessed_data['STRIKE_DISTANCE'] ** 2
        self.preprocessed_data['RESIDUAL_MATURITY_SQR'] = self.preprocessed_data['Residual_Maturity'] ** 2

        # Drop NaN rows
        self.preprocessed_data = self.preprocessed_data.dropna(axis=0, how='any')

        return self.preprocessed_data

    def fit_neural_network(self, preprocessed_data, epochs=1000, batch_size=200, lr=0.001, lambda_weight=0.1):
        # Features used for training
        features = ['STRIKE_DISTANCE_NORM', 'RESIDUAL_MATURITY_NORM', 'STRIKE_DISTANCE_SQR', 'RESIDUAL_MATURITY_SQR']
        inputs = torch.tensor(preprocessed_data[features].values, dtype=torch.float32).to(self.device)
        targets = torch.tensor(preprocessed_data["Implied_Volatility"].values, dtype=torch.float32).view(-1, 1).to(self.device)

        # Create weights for Residual Maturity
        weights = torch.exp(-lambda_weight * torch.tensor(preprocessed_data["Residual_Maturity"].values, dtype=torch.float32)).view(-1, 1).to(self.device)

        # DataLoader
        dataset = TensorDataset(inputs, targets, weights)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the model
        model = VolatilityNN(input_dim=len(features)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training Loop
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch in dataloader:
                x, y, w = batch
                optimizer.zero_grad()
                y_pred = model(x)
                loss = (w * (y_pred - y) ** 2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 50 == 0:
                logging.info(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f}")

        self.fitted_model = model

    def plot_individual_smiles(self, preprocessed_data, num_cols=4):
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")

        maturities = preprocessed_data["Residual_Maturity"].unique()
        num_rows = -(-len(maturities) // num_cols)  # Calculate rows for subplots

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
        axs = axs.flatten()

        for ax, maturity in zip(axs, maturities):
            # Filter data for this maturity
            data = preprocessed_data[preprocessed_data["Residual_Maturity"] == maturity]
            strike_distance = torch.tensor(data["STRIKE_DISTANCE"].values, dtype=torch.float32).view(-1, 1).to(self.device)
            implied_volatility = data["Implied_Volatility"].values

            # Predict using the fitted model
            features = ['STRIKE_DISTANCE_NORM', 'RESIDUAL_MATURITY_NORM', 'STRIKE_DISTANCE_SQR', 'RESIDUAL_MATURITY_SQR']
            inputs = torch.tensor(data[features].values, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                fitted_volatility = self.fitted_model(inputs).cpu().numpy()

            # Plot observed and fitted data
            ax.scatter(strike_distance.cpu().numpy(), implied_volatility, label="Observed", color="blue", alpha=0.6)
            ax.plot(strike_distance.cpu().numpy(), fitted_volatility, label="Fitted", color="red")
            ax.set_title(f"Maturity: {maturity:.2f}")
            ax.set_xlabel("Strike Distance")
            ax.set_ylabel("Implied Volatility")
            ax.legend()

        # Hide unused axes
        for unused_ax in axs[len(maturities):]:
            unused_ax.axis('off')

        plt.tight_layout()
        output_file = os.path.join(self.output_folder, f"fitted_smiles_{self.date}.png")
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Individual smiles saved to {output_file}")

    def run(self, output_folder=None, plot_type="surface"):
        fetched_data = self.fetch_data()
        if fetched_data.empty:
            logging.warning("No data to process.")
            return

        preprocessed_data = self.process_data(fetched_data)
        self.fit_neural_network(preprocessed_data)

        if output_folder is None:
            output_folder = self.output_folder

        if plot_type == "surface":
            self.plot_fitted_surface(preprocessed_data)
        elif plot_type == "subplots":
            self.plot_individual_smiles(preprocessed_data)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

    pipeline = VolatilityNNPipeline(DatabaseFetcher, IVPreprocessor)
    pipeline.run("D://", plot_type="subplots")
