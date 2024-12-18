import sys
import os
import numpy as np
import pyodbc
import pandas as pd
from datetime import datetime
import pytz
import logging
import matplotlib.pyplot as plt
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor
from financial_modelling.data_acquisition.database_fetcher import DataFetcher
from financial_modelling.modelling.SVIModel import SVIModel

class SVICalibrationDataPipeline:
    def __init__(self, data_fetcher: DataFetcher, date = "1546439410", output_folder = "E:\OutputParamsFiles\OutputFiles"):
        """
        Initialize the pipeline.

        Args:
        - date in unixtime format (int)
        """
        self.date = date
        self.db_config = {
            'server': 'DESKTOP-DK79R4I',  # Your server name
            'database': 'DataMining',     # Your database name
            }
        # Define pyodbc-compatible connection string
        self.connection_string = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={self.db_config['server']};"
            f"DATABASE={self.db_config['database']};"
            f"Trusted_Connection=yes;"
            )
        
        self.fetcher = data_fetcher(self.connection_string, use_sqlalchemy=False)
        self.data = None
        self.preprocessor = None
        self.preprocessed_data = None
        self.call_limits = None
        self.put_limits = None
        self.train_data = None
        self.volume_weights = None
        self.model_params = None
        self.output_folder = output_folder

    def fetch_data(self):
        """
        Fetch data from the database for a given list of dates.

        Returns:
        - pd.DataFrame: Fetched data.
        """
        query = f"""
        SELECT TOP (6302) *
        FROM [DataMining].[dbo].[OptionData]
        WHERE [QUOTE_UNIXTIME]= '{self.date}'
        """
        # Fuseau horaire US Eastern Time (New York)
        us_eastern = pytz.timezone("US/Eastern")

        self.readable_time = datetime.fromtimestamp(int(self.date), us_eastern).strftime('%d-%m-%Y %H:%M')
        print(f"Fetching data for date: {self.readable_time}")
        try:
            data = self.fetcher.fetch(query)
        except ValueError as e:
            print(f"Failed to fetch data for {self.readable_time}: {e}")
        
        if not data.empty:
            self.data = data
            print(f"Fetched data for: {self.readable_time}")
        else:
            self.data = pd.DataFrame()
            print("No data fetched.")
        return self.data
        
    def process_data(self, data, expiry, call_limits = (0.90, 1), put_limits = (1, 1.10)):
        """
        Process the fetched data.
        Args:
        - data (pd.DataFrame): Raw data fetched from the database.
        - call_limits (float, float) : Simple Moneyness K/Spot limits for calls
        - put_limits (float, float) : Simple Moneyness K/Spot limits for puts
        Returns:
        - pd.DataFrame: Processed data.
        """
        expiry_specific_data = data[data["EXPIRE_UNIX"]==expiry]
        self.call_limits = call_limits
        self.put_limits = put_limits
        self.preprocessor = IVPreprocessor(expiry_specific_data)
        self.preprocessed_data = self.preprocessor.preprocess(self.call_limits, self.put_limits, mode = "split")

        return self.preprocessed_data

    def fit_model(self, preprocessed_data , model = SVIModel()):
        """
        Fits the processed data to the input model for every Expiry.
        Args:
        - preprocessed_data (pd.DataFrame): preprocessed data.
        - model (SVIModel) : Stochastic Volatility Inspired Model
        Returns:
        - dict : fitted model params.
        """
        
        self.train_data = pd.DataFrame({'Log_Moneyness': preprocessed_data["Log_Moneyness"],
                             'Implied_Volatility': preprocessed_data["Implied_Volatility"],
                             'Volume': preprocessed_data["Volume"]})
        # Volume weights
        self.volume_weights = self.train_data['Volume'].values / self.train_data['Volume'].sum()
        model.fit(self.train_data, volume_weights=self.volume_weights)
        self.model_params = model.get_params()
        return self.model_params

    def plot_fitted_models(self, results, plot = "off"):
        """
        Plots the fitted SVI models and training data in a single Matplotlib window.

        Args:
        - results (list): List of tuples, where each tuple contains:
            (expiry, train_data, model_params)

        Returns:
        - None: Displays the plots.
        """
        if plot == "off":
            return ""
        num_expiries = len(results)
        cols = 12  # Number of columns in the subplot grid
        rows = (num_expiries + cols - 1) // cols  # Calculate rows dynamically

        # Create a shared figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        axes = axes.flatten()  # Flatten the grid for easy iteration

        for i, (expiry, train_data, model_params) in enumerate(results):
            # Check if train_data is empty
            if train_data.empty:
                print(f"Skipping expiry {expiry}: No training data available.")
                axes[i].axis("off")  # Hide the subplot for this expiry
                continue

            # Extract log-moneyness and implied volatility from training data
            log_moneyness = train_data["Log_Moneyness"].values
            implied_volatility = train_data["Implied_Volatility"].values

            # Check if log_moneyness has values
            if len(log_moneyness) == 0:
                print(f"Skipping expiry {expiry}: No valid Log_Moneyness values.")
                axes[i].axis("off")
                continue

            # Extract SVI parameters
            a = model_params["a"]
            b = model_params["b"]
            rho = model_params["rho"]
            m = model_params["m"]
            sigma = model_params["sigma"]

            # Define the SVI formula for total implied variance
            def svi_formula(k, a, b, rho, m, sigma):
                return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

            # Generate SVI curve for the fitted model
            log_moneyness_range = np.linspace(log_moneyness.min() - 0.2, log_moneyness.max() + 0.2, 500)
            implied_variance = svi_formula(log_moneyness_range, a, b, rho, m, sigma)
            fitted_volatility = np.sqrt(np.maximum(implied_variance, 1e-8))  # Avoid negative values

            # Plot in the corresponding subplot
            ax = axes[i]
            ax.scatter(log_moneyness, implied_volatility, color="blue", label="Training Data", alpha=0.7)
            ax.plot(log_moneyness_range, fitted_volatility, color="red", label="Fitted SVI Model", linewidth=2)
            #ax.set_title(f"Expiry: {expiry}", fontsize=14)
            #ax.set_xlabel("Log-Moneyness (log(K / Spot))", fontsize=12)
            #ax.set_ylabel("Implied Volatility", fontsize=12)
            #ax.legend(fontsize=10)
            #ax.grid(alpha=0.3)

        # Hide unused subplots (if any)
        for j in range(len(results), len(axes)):
            axes[j].axis("off")

        # Adjust layout and show the figure
        #plt.tight_layout()
        plt.show()


    def run(self, output_folder):
        """
        Run the entire pipeline.

        Args:
        - dates (list): List of dates to process.
        Returns:
        - csv file : List of params for each (date, expiry)
        """
        output_dataframe = pd.DataFrame(columns = ["QUOTE_UNIXTIME", "EXPIRE_UNIX", "a", "b", "rho", "m", "sigma"])

        # Fetch data
        fetched_data = self.fetch_data()

        # Get the unique expiries that quote at the specific date
        expiries = self.data["EXPIRE_UNIX"].unique().tolist()
        # For plotting purposes 
        results = []
        for expiry in expiries:
            # Process data
            self.process_data(self.data, expiry)
            # Fit the model
            self.fit_model(self.preprocessed_data)
            # Collect results for plotting
            results.append((expiry, self.train_data, self.model_params))

            # Prepare a new row
            new_row = {
                "QUOTE_UNIXTIME": self.date,
                "EXPIRE_UNIX": expiry,
                "a": self.model_params["a"],
                "b": self.model_params["b"],
                "rho": self.model_params["rho"],
                "m": self.model_params["m"],
                "sigma": self.model_params["sigma"]
                }
            output_dataframe = pd.concat([output_dataframe,
                                          pd.DataFrame([new_row])], 
                                          ignore_index=True
                                          )
        self.plot_fitted_models(results)
        
        # Get the right output folder
        output_folder = output_folder
        output_file = os.path.join(output_folder, f"output {self.date}.csv")
        # Save to CSV
        output_dataframe.to_csv(output_file, index=False)


# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(level=logging.INFO,  # Adjust level (e.g., DEBUG for detailed logs, INFO for less verbosity)
#                         format="%(asctime)s - %(levelname)s - %(message)s"
#                         )
#     # Run the pipeline
#     from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher

#     pipeline = SVICalibrationDataPipeline(DatabaseFetcher)
#     pipeline.run("D://")