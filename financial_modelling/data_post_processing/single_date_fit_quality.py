# Package imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from financial_modelling.modelling.SVIModel import SVIModel as svi
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher as dbf
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor as ivp

# Constants
FOLDER = r"E:\OutputParamsFiles\OutputFiles"
DB_CONFIG = {
    'server': 'DESKTOP-DK79R4I',
    'database': 'DataMining',
}
CONNECTION_STRING = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={DB_CONFIG['server']};"
    f"DATABASE={DB_CONFIG['database']};"
    f"Trusted_Connection=yes;"
)

# Helper Functions
def load_fitted_params(date, folder):
    """
    Load the fitted SVI parameters for a specific date.
    """
    file_path = os.path.join(folder, f"output_{date}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    return pd.read_csv(file_path, sep=",")

def extract_fitted_params(fitted_params_file, expiry):
    """
    Extract SVI parameters and maturity for a given expiry.
    """
    params_row = fitted_params_file[fitted_params_file["EXPIRE_DATE"] == expiry]
    if params_row.empty:
        raise ValueError(f"No parameters found for expiry: {expiry}")
    maturity = params_row['Maturity'].values[0]
    params = {key: params_row[key].values[0] for key in ["a", "b", "rho", "m", "sigma"]}
    return params, maturity

def compute_metrics(market_iv, fitted_iv):
    """
    Compute R² and RMSE metrics.
    """
    r2 = r2_score(market_iv, fitted_iv)
    rmse = np.sqrt(mean_squared_error(market_iv, fitted_iv))
    return r2, rmse

def plot_results(log_moneyness, market_iv, fitted_iv, expiry, date):
    """
    Plot the market IV against the fitted IV for a given expiry.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(log_moneyness, market_iv, color="red", label="Market IV")
    plt.scatter(log_moneyness, fitted_iv, color="blue", label="Fitted IV Model")
    plt.title(f"Implied Volatility | Date: {date}, Expiry: {expiry}")
    plt.xlabel("Log-Moneyness")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Script
if __name__ == "__main__":
    # Fetch list of dates from CSV files in the folder
    csv_files = [f for f in os.listdir(FOLDER) if f.startswith("output_") and f.endswith(".csv")]
    dates = [file.split("output_")[1].split(".csv")[0] for file in csv_files]
    
    # Select the specific date to process
    date = dates[2]  # Example: Take the first date
    print(f"Processing date: {date}")
    
    # Load the fitted parameters for the selected date
    fitted_params_file = load_fitted_params(date, FOLDER)
    expiries = fitted_params_file["EXPIRE_DATE"].unique()
    
    # Database query template
    QUERY = f"""
    SELECT *
    FROM [DataMining].[dbo].[OptionData]
    WHERE [QUOTE_UNIXTIME] = '{date}'
    """
    
    # Fetch raw data from the database
    database_fetcher = dbf(CONNECTION_STRING)
    raw_data = database_fetcher.fetch(QUERY)
    database_fetcher.close()
    
    # Preprocess the raw data
    processed_data = ivp(raw_data).preprocess()

    # Process each expiry
    for expiry in expiries:
        try:
            # Filter data for the specific expiry
            expiry_data = processed_data[processed_data["EXPIRE_UNIX"] == expiry]
            log_moneyness = expiry_data["Log_Moneyness"].values
            market_iv = expiry_data["Implied_Volatility"].values
            
            # Extract SVI parameters and maturity
            params, maturity = extract_fitted_params(fitted_params_file, expiry)
            
            # Compute fitted IV using SVI
            fitted_iv = (1 / np.sqrt(maturity)) * np.sqrt(
                svi().svi(log_moneyness, params["a"], params["b"], params["rho"], params["m"], params["sigma"])
            )
            
            # Compute metrics
            r2, rmse = compute_metrics(market_iv, fitted_iv)
            print(f"Expiry: {expiry}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
            
            # Plot results
            #plot_results(log_moneyness, market_iv, fitted_iv, expiry, date)
        
        except Exception as e:
            print(f"Error processing expiry {expiry}: {e}")
