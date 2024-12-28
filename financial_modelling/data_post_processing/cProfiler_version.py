import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from financial_modelling.utils.utils import get_unixtimestamp_readable
from financial_modelling.modelling.SVIModel import SVIModel as svi
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher as dbf
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor as ivp
from financial_modelling.data_acquisition.list_of_files_fetcher import ListOfFilesFetcher as loff

# Function to compute R2 and RMSE
def compute_metrics(predicted, actual):
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return r2, rmse

# Function to get the fitted SVI parameters
folder = r"E:\OutputParamsFiles\OutputFiles"

def get_the_fitted_params_file(date, folder):
    file_path = os.path.join(folder, f"output_{date}.csv")
    return pd.read_csv(file_path, sep=",")

def get_the_fitted_params(dataframe, expiry, date):
    fitted_params_file = get_the_fitted_params_file(date, folder)
    expiry_specific_params = fitted_params_file[fitted_params_file["EXPIRE_DATE"] == expiry]
    maturity = np.array(expiry_specific_params['Maturity'])
    params = {
        "a": np.array(expiry_specific_params["a"]),
        "b": np.array(expiry_specific_params["b"]),
        "rho": np.array(expiry_specific_params["rho"]),
        "m": np.array(expiry_specific_params["m"]),
        "sigma": np.array(expiry_specific_params["sigma"])
    }
    return params, maturity

def process_single_date(date):
    # Connection configuration
    DB_CONFIG = {
        'server': 'DESKTOP-DK79R4I',  # Your server name
        'database': 'DataMining',     # Your database name
    }

    connection_string = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"Trusted_Connection=yes;"
    )

    query = f"""
        SELECT TOP(6302) *
        FROM [DataMining].[dbo].[OptionData]
        WHERE [QUOTE_UNIXTIME] = '{date}'
    """
    
    database_fetcher = dbf(connection_string)
    raw_data = database_fetcher.fetch(query)
    database_fetcher.close()

    if raw_data.empty:
        logging.warning(f"No data available for date: {date}")
        return []

    processed_data = ivp(raw_data).preprocess()

    results = []
    for expiry in processed_data["EXPIRE_UNIX"].unique():
        expiry_specific_processed_data = processed_data[processed_data["EXPIRE_UNIX"] == expiry]
        log_moneyness = np.nan_to_num(np.array(expiry_specific_processed_data["Log_Moneyness"]), nan=0.0)

        if log_moneyness.size == 0:
            logging.warning(f"No valid data for expiry {expiry} on date {date}. Skipping.")
            continue

        params, maturity = get_the_fitted_params(expiry_specific_processed_data, expiry, date)
        maturity = np.nan_to_num(maturity, nan=0.0)

        svi_values = (1 / np.sqrt(maturity)) * np.sqrt(svi().svi(
            log_moneyness,
            np.nan_to_num(params["a"], nan=0.0),
            np.nan_to_num(params["b"], nan=0.0),
            np.nan_to_num(params["rho"], nan=0.0),
            np.nan_to_num(params["m"], nan=0.0),
            np.nan_to_num(params["sigma"], nan=0.0)
        ))

        svi_values = np.nan_to_num(svi_values, nan=0.0)
        actual_iv = np.nan_to_num(np.array(expiry_specific_processed_data["Implied_Volatility"]), nan=0.0)

        if np.isnan(svi_values).any() or np.isnan(actual_iv).any():
            logging.warning(f"NaN encountered in SVI values or Actual IV for expiry {expiry} on date {date}. Skipping.")
            continue

        r2, rmse = compute_metrics(svi_values, actual_iv)

        results.append({
            "QUOTE_UNIXTIME": date,
            "EXPIRE_UNIX": expiry,
            "R2": r2,
            "RMSE": rmse
        })

    return results

def main():
    from financial_modelling.data_acquisition.list_of_files_fetcher import ListOfFilesFetcher as loff
    
    folder = r"E:\OutputParamsFiles\OutputFiles"
    output_folder = r"E:\OutputParamsFiles\SVI quality fit"
    output_csv_path = os.path.join(output_folder, "svi_results.csv")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Fetch list of fitted dates
    loff = loff()
    loff.fetch(folder)
    loff.get_unixtimestamp()
    list_of_fitted_dates = loff.list_of_dates

    # Initialize progress bar
    results = []
    for date in tqdm(list_of_fitted_dates, desc="Processing Dates"):
        results.extend(process_single_date(date))

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")

if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    with open("profile_output.prof", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()

    print("Profile saved to profile_output.prof")
