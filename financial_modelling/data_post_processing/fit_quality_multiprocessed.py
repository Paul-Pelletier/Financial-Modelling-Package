import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from multiprocessing import Pool, cpu_count, Manager
from financial_modelling.utils.utils import get_unixtimestamp_readable
from financial_modelling.modelling.SVIModel import SVIModel as svi
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher as dbf
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor as ivp
from financial_modelling.data_acquisition.list_of_files_fetcher import ListOfFilesFetcher as loff
import warnings
from tqdm import tqdm 

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

def process_single_date(date, output_folder):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
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
        return None

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

    if results:
        intermediate_csv = os.path.join(output_folder, f"svi_results_{date}.csv")
        pd.DataFrame(results).to_csv(intermediate_csv, index=False)
        return intermediate_csv

    return None

def process_single_date_wrapper(args):
    date, output_folder = args
    return process_single_date(date, output_folder)

def main():
    folder = r"E:\OutputParamsFiles\OutputFiles"
    output_folder = r"E:\OutputParamsFiles\SVI_quality_fit"
    consolidated_csv_path = os.path.join(output_folder, "svi_results.csv")
    os.makedirs(output_folder, exist_ok=True)
    
    # Fetch list of fitted dates
    loff_instance = loff()
    loff_instance.fetch(folder)
    loff_instance.get_unixtimestamp()
    list_of_fitted_dates = loff_instance.list_of_dates

    # Initialize the progress bar
    with Manager() as manager:
        progress_bar = tqdm(total=len(list_of_fitted_dates), desc="Processing Dates")
        results = []

        def update_progress_bar(result):
            progress_bar.update()
            if result:
                results.append(result)

        # Prepare arguments for the wrapper function
        tasks = [(date, output_folder) for date in list_of_fitted_dates]

        # Initialize the Pool
        with Pool(cpu_count()) as pool:
            for result in pool.imap_unordered(process_single_date_wrapper, tasks):
                update_progress_bar(result)

        progress_bar.close()

    # Consolidate all intermediate files
    intermediate_files = [file for file in results if file is not None]
    consolidated_data = pd.concat(
        [pd.read_csv(file) for file in intermediate_files], ignore_index=True
    )
    consolidated_data.to_csv(consolidated_csv_path, index=False)
    print(f"Consolidated results saved to: {consolidated_csv_path}")

    # Clean up intermediate files
    for file in intermediate_files:
        os.remove(file)
    print("Intermediate files have been deleted.")

if __name__ == "__main__":
    import cProfile
    import pstats
    import io

    # Profile the main function
    profiler = cProfile.Profile()
    profiler.enable()

    main()  # The function you want to profile

    profiler.disable()

    # Output profiling stats to a readable format
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()

    # Save the profiling results to a file
    with open("profiling_results.txt", "w") as f:
        f.write(s.getvalue())