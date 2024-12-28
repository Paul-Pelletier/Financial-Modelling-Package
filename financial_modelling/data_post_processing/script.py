#Package imports
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from financial_modelling.utils.utils import get_unixtimestamp_readable
from financial_modelling.modelling.SVIModel import SVIModel as svi
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher as dbf
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor as ivp
from financial_modelling.data_acquisition.list_of_files_fetcher import ListOfFilesFetcher as loff
from financial_modelling.utils.utils import get_unixtimestamp_readable

#Get the fitted SVI parameters
def get_the_fitted_params_file(date, folder):
    file_path = os.path.join(folder, f"output_{date}.csv")
    fitted_params_file = pd.read_csv(file_path, sep = ",")
    return fitted_params_file

#Get the fitted SVI parameters & Maturity
def get_the_fitted_params(dataframe, expiry):
    fitted_params_file = get_the_fitted_params_file(date, folder)
    expriry_specific_params = fitted_params_file[fitted_params_file["EXPIRE_DATE"] == expiry]
    maturity = np.array(expriry_specific_params['Maturity'])
    params = {"a":expriry_specific_params["a"],
            "b":expriry_specific_params["b"],
            "rho":expriry_specific_params["rho"],
            "m":expriry_specific_params["m"],
            "sigma":expriry_specific_params["sigma"]}
    for i in params.keys():
        params[i] = np.array(params[i])
    return params, maturity

# Function to compute R2 and RMSE
def compute_metrics(predicted, actual):
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return r2, rmse

#List of fitted SVI parameters
folder = r"E:\OutputParamsFiles\OutputFiles"
loff = loff()
loff.fetch(folder)
loff.get_unixtimestamp()
list_of_fitted_dates = loff.list_of_dates

def process_data(date):
    #Connection configuration
    DB_CONFIG = {
        'server': 'DESKTOP-DK79R4I',  # Your server name
        'database': 'DataMining',     # Your database name
        }

    #Define pyodbc-compatible connection string
    connection_string = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"Trusted_Connection=yes;"
            )

    #Define the query string
    query = f"""
            SELECT TOP(6302) *
            FROM [DataMining].[dbo].[OptionData]
            WHERE [QUOTE_UNIXTIME] = '{date}'
            """
    database_fetcher = dbf(connection_string)
    raw_data = database_fetcher.fetch(query)
    database_fetcher.close()

    #Extract the needed data from raw_data
    
    #Preprocessor preprocessing rawa data
    processed_data = ivp(raw_data).preprocess()
    unique_expiries = processed_data["EXPIRE_UNIX"].unique()
    unique_expiries.sort()
    for i, expiry in enumerate(unique_expiries):
        expiry_specific_processed_data = processed_data[processed_data["EXPIRE_UNIX"] == expiry]
        log_moneyness = np.array(expiry_specific_processed_data["Log_Moneyness"])

        params, maturity = get_the_fitted_params(expiry_specific_processed_data, expiry)
        svi_values = (1/np.sqrt(maturity))*np.sqrt(svi().svi(log_moneyness, params["a"], params["b"], params["rho"], params["m"], params["sigma"]))
        svi_values = np.nan_to_num(svi_values, nan=0.0)
        actual_iv = np.array(expiry_specific_processed_data["Implied_Volatility"])
        r2, rmse = compute_metrics(svi_values, actual_iv)

        # Plot the fitted model data against the raw data IV
        plt.scatter(log_moneyness, svi_values, color = "blue", label = "Fitted IV Model")
        plt.scatter(log_moneyness, actual_iv, color = "red", label = "Market IV")
        plt.title(f"Implied Volatility | STE: {expiry-date},Date: {get_unixtimestamp_readable(date)}, Expiry: {get_unixtimestamp_readable(expiry)}, R2: {r2}, RMSE: {rmse}")
        plt.show()

if __name__ == "__main__":
    import logging
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import r2_score, mean_squared_error
    from financial_modelling.utils.utils import get_unixtimestamp_readable
    from financial_modelling.modelling.SVIModel import SVIModel as svi
    from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher as dbf
    from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor as ivp
    from financial_modelling.data_acquisition.list_of_files_fetcher import ListOfFilesFetcher as loff
    #List of fitted SVI parameters
    folder = r"E:\OutputParamsFiles\OutputFiles"
    loff = loff()
    loff.fetch(folder)
    loff.get_unixtimestamp()
    list_of_fitted_dates = loff.list_of_dates
    index_of_quote_date = 0
    date = list_of_fitted_dates[index_of_quote_date]
    process_data(date)