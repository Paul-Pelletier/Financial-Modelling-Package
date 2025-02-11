from abc import ABC, abstractmethod
from typing import Any, List, Dict
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.ForwardComputationPreprocessor import ForwardComputationPreprocessor
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_CONFIG = {
    'server': 'DESKTOP-DK79R4I',
    'database': 'DataMining',
}

connection_string = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={DB_CONFIG['server']};"
    f"DATABASE={DB_CONFIG['database']};"
    f"Trusted_Connection=yes;"
    f"Connection Timeout=60;"
    f"Encrypt=no;"
    f"Network Library=dbmssocn;"
)

connection_string = (
    "mssql+pyodbc://@DESKTOP-DK79R4I/DataMining?"
    "driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"
)

fetcher = DatabaseFetcher(connection_string, use_sqlalchemy=True)

def compute_forward_and_discountFactor(expiry_raw_data_tuple):
    expiry, raw_data = expiry_raw_data_tuple
    expiry_data = raw_data[raw_data['EXPIRE_UNIX'] == expiry].copy()
    
    if expiry_data.empty:
        logging.warning("Skipping expiry %d due to empty dataset.", expiry)
        return {
            "EXPIRE_UNIX": expiry,
            "FORWARD": None,
            "DISCOUNT_FACTOR": None
        }
    expiry_data.loc[:, 'MidCallMidPutPArity'] = expiry_data['C_MID'] - expiry_data['P_MID']

    #Compute sample weights based on volume and bid-ask spread
    expiry_data['CALL_WEIGHT'] = expiry_data['C_VOLUME'] / (expiry_data['C_ASK'] - expiry_data['C_BID']).replace(0, 1)
    expiry_data['PUT_WEIGHT'] = expiry_data['P_VOLUME'] / (expiry_data['P_ASK'] - expiry_data['P_BID']).replace(0, 1)
    expiry_data['WEIGHT'] = np.abs(expiry_data['CALL_WEIGHT'] + expiry_data['PUT_WEIGHT'])
    
    model = LinearRegression()
    model.fit(expiry_data[['STRIKE']], expiry_data['MidCallMidPutPArity'], sample_weight=expiry_data['WEIGHT'])
    discountedForward, discountFactor = model.intercept_, -model.coef_[0]
    forward = discountedForward / discountFactor
    return {
        "EXPIRE_UNIX": expiry,
        "FORWARD": forward,
        "DISCOUNT_FACTOR": discountFactor
    }

if __name__ == "__main__":
    csv_file = "E:\\ForwardComputations\\outputDistinctQuotesTimes.csv"
    try:
        logging.info("Reading from %s", csv_file)
        distinct_quote_unixtime = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error("File not found: %s", csv_file)
        unique_dates_query_string = """SELECT DISTINCT QUOTE_UNIXTIME FROM [DataMining].[dbo].[OptionData]"""
        distinct_quote_unixtime = fetcher.fetch(unique_dates_query_string)
    
    for QUOTE_UNIXTIME in distinct_quote_unixtime['QUOTE_UNIXTIME']:
        results_list = []
        query = f"""
                SELECT TOP(6302) *
                FROM [DataMining].[dbo].[OptionData]
                WHERE [QUOTE_UNIXTIME] = '{QUOTE_UNIXTIME}'
                """
        raw_data = fetcher.fetch(query)
        expiries = raw_data['EXPIRE_UNIX'].unique()

        drop_criteria = {
            'C_IV': lambda x: x.notna() & (x > 0.05),
            'P_IV': lambda x: x.notna() & (x > 0.05),
            'C_VOLUME': lambda x: x.notna() & (x >= 1),
            'P_VOLUME': lambda x: x.notna() & (x >= 1)
        }
        preprocessor = ForwardComputationPreprocessor(raw_data)
        filtered_data = preprocessor.preprocess(drop_criteria)
        
        with ProcessPoolExecutor(max_workers = 20) as executor:
            results = list(executor.map(compute_forward_and_discountFactor, [(expiry, filtered_data) for expiry in expiries]))
        
        for result in results:
            result["QUOTE_UNIXTIME"] = QUOTE_UNIXTIME
            logging.info("Forward: %s for expiry: %d", str(result["FORWARD"]), result["EXPIRE_UNIX"])
            results_list.append(result)
        
        output_csv = f"E:\\ForwardComputations\\forward_computation_{QUOTE_UNIXTIME}.csv"
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv, index=False)
        logging.info("Results exported to %s", output_csv)
