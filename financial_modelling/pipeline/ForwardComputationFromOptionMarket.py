from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from financial_modelling.data_acquisition.file_fetcher import FileFetcher
from financial_modelling.data_pre_processing.ForwardComputationPreprocessor import ForwardComputationPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed number of workers
MAX_WORKERS = 12

# File path (replace with your actual file path)
FILE_PATH = 'F:\\SPX Data\\2019\\spx_01x_201910.txt'
OUTPUT_DIR = "E:\\ForwardComputations\\FittedData\\"
SEPARATOR = ','  # Adjust if necessary

# Initialize fetcher
fetcher = FileFetcher()

def compute_forward_and_discountFactor(expiry_raw_data_tuple):
    """Compute forward price, discount factor, and R-squared."""
    expiry, raw_data = expiry_raw_data_tuple
    expiry_data = raw_data[raw_data['EXPIRE_UNIX'] == expiry].copy()
    
    if expiry_data.empty:
        logging.warning("Skipping expiry %d due to empty dataset.", expiry)
        return {
            "EXPIRE_UNIX": expiry,
            "FORWARD": None,
            "DISCOUNT_FACTOR": None,
            "R_SQUARED": None
        }
    
    expiry_data['C_MID'] = (expiry_data['C_BID'] + expiry_data['C_ASK']) / 2
    expiry_data['P_MID'] = (expiry_data['P_BID'] + expiry_data['P_ASK']) / 2
    expiry_data['MidCallMidPutParity'] = expiry_data['C_MID'] - expiry_data['P_MID']

    expiry_data['CALL_WEIGHT'] = expiry_data['C_VOLUME'] / (expiry_data['C_ASK'] - expiry_data['C_BID']).replace(0, 1)
    expiry_data['PUT_WEIGHT'] = expiry_data['P_VOLUME'] / (expiry_data['P_ASK'] - expiry_data['P_BID']).replace(0, 1)
    expiry_data['WEIGHT'] = np.abs(expiry_data['CALL_WEIGHT'] + expiry_data['PUT_WEIGHT'])

    model = LinearRegression()
    model.fit(expiry_data[['STRIKE']], expiry_data['MidCallMidPutParity'], sample_weight=expiry_data['WEIGHT'])
    
    discountedForward, discountFactor = model.intercept_, -model.coef_[0]
    forward = discountedForward / discountFactor
    r_squared = model.score(expiry_data[['STRIKE']], expiry_data['MidCallMidPutParity'], sample_weight=expiry_data['WEIGHT'])
    
    return {
        "EXPIRE_UNIX": expiry,
        "FORWARD": forward,
        "DISCOUNT_FACTOR": discountFactor,
        "R_SQUARED": r_squared
    }

def process_quote_time(quote_unixtime, raw_data):
    """Process a single QUOTE_UNIXTIME value."""
    logging.info("Processing QUOTE_UNIXTIME: %s", quote_unixtime)
    results_list = []
    quote_data = raw_data[raw_data['QUOTE_UNIXTIME'] == quote_unixtime]
    
    # Apply preprocessor
    preprocessor = ForwardComputationPreprocessor(quote_data)
    filtered_data = preprocessor.preprocess({
        'C_IV': lambda x: x.notna() & (x > 0.05),
        'P_IV': lambda x: x.notna() & (x > 0.05),
        'C_VOLUME': lambda x: x.notna() & (x >= 1),
        'P_VOLUME': lambda x: x.notna() & (x >= 1)
    })
    
    expiries = filtered_data['EXPIRE_UNIX'].unique()
    
    for expiry in expiries:
        result = compute_forward_and_discountFactor((expiry, filtered_data))
        result["QUOTE_UNIXTIME"] = quote_unixtime
        results_list.append(result)
    
    output_csv = f"{OUTPUT_DIR}forward_computation_{quote_unixtime}.csv"
    pd.DataFrame(results_list).to_csv(output_csv, index=False)
    logging.info("Results exported to %s", output_csv)

def main():
    """Main function to load data and process QUOTE_UNIXTIME values in parallel."""
    logging.info("Fetching data from file: %s", FILE_PATH)
    raw_data = fetcher.fetch(filepath=FILE_PATH, separator=SEPARATOR)
    
    unique_quote_unixtime = raw_data['QUOTE_UNIXTIME'].unique()
    logging.info("Found %d unique QUOTE_UNIXTIME values", len(unique_quote_unixtime))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda q: process_quote_time(q, raw_data), unique_quote_unixtime)
    
if __name__ == "__main__":
    main()
