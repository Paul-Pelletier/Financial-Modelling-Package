import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from financial_modelling.modelling.nss_model import NelsonSiegelSvensson

# Base directories
INPUT_DIR = "E:\\ForwardComputations\\FittedData"
OUTPUT_DIR = "E:\\ForwardParametrization\\FittedParams"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(file_path):
    """Fits NSS model to a forward term structure and saves parameters."""
    try:
        # Load CSV file
        df = pd.read_csv(file_path, sep=",")
        df = df[-np.isinf(df['FORWARD'])]  # Drop invalid rows

        # Compute maturities
        maturities = (1 / (3600 * 24 * 365)) * (df["EXPIRE_UNIX"] - df["QUOTE_UNIXTIME"])
        forward_rates = df["FORWARD"]

        # Fit NSS model
        nss = NelsonSiegelSvensson(beta0=forward_rates.iloc[0], beta1=forward_rates.iloc[0],
                                   beta2=forward_rates.iloc[0], lambda1=0.1)
        nss.fit(maturities, forward_rates)

        # Prepare output file path (mirror input folder structure)
        relative_path = os.path.relpath(file_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, os.path.dirname(relative_path))
        os.makedirs(output_path, exist_ok=True)

        # Save fitted parameters
        output_file = os.path.join(output_path, f"fitted_params_{os.path.basename(file_path)}")
        fitted_params = pd.DataFrame([{
            "beta0": nss.beta0, "beta1": nss.beta1, "beta2": nss.beta2,
            "beta3": nss.beta3, "lambda1": nss.lambda1, "lambda2": nss.lambda2
        }])
        fitted_params.to_csv(output_file, index=False)

        print(f"✅ Processed: {file_path} → {output_file}")
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def scan_and_process():
    """Scans all CSV files and processes them in parallel."""
    file_paths = []
    
    # Recursively find all CSV files
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".csv"):
                file_paths.append(os.path.join(root, file))
    
    print(f"Found {len(file_paths)} files. Starting multiprocessing...")

    # Use multiprocessing to fit NSS models in parallel
    with mp.Pool(processes=20) as pool:
        pool.map(process_file, file_paths)

if __name__ == "__main__":
    scan_and_process()
