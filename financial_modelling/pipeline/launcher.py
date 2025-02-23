import os
import subprocess
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
CSV_FILE = 'F:\\SPX Data\\filePaths.csv'  # CSV containing raw data file paths
OUTPUT_BASE_DIR = "E:\\ForwardComputations\\FittedData\\"  # Base output directory
PROCESSING_SCRIPT = "C:\\Users\\paula\\OneDrive\\Documents\\Financial Modelling Package\\financial_modelling\\pipeline\\ForwardComputationFromOptionMarket.py"  # Path to your processing script
JOURNAL_FILE = "E:\\ForwardComputations\\processing_journal.csv"  # Log of processed files

# Load file list from CSV
file_list_df = pd.read_csv(CSV_FILE)
if 'FilePath' not in file_list_df.columns:
    raise ValueError("CSV must contain a 'FilePath' column with raw data file paths.")

# Load processing journal
if os.path.exists(JOURNAL_FILE):
    processed_df = pd.read_csv(JOURNAL_FILE)
    processed_set = set(processed_df['YearMonth'])
else:
    processed_set = set()

# Process each file
processed_entries = []
for file_path in file_list_df['FilePath']:
    # Extract year and month from filename
    filename = os.path.basename(file_path)
    year_month = filename.split("_")[-1].split(".")[0]  # Extract YYYYMM
    year, month = year_month[:4], year_month[4:]
    
    # Skip if already processed
    if year_month in processed_set:
        logging.info(f"Skipping already processed: {year_month}")
        continue
    
    # Define the output directory for this file
    output_dir = os.path.join(OUTPUT_BASE_DIR, year, month)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    # Launch the processing script
    logging.info(f"Processing file: {file_path}, saving results in {output_dir}")
    try:
        subprocess.run(["python", PROCESSING_SCRIPT, file_path, output_dir], check=True)
        processed_entries.append([year_month])
    except subprocess.CalledProcessError:
        logging.error(f"Error processing file: {file_path}")

# Update journal file
if processed_entries:
    journal_df = pd.DataFrame(processed_entries, columns=['YearMonth'])
    journal_df.to_csv(JOURNAL_FILE, mode='a', header=not os.path.exists(JOURNAL_FILE), index=False)
    logging.info("Processing journal updated.")
