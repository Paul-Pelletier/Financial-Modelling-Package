import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import torch
import logging
from financial_modelling.pipeline.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor
from RegularizedSVIPipelineOptGPU import RegularizedSVICalibrationPipeline


def process_date_gpu(date, output_folder, connection_string):
    """
    Process a single date using the RegularizedSVICalibrationPipeline with GPU support.

    Args:
    - date (str): QUOTE_UNIXTIME to process.
    - output_folder (str): Path to save results.
    - connection_string (str): Database connection string.
    """
    try:
        fetcher = DatabaseFetcher(connection_string=connection_string, use_sqlalchemy=False)
        pipeline = RegularizedSVICalibrationPipeline(
            connection_string=connection_string,
            preprocessor=IVPreprocessor,
            date=str(date),
            output_folder=output_folder,
        )
        pipeline.run()
        fetcher.close()  # Ensure database connection is closed
    except Exception as e:
        logging.error(f"Error processing date {date}: {e}")


def main_gpu():
    """
    Main pipeline function for processing multiple dates with GPU acceleration.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Connection string for database
    connection_string = (
        "DRIVER={SQL Server};"
        "SERVER=DESKTOP-DK79R4I;"
        "DATABASE=DataMining;"
        "Trusted_Connection=yes;"
    )

    # Define output folder
    output_folder = "E:/OutputParamsFiles/OutputFiles_test"

    # Step 1: Split large dataset into individual files by QUOTE_UNIXTIME
    fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates(DatabaseFetcher(connection_string))
    fetcher_and_splitter.run()

    # Step 2: Retrieve folder path and list of files
    files_list = get_file_names(fetcher_and_splitter.folder_path)

    # Step 3: Process each file and each unique QUOTE_UNIXTIME
    for file_name in files_list:
        file_path = os.path.join(fetcher_and_splitter.folder_path, file_name)
        logging.info(f"Processing file: {file_name}")

        try:
            file_df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            continue

        # Step 4: Extract unique QUOTE_UNIXTIME values and parallelize processing
        unique_dates = file_df["QUOTE_UNIXTIME"].unique()

        with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as per your system
            executor.map(
                process_date_gpu,
                unique_dates,
                [output_folder] * len(unique_dates),
                [connection_string] * len(unique_dates),
            )


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    try:
        main_gpu()
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
    finally:
        logging.info(f"Total elapsed time: {datetime.now() - start_time}")
