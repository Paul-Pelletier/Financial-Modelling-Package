import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import torch

# Replace the following imports with the actual modules/paths in your project
from financial_modelling.pipeline.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor
import warnings

# Import the RegularizedSVICalibrationPipeline from your codebase
from financial_modelling.pipeline.RegularizedSVIPipelineOptGPU import RegularizedSVICalibrationPipeline


def process_date_gpu(date, output_folder, device):
    """
    Process a single date using RegularizedSVICalibrationPipeline with GPU support.

    Args:
    - date (str): QUOTE_UNIXTIME to process.
    - output_folder (str): Folder to store output results.
    - device (torch.device): GPU device to use for processing.
    """
    try:
        pipeline = RegularizedSVICalibrationPipeline(
            data_fetcher=DatabaseFetcher,
            preprocessor=IVPreprocessor,
            date=str(date),
            output_folder=output_folder,
        )
        pipeline.device = device  # Ensure GPU device is set
        pipeline.run(output_folder=output_folder)
    except Exception as e:
        print(f"Error processing date {date}: {e}")


def main_gpu():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    """
    Main pipeline function optimized for GPU processing.
    """
    # Ensure GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This pipeline requires GPU acceleration.")

    # GPU device
    device = torch.device("cuda")

    # 1) Split the large file into multiple CSVs, each containing a unique QUOTE_UNIXTIME
    fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates(DatabaseFetcher)
    fetcher_and_splitter.run()

    # 2) Retrieve folder path and get the list of newly created CSV files
    source_folder_path = fetcher_and_splitter.folder_path
    files_list = get_file_names(source_folder_path)

    # 3) Define your output folder for storing calibration results
    output_folder = "E://OutputParamsFiles//OutputFiles_test"

    # 4) Iterate over each CSV file
    for i, file_name in enumerate(files_list):
        file_path = os.path.join(source_folder_path, file_name)
        print(f"Processing file {i + 1}/{len(files_list)}: {file_name}")

        # Read the CSV into a pandas DataFrame
        try:
            file_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # 5) Collect unique QUOTE_UNIXTIME values
        quote_times = file_df["QUOTE_UNIXTIME"].unique()

        # 6) Parallelize over all unique QUOTE_UNIXTIME using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers = 22) as executor:
            executor.map(process_date_gpu, quote_times, [output_folder] * len(quote_times), [device] * len(quote_times))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from datetime import datetime

    start_time = datetime.now()
    try:
        main_gpu()
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
    finally:
        print("Elapsed time: ", datetime.now() - start_time)
