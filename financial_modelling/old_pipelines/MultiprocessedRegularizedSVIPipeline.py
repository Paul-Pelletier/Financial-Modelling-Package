import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Replace the following imports with the actual modules/paths in your project
from financial_modelling.big_data_pipelines.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

# Import the RegularizedSVICalibrationPipeline from your codebase
from financial_modelling.big_data_pipelines.RegularizedSVIPipeline import RegularizedSVICalibrationPipeline


def process_date(date, output_folder):
    """
    Process a single date using RegularizedSVICalibrationPipeline.
    """
    try:
        pipeline = RegularizedSVICalibrationPipeline(
            data_fetcher=DatabaseFetcher,
            preprocessor=IVPreprocessor,
            date=str(date),
        )
        pipeline.run(output_folder=output_folder)
    except Exception as e:
        print(f"Error processing date {date}: {e}")


def main():
    # 1) Split the large file into multiple CSVs, each containing a unique QUOTE_UNIXTIME
    fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates(DatabaseFetcher)
    fetcher_and_splitter.run()

    # 2) Retrieve folder path and get the list of newly created CSV files
    source_folder_path = fetcher_and_splitter.folder_path
    files_list = get_file_names(source_folder_path)

    # 3) Define your output folder for storing calibration results
    output_folder = "E://OutputParamsFiles//OutputFiles"

    # 4) Iterate over each CSV file
    for i, file_name in enumerate(files_list):
        file_path = os.path.join(source_folder_path, file_name)
        print(f"Processing file {i+1}/{len(files_list)}: {file_name}")

        # Read the CSV into a pandas DataFrame
        try:
            file_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # 5) Parallelize over all rows (unique QUOTE_UNIXTIME) in the CSV
        quote_times = file_df["QUOTE_UNIXTIME"]
        with ProcessPoolExecutor() as executor:
            executor.map(process_date, quote_times, [output_folder] * len(quote_times))


if __name__ == "__main__":
    main()
