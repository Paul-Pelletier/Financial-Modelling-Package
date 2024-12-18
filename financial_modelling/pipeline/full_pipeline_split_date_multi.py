from financial_modelling.pipeline.svi_parameters_pipeline import SVICalibrationPipeline
from financial_modelling.pipeline.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

# Splits the big unique file
fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates(DatabaseFetcher)
fetcher_and_splitter.run()

# Get where the files are created
source_folder_path = fetcher_and_splitter.folder_path
files_list = get_file_names(source_folder_path)

# Where the files will be dumped
output_folder = "E://OutputParamsFiles//OutputFiles"

# Function to process a single date
def process_date(date, output_folder):
    # Instantiate and run the pipeline
    SVICalibrationPipeline(data_fetcher = DatabaseFetcher, preprocessor = IVPreprocessor, date=str(date)).run(output_folder)

# Main function
def main(source_folder_path, files_list, output_folder):
    for i, file_name in enumerate(files_list):
        # Read the file
        file_path = os.path.join(source_folder_path, file_name)
        file = pd.read_csv(file_path)

        print(f"Processing file {i+1}/{len(files_list)}: {file_name}")

        # Use ProcessPoolExecutor to parallelize the inner loop
        with ProcessPoolExecutor() as executor:
            executor.map(process_date, file["QUOTE_UNIXTIME"], [output_folder] * len(file["QUOTE_UNIXTIME"]))

if __name__ == "__main__":
    # Paths and file list
    source_folder_path = source_folder_path   # Replace with your folder path
    output_folder = output_folder        # Replace with your output folder
    files_list = files_list        # Replace with your list of files

    # Run the main function
    main(source_folder_path, files_list, output_folder)
