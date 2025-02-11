from financial_modelling.big_data_pipelines.SVICalibrationPipelineWithDifferentialEvolution import SVICalibrationPipelineWithDE
from financial_modelling.big_data_pipelines.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

# Splits the big unique file
fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates(DatabaseFetcher)
fetcher_and_splitter.run()

# Get where the files are created
source_folder_path = fetcher_and_splitter.folder_path
files_list = get_file_names(source_folder_path)

# Where the files will be dumped
output_folder = "E://OutputParamsFiles//OutputFilesTestDE"

# Function to process a single date
def process_date(date, output_folder):
    """
    Process a single date using SVICalibrationPipelineWithDE.
    """
    try:
        svi_calibration = SVICalibrationPipelineWithDE(
            data_fetcher=DatabaseFetcher,
            preprocessor=IVPreprocessor,
            date=str(date)
        )
        svi_calibration.run(output_folder)
    except Exception as e:
        print(f"Error processing date {date}: {e}")

# Main function
def main(source_folder_path, files_list, output_folder):
    for i, file_name in enumerate(files_list):
        # Read the file
        file_path = os.path.join(source_folder_path, file_name)
        file = pd.read_csv(file_path)

        print(f"Processing file {i+1}/{len(files_list)}: {file_name}")

        # Use ProcessPoolExecutor to parallelize the inner loop
        with ProcessPoolExecutor() as executor:
            executor.map(
                process_date, 
                file["QUOTE_UNIXTIME"], 
                [output_folder] * len(file["QUOTE_UNIXTIME"])
            )

if __name__ == "__main__":
    # Run the main function
    main(source_folder_path, files_list, output_folder)
