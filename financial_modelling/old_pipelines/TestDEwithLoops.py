from financial_modelling.big_data_pipelines.SVICalibrationPipelineWithDifferentialEvolution import SVICalibrationPipelineWithDE
from financial_modelling.big_data_pipelines.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor
import pandas as pd
import os

# Splits the big unique file
fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates(DatabaseFetcher)
fetcher_and_splitter.run()

# Get where the files are created
source_folder_path = fetcher_and_splitter.folder_path
files_list = get_file_names(source_folder_path)

# Where the files will be dumped
output_folder = "E://OutputParamsFiles//OutputFilesTestDE"

# Main loop: Process files sequentially
for i, file in enumerate(files_list):
    # Read each file
    file_path = os.path.join(source_folder_path, file)
    file_data = pd.read_csv(file_path)

    print(f"Processing file {i+1}/{len(files_list)}: {file}")

    # Process each date in the file sequentially
    for date in file_data["QUOTE_UNIXTIME"]:
        print(f"Processing date: {date}")
        svi_calibration = SVICalibrationPipelineWithDE(
            data_fetcher=DatabaseFetcher,
            preprocessor=IVPreprocessor,
            date=str(date)
        )
        svi_calibration.run(output_folder)
