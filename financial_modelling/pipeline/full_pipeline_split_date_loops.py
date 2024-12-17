from financial_modelling.pipeline.svi_parameters_pipeline import SVICalibrationDataPipeline
from financial_modelling.pipeline.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
from financial_modelling.utils.utils import get_file_names
import pandas as pd
import os

# Splits the big unique file
fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates()
fetcher_and_splitter.run()

# Get where the files are created
source_folder_path = fetcher_and_splitter.folder_path
files_list = get_file_names(source_folder_path)

# Where the files will be dumped
output_folder = "E://OutputParamsFiles//OutputFiles"

for i,file in enumerate(files_list):
    file = pd.read_csv(os.path.join(source_folder_path, file))
    for date in file["QUOTE_UNIXTIME"]:
        svi_calibration = SVICalibrationDataPipeline(date = str(date)).run(output_folder)
