import pandas as pd
import numpy as np
import os
from financial_modelling.utils.utils import check_distinct_quotetime_file_presence
from financial_modelling.pipeline.export_distinct_unixtimequotedate_pipeline import UniqueQuoteTimeFetcher
from financial_modelling.data_acquisition.base_fetcher import DataFetcher

class FetchAndSplitToMultipleQuoteDates:
    """
    Creates an arbitrary number of split from a csv file that contains a list of unique QUOTE_UNIXTIME
    """
    def __init__(self, generic_fetcher: DataFetcher, number_of_splits = 10):
        self.uniqueQuoteDates = None
        self.folder_path = None
        self.splitted_dataframes = None
        self.number_of_splits = number_of_splits
        self.data_fetcher = generic_fetcher

    def check_file_presence_and_load(self):
        # File fetcher in case it's missing
        fetcher = UniqueQuoteTimeFetcher(self.data_fetcher)
        
        # By default fetcher.output_path is E://OutputParamsFiles//
        self.folder_path = os.path.join(fetcher.output_path, "SplittedDistinct")
        # Files presence check
        if not check_distinct_quotetime_file_presence:
            fetcher.Get_Unique_QuoteTime_File()

        # Load the csv file
        self.uniqueQuoteDates = pd.read_csv(fetcher.output_file)

    def split_calibrate_and_dump(self):
        
        # Split the DataFrame into `num_splits` parts
        self.splitted_dataframes = np.array_split(self.uniqueQuoteDates, self.number_of_splits)
        for i, split in enumerate(self.splitted_dataframes):
            output_file = os.path.join(self.folder_path, f"outputDistinctQuotesTimesStartWith {i}.csv")

            # Drops the file in the E://OutputParamsFiles//SplittedDistinct folder
            split.to_csv(output_file, index = False)

    def run(self):
        self.check_file_presence_and_load()
        self.split_calibrate_and_dump()


import pandas as pd
import numpy as np
import os

class FetchAndSplitPipeline(PipelineBase):
    def __init__(self, fetcher, output_folder="E://OutputParamsFiles//SplittedDistinct", num_splits=10):
        super().__init__(fetcher, output_folder)
        self.num_splits = num_splits

    def fetch_data(self, **kwargs):
        # Load the unique quote times file
        input_file = kwargs.get("input_file", os.path.join(self.output_folder, "outputDistinctQuotesTimes.csv"))
        return pd.read_csv(input_file)

    def process_data(self, data, **kwargs):
        # Split the data into multiple parts
        return np.array_split(data, self.num_splits)

    def save_output(self, data, **kwargs):
        for i, split in enumerate(data):
            output_file = os.path.join(self.output_folder, f"split_{i}.csv")
            split.to_csv(output_file, index=False)
            print(f"Split saved to {output_file}")
