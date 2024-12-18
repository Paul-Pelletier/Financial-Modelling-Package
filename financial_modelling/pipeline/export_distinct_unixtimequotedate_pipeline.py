from financial_modelling.data_acquisition.database_fetcher import DataFetcher
import os

class UniqueQuoteTimeFetcher:
    def __init__(self, generic_fetcher: DataFetcher, output_path = "E://OutputParamsFiles//", file_name = "outputDistinctQuotesTimes.csv"):
        self.output_path = output_path
        self.file_name = file_name
        self.output_file = os.path.join(output_path, "outputDistinctQuotesTimes.csv")
        self.data_fetcher = generic_fetcher

    def Get_Unique_QuoteTime_File(self):
        DB_CONFIG = {
            'server': 'DESKTOP-DK79R4I',  # Your server name
            'database': 'DataMining',     # Your database name
            }

        # Define pyodbc-compatible connection string
        connection_string = (
                f"DRIVER={{SQL Server}};"
                f"SERVER={DB_CONFIG['server']};"
                f"DATABASE={DB_CONFIG['database']};"
                f"Trusted_Connection=yes;"
        )

        # Initialize the fetcher
        fetcher = self.data_fetcher(connection_string, use_sqlalchemy=False)

        # Define a SQL query
        query = """
        SELECT DISTINCT [QUOTE_UNIXTIME]
        FROM [DataMining].[dbo].[OptionData];
        """

        # Fetch data
        data = fetcher.fetch(query)
        data.to_csv(self.output_file, index=False)
        print("Files has been dropped to the default folder")

import os

class UniqueQuoteTimeFetcherPipeline(PipelineBase):
    def __init__(self, fetcher, output_folder="E://OutputParamsFiles//"):
        super().__init__(fetcher, output_folder)

    def fetch_data(self, **kwargs):
        query = """
        SELECT DISTINCT [QUOTE_UNIXTIME]
        FROM [DataMining].[dbo].[OptionData];
        """
        return self.fetcher.fetch(query)

    def process_data(self, data, **kwargs):
        # No processing required for this pipeline.
        return data

    def save_output(self, data, **kwargs):
        output_file = os.path.join(self.output_folder, "outputDistinctQuotesTimes.csv")
        data.to_csv(output_file, index=False)
        print(f"File saved to {output_file}")
