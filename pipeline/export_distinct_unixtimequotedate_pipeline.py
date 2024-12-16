import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_acquisition.database_fetcher import DatabaseFetcher
import time

class UniqueQuoteTimeFetcher:
    def __init__(self, output_path = "E://OutputParamsFiles//", file_name = "outputDistinctQuotesTimes.csv"):
        self.output_path = output_path
        self.file_name = file_name
        self.output_file = os.path.join(output_path, "outputDistinctQuotesTimes.csv")
        pass

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

        # Initialize the DatabaseFetcher
        fetcher = DatabaseFetcher(connection_string, use_sqlalchemy=False)

        # Define a SQL query
        query = """
        SELECT DISTINCT [QUOTE_UNIXTIME]
        FROM [DataMining].[dbo].[OptionData];
        """

        # Fetch data
        data = fetcher.fetch(query)
        data.to_csv(self.output_file, index=False)
        print("Files has been dropped to the default folder")