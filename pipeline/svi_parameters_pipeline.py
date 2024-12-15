import sys
import os

#Adds the absolute path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pre_processing.IVPreprocessor import IVPreprocessor
from data_acquisition.database_fetcher import DatabaseFetcher

import pyodbc
import pandas as pd

class DataPipeline:
    def __init__(self, date_list):
        """
        Initialize the pipeline.

        Args:
        - date_list (list)
        """
        self.date_list = date_list
        self.db_config = {
            'server': 'DESKTOP-DK79R4I',  # Your server name
            'database': 'DataMining',     # Your database name
            }
        # Define pyodbc-compatible connection string
        self.connection_string = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={self.db_config['server']};"
            f"DATABASE={self.db_config['database']};"
            f"Trusted_Connection=yes;"
            )
        
        self.fetcher = DatabaseFetcher(self.connection_string, use_sqlalchemy=False)
        self.query = ["""
                      SELECT TOP (100) [QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE]
                      FROM [DataMining].[dbo].[OptionData]
                      """]

    def fetch_data(self):
        """
        Fetch data from the database for a given list of dates.

        Returns:
        - pd.DataFrame: Fetched data.
        """
        

        for date in dates:
            query = f"""
                SELECT * FROM your_table
                WHERE CAST(quote_date_converted AS DATE) = '{date}'
            """
            data = pd.read_sql_query(query, connection)
            all_data.append(data)

        connection.close()
        return pd.concat(all_data, ignore_index=True)

    def process_data(self, data):
        """
        Process the fetched data.

        Args:
        - data (pd.DataFrame): Raw data fetched from the database.

        Returns:
        - pd.DataFrame: Processed data.
        """
        # Select only the specified columns
        processed_data = data[self.columns]
        return processed_data

    def save_to_csv(self, data):
        """
        Save the processed data to a CSV file.

        Args:
        - data (pd.DataFrame): Data to save.
        """
        data.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")

    def run(self, dates):
        """
        Run the entire pipeline.

        Args:
        - dates (list): List of dates to process.
        """
        # Fetch data
        raw_data = self.fetch_data(dates)

        # Process data
        processed_data = self.process_data(raw_data)

        # Save to CSV
        self.save_to_csv(processed_data)

# Configuration
db_connection_str = "DRIVER={SQL Server};SERVER=your_server;DATABASE=your_db;UID=your_user;PWD=your_password"
output_file = "output.csv"
columns = ["column1", "column2", "column3"]

# Run the pipeline
pipeline = DataPipeline(db_connection_str, output_file, columns)
dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
pipeline.run(dates)
