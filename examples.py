from data_acquisition.database_fetcher import DatabaseFetcher
import time

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
SELECT TOP (100) [QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE]
FROM [DataMining].[dbo].[OptionData]
"""

# Fetch data
try:
    start_time = time.time()
    data = fetcher.fetch(query)
    execution_time = time.time() - start_time
    print(data.head())
    print(f"Query executed in {execution_time:.4f} seconds")
except ValueError as e:
    print(f"Error: {e}")
