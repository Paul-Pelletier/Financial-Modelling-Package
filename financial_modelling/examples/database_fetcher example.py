from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
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
queries = ["""
SELECT TOP (100) [QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE]
FROM [DataMining].[dbo].[OptionData]
"""]*10

# Fetch data
try:
    for query_number, query in enumerate(queries):
        start_time = time.time()
        data = fetcher.fetch(query)
        execution_time = time.time() - start_time
        #print(data_0.head())
        #print(f"Query number {query_number+1} executed in {execution_time:.4f} seconds")
except ValueError as e:
    print(f"Error: {e}")
