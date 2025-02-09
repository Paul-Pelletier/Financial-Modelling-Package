from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.ForwardComputationPreprocessor import ForwardComputationPreprocessor

QUOTE_UNIXTIME = 1546439410

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
query = f"""
        SELECT TOP(6302) *
        FROM [DataMining].[dbo].[OptionData]
        WHERE [QUOTE_UNIXTIME] = '{QUOTE_UNIXTIME}'
        """

raw_data = fetcher.fetch(query)

drop_criteria = {
    'C_IV': lambda x: x.notna() & (x > 0.05),  # Keep rows where Call IV is not NaN and greater than 0.05
    'P_IV': lambda x: x.notna() & (x > 0.04),  # Keep rows where Put IV is not NaN and greater than 0.04
    'C_VOLUME': lambda x: x.notna() & (x >= 1),  # Keep rows where Call Volume is not NaN and at least 10
    'P_VOLUME': lambda x: x.notna() & (x >= 1)  # Keep rows where Put Volume is not NaN and at least 10
}
preprocessor = ForwardComputationPreprocessor(raw_data)
filtered_data = preprocessor.preprocess(drop_criteria)
print(filtered_data.head)
