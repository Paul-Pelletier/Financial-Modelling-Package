from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
from financial_modelling.data_pre_processing.ForwardComputationPreprocessor import ForwardComputationPreprocessor
import logging
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Activate logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
forward_list = []
expiries = filtered_data['EXPIRE_UNIX'].unique()
expiries_list = []
for expiry in expiries:
    expiry_specific_data = filtered_data[filtered_data['EXPIRE_UNIX'] == expiry]
    expiry_specific_data.loc[:, 'MidCallMidPutPArity'] = expiry_specific_data['C_MID'] - expiry_specific_data['P_MID']
    model = LinearRegression()
    model.fit(expiry_specific_data[['STRIKE']], expiry_specific_data['MidCallMidPutPArity'])
    # Print model coefficients
    logging.info("Intercept: %f", model.intercept_)
    logging.info("Slope: %f", model.coef_[0])
    discountedForward, discountFactor  = model.intercept_, -model.coef_[0]
    Forward = discountedForward/discountFactor
    forward_list.append(Forward)
    expiries_list.append((expiry-QUOTE_UNIXTIME)/365)
    logging.info("Forward: %f for expiry: %d", Forward, expiry)

plt.plot(expiries, forward_list, label='Forward')
plt.show()