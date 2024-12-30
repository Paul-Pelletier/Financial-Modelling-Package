import numpy as np
import pandas as pd
from financial_modelling.data_acquisition.file_fetcher import FileFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor

#Fetch file
data = FileFetcher().fetch(filepath = r"raw_data.csv", separator= ";")

#Pre-process file mostly trimming data for calibration purposes

#The result is a Pandas DataFrame with following columns :
#["Log-Moneyness Log(K/S)", "Implied_Volatility", "Volume", "Option Type",
#"Residual_Maturity", "STRIKE_DISTANCE", "QUOTE_UNIXTIME", "EXPIRE_UNIX"]
pre_processed_data = IVPreprocessor(data).preprocess()

#Splits the pre-processed data in 
per_maturity_split_data = {key: subset for key, subset in data.groupby(['EXPIRE_UNIX'])}

