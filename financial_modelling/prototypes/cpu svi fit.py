import numpy as np
import pandas as pd
from financial_modelling.data_acquisition.file_fetcher import FileFetcher
from financial_modelling.data_pre_processing.IVPreprocessor import IVPreprocessor
from NonLinearSVICPUModel import NonLinearModel

#Fetch file
data = FileFetcher().fetch(filepath = r"raw_data.csv", separator= ";")

#Pre-process file mostly trimming data for calibration purposes

#The result is a Pandas DataFrame with following columns :
#["Log-Moneyness Log(K/S)", "Implied_Volatility", "Volume", "Option Type",
#"Residual_Maturity", "STRIKE_DISTANCE", "QUOTE_UNIXTIME", "EXPIRE_UNIX"]
pre_processed_data = IVPreprocessor(data).preprocess()

#Splits the pre-processed on a expire_unix-wise way
#, subset[0]["Implied_Volatility"],  subset[0]["Residual_Maturity"].unique() 
data_split = [(subset[1]["Log_Moneyness"], subset[1]["Implied_Volatility"]) 
              for subset in pre_processed_data.groupby(['EXPIRE_UNIX'])]
data_split = data_split.T
x_train_split_data, y_train_split_data = data_split
initial_params = np.array([[0.05, 0.2, 0.0, 0.0, 0.1] for i in maturities.keys()])
model = NonLinearModel(initial_params = initial_params)
y_pred = model.functional_form(x_train_split_data, initial_params, maturities) 
