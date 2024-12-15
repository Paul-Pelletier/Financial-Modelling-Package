import pandas as pd
import os
import sys
#Allows for importing neighbouring packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pre_processing.IVPreprocessor import IVPreprocessor

# Example DataFrame
data = pd.DataFrame({
    'Spot': [100, 100, 100, 100, 100],
    'Strike': [90, 95, 100, 105, 110],
    'Call_IV': [0.2, 0.18, 0.15, 0.13, 0.12],
    'Put_IV': [0.25, 0.22, 0.2, 0.18, 0.16],
    'Call_Volume': [1000, 800, 600, 500, 400],
    'Put_Volume': [1200, 1000, 700, 600, 500]
})

# Initialize preprocessor
preprocessor = IVPreprocessor(
    data,
    spot_col='Spot',
    strike_col='Strike',
    call_iv_col='Call_IV',
    put_iv_col='Put_IV',
    call_vol_col='Call_Volume',
    put_vol_col='Put_Volume'
)

# Define limits
call_limits = (0.9, 1.1)
put_limits = (0.85, 1.05)

# Process data
processed_data = preprocessor.preprocess(call_limits, put_limits, mode="split")
print(processed_data)