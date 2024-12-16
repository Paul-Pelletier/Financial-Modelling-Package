import os
from financial_modelling.indicators.IVRawIndicators import ImpliedVolatilitySmileIndicators
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the CSV file in the same folder
csv_path = os.path.join(script_dir, "raw_IV_data.csv")

data = pd.read_csv(csv_path, sep = ";")
print(data.columns)
first_smile = data[data["YTE"] == pd.unique(data["YTE"])[0]][["Moneyness", "C_IV", "P_IV"]]
print(first_smile)
moneyness = first_smile['Moneyness']
call_implied_volatility = first_smile['C_IV']
put_implied_volatility = first_smile['P_IV']
indicators = ImpliedVolatilitySmileIndicators(moneyness, call_implied_volatility, put_implied_volatility)
plt.scatter(moneyness, put_implied_volatility, label = "puts")
plt.scatter(moneyness, call_implied_volatility, label = "calls")
plt.scatter(moneyness, indicators.call_minus_put_iv, label = "indicator")
plt.legend()
plt.grid()
plt.show()

#indicators = ImpliedVolatilitySmileIndicators(moneyness, call_implied_volatility, put_implied_volatility)
#attributes = vars(indicators)
#print(attributes)
