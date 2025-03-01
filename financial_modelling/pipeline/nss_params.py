from financial_modelling.modelling.nss_model import NelsonSiegelSvensson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "E:\\ForwardComputations\\FittedData\\2021\\04\\forward_computation_1617287520.csv"
file = pd.read_csv(file_path, sep = ",")
file = file[-np.isinf(file['FORWARD'])]
print(file)
maturities = (1/(3600*24*365))*(file["EXPIRE_UNIX"]-file["QUOTE_UNIXTIME"])
forward_rates = file["FORWARD"]
nss = NelsonSiegelSvensson()
nss.fit(maturities, forward_rates)
npmaturities = np.linspace(0, 3, 100)
plt.plot(maturities, forward_rates, label="Observed")
plt.show()
predicted_rates = nss.predict(maturities)
nss.plot(npmaturities, forward_rates)