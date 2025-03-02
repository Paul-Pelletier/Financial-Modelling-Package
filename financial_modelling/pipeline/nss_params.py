from financial_modelling.modelling.nss_model import NelsonSiegelSvensson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#file_path = "E:\\ForwardComputations\\FittedData\\2021\\04\\forward_computation_1617287520.csv"
file_path = "E:\\ForwardComputations\\FittedData\\2023\\07\\forward_computation_1688391360.csv"
#file_path = "E:\\ForwardComputations\\FittedData\\2022\\06\\forward_computation_1655390220.csv"
file = pd.read_csv(file_path, sep = ",")
file = file[-np.isinf(file['FORWARD'])]
print(file)
maturities = (1/(3600*24*365))*(file["EXPIRE_UNIX"]-file["QUOTE_UNIXTIME"])
forward_rates = file["FORWARD"]
nss = NelsonSiegelSvensson(beta0=forward_rates[0], beta1=forward_rates[0], beta2=forward_rates[0], lambda1=0.1)
nss.fit(maturities, forward_rates)
print(nss.beta0, nss.beta1, nss.beta2, nss.beta3, nss.lambda1, nss.lambda2)
predicted_rates = nss.predict(maturities)
nss.plot(maturities, forward_rates)
maturities_extended = np.linspace(0, 5, 100)
predicted_rates = nss.predict(maturities)
plt.plot(maturities_extended, nss.predict(maturities_extended), label="NSS forward curve")
plt.plot(maturities, forward_rates, "o", label="Observed rates")
plt.show()