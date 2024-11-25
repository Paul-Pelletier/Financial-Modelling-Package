from indicators.IVRawIndicators import ImpliedVolatilitySmileIndicators
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('raw_IV_data.csv', sep = ";")
print(data.columns)
for expiry in pd.unique(data["YTE"]):
    first_smile = data[data["YTE"] == expiry][["Moneyness", "C_IV", "P_IV"]]
    print(first_smile)
    moneyness = first_smile['Moneyness']
    call_implied_volatility = first_smile['C_IV']
    put_implied_volatility = first_smile['P_IV']
    plt.scatter(moneyness, put_implied_volatility, label = "puts")
    plt.scatter(moneyness, call_implied_volatility, label = "calls")
    plt.legend()
    plt.grid()
    plt.show()

indicators = ImpliedVolatilitySmileIndicators(moneyness, call_implied_volatility, put_implied_volatility)
attributes = vars(indicators)
print(attributes)
