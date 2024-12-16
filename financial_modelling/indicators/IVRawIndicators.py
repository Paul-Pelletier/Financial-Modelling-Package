import numpy as np
from scipy.stats import skew, kurtosis

class ImpliedVolatilitySmileIndicators():
    """
    Allows to compute statistical indicators that can capture several IV smiles configuration
    """
    
    def __init__(self, moneyness_column, call_implied_volatility,put_implied_volatility):
        """
        Initialize the Implied Volatility Smile Indicators.
        : param call_implied_volatility : Call implied volatility market data
        : param put_implied_volatility : put implied volatility market data
        : param moneyness : moneyness from the available options chain
        """
        self.call_implied_volatility = call_implied_volatility
        self.put_implied_volatility = put_implied_volatility
        self.moneyness = moneyness_column
        self.maximum_moneyness = self.moneyness.max()
        self.minimum_moneyness = self.moneyness.min()
        self.call_minus_put_iv = self.put_implied_volatility - self.call_implied_volatility
        self.median = np.median(self.call_minus_put_iv)
        self.standard_dev = np.std(self.call_minus_put_iv, ddof = 1)
        self.skew = skew(self.call_minus_put_iv)
        self.kurtosis = kurtosis(self.call_minus_put_iv)