from scipy.stats import skew, kurtosis

class ImpliedVolatilitySmileIndicators(AbstractIndicators):
    """
    Implements statistical indicators to analyze IV smile configurations.
    """

    def __init__(self, moneyness_column, call_implied_volatility, put_implied_volatility):
        super().__init__(moneyness_column, call_implied_volatility, put_implied_volatility)
        self.maximum_moneyness = self.moneyness.max()
        self.minimum_moneyness = self.moneyness.min()

    def calculate_median(self):
        """
        Calculate the median of the call-put IV difference.
        """
        self.median = np.median(self.call_minus_put_iv)
        return self.median

    def calculate_standard_deviation(self):
        """
        Calculate the standard deviation of the call-put IV difference.
        """
        self.standard_dev = np.std(self.call_minus_put_iv, ddof=1)
        return self.standard_dev

    def calculate_skewness(self):
        """
        Calculate the skewness of the call-put IV difference.
        """
        self.skew = skew(self.call_minus_put_iv)
        return self.skew

    def calculate_kurtosis(self):
        """
        Calculate the kurtosis of the call-put IV difference.
        """
        self.kurtosis = kurtosis(self.call_minus_put_iv)
        return self.kurtosis

    def calculate_moneyness_range(self):
        """
        Calculate the range of moneyness.
        """
        self.moneyness_range = self.maximum_moneyness - self.minimum_moneyness
        return self.moneyness_range
