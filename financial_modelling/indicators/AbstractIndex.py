from abc import ABC, abstractmethod
import numpy as np

class AbstractIndicators(ABC):
    """
    Abstract base class for statistical indicators.
    Defines the interface for classes that compute statistical indicators.
    """

    def __init__(self, moneyness_column, call_implied_volatility, put_implied_volatility):
        """
        Initialize the base class with common data.
        """
        self.moneyness = moneyness_column
        self.call_implied_volatility = call_implied_volatility
        self.put_implied_volatility = put_implied_volatility
        self.call_minus_put_iv = self.put_implied_volatility - self.call_implied_volatility

    @abstractmethod
    def calculate_median(self):
        """
        Calculate the median of the call-put IV difference.
        """
        pass

    @abstractmethod
    def calculate_standard_deviation(self):
        """
        Calculate the standard deviation of the call-put IV difference.
        """
        pass

    @abstractmethod
    def calculate_skewness(self):
        """
        Calculate the skewness of the call-put IV difference.
        """
        pass

    @abstractmethod
    def calculate_kurtosis(self):
        """
        Calculate the kurtosis of the call-put IV difference.
        """
        pass

    @abstractmethod
    def calculate_moneyness_range(self):
        """
        Calculate the range of moneyness (max - min).
        """
        pass
