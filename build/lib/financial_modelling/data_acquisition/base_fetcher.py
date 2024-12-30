from abc import ABC, abstractmethod
import pandas as pd

class DataFetcher(ABC):
    """
    Abstract base class for fetching data from various sources.
    """
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetches data and returns it as a Pandas DataFrame.
        
        Returns:
        -------
        pd.DataFrame
            The fetched data.
        """
        pass
