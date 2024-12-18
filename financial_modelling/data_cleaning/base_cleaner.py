from abc import ABC, abstractmethod
import pandas as pd


class BaseCleaner(ABC):
    """
    Abstract base class for cleaning data.
    All derived classes must implement these methods.
    """

    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame and returns a cleaned DataFrame.

        Parameters:
        ----------
        data : pd.DataFrame
            The raw input data to clean.

        Returns:
        -------
        pd.DataFrame
            A cleaned DataFrame.
        """
        pass
