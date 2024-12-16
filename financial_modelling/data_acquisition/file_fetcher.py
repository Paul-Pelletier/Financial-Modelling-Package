import pandas as pd
from .base_fetcher import DataFetcher

class FileFetcher(DataFetcher):
    """
    Fetches data from CSV or TXT files.
    """
    def fetch(self, filepath: str, separator: str = ",", **kwargs) -> pd.DataFrame:
        """
        Loads data from a file into a Pandas DataFrame.
        
        Parameters:
        ----------
        filepath : str
            The path to the file.
        separator : str, optional
            The delimiter for the file (default is ",").
        
        Returns:
        -------
        pd.DataFrame
            The loaded data.
        
        Raises:
        ------
        ValueError
            If the file cannot be loaded.
        """
        try:
            return pd.read_csv(filepath, sep=separator, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")
