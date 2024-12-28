import pandas as pd
from .base_fetcher import DataFetcher
import os

class ListOfFilesFetcher(DataFetcher):
    """
    Fetches data from CSV or TXT files.
    """
    def __init__(self):
        self.list_of_files = []
        self.list_of_dates = []

    def fetch(self, folderpath: str, **kwargs) ->list :
        """
        Loads list if files from a folder into a list.
        
        Parameters:
        ----------
        filepath : str
            The path to the folder.
        
        Returns:
        -------
        pd.DataFrame
            The list of files.
        
        Raises:
        ------
        ValueError
            If the fomder cannot be found.
        """
        try:
            self.list_of_files = os.listdir(folderpath)
            return self.list_of_files
        except Exception as e:
            raise ValueError(f"Failed to find folder: {e}")
    
    def get_unixtimestamp(self):
        list_of_files = self.list_of_files
        list_of_files.remove(list_of_files[0])
        list_of_dates = [int(list_of_files[i].split("_")[1].split(".")[0]) for i in range(len(list_of_files))]
        list_of_dates.sort()
        self.list_of_dates = list_of_dates
        return list_of_dates
