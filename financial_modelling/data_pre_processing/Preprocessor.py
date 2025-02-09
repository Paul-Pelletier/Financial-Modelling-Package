from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Preprocessor(ABC):
    """
    Abstract base class for data preprocessing.
    Defines a standard interface for preprocessing steps.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Preprocessor with the input DataFrame.
        
        Parameters:
        - data (pd.DataFrame): The input DataFrame to be processed.
        """
        self.data = data.copy()
    
    @abstractmethod
    def preprocess(self, **kwargs) -> pd.DataFrame:
        """
        Preprocess the data.
        This method should be overridden by subclasses to implement specific preprocessing logic.
        
        Parameters:
        - kwargs: Additional arguments for preprocessing.
        
        Returns:
        - pd.DataFrame: The preprocessed DataFrame.
        """
        pass
    
    def validate_data(self, required_columns: list):
        """
        Validate that the required columns exist in the data.
        
        Parameters:
        - required_columns (list): A list of column names that must exist in the DataFrame.
        
        Raises:
        - ValueError: If any of the required columns are missing.
        """
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing: {missing_columns}")

    @abstractmethod
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        pass

    @abstractmethod
    def normalize_data(self) -> pd.DataFrame:
        """Normalize numerical features."""
        pass

    @abstractmethod
    def encode_categorical(self) -> pd.DataFrame:
        """Encode categorical variables."""
        pass

    @abstractmethod
    def remove_outliers(self) -> pd.DataFrame:
        """Remove or adjust outliers in the dataset."""
        pass
