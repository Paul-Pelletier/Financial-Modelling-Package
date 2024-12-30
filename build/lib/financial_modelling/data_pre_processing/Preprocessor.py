import pandas as pd

class Preprocessor:
    def __init__(self, data):
        """
        Initialize the GenericPreprocessor with the input DataFrame.

        Parameters:
        - data (pd.DataFrame): The input DataFrame to be processed.
        """
        self.data = data.copy()

    def preprocess(self, **kwargs):
        """
        Preprocess the data.

        This method should be overridden by subclasses to implement specific preprocessing logic.

        Parameters:
        - kwargs: Additional arguments for preprocessing.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame.
        """
        raise NotImplementedError("Subclasses must implement the preprocess method.")

    def validate_data(self, required_columns):
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
