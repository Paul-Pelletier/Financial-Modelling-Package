import pandas as pd
from .Preprocessor import Preprocessor

class ForwardComputationPreprocessor(Preprocessor):
    """
    Preprocessor for preparing data used in forward computation from call and put options.
    """
    
    def preprocess(self, drop_criteria: dict) -> pd.DataFrame:
        """
        Preprocess the data by applying filtering criteria.
        
        Parameters:
        - drop_criteria (dict): A dictionary where keys are column names and values are lambda functions
                               specifying filtering conditions.

        Returns:
        - pd.DataFrame: The filtered DataFrame.
        """
        self.validate_data([
            'QUOTE_UNIXTIME', 'UNDERLYING_LAST', 'EXPIRE_UNIX', 'DTE',
            'C_IV', 'P_IV', 'C_VOLUME', 'P_VOLUME', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK'
        ])
        
        df = self.data.copy()
        
        # Apply filtering criteria
        for column, condition in drop_criteria.items():
            if column in df.columns:
                df = df[condition(df[column])]
        
        return df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Fill missing values with column mean for numerical columns."""
        self.data = self.data.fillna(self.data.mean())
        return self.data

    def normalize_data(self) -> pd.DataFrame:
        """Normalize numerical features using min-max scaling."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].min()) / (self.data[numeric_cols].max() - self.data[numeric_cols].min())
        return self.data

    def encode_categorical(self) -> pd.DataFrame:
        """Convert categorical variables using one-hot encoding."""
        self.data = pd.get_dummies(self.data, drop_first=True)
        return self.data

    def remove_outliers(self) -> pd.DataFrame:
        """Remove outliers beyond 3 standard deviations from the mean."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data = self.data[(np.abs(self.data[numeric_cols] - self.data[numeric_cols].mean()) <= (3 * self.data[numeric_cols].std())).all(axis=1)]
        return self.data