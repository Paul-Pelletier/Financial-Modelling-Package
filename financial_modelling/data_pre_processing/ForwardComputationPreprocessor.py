import pandas as pd
from .Preprocessor import Preprocessor
import logging
import logging

# Activate logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info("Starting preprocessing with drop criteria: %s", drop_criteria.keys())
        self.validate_data([
            'QUOTE_UNIXTIME', 'UNDERLYING_LAST', 'EXPIRE_UNIX', 'DTE',
            'C_IV', 'P_IV', 'C_VOLUME', 'P_VOLUME', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK'
        ])
        
        df = self.data.copy()
        logging.info("Initial data shape before filtering: %s", df.shape)
        
        # Convert numeric fields to appropriate types to avoid TypeError
        numeric_cols = ['QUOTE_UNIXTIME', 'UNDERLYING_LAST', 'EXPIRE_UNIX', 'DTE',
                        'C_IV', 'P_IV', 'STRIKE', 'EXPIRE_UNIX', 'C_VOLUME', 'P_VOLUME', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info("Converted column '%s' to numeric.", col)
        
        # Drop NaN values that result from failed conversions
        df = df.dropna(subset=numeric_cols)
        logging.info("Dropped rows with NaN values in numeric columns.")
        
        # Apply filtering criteria
        for column, condition in drop_criteria.items():
            if column in df.columns:
                before_filtering = df.shape[0]
                df = df[condition(df[column])]
                after_filtering = df.shape[0]
                logging.info("Filtered column '%s': %d -> %d rows remaining", column, before_filtering, after_filtering)
        
        logging.info("Final data shape after preprocessing: %s", df.shape)
        return df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Fill missing values with column mean for numerical columns."""
        self.data = self.data.fillna(self.data.mean())
        logging.info("Handled missing values in data.")
        return self.data

    def normalize_data(self) -> pd.DataFrame:
        """Normalize numerical features using min-max scaling."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].min()) / (self.data[numeric_cols].max() - self.data[numeric_cols].min())
        logging.info("Normalized numerical data.")
        return self.data

    def encode_categorical(self) -> pd.DataFrame:
        """Convert categorical variables using one-hot encoding."""
        self.data = pd.get_dummies(self.data, drop_first=True)
        logging.info("Encoded categorical variables.")
        return self.data

    def remove_outliers(self) -> pd.DataFrame:
        """Remove outliers beyond 3 standard deviations from the mean."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        before_filtering = self.data.shape[0]
        self.data = self.data[(np.abs(self.data[numeric_cols] - self.data[numeric_cols].mean()) <= (3 * self.data[numeric_cols].std())).all(axis=1)]
        after_filtering = self.data.shape[0]
        logging.info("Removed outliers: %d -> %d rows remaining", before_filtering, after_filtering)
        return self.data
