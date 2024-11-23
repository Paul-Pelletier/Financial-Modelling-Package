import pandas as pd
import numpy as np

class DataCleaner:
    """
    Cleans raw data fetched from the data_acquisition module.
    """

    def __init__(self):
        pass

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame and returns a processed DataFrame.

        Parameters:
        ----------
        data : pd.DataFrame
            Raw data to clean.

        Returns:
        -------
        pd.DataFrame
            Cleaned data.
        """
        # Step 1: Remove duplicates
        data = data.drop_duplicates()

        # Step 2: Handle missing values
        missing_threshold = 0.2  # Drop columns with >20% missing values
        data = data.loc[:, data.isnull().mean() < missing_threshold]  # Drop columns
        data = data.fillna(method='ffill').fillna(method='bfill')  # Fill remaining missing values

        # Step 3: Ensure correct data types
        for column in data.select_dtypes(include=['object']).columns:
            if column.lower().endswith("date"):
                data[column] = pd.to_datetime(data[column], errors='coerce')
            else:
                try:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                except Exception:
                    pass

        # Step 4: Filter outliers
        for column in data.select_dtypes(include=[np.number]).columns:
            data = self.filter_outliers(data, column)

        # Step 5: Normalize column names
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

        return data

    def filter_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Removes extreme outliers from a numeric column using the IQR method.

        Parameters:
        ----------
        data : pd.DataFrame
            The DataFrame containing the column to filter.
        column : str
            The name of the column to process.

        Returns:
        -------
        pd.DataFrame
            DataFrame with outliers removed.
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    def validate_computed_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the correctness of pre-computed columns, such as bid-ask spreads.

        Parameters:
        ----------
        data : pd.DataFrame
            The cleaned data with pre-computed columns.

        Returns:
        -------
        pd.DataFrame
            Data with validated columns.
        """
        if 'c_bidaskspread' in data.columns and 'c_bid' in data.columns and 'c_ask' in data.columns:
            data['c_bidaskspread_check'] = data['c_ask'] - data['c_bid']
            data['c_bidaskspread_valid'] = np.isclose(data['c_bidaskspread'], data['c_bidaskspread_check'])
        
        if 'p_bidaskspread' in data.columns and 'p_bid' in data.columns and 'p_ask' in data.columns:
            data['p_bidaskspread_check'] = data['p_ask'] - data['p_bid']
            data['p_bidaskspread_valid'] = np.isclose(data['p_bidaskspread'], data['p_bidaskspread_check'])

        return data
