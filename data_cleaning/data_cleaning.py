import pandas as pd
import numpy as np

class DataCleaner:
    """
    Cleans raw data, trims values by percentiles, and computes weighted averages
    for observations with the same key column (e.g., YTE), while excluding single data points.
    Additionally, it includes checks and enhancements specific to SABR model fitting.
    """

    def __init__(self, key_column="YTE", value_column="calc", weight_column="volume"):
        """
        Initializes the DataCleaner with customizable column names.

        Parameters:
        ----------
        key_column : str
            The column name for the key grouping (e.g., YTE).
        value_column : str
            The column name for the values to trim and average (e.g., calc).
        weight_column : str
            The column name for the weights used in averaging (e.g., volume).
        """
        self.key_column = key_column
        self.value_column = value_column
        self.weight_column = weight_column

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by trimming and applying weighted aggregation.
        Additional filtering specific to SABR model fitting is also applied.

        Parameters:
        ----------
        data : pd.DataFrame
            The raw input data to clean.

        Returns:
        -------
        pd.DataFrame
            A cleaned DataFrame with unique key_column values and their weighted averages.
        """
        # Step 1: Remove groups with a single data point
        data = self._remove_single_data_points(data)

        # Step 2: Trim data by percentiles (25th and 75th)
        data = self._trim_by_percentiles(data)

        # Step 3: Apply SABR-specific data cleaning (e.g., IV filtering, strike/underlying filtering)
        data = self._apply_sabr_data_cleaning(data)

        # Step 4: Compute weighted average for each unique key_column
        cleaned_data = self._compute_weighted_average(data)

        return cleaned_data

    def _remove_single_data_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows belonging to groups in key_column with only a single data point.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame excluding groups with a single data point.
        """
        # Filter out groups with only one data point
        group_sizes = data.groupby(self.key_column).size()
        valid_keys = group_sizes[group_sizes > 1].index
        return data[data[self.key_column].isin(valid_keys)]

    def _trim_by_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Trims the value_column by removing rows outside the 25th-75th percentile range.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        pd.DataFrame
            A trimmed DataFrame.
        """
        trimmed_data = []
        for key, group in data.groupby(self.key_column):
            # Compute percentiles
            Q1 = group[self.value_column].quantile(0.25)
            Q3 = group[self.value_column].quantile(0.75)

            # Filter rows within the percentile range
            trimmed_group = group[(group[self.value_column] >= Q1) & (group[self.value_column] <= Q3)]
            trimmed_data.append(trimmed_group)

        return pd.concat(trimmed_data, ignore_index=True)

    def _apply_sabr_data_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies SABR-specific data cleaning steps such as filtering out rows with invalid
        implied volatilities, strike prices, and underlying prices.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        pd.DataFrame
            A cleaned DataFrame with valid implied volatilities and prices.
        """
        # Step 1: Filter for valid implied volatilities (positive and reasonable)
        data = data[data['C_IV'] > 0]  # Calls implied volatility
        data = data[data['P_IV'] > 0]  # Puts implied volatility

        # Step 2: Filter for valid underlying prices and strike prices (non-zero)
        data = data[data['UNDERLYING_LAST'] > 0]  # Ensure non-zero underlying price
        data = data[data['STRIKE'] > 0]  # Ensure valid strike prices

        # Step 3: Use existing mid prices for filtering if necessary
        data = data[data['C_MID'] > 0]  # Ensure valid call mid prices
        data = data[data['P_MID'] > 0]  # Ensure valid put mid prices

        # Step 4: Filter for non-zero bid prices (optional, for stricter quality)
        data = data[data['C_BID'] > 0]  # Valid call options
        data = data[data['P_BID'] > 0]  # Valid put options

        return data

    def _compute_weighted_average(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the weighted average of value_column using weight_column for each unique key_column.

        Parameters:
        ----------
        data : pd.DataFrame
            The trimmed DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with unique key_column values and their weighted averages.
        """
        # Weighted average calculation
        aggregated_data = data.groupby(self.key_column).apply(
            lambda group: pd.Series({
                self.value_column: np.average(group[self.value_column], weights=group[self.weight_column]),
                self.weight_column: group[self.weight_column].sum()  # Optional: Keep total weight if needed
            })
        ).reset_index()

        return aggregated_data
