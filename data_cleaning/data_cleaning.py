import pandas as pd
import numpy as np

class DataCleaner:
    """
    Cleans raw data, trims values by percentiles, and computes weighted averages
    for observations with the same key column (e.g., YTE), while excluding single data points.
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

        # Step 3: Compute weighted average for each unique key_column
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
