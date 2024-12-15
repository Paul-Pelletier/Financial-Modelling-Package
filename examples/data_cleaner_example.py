import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

#Allows for importing neighbouring packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_cleaning.data_cleaning import DataCleaner
from utils.utils import replace_commas

# Sample dataset
converters = {"C_IV": replace_commas, "P_IV": replace_commas}

data = pd.read_csv("raw_data.csv", sep=";", converters = converters)

# Initialize the cleaner with custom column names (if needed)
cleaner = DataCleaner(key_column="YTE", value_column="calc", weight_column="volume")

# Clean the data
cleaned_data = cleaner.clean_data(data)

# Display the cleaned data
print("Cleaned Data:")
print(cleaned_data)

# Scatter plot function to compare raw vs cleaned data
def plot_raw_vs_cleaned(raw_data, cleaned_data, x_column, y_column):
    """
    Plots a scatter plot comparing raw vs cleaned data.

    Parameters:
    ----------
    raw_data : pd.DataFrame
        The original raw dataset.
    cleaned_data : pd.DataFrame
        The cleaned dataset.
    x_column : str
        The column to use as the x-axis (e.g., "YTE").
    y_column : str
        The column to use as the y-axis (e.g., "calc").
    """
    plt.figure(figsize=(10, 6))

    # Plot raw data
    plt.scatter(raw_data[x_column], raw_data[y_column], alpha=0.5, label="Raw Data", color="red")

    # Plot cleaned data
    plt.scatter(cleaned_data[x_column], cleaned_data[y_column], alpha=0.8, label="Cleaned Data", color="blue")

    # Labels and legend
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"Comparison of Raw vs Cleaned Data ({x_column} vs {y_column})")
    plt.legend()
    plt.grid()
    plt.show()


# Call the plot function
plot_raw_vs_cleaned(data, cleaned_data, "YTE", "calc")
