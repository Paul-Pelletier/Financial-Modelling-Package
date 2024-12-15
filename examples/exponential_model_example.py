import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from modelling.Exponential_model import ExponentialModel

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the CSV file in the same folder
csv_path = os.path.join(script_dir, "cleaned_data.csv")

# Load the cleaned data
data = pd.read_csv(csv_path, sep=";")

# Extract the independent and dependent variables
x = data["YTE"].values
y = data["Forward"].values

model = ExponentialModel(a=2.0, b=-0.5, c=1.0)

# Fit the model to data
optimized_params = model.fit(x, y)

# Print the optimized parameters
print("Optimized Parameters:")
print(f"a = {optimized_params[0]:.4f}, b = {optimized_params[1]:.4f}, c = {optimized_params[2]:.4f}")

# Plot the results
model.plot(x, y)
