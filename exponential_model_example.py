import pandas as pd
from modelling.Exponential_model import ExponentialModel

# Load the cleaned data
data = pd.read_csv("cleaned_data.csv", sep=";")

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
