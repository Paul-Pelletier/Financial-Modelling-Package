import os
import pandas as pd

# Define directories
INPUT_DIR = "E:\\ForwardComputations\\FittedParams"  # Source folder
OUTPUT_DIR = "E:\\ForwardComputations\\FittedAggregated"  # Destination for yearly aggregation

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_quote_unixtime(filename):
    """Extracts quote_unixtime from filenames like 'fitted_params_forward_computation_1550176320.csv'."""
    try:
        return int(filename.split("_")[-1].replace(".csv", ""))
    except ValueError:
        return None  # Skip invalid filenames

def aggregate_yearly_data(year):
    """Aggregates all fitted parameter CSVs for a given year into a single CSV."""
    year_path = os.path.join(INPUT_DIR, str(year))
    if not os.path.exists(year_path):
        print(f"⚠️ No data found for {year}. Skipping...")
        return
    
    aggregated_data = []

    # Traverse all months in the given year
    for month in sorted(os.listdir(year_path)):
        month_path = os.path.join(year_path, month)
        if not os.path.isdir(month_path):  # Skip non-folder files
            continue
        
        # Process all CSV files in each month folder
        for file in os.listdir(month_path):
            if file.startswith("fitted_params_forward_computation") and file.endswith(".csv"):
                file_path = os.path.join(month_path, file)
                quote_unixtime = extract_quote_unixtime(file)
                
                if quote_unixtime:
                    df = pd.read_csv(file_path)
                    df["quote_unixtime"] = quote_unixtime  # Add time marker
                    aggregated_data.append(df)
    
    if aggregated_data:
        # Combine all DataFrames and sort by `quote_unixtime`
        final_df = pd.concat(aggregated_data, ignore_index=True)
        final_df = final_df.sort_values(by="quote_unixtime")

        # Save aggregated file per year
        output_file = os.path.join(OUTPUT_DIR, f"{year}.csv")
        final_df.to_csv(output_file, index=False)
        print(f"✅ Aggregated {len(final_df)} rows for {year} → {output_file}")
    else:
        print(f"⚠️ No valid data found for {year}")

def aggregate_all_years():
    """Automatically detects available years and processes them separately."""
    years = [folder for folder in os.listdir(INPUT_DIR) if folder.isdigit()]
    
    print(f"📌 Found {len(years)} years to process...")
    
    for year in sorted(years):
        aggregate_yearly_data(year)

if __name__ == "__main__":
    aggregate_all_years()
