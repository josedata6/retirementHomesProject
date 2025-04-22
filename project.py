import pandas as pd

# Define file paths for each year's data
files = {
    "2015": "/mnt/data/2015_CostReport.csv",
    "2016": "/mnt/data/2016_CostReport.csv",
    "2017": "/mnt/data/2017_CostReport.csv",
    "2018": "/mnt/data/2018_CostReport.csv",
    "2019": "/mnt/data/2019_CostReport.csv"
}

# Create an empty list to hold each year's DataFrame
all_dataframes = []

# Loop through files, read, and append the year
for year, path in files.items():
    df = pd.read_csv(path, low_memory=False)
    df['Year'] = int(year)
    all_dataframes.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(all_dataframes, ignore_index=True)

# Show the shape or a preview
print("Merged DataFrame shape:", merged_df.shape)
print(merged_df.head())