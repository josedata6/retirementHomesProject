import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import numpy as np

# Define file paths for each year's data
# Creates a dictionary called files that maps each year to the corresponding CSV file name
files = {
    "2015": "2015_CostReport.csv",
    "2016": "2016_CostReport.csv",
    "2017": "2017_CostReport.csv",
    "2018": "2018_CostReport.csv",
    "2019": "2019_CostReport.csv",
    "2020": "2020_CostReport.csv",
    "2021": "2021_CostReport.csv"}

# Create an empty list to hold each year's DataFrame
all_dataframes = []

# Loop through files, read, and append the year
for year, path in files.items():
    df = pd.read_csv(path, low_memory=False)
    df['Year'] = int(year)
    all_dataframes.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(all_dataframes, ignore_index=True)

# # Show the shape or a preview
# print("Merged DataFrame shape:", merged_df.shape)
# print(merged_df.head())

# # Check the columns of the merged DataFrame
# print(merged_df.columns)

# # Get basic statistics
# print(merged_df.describe())

# print(merged_df.groupby("Year")['Net_Income'].mean())  # Yearly average net income

financial_metrics = [
    'net_income',
    'gross_revenue',
    'total_income',
    'total_costs',
    'total_current_assets',
    'total_current_liabilities',
    'total_liabilities',
    'total_fund_balances',
    'total_fixed_assets'
]
# Convert financial metrics to numeric, coercing errors to NaN
# This will ensure that any non-numeric values in these columns are converted to NaN
for col in financial_metrics:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

yearly_summary = merged_df.groupby("Year")[financial_metrics].mean()
print(yearly_summary)

yearly_summary['profit_margin'] = yearly_summary['net_income'] / yearly_summary['gross_revenue']
yearly_summary['roa'] = yearly_summary['net_income'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
yearly_summary['liability_to_asset_ratio'] = yearly_summary['total_liabilities'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
yearly_summary['current_ratio'] = yearly_summary['total_current_assets'] / yearly_summary['total_current_liabilities']
yearly_summary['debt_to_equity_ratio'] = yearly_summary['total_liabilities'] / yearly_summary['total_fund_balances']
yearly_summary['asset_turnover'] = yearly_summary['gross_revenue'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
yearly_summary['return_on_equity'] = yearly_summary['net_income'] / yearly_summary['total_fund_balances']
yearly_summary['return_on_assets'] = yearly_summary['net_income'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
yearly_summary['return_on_investment'] = yearly_summary['net_income'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
yearly_summary['debt_ratio'] = yearly_summary['total_liabilities'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])

# Plot Net Income
yearly_summary['net_income'].plot(marker='o', title='Average Net Income Over Time')
plt.ylabel("Net Income ($)")
plt.grid(True)
plt.show()

# Plot Profit Margin
yearly_summary['profit_margin'].plot(marker='o', title='Profit Margin Over Time')
plt.ylabel("Profit Margin")
plt.grid(True)
plt.show()

# Round for readability
yearly_summary = yearly_summary.round(2)

# Display the summary
print("Yearly Financial Performance Summary:")
print(yearly_summary)


#################
######## printing the summary to a CSV file
output_filename = "yearly_financial_summary.csv"
yearly_summary.to_csv(output_filename)

print(f"\nYearly summary has been successfully saved to {output_filename}")

##############################################################
################ analyzing health deficiencies data ######################
##############################################################

# # ✅ Step 1: Use explicit relative path (./ = current directory)
# health_files = glob.glob("./HealthDeficiencies_*.csv")

# # ✅ Step 2: Function to load and label year
# def load_health_file(file_path):
#     df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
#     year = os.path.basename(file_path).split("_")[1].split(".")[0]
#     df["Year"] = int(year)
#     return df

# # ✅ Step 3: Read and combine all files
# all_dfs = [load_health_file(f) for f in health_files]

# # ✅ Step 4: Check how many files were loaded
# print(f"✅ Loaded {len(all_dfs)} files:")
# for f in health_files:
#     print(f)

# # ✅ Step 5: Merge them
# health_df = pd.concat(all_dfs, ignore_index=True)

# # ✅ Step 6: Summarize
# summary = health_df.groupby("Year").agg(
#     Total_Reports=('provnum', 'count'),
#     Unique_Providers=('provnum', 'nunique'),
#     Complaint_Related=('complaint', lambda x: (x == 'Y').sum())
# )

# summary['Complaint_Ratio'] = (summary['Complaint_Related'] / summary['Total_Reports']).round(2)

# print("\n📊 Health Deficiency Summary by Year (2015–2021):")
# print(summary)



################################################################
### deficiency summary df that contains summary stats per provider per year

# # Step 1: Standardize column names
# health_df.columns = health_df.columns.str.strip().str.lower()

# # Step 2: Group by provider and year to summarize
# deficiency_summary_df = health_df.groupby(['provnum', 'year']).agg(
#     total_deficiencies=('tag', 'count'),
#     complaint_deficiencies=('complaint', lambda x: (x == 'Y').sum())
# ).reset_index()

# # Step 3: Create complaint ratio
# deficiency_summary_df['complaint_ratio'] = (deficiency_summary_df['complaint_deficiencies'] / deficiency_summary_df['total_deficiencies']).round(2)

# # Preview
# print(deficiency_summary_df.head())

# # Ensure both keys are string type
# merged_df['Provider_CCN'] = merged_df['Provider_CCN'].astype(str)
# deficiency_summary_df['provnum'] = deficiency_summary_df['provnum'].astype(str)

# # Merge on Provider ID and Year
# merged_final_df = pd.merge(merged_df, deficiency_summary_df,
#                            left_on=['Provider_CCN', 'Year'],
#                            right_on=['provnum', 'year'],
#                            how='left')




##################################################################
#### Identify which variables (factors) significantly influence Net Income or Profitability of nursing homes.


# # Assume you have deficiency summary per provider-year
# merged_df = pd.merge(merged_df, deficiency_summary_df, left_on=['Provider_CCN', 'Year'], right_on=['provnum', 'Year'], how='left')
