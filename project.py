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


# #################
# ######## printing the summary to a CSV file
# output_filename = "yearly_financial_summary.csv"
# yearly_summary.to_csv(output_filename)

# print(f"\nYearly summary has been successfully saved to {output_filename}")

##############################################################
################ analyzing health deficiencies data ######################
##############################################################

# ✅ Step 1: Use explicit relative path (./ = current directory)
health_files = glob.glob("./HealthDeficiencies_*.csv")

# ✅ Step 2: Function to load and label year
def load_health_file(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    year = os.path.basename(file_path).split("_")[1].split(".")[0]
    df["Year"] = int(year)
    return df

# ✅ Step 3: Read and combine all files
all_dfs = [load_health_file(f) for f in health_files]

# ✅ Step 4: Check how many files were loaded
print(f"✅ Loaded {len(all_dfs)} files:")
for f in health_files:
    print(f)

# ✅ Step 5: Merge them
health_df = pd.concat(all_dfs, ignore_index=True)

# ✅ Step 6: Summarize
summary = health_df.groupby("Year").agg(
    Total_Reports=('provnum', 'count'),
    Unique_Providers=('provnum', 'nunique'),
    Complaint_Related=('complaint', lambda x: (x == 'Y').sum())
)

summary['Complaint_Ratio'] = (summary['Complaint_Related'] / summary['Total_Reports']).round(2)

print("\n📊 Health Deficiency Summary by Year (2015–2021):")
print(summary)



################################################################
### deficiency summary df that contains summary stats per provider per year

# Step 1: Standardize column names
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

# Merge on Provider ID and Year
# merged_final_df = pd.merge(merged_df, deficiency_summary_df,
#                            left_on=['Provider_CCN', 'Year'],
#                            right_on=['provnum', 'year'],
#                            how='left')




##################################################################
#### Identify which variables (factors) significantly influence Net Income or Profitability of nursing homes.


# # Assume you have deficiency summary per provider-year
# merged_df = pd.merge(merged_df, deficiency_summary_df, left_on=['Provider_CCN', 'Year'], right_on=['provnum', 'Year'], how='left')

##########################################
############ getting the average total salaries adjusted by year
# # This code calculates the average total salaries adjusted for each year from 2015 to 2021
# # and plots the results.
# import pandas as pd
# import matplotlib.pyplot as plt

# # List of files and years
# files = {
#     "2015": "2015_CostReport_cleaned.csv",
#     "2016": "2016_CostReport_cleaned.csv",
#     "2017": "2017_CostReport_cleaned.csv",
#     "2018": "2018_CostReport_cleaned.csv",
#     "2019": "2019_CostReport_cleaned.csv",
#     "2020": "2020_CostReport_cleaned.csv",
#     "2021": "2021_CostReport_cleaned.csv"
# }

# # Create an empty list to hold the data
# salary_data = []

# # Loop through each file
# for year, filename in files.items():
#     # Load the data
#     df = pd.read_csv(filename, low_memory=False)
    
#     # Ensure the total_salaries_adjusted column is numeric
#     df['total_salaries_adjusted'] = pd.to_numeric(df['total_salaries_adjusted'], errors='coerce')
    
#     # Calculate the average (mean) salary for that year
#     avg_salary = df['total_salaries_adjusted'].mean()
    
#     # Store the result
#     salary_data.append({
#         "Year": int(year),
#         "Average_Total_Salaries_Adjusted": avg_salary
#     })

# # Convert to a DataFrame
# salary_df = pd.DataFrame(salary_data)

# # Sort by Year
# salary_df = salary_df.sort_values("Year")

# # Print the results
# print("\nAverage Total Salaries Adjusted by Year:")
# print(salary_df)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(salary_df['Year'], salary_df['Average_Total_Salaries_Adjusted'], marker='o')
# plt.title('Average Total Salaries Adjusted (2015-2021)')
# plt.xlabel('Year')
# plt.ylabel('Average Total Salaries ($)')
# plt.grid(True)
# plt.show()

# # Optional: Save to a CSV for further analysis
# salary_df.to_csv("average_total_salaries_adjusted_2015_2021.csv", index=False)
# print("\nResults saved to 'average_total_salaries_adjusted_2015_2021.csv'")


################################################################
##### Compare Total Gross Income vs Total Salaries Adjusted
# This code compares the total gross income and total salaries adjusted for each year from 2015 to 2021
# and plots the results.

# import pandas as pd
# import matplotlib.pyplot as plt

# # Define your files and years
# files = {
#     "2015": "2015_CostReport_cleaned.csv",
#     "2016": "2016_CostReport_cleaned.csv",
#     "2017": "2017_CostReport_cleaned.csv",
#     "2018": "2018_CostReport_cleaned.csv",
#     "2019": "2019_CostReport_cleaned.csv",
#     "2020": "2020_CostReport_cleaned.csv",
#     "2021": "2021_CostReport_cleaned.csv"
# }

# # List to collect the data
# comparison_data = []

# # Loop through each file
# for year, filename in files.items():
#     try:
#         # Load data
#         df = pd.read_csv(filename, low_memory=False)

#         # Make sure both columns exist
#         if 'gross_revenue' in df.columns and 'total_salaries_adjusted' in df.columns:
#             # Convert columns to numeric safely
#             df['gross_revenue'] = pd.to_numeric(df['gross_revenue'], errors='coerce')
#             df['total_salaries_adjusted'] = pd.to_numeric(df['total_salaries_adjusted'], errors='coerce')

#             # Calculate total gross revenue and total salaries adjusted
#             total_gross_income = df['gross_revenue'].sum()
#             total_salaries = df['total_salaries_adjusted'].sum()
#             salary_burden_ratio = total_salaries / total_gross_income if total_gross_income != 0 else None
#             difference = total_gross_income - total_salaries

#             # Save the results
#             comparison_data.append({
#                 'Year': int(year),
#                 'Total_Gross_Income': total_gross_income,
#                 'Total_Salaries_Adjusted': total_salaries,
#                 'Difference_GrossIncome_vs_Salaries': difference,
#                 'Salary_Burden_Ratio': salary_burden_ratio
#             })
#         else:
#             print(f"Missing expected columns in {filename}, skipping...")
#     except Exception as e:
#         print(f"Error processing {filename}: {e}")

# # Convert to DataFrame
# comparison_df = pd.DataFrame(comparison_data)

# # Sort by year
# comparison_df = comparison_df.sort_values('Year')

# # Round the numbers for easier reading
# comparison_df = comparison_df.round(2)

# # Print out the results
# print("\nGross Income vs Salaries Comparison (2015-2021):")
# print(comparison_df)

# # Plotting Gross Income and Salaries
# plt.figure(figsize=(12, 7))
# plt.plot(comparison_df['Year'], comparison_df['Total_Gross_Income'], marker='o', label='Total Gross Income')
# plt.plot(comparison_df['Year'], comparison_df['Total_Salaries_Adjusted'], marker='s', label='Total Salaries Adjusted')

# plt.title('Total Gross Income vs Total Salaries Adjusted (2015-2021)')
# plt.xlabel('Year')
# plt.ylabel('Dollars ($)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plotting Salary Burden Ratio
# plt.figure(figsize=(10, 6))
# plt.plot(comparison_df['Year'], comparison_df['Salary_Burden_Ratio'], marker='^', color='green')
# plt.title('Salary Burden Ratio Over Time (Salaries ÷ Gross Income)')
# plt.xlabel('Year')
# plt.ylabel('Salary Burden Ratio')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Save to CSV if needed
# comparison_df.to_csv('gross_income_vs_salaries_with_burden_ratio.csv', index=False)
# print("\nFull comparison data saved to 'gross_income_vs_salaries_with_burden_ratio.csv'")

##############################################################
################ Average Certified Beds and Residents by Year ##################
# This code calculates the average number of certified beds and residents for each year from 2015 to 2021
# and plots the results.

import pandas as pd
import matplotlib.pyplot as plt

# Files and years mapping
files = {
    "2015": "ProviderInfo_2015.csv",
    "2016": "ProviderInfo_2016.csv",
    "2017": "ProviderInfo_2017.csv",
    "2018": "ProviderInfo_2018.csv",
    "2019": "ProviderInfo_2019.csv",
    "2020": "ProviderInfo_2020.csv",
    "2021": "ProviderInfo_2021.csv"
}

# Empty list to store data
provider_data = []

# Loop through each year and file
for year, filename in files.items():
    # Load the CSV
    df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)
    
    # Ensure numeric values
    df['BEDCERT'] = pd.to_numeric(df['BEDCERT'], errors='coerce')
    df['RESTOT'] = pd.to_numeric(df['RESTOT'], errors='coerce')
    
    # Calculate averages
    avg_beds = df['BEDCERT'].mean()
    avg_residents = df['RESTOT'].mean()
    
    # Save the results
    provider_data.append({
        "Year": int(year),
        "Average_Certified_Beds": avg_beds,
        "Average_Residents": avg_residents
    })

# Convert to DataFrame
provider_df = pd.DataFrame(provider_data)

# Sort by year
provider_df = provider_df.sort_values("Year")

# Show the results
print("\nAverage Certified Beds and Residents by Year:")
print(provider_df)

# Plot Certified Beds over time
plt.figure(figsize=(10, 5))
plt.plot(provider_df['Year'], provider_df['Average_Certified_Beds'], marker='o', label='Certified Beds')
plt.plot(provider_df['Year'], provider_df['Average_Residents'], marker='o', label='Residents')
plt.title('Average Certified Beds and Residents (2015–2021)')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

# # Optional: Save to CSV
# provider_df.to_csv("average_beds_residents_2015_2021.csv", index=False)
# print("\nResults saved to 'average_beds_residents_2015_2021.csv'")


###################################

