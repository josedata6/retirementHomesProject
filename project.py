import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import numpy as np

# Define file paths for each year's data
# Creates a dictionary called files that maps each year to the corresponding CSV file name
# files = {
#     "2015": "2015_CostReport.csv",
#     "2016": "2016_CostReport.csv",
#     "2017": "2017_CostReport.csv",
#     "2018": "2018_CostReport.csv",
#     "2019": "2019_CostReport.csv",
#     "2020": "2020_CostReport.csv",
#     "2021": "2021_CostReport.csv"}

# # Create an empty list to hold each year's DataFrame
# all_dataframes = []

# # Loop through files, read, and append the year
# for year, path in files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['Year'] = int(year)
#     all_dataframes.append(df)

# # Concatenate all DataFrames into one
# merged_df = pd.concat(all_dataframes, ignore_index=True)

# # # Show the shape or a preview
# # print("Merged DataFrame shape:", merged_df.shape)
# # print(merged_df.head())

# # # Check the columns of the merged DataFrame
# # print(merged_df.columns)

# # # Get basic statistics
# # print(merged_df.describe())

# # print(merged_df.groupby("Year")['Net_Income'].mean())  # Yearly average net income

# financial_metrics = [
#     'net_income',
#     'gross_revenue',
#     'total_income',
#     'total_costs',
#     'total_current_assets',
#     'total_current_liabilities',
#     'total_liabilities',
#     'total_fund_balances',
#     'total_fixed_assets'
# ]
# # Convert financial metrics to numeric, coercing errors to NaN
# # This will ensure that any non-numeric values in these columns are converted to NaN
# for col in financial_metrics:
#     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# yearly_summary = merged_df.groupby("Year")[financial_metrics].mean()
# print(yearly_summary)

# yearly_summary['profit_margin'] = yearly_summary['net_income'] / yearly_summary['gross_revenue']
# yearly_summary['roa'] = yearly_summary['net_income'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
# yearly_summary['liability_to_asset_ratio'] = yearly_summary['total_liabilities'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
# yearly_summary['current_ratio'] = yearly_summary['total_current_assets'] / yearly_summary['total_current_liabilities']
# yearly_summary['debt_to_equity_ratio'] = yearly_summary['total_liabilities'] / yearly_summary['total_fund_balances']
# yearly_summary['asset_turnover'] = yearly_summary['gross_revenue'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
# yearly_summary['return_on_equity'] = yearly_summary['net_income'] / yearly_summary['total_fund_balances']
# yearly_summary['return_on_assets'] = yearly_summary['net_income'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
# yearly_summary['return_on_investment'] = yearly_summary['net_income'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])
# yearly_summary['debt_ratio'] = yearly_summary['total_liabilities'] / (yearly_summary['total_current_assets'] + yearly_summary['total_fixed_assets'])

# # Plot Net Income
# yearly_summary['net_income'].plot(marker='o', title='Average Net Income Over Time')
# plt.ylabel("Net Income ($)")
# plt.grid(True)
# plt.show()

# # Plot Profit Margin
# yearly_summary['profit_margin'].plot(marker='o', title='Profit Margin Over Time')
# plt.ylabel("Profit Margin")
# plt.grid(True)
# plt.show()

# # Round for readability
# yearly_summary = yearly_summary.round(2)

# # Display the summary
# print("Yearly Financial Performance Summary:")
# print(yearly_summary)


# #################
# ######## printing the summary to a CSV file
# output_filename = "yearly_financial_summary.csv"
# yearly_summary.to_csv(output_filename)

# print(f"\nYearly summary has been successfully saved to {output_filename}")

##############################################################
################ analyzing health deficiencies data ######################
##############################################################

# # # ✅ Step 1: Use explicit relative path (./ = current directory)
# health_files = glob.glob("./HealthDeficiencies_*.csv")

# # ✅ Step 2: Function to load and label year
# def load_health_file(file_path):
#     df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
#     year = os.path.basename(file_path).split("_")[1].split(".")[0]
#     df["Year"] = int(year)
#     return df

# ✅ Step 3: Read and combine all files
# all_dfs = [load_health_file(f) for f in health_files]

# # ✅ Step 4: Check how many files were loaded
# print(f"✅ Loaded {len(all_dfs)} files:")
# for f in health_files:
#     print(f)

# ✅ Step 5: Merge them
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

# # Merge on Provider ID and Year
# merged_final_df = pd.merge(merged_df, deficiency_summary_df,
#                            left_on=['Provider_CCN', 'Year'],
#                            right_on=['provnum', 'year'],
#                            how='left')




##################################################################
#### Identify which variables (factors) significantly influence Net Income or Profitability of nursing homes.


# # Assume you have deficiency summary per provider-year
# merged_df = pd.merge(merged_df, deficiency_summary_df, left_on=['Provider_CCN', 'Year'], right_on=['provnum', 'Year'], how='left')



##############################
## adds map with homes #####
# 📦 First install necessary packages if you haven't
# pip install pandas folium

# import pandas as pd
# import glob
# import folium
# from folium.plugins import MarkerCluster

# # ✅ Step 1: Load all QualityMsrMDS files
# files = glob.glob("QualityMsrMDS_20*_Cleaned.csv")  # Adjust path if needed

# def load_address_info(filepath):
#     df = pd.read_csv(filepath, encoding='ISO-8859-1', low_memory=False)
#     df.columns = df.columns.str.strip().str.lower()
#     required_cols = ['address', 'city', 'state', 'zip']
#     if all(col in df.columns for col in required_cols):
#         return df[required_cols]
#     else:
#         print(f"⚠️ Skipping file {filepath}: required address columns not found.")
#         return pd.DataFrame()

# # ✅ Step 2: Combine and deduplicate addresses
# address_df = pd.concat([load_address_info(f) for f in files], ignore_index=True)
# address_df.drop_duplicates(subset=['address', 'city', 'state', 'zip'], inplace=True)

# # ✅ Step 3: Load ZIP code coordinates
# zip_latlng_df = pd.read_csv('uszips.csv')  # Must contain 'zip', 'lat', 'lng' columns
# zip_latlng_df['zip'] = zip_latlng_df['zip'].astype(str).str.zfill(5)
# address_df['zip'] = address_df['zip'].astype(str).str.zfill(5)

# # Step 4: Merge address info with ZIP coordinates
# merged_df = pd.merge(
#     address_df,               # Keep all original columns: address, city, state, zip
#     zip_latlng_df[['zip', 'lat', 'lng']],  # Only bring in lat/lng
#     how='inner',
#     on='zip'
# )
# # ✅ Step 5: Create the Map
# m = folium.Map(location=[merged_df['lat'].mean(), merged_df['lng'].mean()], zoom_start=5)

# # ✅ Step 6: Add Clustering
# marker_cluster = MarkerCluster().add_to(m)

# for _, row in merged_df.iterrows():
#     folium.Marker(
#         location=[row['lat'], row['lng']],
#         popup=f"{row['address']}, {row['city']}, {row['state']} {row['zip']}",
#         icon=folium.Icon(color='blue', icon='home')
#     ).add_to(marker_cluster)

# # ✅ Step 7: Save Map
# m.save("nursing_homes_map_clustered.html")
# print("✅ Map saved as 'nursing_homes_map_clustered.html'. Open it in your browser!")


######################################################
##### Map with nursing homes by year and legend #####

# import pandas as pd
# import glob
# import folium
# from folium.plugins import MarkerCluster

# # Step 1: Load all QualityMsrMDS files
# files = glob.glob("QualityMsrMDS_20*_Cleaned.csv")

# def load_address_info(filepath):
#     df = pd.read_csv(filepath, encoding='ISO-8859-1', low_memory=False)
#     df.columns = df.columns.str.strip().str.lower()
#     year = os.path.basename(filepath).split("_")[1]
#     df['year'] = int(year)
#     required_cols = ['address', 'city', 'state', 'zip', 'year']
#     if all(col in df.columns for col in ['address', 'city', 'state', 'zip']):
#         return df[required_cols]
#     else:
#         print(f"⚠️ Skipping file {filepath}: required address columns not found.")
#         return pd.DataFrame()

# # Combine and deduplicate
# address_df = pd.concat([load_address_info(f) for f in files], ignore_index=True)
# address_df.drop_duplicates(subset=['address', 'city', 'state', 'zip'], inplace=True)

# # Load ZIP code coordinates
# zip_latlng_df = pd.read_csv('uszips.csv')
# zip_latlng_df['zip'] = zip_latlng_df['zip'].astype(str).str.zfill(5)
# address_df['zip'] = address_df['zip'].astype(str).str.zfill(5)

# # Merge
# merged_df = pd.merge(
#     address_df,
#     zip_latlng_df[['zip', 'lat', 'lng']],
#     how='inner',
#     on='zip'
# )

# print(f"✅ Total records after merging with ZIP coordinates: {len(merged_df)}")

# # Color palette for each year
# year_colors = {
#     2015: 'blue',
#     2016: 'green',
#     2017: 'purple',
#     2018: 'orange',
#     2019: 'red',
#     2020: 'pink',
#     2021: 'cadetblue'
# }

# # Create base map
# m = folium.Map(location=[merged_df['lat'].mean(), merged_df['lng'].mean()], zoom_start=5)

# # Create a FeatureGroup for each year
# for year, color in year_colors.items():
#     year_data = merged_df[merged_df['year'] == year]
#     fg = folium.FeatureGroup(name=f"{year}", show=True)
#     cluster = MarkerCluster().add_to(fg)

#     for _, row in year_data.iterrows():
#         folium.Marker(
#             location=[row['lat'], row['lng']],
#             popup=f"{row['address']}, {row['city']}, {row['state']} {row['zip']} ({row['year']})",
#             icon=folium.Icon(color=color, icon='home')
#         ).add_to(cluster)

#     fg.add_to(m)

# # Add layer control
# folium.LayerControl(collapsed=False).add_to(m)

# # Save the map
# m.save("nursing_homes_toggle_by_year.html")
# print("✅ Interactive toggle map saved as 'nursing_homes_toggle_by_year.html'")

###################################
######### Regression Analysis ######

# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# files = {
#     "2015": "2015_CostReport_cleaned.csv",
#     "2016": "2016_CostReport_cleaned.csv",
#     "2017": "2017_CostReport_cleaned.csv",
#     "2018": "2018_CostReport_cleaned.csv",
#     "2019": "2019_CostReport_cleaned.csv",
#     "2020": "2020_CostReport_cleaned.csv",
#     "2021": "2021_CostReport_cleaned.csv"}

# # Create an empty list to hold each year's DataFrame
# all_dataframes = []

# # Loop through files, read, and append the year
# for year, path in files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['Year'] = int(year)
#     all_dataframes.append(df)

# # Concatenate all DataFrames into one
# merged_df = pd.concat(all_dataframes, ignore_index=True)

# health_files = glob.glob("./HealthDeficiencies_*.csv")

# # ✅ Step 2: Function to load and label year
# def load_health_file(file_path):
#     df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
#     year = os.path.basename(file_path).split("_")[1].split(".")[0]
#     df["Year"] = int(year)
#     return df

# # ✅ Step 3: Read and combine all files
# all_dfs = [load_health_file(f) for f in health_files]


# # ✅ Step 5: Merge them
# health_df = pd.concat(all_dfs, ignore_index=True)


# # ✅ Step 1: Select your features and target
# # Make sure the following columns exist in your merged DataFrame
# features = [
#     'occupancy_rate',
#     'staffing_rating',
#     'rn_staffing_rating',
#     'total_liabilities',
#     'total_salaries_from_worksheet_a',
#     'bedcert'
# ]

# target = 'net_income'

# # ✅ Step 2: Drop rows with missing data
# data = merged_df.dropna(subset=features + [target])

# X = data[features]
# y = data[target]

# # ✅ Step 3: Add constant term for intercept
# X = sm.add_constant(X)

# # ✅ Step 4: Fit the model using statsmodels
# model = sm.OLS(y, X).fit()

# # ✅ Step 5: Print summary of results
# print(model.summary())

##################################################
#### regression analysis with all years combined ####

# import pandas as pd
# import statsmodels.api as sm
# import glob
# import os

# # Step 1: Load all CostReport CSVs (2015–2021)
# cost_files = {
#     "2015": "2015_CostReport.csv",
#     "2016": "2016_CostReport.csv",
#     "2017": "2017_CostReport.csv",
#     "2018": "2018_CostReport.csv",
#     "2019": "2019_CostReport.csv",
#     "2020": "2020_CostReport.csv",
#     "2021": "2021_CostReport.csv"
# }

# cost_dfs = []
# for year, path in cost_files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['year'] = int(year)
#     df['provider_ccn'] = df['provider_ccn'].astype(str).str.zfill(6)
#     cost_dfs.append(df)

# cost_df = pd.concat(cost_dfs, ignore_index=True)

# # Step 2: Load ProviderInfo files
# provider_files = glob.glob("ProviderInfo_20*.csv")
# provider_dfs = []

# for file in provider_files:
#     df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
#     year = int(os.path.basename(file).split("_")[1].split(".")[0])
#     df['year'] = year

#     # ✅ Use exact casing: "Federal Provider Number"
#     if "Federal Provider Number" in df.columns:
#         df['Federal Provider Number'] = df['Federal Provider Number'].astype(str).str.zfill(6)
#         df = df.rename(columns={"Federal Provider Number": "provider_ccn"})
#         provider_dfs.append(df[['provider_ccn', 'year', 'staffing_rating', 'rn_staffing_rating']])
#     else:
#         print(f"⚠️ Skipping {file}: 'Federal Provider Number' column not found.")

# provider_df = pd.concat(provider_dfs, ignore_index=True)

# # Step 3: Merge cost and provider info on provider_ccn + year
# merged_df = pd.merge(
#     cost_df,
#     provider_df,
#     on=['provider_ccn', 'year'],
#     how='inner'
# )

# # Step 4: Compute Occupancy Rate
# merged_df['occupancy_rate'] = merged_df['total_days_total'] / merged_df['total_bed_days_available']

# # Step 5: Estimate bedcert if needed
# if 'bedcert' not in merged_df.columns:
#     if 'number_of_beds' in merged_df.columns:
#         merged_df['bedcert'] = merged_df['number_of_beds']
#     elif 'total_bed_days_available' in merged_df.columns:
#         merged_df['bedcert'] = merged_df['total_bed_days_available'] / 365
#     else:
#         raise ValueError("❌ Could not compute 'bedcert': missing both 'number_of_beds' and 'total_bed_days_available'")


# # Step 6: Define features and target
# features = [
#     'occupancy_rate',
#     'staffing_rating',
#     'rn_staffing_rating',
#     'total_liabilities',
#     'total_salaries_from_worksheet_a',
#     'bedcert'
# ]
# target = 'net_income'

# # Step 7: Clean and fit model
# for col in features + [target]:
#     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# data = merged_df.dropna(subset=features + [target])
# X = sm.add_constant(data[features])
# y = data[target]

# model = sm.OLS(y, X).fit()

# # Step 8: Show regression summary
# print(model.summary())

#######################################
# Regression Analysis with all years combined
# This code assumes you have already cleaned the CostReport and ProviderInfo files

import pandas as pd
import statsmodels.api as sm
import glob
import os

# Step 1: Load all cleaned CostReport CSVs (2015–2021)
cost_files = {
    "2015": "2015_CostReport_cleaned.csv",
    "2016": "2016_CostReport_cleaned.csv",
    "2017": "2017_CostReport_cleaned.csv",
    "2018": "2018_CostReport_cleaned.csv",
    "2019": "2019_CostReport_cleaned.csv",
    "2020": "2020_CostReport_cleaned.csv",
    "2021": "2021_CostReport_cleaned.csv"
}

cost_dfs = []
for year, path in cost_files.items():
    df = pd.read_csv(path, low_memory=False)
    df['year'] = int(year)
    df['provider_ccn'] = df['provider_ccn'].astype(str).str.zfill(6)
    cost_dfs.append(df)

cost_df = pd.concat(cost_dfs, ignore_index=True)

# Step 2: Load ProviderInfo files
provider_files = glob.glob("ProviderInfo_20*.csv")
provider_dfs = []

for file in provider_files:
    df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
    year = int(os.path.basename(file).split("_")[1].split(".")[0])
    df['year'] = year

    # Use exact casing: "Federal Provider Number"
    if "Federal Provider Number" in df.columns:
        df['Federal Provider Number'] = df['Federal Provider Number'].astype(str).str.zfill(6)
        df = df.rename(columns={"Federal Provider Number": "provider_ccn"})
        provider_dfs.append(df[['provider_ccn', 'year', 'staffing_rating', 'rn_staffing_rating']])
    else:
        print(f"⚠️ Skipping {file}: 'Federal Provider Number' column not found.")

provider_df = pd.concat(provider_dfs, ignore_index=True)

# Step 3: Merge cost and provider info on provider_ccn + year
merged_df = pd.merge(
    cost_df,
    provider_df,
    on=['provider_ccn', 'year'],
    how='inner'
)

# Step 4: Compute Occupancy Rate
merged_df['occupancy_rate'] = merged_df['total_days_total'] / merged_df['total_bed_days_available']

# Step 5: Estimate bedcert if needed
if 'bedcert' not in merged_df.columns:
    if 'number_of_beds' in merged_df.columns:
        merged_df['bedcert'] = merged_df['number_of_beds']
    elif 'total_bed_days_available' in merged_df.columns:
        merged_df['bedcert'] = merged_df['total_bed_days_available'] / 365
    else:
        raise ValueError("❌ Could not compute 'bedcert': missing both 'number_of_beds' and 'total_bed_days_available'")

# Step 6: Define features and target (updated feature name here)
features = [
    'occupancy_rate',
    'staffing_rating',
    'rn_staffing_rating',
    'total_liabilities',
    'total_salaries_adjusted',  # <- updated here
    'bedcert'
]
target = 'net_income'

# Step 7: Clean and fit model
for col in features + [target]:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

data = merged_df.dropna(subset=features + [target])
X = sm.add_constant(data[features])
y = data[target]

model = sm.OLS(y, X).fit()

# Step 8: Show regression summary
print(model.summary())



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Step 1: Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Step 2: Predict
y_pred = rf.predict(X)

# Step 3: Evaluate performance
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\n🌲 Random Forest Regressor Results:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:,.2f}")

# Step 4: Feature Importance
import matplotlib.pyplot as plt

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = X.columns[indices]

plt.figure(figsize=(8, 6))
plt.title("Feature Importance (Random Forest)")
plt.barh(features_sorted, importances[indices], color="skyblue")
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
