import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import numpy as np

## Define file paths for each year's data
## Creates a dictionary called files that maps each year to the corresponding CSV file name
# files = {
#     "2015": "2015_CostReport.csv",
#     "2016": "2016_CostReport.csv",
#     "2017": "2017_CostReport.csv",
#     "2018": "2018_CostReport.csv",
#     "2019": "2019_CostReport.csv",
#     "2020": "2020_CostReport.csv",
#     "2021": "2021_CostReport.csv"
# }

# Create an empty list to hold each year's DataFrame
# all_dataframes = []

# Loop through files, read, and append the year
# for year, path in files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['Year'] = int(year)
#     all_dataframes.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(all_dataframes, ignore_index=True)

# Show the shape or a preview
print("Merged DataFrame shape:", merged_df.shape)
print(merged_df.head())

# Check the columns of the merged DataFrame
print(merged_df.columns)

# # Get basic statistics
print(merged_df.describe())

# Yearly average net income
# # print(merged_df.groupby("Year")['Net_Income'].mean())

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

# # #Use explicit relative path (./ = current directory)
# health_files = glob.glob("./HealthDeficiencies_*.csv")

# # Function to load and label year
# def load_health_file(file_path):
#     df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
#     year = os.path.basename(file_path).split("_")[1].split(".")[0]
#     df["Year"] = int(year)
#     return df

## Read and combine all files
# all_dfs = [load_health_file(f) for f in health_files]

# ## Check how many files were loaded
# print(f"Loaded {len(all_dfs)} files:")
# for f in health_files:
#     print(f)

## Merge them
# health_df = pd.concat(all_dfs, ignore_index=True)

# ## Summarize
# summary = health_df.groupby("Year").agg( # Group by year
#     Total_Reports=('provnum', 'count'), # Count total reports
#     Unique_Providers=('provnum', 'nunique'), # Count unique providers
#     Complaint_Related=('complaint', lambda x: (x == 'Y').sum()) # # Count complaint-related reports
# )
## The proportion of health deficiency reports that were complaint-related, rounded to 2 decimal places.
# summary['Complaint_Ratio'] = (summary['Complaint_Related'] / summary['Total_Reports']).round(2)

# print("\nHealth Deficiency Summary by Year (2015–2021):")
# print(summary)



################################################################
### deficiency summary df that contains summary stats per provider per year

### Standardize column names
# health_df.columns = health_df.columns.str.strip().str.lower()

# ## Group by provider and year to summarize
# deficiency_summary_df = health_df.groupby(['provnum', 'year']).agg( 
#     total_deficiencies=('tag', 'count'), # the total number of deficiencies
#     complaint_deficiencies=('complaint', lambda x: (x == 'Y').sum()) ## the number of complaint-related deficiencies
# ).reset_index() ## reset index to get a flat DataFrame

##vCreate complaint ratio
# deficiency_summary_df['complaint_ratio'] = (deficiency_summary_df['complaint_deficiencies'] / deficiency_summary_df['total_deficiencies']).round(2)

# # Preview
# print(deficiency_summary_df.head())

# # Ensure both keys are string type
# merged_df['Provider_CCN'] = merged_df['Provider_CCN'].astype(str)
# deficiency_summary_df['provnum'] = deficiency_summary_df['provnum'].astype(str)

# # Merge on Provider ID and Year
# merged_final_df = pd.merge(merged_df, deficiency_summary_df, ## merge the two DataFrames
#                            left_on=['Provider_CCN', 'Year'], ## left_on is the key in merged_df
#                            right_on=['provnum', 'year'], ## right_on is the key in deficiency_summary_df
#                            how='left') ## left join to keep all records from merged_df




##################################################################
#### Identify which variables (factors) significantly influence Net Income or Profitability of nursing homes.


# # Assume you have deficiency summary per provider-year
# merged_df = pd.merge(merged_df, deficiency_summary_df, left_on=['Provider_CCN', 'Year'], right_on=['provnum', 'Year'], how='left')



##############################
## adds map with homes #####

# import pandas as pd
# import glob
# import folium
# from folium.plugins import MarkerCluster

# ## Load all QualityMsrMDS files
# files = glob.glob("QualityMsrMDS_20*_Cleaned.csv")  # gets all files matching the pattern

## function to load address info
# def load_address_info(filepath):
#     df = pd.read_csv(filepath, encoding='ISO-8859-1', low_memory=False)
#     df.columns = df.columns.str.strip().str.lower()
#     required_cols = ['address', 'city', 'state', 'zip']
#     if all(col in df.columns for col in required_cols):
#         return df[required_cols]
#     else:
#         print(f"Skipping file {filepath}: required address columns not found.")
#         return pd.DataFrame()

# # Combine and deduplicate addresses
# address_df = pd.concat([load_address_info(f) for f in files], ignore_index=True)
# address_df.drop_duplicates(subset=['address', 'city', 'state', 'zip'], inplace=True)

# # Load ZIP code coordinates
# zip_latlng_df = pd.read_csv('uszips.csv')  # Must contain 'zip', 'lat', 'lng' columns 
## takes variable zip and converts it to string, then pads with leading zeros to make it 5 digits
# zip_latlng_df['zip'] = zip_latlng_df['zip'].astype(str).str.zfill(5) 
## takes variable zip and converts it to string, then pads with leading zeros to make it 5 digits
# address_df['zip'] = address_df['zip'].astype(str).str.zfill(5)

# # Step 4: Merge address info with ZIP coordinates
# merged_df = pd.merge(
#     address_df,               # Keep all original columns: address, city, state, zip
#     zip_latlng_df[['zip', 'lat', 'lng']],  # Only bring in lat/lng
#     how='inner',
#     on='zip'
# )
# ## Create the Map
# m = folium.Map(location=[merged_df['lat'].mean(), merged_df['lng'].mean()], zoom_start=5)

# ## Add Clustering
# marker_cluster = MarkerCluster().add_to(m)

# for _, row in merged_df.iterrows():
#     folium.Marker(
#         location=[row['lat'], row['lng']],
#         popup=f"{row['address']}, {row['city']}, {row['state']} {row['zip']}",
#         icon=folium.Icon(color='blue', icon='home')
#     ).add_to(marker_cluster)

### Save Map html file
# m.save("nursing_homes_map_clustered.html")
# print("Map saved as 'nursing_homes_map_clustered.html'. Open it in browser to view.")


######################################################
##### Map with nursing homes by year and legend #####

import pandas as pd
import glob
import folium
from folium.plugins import MarkerCluster

### Load all QualityMsrMDS files
files = glob.glob("QualityMsrMDS_20*_Cleaned.csv")

## Function to load address info
def load_address_info(filepath):
    ## reads file and loads it into a DataFrame
    df = pd.read_csv(filepath, encoding='ISO-8859-1', low_memory=False) 
    ## Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    ### Extract year from filename
    year = os.path.basename(filepath).split("_")[1]
    df['year'] = int(year)
    ### Check for required columns
    required_cols = ['address', 'city', 'state', 'zip', 'year']
    if all(col in df.columns for col in ['address', 'city', 'state', 'zip']):
        return df[required_cols]
    else:
        print(f" Skipping file {filepath}: required address columns not found.")
        return pd.DataFrame()

# Combine and deduplicate
address_df = pd.concat([load_address_info(f) for f in files], ignore_index=True)
address_df.drop_duplicates(subset=['address', 'city', 'state', 'zip'], inplace=True)

# Load ZIP code coordinates
zip_latlng_df = pd.read_csv('uszips.csv')
zip_latlng_df['zip'] = zip_latlng_df['zip'].astype(str).str.zfill(5)
address_df['zip'] = address_df['zip'].astype(str).str.zfill(5)

# Merge dataframes
# Merge address info with ZIP coordinates
merged_df = pd.merge(
    address_df,
    zip_latlng_df[['zip', 'lat', 'lng']],
    how='inner',
    on='zip'
)

print(f"Total records after merging with ZIP coordinates: {len(merged_df)}")

# Color palette for each year
year_colors = {
    2015: 'blue',
    2016: 'green',
    2017: 'purple',
    2018: 'orange',
    2019: 'red',
    2020: 'pink',
    2021: 'cadetblue'
}

# Create base map
m = folium.Map(location=[merged_df['lat'].mean(), merged_df['lng'].mean()], zoom_start=5)

# Create a FeatureGroup for each year
for year, color in year_colors.items():
    year_data = merged_df[merged_df['year'] == year]
    fg = folium.FeatureGroup(name=f"{year}", show=True)
    cluster = MarkerCluster().add_to(fg)

#### loop through each row in the year_data DataFrame
    for _, row in year_data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"{row['address']}, {row['city']}, {row['state']} {row['zip']} ({row['year']})",
            icon=folium.Icon(color=color, icon='home')
        ).add_to(cluster)

    fg.add_to(m)

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Save the map in html file
m.save("nursing_homes_toggle_by_year.html")
print("Interactive toggle map saved as 'nursing_homes_toggle_by_year.html'")

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

# # Function to load and label year
# def load_health_file(file_path):
#     df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
#     year = os.path.basename(file_path).split("_")[1].split(".")[0]
#     df["Year"] = int(year)
#     return df

# # Read and combine all files
# all_dfs = [load_health_file(f) for f in health_files]


# # Merge dataframe
# health_df = pd.concat(all_dfs, ignore_index=True)


# # Select your features and target
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

# ## Drop rows with missing data
# data = merged_df.dropna(subset=features + [target])

### data features and target
# X = data[features]
# y = data[target]

# # Add constant term for intercept
# X = sm.add_constant(X)

# # Fit the model using statsmodels
# model = sm.OLS(y, X).fit()

# # Print summary of results
# print(model.summary())

##################################################
#### regression analysis with all years combined ####

# import pandas as pd
# import statsmodels.api as sm
# import glob
# import os

# # Load all CostReport CSVs (2015–2021)
# cost_files = {
#     "2015": "2015_CostReport.csv",
#     "2016": "2016_CostReport.csv",
#     "2017": "2017_CostReport.csv",
#     "2018": "2018_CostReport.csv",
#     "2019": "2019_CostReport.csv",
#     "2020": "2020_CostReport.csv",
#     "2021": "2021_CostReport.csv"
# }

### Create an empty list to hold each year's DataFrame
# cost_dfs = []
### Loop through files, read, and append the year
# for year, path in cost_files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['year'] = int(year)
## takes variable provider_ccn and converts it to string, then pads with leading zeros to make it 6 digits
#     df['provider_ccn'] = df['provider_ccn'].astype(str).str.zfill(6)
#     cost_dfs.append(df)
### # Concatenate all DataFrames into one
# cost_df = pd.concat(cost_dfs, ignore_index=True)

### Load ProviderInfo files
# provider_files = glob.glob("ProviderInfo_20*.csv")
# provider_dfs = []

# for file in provider_files:
#     df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
#     year = int(os.path.basename(file).split("_")[1].split(".")[0])
#     df['year'] = year

#     # Use exact casing: "Federal Provider Number"
#     if "Federal Provider Number" in df.columns:
#### takes variable Federal Provider Number and converts it to string, then pads with leading zeros to make it 6 digits
#         df['Federal Provider Number'] = df['Federal Provider Number'].astype(str).str.zfill(6)
### #         # Rename the column to match cost_df
#         df = df.rename(columns={"Federal Provider Number": "provider_ccn"})
#### #         # Append relevant columns
#         provider_dfs.append(df[['provider_ccn', 'year', 'staffing_rating', 'rn_staffing_rating']])
#     else:
#         print(f"Skipping {file}: 'Federal Provider Number' column not found.")
### # # Concatenate all ProviderInfo DataFrames into one
# provider_df = pd.concat(provider_dfs, ignore_index=True)

# # Merge cost and provider info on provider_ccn + year
# merged_df = pd.merge(
#     cost_df,
#     provider_df,
#     on=['provider_ccn', 'year'],
#     how='inner'
# )

# # Compute Occupancy Rate
# merged_df['occupancy_rate'] = merged_df['total_days_total'] / merged_df['total_bed_days_available']

# ## Estimate bedcert if needed
# if 'bedcert' not in merged_df.columns:
#     if 'number_of_beds' in merged_df.columns:
#         merged_df['bedcert'] = merged_df['number_of_beds']
#     elif 'total_bed_days_available' in merged_df.columns:
#         merged_df['bedcert'] = merged_df['total_bed_days_available'] / 365
#     else:
#         raise ValueError("Could not compute 'bedcert': missing both 'number_of_beds' and 'total_bed_days_available'")


# # Define features and target
# features = [
#     'occupancy_rate',
#     'staffing_rating',
#     'rn_staffing_rating',
#     'total_liabilities',
#     'total_salaries_from_worksheet_a',
#     'bedcert'
# ]
# target = 'net_income'

# # Clean and fit model
# for col in features + [target]:
#     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# data = merged_df.dropna(subset=features + [target])
# X = sm.add_constant(data[features])
# y = data[target]

# model = sm.OLS(y, X).fit()

# # Show regression summary
# print(model.summary())

#######################################
# Regression Analysis with all years combined

# import pandas as pd
# import statsmodels.api as sm
# import glob
# import os

# ## Load all cleaned CostReport CSVs (2015–2021)
# cost_files = {
#     "2015": "2015_CostReport_cleaned.csv",
#     "2016": "2016_CostReport_cleaned.csv",
#     "2017": "2017_CostReport_cleaned.csv",
#     "2018": "2018_CostReport_cleaned.csv",
#     "2019": "2019_CostReport_cleaned.csv",
#     "2020": "2020_CostReport_cleaned.csv",
#     "2021": "2021_CostReport_cleaned.csv"
# }

## create an empty list to hold each year's DataFrame
# cost_dfs = []
### Loop through files, read, and append the year
# for year, path in cost_files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['year'] = int(year)
#     df['provider_ccn'] = df['provider_ccn'].astype(str).str.zfill(6)
#     cost_dfs.append(df)

### Concatenate all DataFrames into one
# cost_df = pd.concat(cost_dfs, ignore_index=True)

# # Load ProviderInfo files
# provider_files = glob.glob("ProviderInfo_20*.csv")
# provider_dfs = []

## loop through files, read, and append the year
# for file in provider_files:
#     df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
#     year = int(os.path.basename(file).split("_")[1].split(".")[0])
#     df['year'] = year

#     # Use exact casing: "Federal Provider Number"
#     if "Federal Provider Number" in df.columns:
#         df['Federal Provider Number'] = df['Federal Provider Number'].astype(str).str.zfill(6)
#         df = df.rename(columns={"Federal Provider Number": "provider_ccn"})
#         provider_dfs.append(df[['provider_ccn', 'year', 'staffing_rating', 'rn_staffing_rating']])
#     else:
#         print(f"Skipping {file}: 'Federal Provider Number' column not found.")

# provider_df = pd.concat(provider_dfs, ignore_index=True)

# # Merge cost and provider info on provider_ccn + year
# merged_df = pd.merge(
#     cost_df,
#     provider_df,
#     on=['provider_ccn', 'year'],
#     how='inner'
# )

# # Compute Occupancy Rate
# merged_df['occupancy_rate'] = merged_df['total_days_total'] / merged_df['total_bed_days_available']

# # Estimate bedcert if needed
# if 'bedcert' not in merged_df.columns:
#     if 'number_of_beds' in merged_df.columns:
#         merged_df['bedcert'] = merged_df['number_of_beds']
#     elif 'total_bed_days_available' in merged_df.columns:
#         merged_df['bedcert'] = merged_df['total_bed_days_available'] / 365
#     else:
#         raise ValueError("Could not compute 'bedcert': missing both 'number_of_beds' and 'total_bed_days_available'")

# # Define features and target (updated feature name here)
# features = [
#     'occupancy_rate',
#     'staffing_rating',
#     'rn_staffing_rating',
#     'total_liabilities',
#     'total_salaries_adjusted',  # <- updated here
#     'bedcert'
# ]
# target = 'net_income'

# # Clean and fit model
# for col in features + [target]:
#     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# data = merged_df.dropna(subset=features + [target])
# X = sm.add_constant(data[features])
# y = data[target]

# model = sm.OLS(y, X).fit()

# ## Show regression summary
# print(model.summary())

######################################################
######## ### Random Forest Regression Analysis ######

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_squared_error
# import numpy as np

# ###  Train the model
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
### Fit the model
# rf.fit(X, y)

# ## Predict
# y_pred = rf.predict(X)

# ## Evaluate performance
# r2 = r2_score(y, y_pred)
### Calculate RMSE
# rmse = np.sqrt(mean_squared_error(y, y_pred))

# print(f"\n Random Forest Regressor Results:")
# print(f"R² Score: {r2:.3f}")
# print(f"RMSE: {rmse:,.2f}")

# ## Feature Importance
# import matplotlib.pyplot as plt

# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# features_sorted = X.columns[indices]

### Plot feature importance
# plt.figure(figsize=(8, 6))
# plt.title("Feature Importance (Random Forest)")
### Bar plot of feature importance
# plt.barh(features_sorted, importances[indices], color="skyblue")
# plt.xlabel("Relative Importance")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()



######################

### LightGBM Regression Analysis
# import pandas as pd
# import numpy as np
# import os
# import glob
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt
# from lightgbm import LGBMRegressor, plot_importance

# ## Load all cleaned CostReport CSVs (2015–2021)
# cost_files = {
#     "2015": "2015_CostReport_cleaned.csv",
#     "2016": "2016_CostReport_cleaned.csv",
#     "2017": "2017_CostReport_cleaned.csv",
#     "2018": "2018_CostReport_cleaned.csv",
#     "2019": "2019_CostReport_cleaned.csv",
#     "2020": "2020_CostReport_cleaned.csv",
#     "2021": "2021_CostReport_cleaned.csv"
# }

### Create an empty list to hold each year's DataFrame
# all_dataframes = []
### Loop through files, read, and append the year
# for year, path in cost_files.items():
#     df = pd.read_csv(path, low_memory=False)
#     df['year'] = int(year)
#     all_dataframes.append(df)

# # Combine all years into one DataFrame
# merged_df = pd.concat(all_dataframes, ignore_index=True)

# ## Define features and target
# features = [
#     'occupancy_rate',
#     'staffing_rating',
#     'rn_staffing_rating',
#     'total_liabilities',
#     'total_salaries_adjusted',
#     'bedcert'
# ]
# target = 'net_income'

# ## Ensure columns are numeric
# for col in features + [target]:
#     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# ## Clean the dataset
# data = merged_df.dropna(subset=features + [target])
# X = data[features]
# y = data[target]

# ## Split into train/test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ## Train the LightGBM model
# model = LGBMRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# ## Evaluate the model
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print("\nLightGBM Regression Results:")
# print(f"R² Score: {r2:.4f}")
# print(f"RMSE: ${rmse:,.2f}")

# ## Plot feature importance
# plot_importance(model, title='LightGBM Feature Importance', xlabel='F Score')
# plt.tight_layout()
# plt.show()

######################

# Importing required libraries
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
import xgboost as xgb  # for using XGBoost machine learning model
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for advanced plotting (not used in this script)
from sklearn.model_selection import train_test_split, GridSearchCV  # for splitting data and parameter tuning
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.metrics import mean_squared_error, r2_score  # for evaluating model performance

# Dictionary mapping years to corresponding cleaned CSV file paths
files = {
    "2015": "2015_CostReport_cleaned.csv",
    "2016": "2016_CostReport_cleaned.csv",
    "2017": "2017_CostReport_cleaned.csv",
    "2018": "2018_CostReport_cleaned.csv",
    "2019": "2019_CostReport_cleaned.csv",
    "2020": "2020_CostReport_cleaned.csv",
    "2021": "2021_CostReport_cleaned.csv"
}

# List to store dataframes for each year
all_dataframes = []

# Loop through each file, read it, add a 'Year' column, and append to the list
for year, path in files.items():
    df = pd.read_csv(path, low_memory=False)  # read CSV file
    df['Year'] = int(year)  # add a 'Year' column for reference
    all_dataframes.append(df)  # append the dataframe to the list

# Combine all yearly dataframes into one single dataframe
merged_df = pd.concat(all_dataframes, ignore_index=True)

# Selecting relevant features for the model (commented out features are excluded)
X = merged_df[[
    # 'cash_on_hand_and_in_banks',
    # 'overhead_non_salary_costs',
    # 'total_fund_balances',
    'gross_revenue',
    'total_income',
    'total_costs',
    'total_current_assets',
    'total_current_liabilities',
    'total_liabilities',
    'total_fund_balances',
    'total_fixed_assets']]

# Define target variable (what we want to predict)
y = merged_df['net_income']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on training data
X_test = scaler.transform(X_test)  # transform test data using the same scaler

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # number of trees
    'learning_rate': [0.05, 0.1, 0.2],  # step size shrinkage
    'max_depth': [3, 4, 5]  # maximum depth of trees
}

# Use GridSearchCV to find the best combination of parameters
grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)  # train with cross-validation

# Retrieve and print the best hyperparameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train final model using the best parameters from GridSearch
model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using common regression metrics
mse = mean_squared_error(y_test, y_pred)  # mean squared error
r2 = r2_score(y_test, y_pred)  # R-squared score
rmse = np.sqrt(mse)  # root mean squared error

# Print evaluation results
print(f"\nFinal RMSE on Test Set: {rmse:.2f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# # Feature importance visualization (currently commented out)
# feature_importance = model.feature_importances_
# sorted_idx = np.argsort(feature_importance)[::-1]

# plt.figure(figsize=(10, 6))
# plt.barh(np.array(X.columns)[sorted_idx], feature_importance[sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("XGBoost Feature Importance")
# plt.show()

# # Scatter plot to compare actual vs predicted values (currently commented out)
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("XGBoost (XGBRegressor): Actual vs Predicted")
# plt.show()

##################################################

# Importing required libraries
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
import xgboost as xgb  # for using XGBoost machine learning model
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for advanced plotting (not used in this script)
from sklearn.model_selection import train_test_split, GridSearchCV  # for splitting data and parameter tuning
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.metrics import mean_squared_error, r2_score  # for evaluating model performance

# Dictionary mapping years to corresponding cleaned CSV file paths
files = {
    "2015": "2015_CostReport_cleaned.csv",
    "2016": "2016_CostReport_cleaned.csv",
    "2017": "2017_CostReport_cleaned.csv",
    "2018": "2018_CostReport_cleaned.csv",
    "2019": "2019_CostReport_cleaned.csv",
    "2020": "2020_CostReport_cleaned.csv",
    "2021": "2021_CostReport_cleaned.csv"
}

# List to store dataframes for each year
all_dataframes = []

# Loop through each file, read it, add a 'Year' column, and append to the list
for year, path in files.items():
    df = pd.read_csv(path, low_memory=False)  # read CSV file
    df['Year'] = int(year)  # add a 'Year' column for reference
    all_dataframes.append(df)  # append the dataframe to the list

# Combine all yearly dataframes into one single dataframe
merged_df = pd.concat(all_dataframes, ignore_index=True)

# Selecting relevant features for the model (commented out features are excluded)
X = merged_df[[
    # 'cash_on_hand_and_in_banks',
    # 'overhead_non_salary_costs',
    # 'total_fund_balances',
    'gross_revenue',
    'total_income',
    'total_costs',
    'total_current_assets',
    'total_current_liabilities',
    'total_liabilities',
    'total_fund_balances',
    'total_fixed_assets']]

# Define target variable (what we want to predict)
y = merged_df['net_income']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on training data
X_test = scaler.transform(X_test)  # transform test data using the same scaler

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # number of trees
    'learning_rate': [0.05, 0.1, 0.2],  # step size shrinkage
    'max_depth': [3, 4, 5]  # maximum depth of trees
}

# Use GridSearchCV to find the best combination of parameters
grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)  # train with cross-validation

# Retrieve and print the best hyperparameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train final model using the best parameters from GridSearch
model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using common regression metrics
mse = mean_squared_error(y_test, y_pred)  # mean squared error
r2 = r2_score(y_test, y_pred)  # R-squared score
rmse = np.sqrt(mse)  # root mean squared error

# Print evaluation results
print(f"\nFinal RMSE on Test Set: {rmse:.2f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# # Feature importance visualization (currently commented out)
# feature_importance = model.feature_importances_
# sorted_idx = np.argsort(feature_importance)[::-1]

# plt.figure(figsize=(10, 6))
# plt.barh(np.array(X.columns)[sorted_idx], feature_importance[sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("XGBoost Feature Importance")
# plt.show()

# # Scatter plot to compare actual vs predicted values (currently commented out)
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("XGBoost (XGBRegressor): Actual vs Predicted")
# plt.show()

#######################################
#####
### Random Forest Regression Analysis ######

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load merged dataset
df = pd.read_csv("Merged_2015_Data.csv")

# Clean target variable
df["WEIGHTED_ALL_CYCLES_SCORE"] = pd.to_numeric(df["WEIGHTED_ALL_CYCLES_SCORE"], errors="coerce")
df = df.dropna(subset=["WEIGHTED_ALL_CYCLES_SCORE"])

# Define features
features = [
    "incident_cnt", "cmplnt_cnt", "FINE_TOT", "TOT_PENLTY_CNT",
    "deficiency_count", "total_fines", "total_payden_days",
    "avg_quality_measure", "total_fund_balances", "total_liabilities", "total_other_assets"
]
df = df.dropna(subset=features)
X = df[features]
y = df["WEIGHTED_ALL_CYCLES_SCORE"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print(f"R²: {r2:.3f}, RMSE: {rmse:.2f}")

# Plot feature importances
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind="barh", figsize=(10, 6), title="Feature Importance (2015)")
plt.tight_layout()
plt.show()

#############################
##### Data Merging and Exporting #####
# Importing required libraries

import pandas as pd

# Load CSV files from your local directory
provider_df = pd.read_csv("Cleaned_ProviderInfo_2015.csv", dtype={"provnum": str})
cost_df = pd.read_csv("2015_CostReport_cleaned.csv", dtype={"provider_ccn": str})
deficiency_df = pd.read_csv("HealthDeficiencies_2015.csv", encoding="latin1", dtype={"provnum": str})
penalty_df = pd.read_csv("Penalties_2015_Clean.csv", dtype={"provnum": str})
quality_df = pd.read_csv("QualityMsrMDS_2015_Cleaned.csv", dtype={"provnum": str})

# Standardize provider ID format
provider_df["provnum"] = provider_df["provnum"].str.zfill(6)
cost_df["provider_ccn"] = cost_df["provider_ccn"].str.zfill(6)
deficiency_df["provnum"] = deficiency_df["provnum"].str.zfill(6)
penalty_df["provnum"] = penalty_df["provnum"].str.zfill(6)
quality_df["provnum"] = quality_df["provnum"].str.zfill(6)

# Aggregate deficiency counts
deficiency_counts = deficiency_df.groupby("provnum").size().reset_index(name="deficiency_count")

# Summarize penalties
penalty_summary = penalty_df.groupby("provnum").agg({
    "fine_amt": "sum",
    "payden_days": "sum"
}).reset_index().rename(columns={"fine_amt": "total_fines", "payden_days": "total_payden_days"})

# Average quality measure
quality_scores = quality_df.groupby("provnum").agg({
    "measure_score_3qtr_avg": "mean"
}).reset_index().rename(columns={"measure_score_3qtr_avg": "avg_quality_measure"})

# Select cost variables
cost_financials = cost_df[[
    "provider_ccn", "total_fund_balances", "total_liabilities", "total_other_assets"
]].rename(columns={"provider_ccn": "provnum"})

# Merge all data
merged_df = provider_df[[
    "provnum", "WEIGHTED_ALL_CYCLES_SCORE", "incident_cnt", "cmplnt_cnt", "FINE_TOT", "TOT_PENLTY_CNT"
]].copy()

merged_df = merged_df.merge(deficiency_counts, on="provnum", how="left")
merged_df = merged_df.merge(penalty_summary, on="provnum", how="left")
merged_df = merged_df.merge(quality_scores, on="provnum", how="left")
merged_df = merged_df.merge(cost_financials, on="provnum", how="left")

# Convert score to numeric
merged_df["WEIGHTED_ALL_CYCLES_SCORE"] = pd.to_numeric(merged_df["WEIGHTED_ALL_CYCLES_SCORE"], errors='coerce')

# Export to local CSV
merged_df.to_csv("Merged_2015_Data.csv", index=False)



