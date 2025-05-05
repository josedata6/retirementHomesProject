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