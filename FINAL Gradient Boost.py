import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load multiple yearly datasets
files = {
    "2015": "2015_CostReport_cleaned.csv",
    "2016": "2016_CostReport_cleaned.csv",
    "2017": "2017_CostReport_cleaned.csv",
    "2018": "2018_CostReport_cleaned.csv",
    "2019": "2019_CostReport_cleaned.csv",
    "2020": "2020_CostReport_cleaned.csv",
    "2021": "2021_CostReport_cleaned.csv"
}

all_dataframes = []

# Read and merge data
for year, path in files.items():
    df = pd.read_csv(path, low_memory=False)
    df['Year'] = int(year)
    all_dataframes.append(df)

merged_df = pd.concat(all_dataframes, ignore_index=True)

# Define features and target
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

y = merged_df['net_income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for numeric values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train XGBoost model with best parameters
model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Predictions and metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nFinal RMSE on Test Set: {rmse:.2f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# # Feature importance visualization
# feature_importance = model.feature_importances_
# sorted_idx = np.argsort(feature_importance)[::-1]

# plt.figure(figsize=(10, 6))
# plt.barh(np.array(X.columns)[sorted_idx], feature_importance[sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("XGBoost Feature Importance")
# plt.show()

# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("XGBoost (XGBRegressor): Actual vs Predicted")
# plt.show()

