import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load and clean data
data = pd.read_csv("data/listings.csv")
data["price"] = data["price"].replace('[\$,]', '', regex=True).astype(float)
data = data.dropna(subset=["price"])

X = data.drop(columns=["price"])
y = data["price"]

# 2. Split into train/val/test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# 3. Identify categorical features
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
low_card_cols = [col for col in object_cols if X_train[col].nunique() < 20]

# 4. One-hot encode low cardinality categorical columns
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_card_cols]))
OH_cols_val = pd.DataFrame(OH_encoder.transform(X_val[low_card_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_card_cols]))

# Restore indices
OH_cols_train.index = X_train.index
OH_cols_val.index = X_val.index
OH_cols_test.index = X_test.index

# Drop categorical columns and combine numeric + encoded categorical columns
num_X_train = X_train.drop(columns=object_cols)
num_X_val = X_val.drop(columns=object_cols)
num_X_test = X_test.drop(columns=object_cols)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_val = pd.concat([num_X_val, OH_cols_val], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

# Convert all column names to strings to avoid sklearn errors
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_val.columns = OH_X_val.columns.astype(str)
OH_X_test.columns = OH_X_test.columns.astype(str)

# Align columns in val and test to train
OH_X_val = OH_X_val.reindex(columns=OH_X_train.columns, fill_value=0)
OH_X_test = OH_X_test.reindex(columns=OH_X_train.columns, fill_value=0)

# 5. Log transform targets
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

# 6. Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"]
}

model = RandomForestRegressor(random_state=0)
grid = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
grid.fit(OH_X_train, y_train_log)

print("Best parameters found:", grid.best_params_)

# 7. Evaluate best model
best_model = grid.best_estimator_
val_preds_log = best_model.predict(OH_X_val)
val_preds = np.expm1(val_preds_log)

test_preds_log = best_model.predict(OH_X_test)
test_preds = np.expm1(test_preds_log)

val_mae = mean_absolute_error(y_val, val_preds)
test_mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print(f"Validation MAE: ${val_mae:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Test R²: {r2:.2f}")

# === PLOTS === #

# 1. Validation Curve for max_depth
param_range = [5, 10, 15, 20, 30, 50]  # 50 instead of None for plotting

train_scores, val_scores = validation_curve(
    RandomForestRegressor(random_state=0, n_estimators=100),
    OH_X_train, y_train_log,
    param_name="max_depth",
    param_range=param_range,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
)

train_mae_mean = -np.mean(train_scores, axis=1)
val_mae_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mae_mean, label="Training MAE", marker='o')
plt.plot(param_range, val_mae_mean, label="Validation MAE", marker='o')
plt.xlabel("max_depth")
plt.ylabel("MAE")
plt.title("Validation Curve for max_depth")
plt.legend()
plt.grid(True)
plt.show()

# 2. Grid Search Heatmap (max_depth vs min_samples_leaf)

results = pd.DataFrame(grid.cv_results_)
pivot_table = results.pivot_table(
    index='param_max_depth',
    columns='param_min_samples_leaf',
    values='mean_test_score'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
plt.title("Grid Search: mean_test_score (neg MAE) by max_depth and min_samples_leaf")
plt.xlabel("min_samples_leaf")
plt.ylabel("max_depth")
plt.show()

# 3. Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    OH_X_train, y_train_log,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
)

train_mae_mean = -np.mean(train_scores, axis=1)
val_mae_mean = -np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mae_mean, label="Training MAE", marker='o')
plt.plot(train_sizes, val_mae_mean, label="Validation MAE", marker='o')
plt.xlabel("Training set size")
plt.ylabel("MAE")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

# 4. Feature Importance Plot (top 20 features)
importances = best_model.feature_importances_
feature_names = OH_X_train.columns

sorted_idx = np.argsort(importances)[::-1][:20]  # top 20 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], align='center')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances")
plt.grid(True)
plt.show()

# 5. Residual Plot (Test Set)
residuals = y_test - test_preds

plt.figure(figsize=(8, 6))
plt.scatter(test_preds, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.show()
