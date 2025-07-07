import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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

# 6. Prepare DMatrix for xgboost.cv
dtrain = xgb.DMatrix(OH_X_train, label=y_train_log)

# 7. Define parameters
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'eval_metric': 'rmse'
}

# 8. Run cross-validation with early stopping
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=5,
    metrics="rmse",
    seed=42,
    verbose_eval=50
)

best_n_estimators = len(cv_results)
print(f"Best number of boosting rounds: {best_n_estimators}")

# 9. Plot CV learning curve (RMSE vs boosting round)
plt.figure(figsize=(10, 6))
plt.plot(cv_results['train-rmse-mean'], label='Train RMSE')
plt.plot(cv_results['test-rmse-mean'], label='CV RMSE')
plt.fill_between(cv_results.index,
                 cv_results['test-rmse-mean'] - cv_results['test-rmse-std'],
                 cv_results['test-rmse-mean'] + cv_results['test-rmse-std'],
                 alpha=0.2)
plt.axvline(best_n_estimators, color='r', linestyle='--', label='Best Iteration')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('XGBoost Cross-Validation RMSE')
plt.legend()
plt.grid(True)
plt.show()

# 10. Train final XGBRegressor model
my_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=0.05,
    n_jobs=4
)

my_model.fit(OH_X_train, y_train_log)

# 11. Evaluate on validation set
y_val_pred_log = my_model.predict(OH_X_val)
y_val_pred = np.expm1(y_val_pred_log)  # revert log1p

rmse = np.sqrt(mean_squared_error(np.expm1(y_val_log), y_val_pred))
print(f"Validation RMSE: {rmse:.2f}")

# 12. Scatter plot of true vs predicted prices on validation set
plt.figure(figsize=(8, 8))
sns.scatterplot(x=np.expm1(y_val_log), y=y_val_pred, alpha=0.5)
plt.plot([0, y_val_pred.max()], [0, y_val_pred.max()], 'r--')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title('Validation Set: True vs Predicted Prices')
plt.grid(True)
plt.show()
