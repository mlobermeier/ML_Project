import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and clean data
data = pd.read_csv("data/listings.csv")
data["price"] = data["price"].replace('[\$,]', '', regex=True).astype(float)
data = data.dropna(subset=["price"])
X = data.drop(columns=["price"])
y = data["price"]

# 2. Split into train/val/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42)

# 3. Identify categorical features
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
low_card_cols = [col for col in object_cols if X_train[col].nunique() < 20]
high_card_cols = list(set(object_cols) - set(low_card_cols))


# 4. Frequency encode high-cardinality categorical columns
def freq_encode(train, val, test, cols):
    for col in cols:
        freq = train[col].value_counts() / len(train)
        train[col] = train[col].map(freq)
        val[col] = val[col].map(freq).fillna(0)
        test[col] = test[col].map(freq).fillna(0)
    return train, val, test

X_train_fe, X_val_fe, X_test_fe = freq_encode(
    X_train.copy(), X_val.copy(), X_test.copy(), high_card_cols)

# 5. One-hot encode low-cardinality categorical columns
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_fe[low_card_cols]))
OH_cols_val = pd.DataFrame(OH_encoder.transform(X_val_fe[low_card_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test_fe[low_card_cols]))

# Restore indices for concatenation
OH_cols_train.index = X_train_fe.index
OH_cols_val.index = X_val_fe.index
OH_cols_test.index = X_test_fe.index

# 6. Drop low-cardinality categorical columns from numeric sets
num_X_train = X_train_fe.drop(columns=low_card_cols)
num_X_val = X_val_fe.drop(columns=low_card_cols)
num_X_test = X_test_fe.drop(columns=low_card_cols)

# 7. Combine numeric + frequency encoded + one-hot encoded features
X_train_final = pd.concat([num_X_train, OH_cols_train], axis=1)
X_val_final = pd.concat([num_X_val, OH_cols_val], axis=1)
X_test_final = pd.concat([num_X_test, OH_cols_test], axis=1)

# Convert all column names to strings
X_train_final.columns = X_train_final.columns.astype(str)
X_val_final.columns = X_val_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

# Ensure the same columns order and types across datasets
X_val_final = X_val_final.reindex(columns=X_train_final.columns, fill_value=0)
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

# 8. Log-transform target variable
y_train_log = np.log1p(y_train)  # log(1 + price)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

# 9. Train model on log-transformed targets
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train_final, y_train_log)

# 10. Predict and inverse-transform predictions
val_preds_log = model.predict(X_val_final)
val_preds = np.expm1(val_preds_log)

test_preds_log = model.predict(X_test_final)
test_preds = np.expm1(test_preds_log)

# 11. Evaluate performance on original scale
val_mae = mean_absolute_error(y_val, val_preds)
test_mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print("MAE on validation set (original $):", val_mae)
print("MAE on test set (original $):", test_mae)
print("R² on test set:", r2)

# 12. Plot results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_preds, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted Prices (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.text(50, 900, f'MAE: ${test_mae:.2f}\nR²: {r2:.2f}', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.5))
plt.show()
