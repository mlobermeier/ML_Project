import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

def run_random_forest(csv_file, image_predictions=None):
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
    fig_val_curve = plt.plot(param_range, train_mae_mean, label="Training MAE", marker='o')
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

    fig_gridheatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", ax=ax_heatmap)
    ax_heatmap.set_title("Grid Search: mean_test_score (neg MAE) by max_depth and min_samples_leaf")
    ax_heatmap.set_xlabel("min_samples_leaf")
    ax_heatmap.set_ylabel("max_depth")

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

   
    fig_learning_curve, ax_lc = plt.subplots(figsize=(8, 5))
    ax_lc.plot(train_sizes, train_mae_mean, label="Training MAE", marker='o')
    ax_lc.plot(train_sizes, val_mae_mean, label="Validation MAE", marker='o')
    ax_lc.set_xlabel("Training set size")
    ax_lc.set_ylabel("MAE")
    ax_lc.set_title("Learning Curve")
    ax_lc.legend()
    ax_lc.grid(True)
    plt.show()

    # 4. Feature Importance Plot (top 20 features)
    importances = best_model.feature_importances_
    feature_names = OH_X_train.columns

    sorted_idx = np.argsort(importances)[::-1][:20]  # top 20 features

    fig_importance, ax_imp = plt.subplots(figsize=(10, 6))
    ax_imp.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], align='center')
    ax_imp.yticks(range(len(sorted_idx)))
    ax_imp.set_yticklabels(feature_names[sorted_idx][::-1])
    ax_imp.set_xlabel("Feature Importance")
    ax_imp.set_title("Top 20 Feature Importances")
    plt.grid(True)
    plt.show()

    # 5. Residual Plot (Test Set)
    residuals = y_test - test_preds

    fig_resid, ax_resid = plt.subplots(figsize=(8, 6))
    ax_resid.scatter(test_preds, residuals, alpha=0.3)
    ax_resid.axhline(y=0, color='r', linestyle='--')
    ax_resid.set_xlabel("Predicted Price")
    ax_resid.set_ylabel("Residuals (Actual - Predicted)")
    ax_resid.set_title("Residual Plot")
    ax_resid.grid(True)

    plt.show()


    # 6. Prepare results for return
    metrics = {
        "MAE": test_mae,
        "R2": r2,
        "RMSE": np.sqrt(mean_absolute_error(y_test, test_preds)**2),
    }
    preds = {
        "val_preds": val_preds,
        "test_preds": test_preds
    }
    figs = {
        "validation_curve": fig_val_curve,
        "grid_search_heatmap": fig_gridheatmap,
        "learning_curve": fig_learning_curve,
        "feature_importance": fig_importance,
        "residual_plot": fig_resid
    }
     
    return {best_model, metrics, preds, figs}
