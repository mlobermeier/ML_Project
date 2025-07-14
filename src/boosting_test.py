import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_boosting(csv_file, image_predictions=None):
    # 1. Load and clean data
    data = pd.read_csv(csv_file)
    if image_predictions is not None:
        # Merge image predictions if provided
        data = data.merge(image_predictions, left_on="id", right_on="listing_id", how="left")
        # Drop rows with missing image predictions
        data = data.dropna(subset=["predicted_class"])

    
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
    #print(f"Best number of boosting rounds: {best_n_estimators}")

    # 9. Plot CV learning curve (RMSE vs boosting round)
    fig_cv_rmse, ax_cv_rmse = plt.subplots(figsize=(10, 6))
    ax_cv_rmse.plot(cv_results['train-rmse-mean'], label='Train RMSE')
    ax_cv_rmse.plot(cv_results['test-rmse-mean'], label='CV RMSE')

    ax_cv_rmse.fill_between(
        cv_results.index,
        cv_results['test-rmse-mean'] - cv_results['test-rmse-std'],
        cv_results['test-rmse-mean'] + cv_results['test-rmse-std'],
        alpha=0.2
    )

    ax_cv_rmse.axvline(best_n_estimators, color='r', linestyle='--', label='Best Iteration')
    ax_cv_rmse.set_xlabel('Boosting Round')
    ax_cv_rmse.set_ylabel('RMSE')
    ax_cv_rmse.set_title('XGBoost Cross-Validation RMSE')
    ax_cv_rmse.legend()
    ax_cv_rmse.grid(True)

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
    #print(f"Validation RMSE: {rmse:.2f}")

    # 12. Scatter plot of true vs predicted prices on validation set
    fig_val_scatter, ax_val_scatter = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=np.expm1(y_val_log), y=y_val_pred, alpha=0.5, ax=ax_val_scatter)
    ax_val_scatter.plot([0, y_val_pred.max()], [0, y_val_pred.max()], 'r--')
    ax_val_scatter.set_xlabel('True Price')
    ax_val_scatter.set_ylabel('Predicted Price')
    ax_val_scatter.set_title('Validation Set: True vs Predicted Prices')
    ax_val_scatter.grid(True)

    mae = np.mean(np.abs(np.expm1(y_val_log) - y_val_pred))
    r2 = r2_score(np.expm1(y_val_log), y_val_pred) 

    #metrics box
    metrics_text = f"MAE = {mae:.2f}\nR² = {r2:.2f}"
    ax_val_scatter.text(0.05, 0.95, metrics_text, transform=ax_val_scatter.transAxes,
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.5))

    # 13. Top 20 Feature Importances Plot
    importances = my_model.feature_importances_
    feature_names = np.array(OH_X_train.columns)

    sorted_idx = np.argsort(importances)[::-1][:20]  # top 20 features

    fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(10, 6))
    ax_feat_imp.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], align='center')
    ax_feat_imp.set_yticks(range(len(sorted_idx)))
    ax_feat_imp.set_yticklabels(feature_names[sorted_idx][::-1])
    ax_feat_imp.set_xlabel("Feature Importance")
    ax_feat_imp.set_title("Top 20 Feature Importances")
    ax_feat_imp.grid(True)

    # 14. Residual pplot (Validation Set)
    residuals = np.expm1(y_val_log) - y_val_pred  # true - predicted

    fig_residuals, ax_residuals = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_val_pred, y=residuals, ax=ax_residuals, alpha=0.5)
    ax_residuals.axhline(0, color='red', linestyle='--', linewidth=1)
    ax_residuals.set_xlabel('Predicted Price')
    ax_residuals.set_ylabel('Residuals (True - Predicted)')
    ax_residuals.set_title('Residual Plot (Validation Set)')
    ax_residuals.grid(True)


    # 15. Save model
    # my_model.save_model("xgboost_model.json")  # Uncomment to save model

    metrics = {
        "Best CV RMSE": float(cv_results['test-rmse-mean'].min()),
        "Best Iteration": int(best_n_estimators),
        "Final Train RMSE": float(cv_results['train-rmse-mean'].min()),
        "Final Test RMSE": float(cv_results['test-rmse-mean'].min()),
        "Validation RMSE": rmse,
        "MAE": np.mean(np.abs(np.expm1(y_val_log) - y_val_pred))
    }

    preds = {
        "y_val_pred": y_val_pred,
        "y_val_true": np.expm1(y_val_log)
    }

    figs = {
        "val_scatter": fig_val_scatter,
        "cv_rmse": fig_cv_rmse,
        "feature_importance": fig_feat_imp,
        "residuals": fig_residuals

    }

    log = {
        "best_n_estimators": best_n_estimators,
        "validation_rmse": rmse
    }

    features = {
        "importances": my_model.feature_importances_,
        "names": OH_X_train.columns.tolist()
    }

    return{
        "model": my_model,
        "metrics": metrics,
        "preds": preds,
        "figs": figs,
        "log": log,
        "features": features
    } 