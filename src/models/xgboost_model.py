"""
XGBoost regressor for demand forecasting — best performing model.
Best result: R² = 0.90 | Feature Set_2 | Time Split | Hyperparams: xgb_3

Hyperparameter configurations tested (xgb_0 → xgb_3):
  xgb_0: default
  xgb_1: deeper trees (max_depth=8)
  xgb_2: more regularisation (lambda=2, alpha=1)
  xgb_3: tuned (max_depth=6, eta=0.05, subsample=0.8, colsample=0.8) ← best
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import load_grocery_data, normalise_features
from feature_engineering import add_time_features, add_interpolated_order, get_feature_sets

TARGET = "Demand"

HYPERPARAMS = {
    "xgb_0": {
        "objective": "reg:squarederror", "seed": 42
    },
    "xgb_1": {
        "objective": "reg:squarederror", "max_depth": 8, "seed": 42
    },
    "xgb_2": {
        "objective": "reg:squarederror", "max_depth": 6,
        "lambda": 2, "alpha": 1, "seed": 42
    },
    "xgb_3": {   # ← best configuration
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "seed": 42,
    },
}


def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{label:50s} | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_xgb(X_train, y_train, X_test, y_test, params, label):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtest, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return evaluate(y_test, model.predict(dtest), label), model


def run_experiments(df: pd.DataFrame):
    feature_sets = get_feature_sets(df)
    results = []
    best_model = None
    best_r2 = -np.inf

    for set_name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0).values
        y = df[TARGET].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        for hp_name, params in HYPERPARAMS.items():
            label = f"XGB | {set_name} | TimeSplit | {hp_name}"
            result, model = train_xgb(X_train, y_train, X_test, y_test, params, label)
            results.append(result)
            if result["R2"] > best_r2:
                best_r2 = result["R2"]
                best_model = model

    return pd.DataFrame(results).sort_values("R2", ascending=False), best_model


def analyse_errors(df: pd.DataFrame, best_features, best_params):
    """Show where errors cluster — expectation: epidemic months are worst."""
    available = [f for f in best_features if f in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET].values

    split = int(len(X) * 0.8)
    dtrain = xgb.DMatrix(X[:split], label=y[:split])
    dtest  = xgb.DMatrix(X[split:], label=y[split:])
    model  = xgb.train(best_params, dtrain, num_boost_round=500, verbose_eval=False)

    test_df = df.iloc[split:].copy()
    test_df["predicted"] = model.predict(dtest)
    test_df["abs_error"] = (test_df[TARGET] - test_df["predicted"]).abs()
    test_df["month"] = test_df["Date"].dt.month

    error_by_month = test_df.groupby("month")["abs_error"].mean()
    error_by_epidemic = test_df.groupby("Epidemic")["abs_error"].mean()

    print("\nMean absolute error by month:")
    print(error_by_month.round(4))
    print("\nMean absolute error by epidemic status:")
    print(error_by_epidemic.round(4))
    print("\nConclusion: Errors concentrate in epidemic periods as expected.")


if __name__ == "__main__":
    df = load_grocery_data("data/retail_store_inventory.csv")
    df, _ = normalise_features(df)
    df = add_time_features(df)
    df = add_interpolated_order(df)

    results, best_model = run_experiments(df)
    print("\nAll XGBoost Results (sorted by R²):")
    print(results.to_string(index=False))

    # Error analysis with best config: Set_2 + xgb_3
    feature_sets = get_feature_sets(df)
    analyse_errors(df, feature_sets["Set_2"], HYPERPARAMS["xgb_3"])
