"""
Linear Regression baseline for demand forecasting.
Best result: R² = 0.74 with all features (Set_4) and time-series split.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import load_grocery_data, normalise_features
from feature_engineering import add_time_features, add_interpolated_order, get_feature_sets

TARGET = "Demand"


def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{label:30s} | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def run_experiments(df: pd.DataFrame):
    feature_sets = get_feature_sets(df)
    results = []

    for set_name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0)
        y = df[TARGET]

        # Experiment A: Hold-out (80/20 chronological split)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        results.append(evaluate(y_test, model.predict(X_test), f"LR | {set_name} | TimesSplit"))

        # Experiment B: 5-Fold CV
        kf = KFold(n_splits=5, shuffle=False)
        preds, actuals = [], []
        for tr, te in kf.split(X):
            m = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
            preds.extend(m.predict(X.iloc[te]))
            actuals.extend(y.iloc[te])
        results.append(evaluate(actuals, preds, f"LR | {set_name} | 5-Fold CV"))

    return pd.DataFrame(results).sort_values("R2", ascending=False)


if __name__ == "__main__":
    df = load_grocery_data("data/retail_store_inventory.csv")
    df, _ = normalise_features(df)
    df = add_time_features(df)
    df = add_interpolated_order(df)

    results = run_experiments(df)
    print("\nAll Linear Regression Results:")
    print(results.to_string(index=False))
