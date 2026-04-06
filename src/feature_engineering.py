"""
Feature engineering for demand forecasting.

Two groups of features:
  1. Time-based: month, week, weekday, is_weekend, is_month_start
  2. Interpolated_Order: smoothed daily order volume for zero-order days

Feature sets (used in model experiments):
  Set_0: Most relevant features (selected manually)
  Set_1: Top features by RF importance
  Set_2: Top original features only (no engineered)  ← best for XGBoost
  Set_3: All top features
  Set_4: All features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from preprocessing import load_grocery_data, normalise_features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]           = df["Date"].dt.year
    df["month"]          = df["Date"].dt.month
    df["week"]           = df["Date"].dt.isocalendar().week.astype(int)
    df["day"]            = df["Date"].dt.day
    df["day_of_week"]    = df["Date"].dt.dayofweek
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["quarter"]        = df["Date"].dt.quarter
    print("Added time features: year, month, week, day, day_of_week, is_weekend, is_month_start, quarter")
    return df


def add_interpolated_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    For days where Units_Ordered == 0, estimate the implied daily order rate
    by spreading the last known order evenly across days since that order.

    Formula: Interpolated_Order = Last_Order_Amount / Days_Since_Last_Order
    """
    df = df.copy().sort_values(["Store_ID", "Product_ID", "Date"])

    df["Last_Order_Date"]   = None
    df["Last_Order_Amount"] = 0.0

    for key, group in df.groupby(["Store_ID", "Product_ID"]):
        mask = df["Store_ID"] == key[0]
        mask &= df["Product_ID"] == key[1]

        last_date   = None
        last_amount = 0.0
        for idx in df[mask].index:
            if df.at[idx, "Units_Ordered"] > 0:
                last_date   = df.at[idx, "Date"]
                last_amount = df.at[idx, "Units_Ordered"]
            df.at[idx, "Last_Order_Date"]   = last_date
            df.at[idx, "Last_Order_Amount"] = last_amount

    df["Last_Order_Date"] = pd.to_datetime(df["Last_Order_Date"])
    df["Days_Since_Last_Order"] = (df["Date"] - df["Last_Order_Date"]).dt.days.fillna(1).clip(lower=1)
    df["Interpolated_Order"] = df["Last_Order_Amount"] / df["Days_Since_Last_Order"]

    print("Added: Days_Since_Last_Order, Interpolated_Order")
    return df


def get_feature_sets(df: pd.DataFrame) -> dict:
    original_features = [
        "Price", "Competitor_Price", "Inventory_Level",
        "Epidemic", "Promotion", "Discount",
        "month", "week", "day_of_week", "is_weekend", "is_month_start"
    ]
    engineered_features = ["Interpolated_Order", "Days_Since_Last_Order"]

    sets = {
        "Set_0": ["Price", "Inventory_Level", "Epidemic", "Promotion", "month", "Interpolated_Order"],
        "Set_1": original_features[:6] + engineered_features,
        "Set_2": original_features,                          # top original only ← XGBoost best
        "Set_3": original_features + engineered_features,
        "Set_4": [c for c in df.columns if c not in ["Date", "Demand", "Store_ID", "Product_ID",
                                                       "Category", "Region", "Season"]],
    }
    return sets


def plot_feature_importance(df: pd.DataFrame, target: str = "Demand"):
    feature_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != target and c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    colors = ["#2ecc71" if f in ["Interpolated_Order", "Days_Since_Last_Order",
                                  "month", "week", "day_of_week", "is_weekend", "is_month_start"]
              else "#3498db" for f in importance.index]
    importance.plot(kind="bar", color=colors)
    plt.title("Feature Importance (Random Forest) — Green = Engineered Features")
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    return importance


if __name__ == "__main__":
    df = load_grocery_data("data/retail_store_inventory.csv")
    df, _ = normalise_features(df)
    df = add_time_features(df)
    df = add_interpolated_order(df)
    importance = plot_feature_importance(df)
    print("\nTop 10 features:")
    print(importance.head(10))
