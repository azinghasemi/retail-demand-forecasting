"""
Data loading, cleaning, and preprocessing for retail demand forecasting.

Steps:
  1. Load and filter to Grocery category
  2. Parse dates
  3. Handle outliers (Z-score & IQR analysis — kept as-is for time-series continuity)
  4. Normalise numeric features (MinMax)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


NUMERIC_COLS = ["Price", "Competitor_Price", "Demand", "Units_Sold", "Inventory_Level"]
ORDINAL_COLS = ["Epidemic", "Promotion", "Discount"]  # treated as ordinal, not normalised


def load_grocery_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df["Category"] == "Groceries"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)
    print(f"Grocery subset shape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Stores: {df['Store_ID'].nunique()} | Products: {df['Product_ID'].nunique()}")
    return df


def detect_outliers(df: pd.DataFrame, col: str):
    """Compare Z-score vs IQR outlier counts for a numeric column."""
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    z_outliers = (z_scores > 3).sum()

    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()

    print(f"{col}: Z-score outliers={z_outliers}, IQR outliers={iqr_outliers}")


def analyse_outliers(df: pd.DataFrame):
    """
    Report outliers but do NOT remove them.
    Rationale: epidemic-driven demand spikes are real events, not noise.
    Removing them would break time-series continuity and destroy epidemic feature signal.
    """
    print("Outlier analysis (rows kept — epidemic spikes are real):")
    for col in NUMERIC_COLS:
        if col in df.columns:
            detect_outliers(df, col)


def normalise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MinMax normalisation for continuous numeric columns.
    Ordinal columns (Epidemic, Promotion, Discount) are left unchanged.
    """
    scaler = MinMaxScaler()
    cols_to_scale = [c for c in NUMERIC_COLS if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"MinMax normalised: {cols_to_scale}")
    return df, scaler


if __name__ == "__main__":
    df = load_grocery_data("data/retail_store_inventory.csv")
    analyse_outliers(df)
    df, scaler = normalise_features(df)
    print(df[NUMERIC_COLS].describe().round(3))
