"""
Discount Calendar Designer

For each Season × Product combination:
  1. Simulate discount scenarios: 0%, 5%, 10%, 15%, 20%
  2. Predict demand for each scenario using trained XGBoost
  3. Calculate revenue: pred_demand × price × (1 - discount/100)
  4. Select the discount that maximises revenue
  5. Output as heatmap calendar + CSV

Revenue formula:
    revenue = pred_demand × price × (1 - disc / 100)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..") + "/src")
from preprocessing import load_grocery_data, normalise_features
from feature_engineering import add_time_features, add_interpolated_order, get_feature_sets

DISCOUNT_LEVELS = [0, 5, 10, 15, 20]
TARGET = "Demand"
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3:  "Spring", 4: "Spring", 5: "Spring",
    6:  "Summer", 7: "Summer", 8: "Summer",
    9:  "Autumn", 10: "Autumn", 11: "Autumn",
}


def prepare_calendar_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add season, holiday flags, and one-hot encode categoricals."""
    df = df.copy()
    df["Season"] = df["Date"].dt.month.map(SEASON_MAP)

    try:
        import holidays
        us_holidays = holidays.US(years=df["Date"].dt.year.unique().tolist())
        df["is_holiday"] = df["Date"].isin(us_holidays).astype(int)
    except ImportError:
        df["is_holiday"] = 0

    cat_cols = df.select_dtypes(include="object").columns.difference(["Date"])
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def train_demand_model(df: pd.DataFrame, feature_cols: list) -> xgb.Booster:
    """Train XGBoost on 80% of the data (time-series split)."""
    X = df[feature_cols].fillna(0).values
    y = df[TARGET].values
    split = int(len(X) * 0.8)
    dtrain = xgb.DMatrix(X[:split], label=y[:split])
    dval   = xgb.DMatrix(X[split:], label=y[split:])

    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    print(f"Model trained on {split} rows, validated on {len(X) - split} rows.")
    return model, feature_cols


def build_discount_calendar(df: pd.DataFrame, model: xgb.Booster, feature_cols: list) -> pd.DataFrame:
    """
    For each Season × Product:
      - Test each discount level
      - Predict demand
      - Calculate revenue
      - Select optimal discount
    """
    results = []
    product_col = [c for c in df.columns if "Product" in c][0]
    season_col  = [c for c in df.columns if "Season" in c or "season" in c.lower()][0] \
                  if any("Season" in c for c in df.columns) else None

    if season_col is None:
        df["Season"] = df["Date"].dt.month.map(SEASON_MAP)
        season_col = "Season"

    for season in df[season_col].unique():
        for product in df[product_col].unique() if product_col in df.columns else ["All"]:
            subset = df[(df[season_col] == season)]
            if product_col in df.columns:
                subset = subset[subset[product_col] == product]
            if subset.empty:
                continue

            avg_price = subset["Price"].mean() if "Price" in subset.columns else 10.0
            best_revenue = -np.inf
            best_discount = 0

            for disc in DISCOUNT_LEVELS:
                row = subset.copy()
                if "Discount" in row.columns:
                    row["Discount"] = disc
                X_scenario = row[feature_cols].fillna(0)
                dmat = xgb.DMatrix(X_scenario.values)
                pred_demand = model.predict(dmat).mean()
                revenue = pred_demand * avg_price * (1 - disc / 100)

                if revenue > best_revenue:
                    best_revenue = revenue
                    best_discount = disc

            results.append({
                "Season": season,
                "Product": product,
                "Optimal_Discount_%": best_discount,
                "Est_Revenue": round(best_revenue, 2),
            })

    return pd.DataFrame(results)


def plot_calendar(calendar_df: pd.DataFrame):
    """Display the discount calendar as a heatmap."""
    pivot = calendar_df.pivot(index="Product", columns="Season", values="Optimal_Discount_%")
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    pivot = pivot[[s for s in season_order if s in pivot.columns]]

    plt.figure(figsize=(10, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, cbar_kws={"label": "Optimal Discount (%)"}
    )
    plt.title("Product × Season Optimal Discount Calendar")
    plt.xlabel("Season")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig("discount_calendar.png", dpi=150)
    plt.show()
    print("Saved: discount_calendar.png")


if __name__ == "__main__":
    df = load_grocery_data("../data/retail_store_inventory.csv")
    df, _ = normalise_features(df)
    df = add_time_features(df)
    df = add_interpolated_order(df)
    df = prepare_calendar_data(df)

    feature_sets = get_feature_sets(df)
    feature_cols = [f for f in feature_sets["Set_2"] if f in df.columns]

    model, feature_cols = train_demand_model(df, feature_cols)
    calendar = build_discount_calendar(df, model, feature_cols)

    print("\nOptimal Discount Calendar:")
    print(calendar.to_string(index=False))

    calendar.to_csv("discount_calendar.csv", index=False)
    print("\nSaved: discount_calendar.csv")

    plot_calendar(calendar)
