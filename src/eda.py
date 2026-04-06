"""
Exploratory Data Analysis for retail demand forecasting.

Covers:
  - Histograms of all numeric features
  - Correlation heatmap
  - Monthly demand heatmaps per store
  - Seasonal demand comparison 2022 vs 2023
  - Epidemic effect on seasonal and monthly demand
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_grocery_data

sns.set(style="whitegrid")


def plot_histograms(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n = len(numeric_cols)
    fig, axes = plt.subplots(nrows=(n // 4) + 1, ncols=4, figsize=(18, 4 * ((n // 4) + 1)))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("All Numeric Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("histograms.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Note: Epidemic [0/1], Promotion [0/1], Discount [0/5/10/15/20] are ordinal.")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4)
    plt.title("Correlation Heatmap — Grocery Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Insight: Select high-correlation features with Demand as model inputs.")


def plot_monthly_demand_heatmap(df: pd.DataFrame):
    df = df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    pivot = df.pivot_table(values="Demand", index="Store_ID", columns="YearMonth", aggfunc="mean")
    plt.figure(figsize=(20, 5))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.1)
    plt.title("Monthly Average Demand per Store")
    plt.xlabel("Month")
    plt.ylabel("Store")
    plt.tight_layout()
    plt.savefig("monthly_demand_heatmap.png", dpi=150)
    plt.show()
    print("Insight: Stores show synchronised demand patterns with sudden spikes/drops.")


def plot_epidemic_effect(df: pd.DataFrame):
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Date"].dt.month.map(
        {12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for epidemic_val, label in [(0, "Non-Epidemic"), (1, "Epidemic")]:
        subset = df[df["Epidemic"] == epidemic_val]
        seasonal = subset.groupby("Season")["Demand"].mean()
        axes[0].bar(seasonal.index, seasonal.values,
                    alpha=0.6, label=label)
    axes[0].set_title("Seasonal Demand: Epidemic vs Non-Epidemic")
    axes[0].legend()

    monthly = df.groupby(["Month", "Epidemic"])["Demand"].mean().unstack()
    monthly.plot(ax=axes[1], marker="o")
    axes[1].set_title("Monthly Demand by Epidemic Status")
    axes[1].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig("epidemic_effect.png", dpi=150)
    plt.show()
    print("Insight: Epidemic periods drive the largest demand anomalies — key feature.")


if __name__ == "__main__":
    df = load_grocery_data("data/retail_store_inventory.csv")
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_monthly_demand_heatmap(df)
    plot_epidemic_effect(df)
