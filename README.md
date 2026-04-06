# Retail Demand Forecasting & Discount Optimisation

A machine learning pipeline to **forecast daily grocery demand** across 5 retail stores and design an **optimal seasonal discount calendar** that maximises revenue while reducing food waste.

---

## Business Problem

> In the EU, over 59 million tonnes of food waste are generated annually (132 kg/inhabitant), valued at €132 billion. At the same time, 42 million people cannot afford a quality meal every second day.

Smarter demand forecasting enables retailers to:
- Set dynamic discounts that clear expiring stock before it becomes waste
- Avoid over-ordering (inventory cost) and under-ordering (lost sales)
- Align promotions with seasonally predicted demand peaks

---

## Dataset

- **Source:** Retail Store Inventory Forecasting dataset (Kaggle)
- **Period:** 2022–2024 (includes COVID-19 epidemic periods)
- **Scope:** Grocery category — 30,400 rows × 16 features, 5 stores
- **Target variable:** `Demand` (daily units demanded per store-product)

Place the dataset CSV in the `data/` folder.

---

## Project Structure

```
retail-demand-forecasting/
├── data/                              ← Place dataset CSV here
├── notebooks/
│   └── demand_forecasting.ipynb      ← Full end-to-end analysis
├── src/
│   ├── preprocessing.py              ← Cleaning, outlier handling, normalisation
│   ├── eda.py                        ← Histograms, heatmaps, seasonal patterns
│   ├── feature_engineering.py        ← Time features + Interpolated_Order metric
│   └── models/
│       ├── linear_regression.py      ← Baseline model
│       ├── random_forest.py          ← Non-linear, feature importance
│       └── xgboost_model.py          ← Best model (R²=0.90)
├── discount_calendar/
│   └── discount_calendar.py          ← Optimal discount per season-product
├── requirements.txt
└── README.md
```

---

## Models & Results

| Model | Best R² | Split Method | Feature Set |
|-------|---------|-------------|-------------|
| Linear Regression | 0.74 | Time Split | All features (Set 4) |
| Random Forest | 0.83 | 5-Fold CV | All features (Set 4) |
| **XGBoost** | **0.90** | **Time Split** | **Set 2 (top original features)** |

**Evaluation metrics used:** MAE · RMSE · R²

**Key finding:** Epidemic periods are the primary driver of forecast errors — the model learns demand well in normal conditions but epidemic events create anomalous spikes.

---

## Feature Engineering

| Feature | Source | Purpose |
|---------|--------|---------|
| `month`, `week`, `day_of_week` | `Date` column | Capture seasonality |
| `is_weekend`, `is_month_start` | `Date` column | Promotion timing signals |
| `Interpolated_Order` | `Units_Ordered` + `Date` | Fill zero-order days with smoothed daily rate |
| `Days_Since_Last_Order` | `Units_Ordered` | Reorder cycle indicator |

Formula for `Interpolated_Order`:
```
Interpolated_Order = Last_Order_Amount / Days_Since_Last_Order
```

---

## Discount Calendar

For each **Season × Product** combination:
1. Simulate discounts: 0%, 5%, 10%, 15%, 20%
2. Predict demand for each scenario using the trained XGBoost model
3. Calculate estimated revenue: `revenue = pred_demand × price × (1 - discount/100)`
4. Select the discount that maximises revenue
5. Export as a heatmap calendar and CSV

---

## Business Recommendations

1. **Discount Calendar** — apply the season-product optimal discount automatically
2. **Reorder Point** — use demand forecasts to trigger restocking: `reorder_point = avg_demand × lead_time + safety_stock`
3. **Bundle Promotions** — pair low-demand perishables with high-demand items to reduce waste and increase basket value

---

## Setup & Usage

```bash
pip install -r requirements.txt

# Run full notebook
jupyter notebook notebooks/demand_forecasting.ipynb

# Or run individual modules
python src/preprocessing.py
python src/feature_engineering.py
python src/models/xgboost_model.py
python discount_calendar/discount_calendar.py
```

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
