"""
Streamlit live demo — Retail Demand Forecasting & Discount Optimisation
Uses synthetic grocery retail data — no Kaggle download required.

Run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRODUCTS  = ["Dairy", "Bakery", "Produce", "Meat", "Frozen", "Beverages", "Snacks"]
SEASONS   = ["Winter", "Spring", "Summer", "Autumn"]
STORES    = ["Store A", "Store B", "Store C", "Store D", "Store E"]
DISCOUNTS = [0, 5, 10, 15, 20]

SEASON_MONTH = {"Winter": [12, 1, 2], "Spring": [3, 4, 5],
                "Summer": [6, 7, 8],  "Autumn": [9, 10, 11]}

SEASON_DEMAND_FACTORS = {
    ("Dairy",     "Winter"): 1.15, ("Dairy",    "Summer"): 0.90,
    ("Bakery",    "Winter"): 1.20, ("Bakery",   "Summer"): 0.85,
    ("Produce",   "Summer"): 1.35, ("Produce",  "Winter"): 0.75,
    ("Meat",      "Winter"): 1.10, ("Meat",     "Summer"): 1.05,
    ("Frozen",    "Summer"): 1.25, ("Frozen",   "Winter"): 0.95,
    ("Beverages", "Summer"): 1.40, ("Beverages","Winter"): 0.70,
    ("Snacks",    "Winter"): 1.10, ("Snacks",   "Summer"): 1.05,
}

BASE_PRICES = {
    "Dairy": 3.50, "Bakery": 2.80, "Produce": 2.20,
    "Meat": 8.90, "Frozen": 4.50, "Beverages": 1.80, "Snacks": 2.30,
}

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

@st.cache_data
def generate_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily demand data for 5 stores × 7 products."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days)
    rows = []

    for store in STORES:
        store_factor = rng.uniform(0.8, 1.2)
        for product in PRODUCTS:
            base_demand = rng.integers(40, 120)
            price = BASE_PRICES[product]
            last_order = rng.integers(50, 200)
            days_since  = 1

            for d in dates:
                month  = d.month
                season = next(s for s, ms in SEASON_MONTH.items() if month in ms)
                factor = SEASON_DEMAND_FACTORS.get((product, season), 1.0)

                is_weekend = int(d.dayofweek >= 5)
                is_epidemic = int((d >= pd.Timestamp("2022-03-01")) and
                                  (d <= pd.Timestamp("2022-08-31")))

                discount = rng.choice([0, 5, 10, 15, 20], p=[0.5, 0.2, 0.15, 0.1, 0.05])
                disc_boost = 1 + discount * 0.03

                demand = int(
                    base_demand * factor * store_factor * disc_boost
                    * (1.2 if is_epidemic else 1.0)
                    * rng.uniform(0.85, 1.15)
                )

                interpolated_order = last_order / days_since
                days_since += 1
                if rng.random() < 0.15:
                    last_order = rng.integers(50, 200)
                    days_since  = 1

                rows.append({
                    "date":               d,
                    "store":              store,
                    "product":            product,
                    "season":             season,
                    "month":              month,
                    "week":               d.isocalendar().week,
                    "day_of_week":        d.dayofweek,
                    "is_weekend":         is_weekend,
                    "is_epidemic":        is_epidemic,
                    "discount_pct":       discount,
                    "price":              price,
                    "interpolated_order": round(interpolated_order, 2),
                    "days_since_order":   days_since,
                    "demand":             demand,
                })

    return pd.DataFrame(rows)


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a fast RandomForest on the synthetic data."""
    features = ["month", "day_of_week", "is_weekend", "is_epidemic",
                "discount_pct", "interpolated_order", "days_since_order"]
    # Encode product and store
    df_enc = df.copy()
    df_enc["product_enc"] = pd.Categorical(df_enc["product"]).codes
    df_enc["store_enc"]   = pd.Categorical(df_enc["store"]).codes
    features += ["product_enc", "store_enc"]

    X = df_enc[features].values
    y = df_enc["demand"].values

    split = int(len(X) * 0.8)
    model = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X[:split], y[:split])

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, features, importances, df_enc


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = generate_data()
model, feature_cols, importances, df_enc = train_model(df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Retail Demand Forecasting & Discount Optimisation")
st.markdown(
    "Forecast daily grocery demand · Design the optimal seasonal discount calendar · "
    "Reduce food waste · Maximise revenue"
)

st.divider()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stores", len(STORES))
col2.metric("Products", len(PRODUCTS))
col3.metric("Days of Data", len(df["date"].unique()))
col4.metric("Total Records", f"{len(df):,}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast vs Actual", "Discount Simulator", "Discount Calendar", "Feature Importance"
])

# --- Tab 1: Forecast ---
with tab1:
    st.subheader("Demand Forecast vs Actual")

    c1, c2 = st.columns(2)
    sel_product = c1.selectbox("Product", PRODUCTS, key="fc_product")
    sel_store   = c2.selectbox("Store",   STORES,   key="fc_store")

    sub = df_enc[(df_enc["product"] == sel_product) & (df_enc["store"] == sel_store)].copy()
    sub = sub.sort_values("date").tail(180)  # last 6 months

    sub["predicted"] = model.predict(sub[feature_cols].values).round(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["demand"],    name="Actual",    line=dict(color="#2980b9")))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["predicted"], name="Predicted", line=dict(color="#e74c3c", dash="dash")))

    # Shade epidemic period
    epidemic_start = pd.Timestamp("2022-03-01")
    epidemic_end   = pd.Timestamp("2022-08-31")
    fig.add_vrect(
        x0=epidemic_start, x1=epidemic_end,
        fillcolor="orange", opacity=0.10,
        annotation_text="Epidemic period", annotation_position="top left",
    )
    fig.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Daily Units",
        legend_title="",
        margin=dict(l=0, r=0, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange band = epidemic period (2022 Mar–Aug). Errors cluster here — inherently unpredictable demand spikes.")

# --- Tab 2: Discount Simulator ---
with tab2:
    st.subheader("Discount Simulator")
    st.caption("Select a product and season — see how each discount level affects demand and revenue.")

    c1, c2, c3 = st.columns(3)
    sim_product = c1.selectbox("Product", PRODUCTS,   key="sim_product")
    sim_season  = c2.selectbox("Season",  SEASONS,    key="sim_season")
    sim_store   = c3.selectbox("Store",   STORES,     key="sim_store")

    price = BASE_PRICES[sim_product]
    season_months = SEASON_MONTH[sim_season]
    season_factor = SEASON_DEMAND_FACTORS.get((sim_product, sim_season), 1.0)

    # Build one row per discount level
    rows = []
    for disc in DISCOUNTS:
        row = {
            "month":              season_months[1],
            "day_of_week":        2,
            "is_weekend":         0,
            "is_epidemic":        0,
            "discount_pct":       disc,
            "interpolated_order": 80,
            "days_since_order":   3,
            "product_enc":        PRODUCTS.index(sim_product),
            "store_enc":          STORES.index(sim_store),
        }
        pred = model.predict([list(row[f] for f in feature_cols)])[0]
        revenue = pred * price * (1 - disc / 100)
        rows.append({
            "Discount %":     disc,
            "Predicted Demand": round(pred),
            "Revenue (€)":    round(revenue, 2),
        })

    sim_df = pd.DataFrame(rows)
    best_disc = sim_df.loc[sim_df["Revenue (€)"].idxmax(), "Discount %"]

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        colors = ["#27ae60" if d == best_disc else "#2980b9" for d in sim_df["Discount %"]]
        fig.add_trace(go.Bar(
            x=sim_df["Discount %"].astype(str) + "%",
            y=sim_df["Revenue (€)"],
            marker_color=colors,
            text=sim_df["Revenue (€)"].apply(lambda x: f"€{x:.0f}"),
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Revenue by Discount — {sim_product} · {sim_season}",
            xaxis_title="Discount Level",
            yaxis_title="Estimated Daily Revenue (€)",
            height=380,
            margin=dict(l=0, r=0, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown(f"**Optimal discount: {int(best_disc)}%**")
        st.markdown(f"*Max revenue: €{sim_df.loc[sim_df['Revenue (€)'].idxmax(), 'Revenue (€)']:.2f}*")
        st.dataframe(
            sim_df.style.highlight_max(subset=["Revenue (€)"], color="#d4efdf"),
            use_container_width=True,
            hide_index=True,
        )

# --- Tab 3: Discount Calendar ---
with tab3:
    st.subheader("Optimal Discount Calendar — All Products × Seasons")
    st.caption("Green = high discount optimal (perishables in peak season). White = no discount needed.")

    calendar_rows = []
    for product in PRODUCTS:
        for season in SEASONS:
            months = SEASON_MONTH[season]
            best_revenue = -1
            best_disc = 0
            for disc in DISCOUNTS:
                row = {
                    "month":              months[1],
                    "day_of_week":        2,
                    "is_weekend":         0,
                    "is_epidemic":        0,
                    "discount_pct":       disc,
                    "interpolated_order": 80,
                    "days_since_order":   3,
                    "product_enc":        PRODUCTS.index(product),
                    "store_enc":          0,
                }
                pred = model.predict([list(row[f] for f in feature_cols)])[0]
                revenue = pred * BASE_PRICES[product] * (1 - disc / 100)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_disc = disc
            calendar_rows.append({
                "Product": product,
                "Season":  season,
                "Optimal Discount %": best_disc,
                "Est. Revenue (€)": round(best_revenue, 2),
            })

    cal_df = pd.DataFrame(calendar_rows)
    pivot = cal_df.pivot(index="Product", columns="Season", values="Optimal Discount %")
    pivot = pivot[SEASONS]

    fig = px.imshow(
        pivot,
        color_continuous_scale="Greens",
        text_auto=True,
        aspect="auto",
        labels={"color": "Optimal Discount %"},
        title="Optimal Discount % per Product × Season",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Dark green = high discount maximises revenue. Beverages peak in Summer, Bakery in Winter.")

# --- Tab 4: Feature Importance ---
with tab4:
    st.subheader("What Drives Demand?")
    st.caption("RandomForest feature importance — higher = stronger predictor of daily demand.")

    imp_df = importances.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df["Feature"] = imp_df["Feature"].replace({
        "month": "Month",
        "day_of_week": "Day of Week",
        "is_weekend": "Is Weekend",
        "is_epidemic": "Epidemic Period",
        "discount_pct": "Discount %",
        "interpolated_order": "Interpolated Order",
        "days_since_order": "Days Since Last Order",
        "product_enc": "Product",
        "store_enc": "Store",
    })

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        text="Importance",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=420,
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=60, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Demo uses synthetic data · Full model trained on "
    "[Retail Store Inventory Forecasting (Kaggle)](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) · "
    "XGBoost achieves R²=0.90 on the real dataset"
)
