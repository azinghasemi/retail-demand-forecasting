# Screenshot Guide

Take these 4 screenshots and save them here with the exact filenames below.
All images should be **1400px wide minimum**, saved as PNG.

---

## 01_forecast_vs_actual.png
**What:** Streamlit app → "Forecast vs Actual" tab
- Select: Product=Dairy or Produce, Store=Store A
- The line chart showing Actual (blue) vs Predicted (red dashed)
- Make sure the orange epidemic period band is visible
- Crop to just the chart area (no sidebar)

## 02_feature_importance.png
**What:** Streamlit app → "Feature Importance" tab
- Default filters
- Horizontal bar chart showing all 9 features ordered by importance
- Should show Interpolated Order and Product near the top

## 03_discount_calendar.png
**What:** Streamlit app → "Discount Calendar" tab
- Default filters
- The full heatmap grid (Products × Seasons)
- Make sure the colour scale is visible (dark green = high discount)

## 04_streamlit_demo.png
**What:** The Streamlit app header + tabs visible
- Run: `streamlit run app.py`
- Capture the full above-the-fold view: title + KPI tiles + tab bar
- Use browser width ~1400px

---

After adding all screenshots, commit and push. The README already references these filenames.
