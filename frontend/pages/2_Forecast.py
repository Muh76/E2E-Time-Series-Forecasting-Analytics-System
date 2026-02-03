"""
Forecast vs Actual — compare actual historical values with forecasts.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import get_forecast_vs_actual
from components.charts import render_forecast_vs_actual_plotly


def main() -> None:
    st.title("Forecast vs Actual")

    # Horizon selector (7–30 days)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, key="forecast_horizon")

    # Fetch data via API (returns mock if unavailable)
    data = get_forecast_vs_actual(horizon=horizon)
    entity_ids = data.get("entity_ids") or []
    entity_id = data.get("entity_id")

    # Entity selector (only if entity_ids exist)
    selected_entity = entity_id
    if entity_ids:
        selected_entity = st.selectbox(
            "Entity",
            options=entity_ids,
            index=entity_ids.index(entity_id) if entity_id in entity_ids else 0,
            key="forecast_entity",
        )
        # Re-fetch for selected entity if different
        if selected_entity != entity_id:
            data = get_forecast_vs_actual(entity_id=selected_entity, horizon=horizon)

    # Build ready-to-plot data (API returns {date, value} lists)
    actual_list = data.get("actual") or []
    forecast_list = data.get("forecast") or []

    # Merge into unified timeline: actual = solid, forecast = dashed
    date_to_actual = {r["date"]: r["value"] for r in actual_list}
    date_to_forecast = {r["date"]: r["value"] for r in forecast_list}
    all_dates = sorted(set(date_to_actual) | set(date_to_forecast))

    actual_vals = [date_to_actual.get(d) for d in all_dates]
    forecast_vals = [date_to_forecast.get(d) for d in all_dates]

    # Chart
    render_forecast_vs_actual_plotly(
        dates=all_dates,
        actual=actual_vals,
        forecast=forecast_vals,
        title="",
    )

    # Metrics below chart (from API; no frontend computation)
    metrics = data.get("metrics") or {}
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    mape = metrics.get("mape")

    st.markdown("---")
    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="MAE", value=f"{mae:.4f}" if mae is not None else "—")
    with col2:
        st.metric(label="RMSE", value=f"{rmse:.4f}" if rmse is not None else "—")
    with col3:
        mape_display = f"{mape * 100:.2f}%" if mape is not None and mape < 1 else f"{mape:.4f}" if mape is not None else "—"
        st.metric(label="MAPE", value=mape_display)


main()
