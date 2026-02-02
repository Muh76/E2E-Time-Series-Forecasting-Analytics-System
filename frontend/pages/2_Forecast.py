"""
Forecast page — generate and explore time series forecasts.
"""

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import get_api_base_url, get_forecasts, get_metrics
from components.charts import render_forecast_vs_actual_plotly


def _compute_mae_rmse(actual: list[float], forecast: list[float]) -> tuple[float, float]:
    """Compute MAE and RMSE from aligned actual and forecast lists."""
    if not actual or not forecast or len(actual) != len(forecast):
        return float("nan"), float("nan")
    n = len(actual)
    mae = sum(abs(a - f) for a, f in zip(actual, forecast)) / n
    rmse = (sum((a - f) ** 2 for a, f in zip(actual, forecast)) / n) ** 0.5
    return mae, rmse


def main() -> None:
    st.title("Forecast")
    st.markdown("Generate and explore forecasts for your time series.")

    api_base = get_api_base_url()
    st.caption(f"API: `{api_base}`")

    # Fetch data
    metrics = get_metrics()
    data = metrics.get("data") or []
    meta = metrics.get("meta") or {}
    series_ids = meta.get("series_ids") or []
    if not series_ids and data:
        series_ids = sorted({r.get("series_id") for r in data if r.get("series_id")})
    if not series_ids:
        series_ids = ["series_001"]

    forecast_resp = get_forecasts(series_ids=series_ids, horizon_steps=14)

    # Entity selector
    st.markdown("---")
    st.subheader("Entity")
    selected_entity = st.selectbox(
        "Select entity (series)",
        options=series_ids,
        index=0,
        key="forecast_entity",
    )

    # Filter data for selected entity
    entity_data = [r for r in data if r.get("series_id") == selected_entity]
    entity_forecast = next(
        (f for f in (forecast_resp.get("forecasts") or []) if f.get("series_id") == selected_entity),
        None,
    )

    # Horizon
    horizon = 0
    if entity_forecast:
        steps = entity_forecast.get("steps") or []
        point_forecast = entity_forecast.get("point_forecast") or []
        horizon = len(steps) or len(point_forecast)
    if horizon == 0:
        horizon = 14

    st.markdown("### Horizon")
    st.metric(label="Forecast steps", value=horizon)

    # Build chart data: actual vs forecast
    dates = [r["date"] for r in entity_data if "date" in r]
    actual_vals = [r["actual"] for r in entity_data if "actual" in r]
    forecast_vals = [r["forecast"] for r in entity_data if "forecast" in r]

    # Append future forecast if available
    if entity_forecast:
        point_forecast = entity_forecast.get("point_forecast") or []
        if point_forecast and dates:
            last_date = datetime.fromisoformat(dates[-1]).date()
            for i in range(len(point_forecast)):
                d = (last_date + timedelta(days=i + 1)).isoformat()
                dates.append(d)
                forecast_vals.append(point_forecast[i])
                if len(actual_vals) < len(dates):
                    actual_vals.append(None)
        elif point_forecast:
            for i, v in enumerate(point_forecast):
                d = (datetime.now(timezone.utc).date() + timedelta(days=i + 1)).isoformat()
                dates.append(d)
                forecast_vals.append(v)

    # Extend actual to match length (use None for future)
    while len(actual_vals) < len(forecast_vals):
        actual_vals.append(None)
    actual_vals = actual_vals[: len(dates)]
    forecast_vals = forecast_vals[: len(dates)]

    # Filter out None for actual when computing metrics
    actual_for_metrics = [a for a in actual_vals if a is not None]
    forecast_for_metrics = forecast_vals[: len(actual_for_metrics)]

    # MAE / RMSE
    mae, rmse = _compute_mae_rmse(actual_for_metrics, forecast_for_metrics)

    st.markdown("---")
    st.subheader("Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="MAE", value=f"{mae:.4f}" if not math.isnan(mae) else "—")
    with col2:
        st.metric(label="RMSE", value=f"{rmse:.4f}" if not math.isnan(rmse) else "—")

    # Line chart: actual vs forecast
    st.markdown("---")
    st.subheader("Actual vs Forecast")
    # For Plotly, use None for missing actual (will show gap)
    actual_plot = [a if a is not None else None for a in actual_vals]
    render_forecast_vs_actual_plotly(
        dates=dates,
        actual=actual_plot,
        forecast=forecast_vals,
        title=f"{selected_entity} — Actual vs Forecast",
    )


main()
