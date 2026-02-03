"""
Forecast vs Actual — compare actual historical values with forecasts.
"""

import streamlit as st

from components.api import get_forecast_vs_actual
from components.charts import render_forecast_vs_actual_plotly
from components.metrics import format_float, format_mape
from components.ui import chart_loading_placeholder, with_loading


def main() -> None:
    st.title("Forecast vs Actual")

    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, key="forecast_horizon")
    compare_baseline = st.checkbox("Compare baseline model", value=False, key="compare_baseline")

    chart_ph = chart_loading_placeholder()

    def _fetch_and_build():
        data = get_forecast_vs_actual(horizon=horizon, include_baseline=compare_baseline)
        entity_ids = data.get("entity_ids") or []
        entity_id = data.get("entity_id")
        selected_entity = entity_id
        if entity_ids:
            selected_entity = st.selectbox(
                "Entity",
                options=entity_ids,
                index=entity_ids.index(entity_id) if entity_id in entity_ids else 0,
                key="forecast_entity",
            )
            if selected_entity != entity_id:
                data = get_forecast_vs_actual(entity_id=selected_entity, horizon=horizon, include_baseline=compare_baseline)
        actual_list = data.get("actual") or []
        forecast_list = data.get("forecast") or []
        baseline_list = (data.get("baseline") or []) if compare_baseline else []
        date_to_actual = {r["date"]: r["value"] for r in actual_list}
        date_to_forecast = {r["date"]: r["value"] for r in forecast_list}
        date_to_baseline = {r["date"]: r["value"] for r in baseline_list} if baseline_list else {}
        all_dates = sorted(set(date_to_actual) | set(date_to_forecast) | set(date_to_baseline))
        actual_vals = [date_to_actual.get(d) for d in all_dates]
        forecast_vals = [date_to_forecast.get(d) for d in all_dates]
        baseline_vals = [date_to_baseline.get(d) for d in all_dates] if date_to_baseline else None
        return data, all_dates, actual_vals, forecast_vals, baseline_vals

    data, all_dates, actual_vals, forecast_vals, baseline_vals = with_loading(_fetch_and_build)
    chart_ph.empty()
    render_forecast_vs_actual_plotly(
        dates=all_dates,
        actual=actual_vals,
        forecast=forecast_vals,
        title="",
        baseline=baseline_vals,
    )

    metrics = data.get("metrics") or {}
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    mape = metrics.get("mape")

    st.markdown("---")
    st.subheader("Metrics")
    if compare_baseline and data.get("baseline_metrics"):
        baseline_metrics = data.get("baseline_metrics") or {}
        model_name = data.get("model_name") or "LightGBM"
        baseline_name = data.get("baseline_model_name") or "Baseline"
        st.dataframe(
            {
                "Model": [baseline_name, model_name],
                "MAE": [format_float(baseline_metrics.get("mae")), format_float(mae)],
                "RMSE": [format_float(baseline_metrics.get("rmse")), format_float(rmse)],
                "MAPE": [format_mape(baseline_metrics.get("mape")), format_mape(mape)],
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="MAE", value=format_float(mae))
        with col2:
            st.metric(label="RMSE", value=format_float(rmse))
        with col3:
            st.metric(label="MAPE", value=format_mape(mape))


main()
