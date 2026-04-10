"""
Monitoring & Drift — performance metrics, drift detection, and alerts.
Read-only; data and charts come from the API.
"""

from datetime import datetime, timezone

import requests
import streamlit as st
from components.api import describe_request_error, get_forecast_evaluation_metrics, get_monitoring_summary
from components.charts import render_drift_bar_chart, render_rolling_metric_chart
from components.ui import chart_loading_placeholder, render_empty, render_error, with_loading


def _summary_for_copilot(summary: dict) -> dict:
    """Build summary-level context for Copilot; exclude raw time-series data."""
    out = {k: v for k, v in summary.items() if k != "rolling_series"}
    return out


def main() -> None:
    st.title("Monitoring & Drift")

    chart_ph = chart_loading_placeholder()
    try:
        summary = with_loading(get_monitoring_summary)
    except requests.RequestException as exc:
        chart_ph.empty()
        render_error(describe_request_error(exc))
        return
    chart_ph.empty()

    # Thresholds from API
    thresholds = summary.get("thresholds") or {}
    mae_alert = thresholds.get("mae_alert", 15.0)
    mape_alert = thresholds.get("mape_alert", 0.20)
    if mape_alert < 1 and mape_alert > 0:
        mape_alert = mape_alert * 100
    drift_threshold = thresholds.get("drift_threshold", 0.25)

    # Alerts from API only (no frontend computation)
    alerts = summary.get("alerts") or {}
    mae_alerted = alerts.get("mae", False)
    mape_alerted = alerts.get("mape", False)
    drift_alerted = alerts.get("drift", False)

    # --- Section 3: Alerts ---
    st.markdown("---")
    st.subheader("Alerts")
    col1, col2, col3 = st.columns(3)
    with col1:
        icon = "❌" if mae_alerted else "✅"
        st.markdown(f"**MAE** {icon}")
        st.caption(f"Threshold: {mae_alert}")
    with col2:
        icon = "❌" if mape_alerted else "✅"
        st.markdown(f"**MAPE** {icon}")
        st.caption(f"Threshold: {mape_alert}%")
    with col3:
        icon = "❌" if drift_alerted else "✅"
        st.markdown(f"**Drift** {icon}")
        st.caption(f"Threshold: {drift_threshold}")

    any_alert = mae_alerted or mape_alerted or drift_alerted
    if any_alert:
        perf = summary.get("performance") or {}
        drift_data = summary.get("drift") or {}
        alerts_list = []
        if mae_alerted:
            alerts_list.append(
                {
                    "alert_type": "MAE",
                    "current_value": perf.get("mae"),
                    "threshold": mae_alert,
                }
            )
        if mape_alerted:
            alerts_list.append(
                {
                    "alert_type": "MAPE",
                    "current_value": perf.get("mape"),
                    "threshold": mape_alert,
                }
            )
        if drift_alerted:
            alerts_list.append(
                {
                    "alert_type": "Drift",
                    "current_value": drift_data.get("overall_score"),
                    "threshold": drift_threshold,
                }
            )

        if st.button("Explain this alert", key="explain_alert_btn"):
            timestamp = summary.get("as_of") or datetime.now(timezone.utc).isoformat()
            if isinstance(timestamp, str) and "Z" in timestamp:
                timestamp = timestamp.replace("Z", "+00:00")
            st.session_state["alert_context"] = {
                "alerts": alerts_list,
                "timestamp": timestamp,
                "monitoring_summary": _summary_for_copilot(summary),
            }
            st.switch_page("pages/4_Copilot.py")

    # --- Section 1: Performance Monitoring (rolling from GET /api/v1/metrics) ---
    st.markdown("---")
    st.subheader("1. Performance monitoring")
    st.caption(
        "Rolling series from stored forecast-vs-actual errors (run forecasts with overlapping actuals to populate)."
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        roll_store = st.number_input(
            "Store ID (rolling)",
            min_value=1,
            value=1,
            step=1,
            key="mon_roll_store",
            help="Filter rolling history to this store.",
        )
    with rc2:
        roll_window = st.slider(
            "Rolling window",
            min_value=2,
            max_value=30,
            value=7,
            key="mon_roll_window",
        )

    rolling_mae_dates: list[str] = []
    rolling_mae_vals: list[float] = []
    rolling_mape_vals: list[float] = []
    try:
        mpack = get_forecast_evaluation_metrics(int(roll_store), window=int(roll_window))
        rser = mpack.get("rolling") or {}
        rolling_mae_dates = list(rser.get("timestamps") or [])
        rolling_mae_vals = [float(x) for x in (rser.get("mae") or [])]
        rolling_mape_vals = [float(x) for x in (rser.get("mape") or [])]
    except requests.RequestException as exc:
        st.warning(describe_request_error(exc))

    col1, col2 = st.columns(2)
    with col1:
        if rolling_mae_dates and rolling_mae_vals and len(rolling_mae_dates) == len(rolling_mae_vals):
            render_rolling_metric_chart(
                dates=rolling_mae_dates,
                values=rolling_mae_vals,
                threshold=mae_alert,
                metric_name="MAE",
                title=f"Rolling MAE (window={roll_window})",
                threshold_label=f"Alert ({mae_alert})",
            )
        else:
            legacy = (summary.get("rolling_series") or {}).get("mae") or []
            if legacy:
                render_rolling_metric_chart(
                    dates=[r["date"] for r in legacy],
                    values=[r["value"] for r in legacy],
                    threshold=mae_alert,
                    metric_name="MAE",
                    title="Rolling MAE (monitoring summary)",
                    threshold_label=f"Alert ({mae_alert})",
                )
            else:
                render_empty("No rolling MAE data yet — generate forecasts with evaluable dates.")
    with col2:
        if rolling_mae_dates and rolling_mape_vals and len(rolling_mae_dates) == len(rolling_mape_vals):
            render_rolling_metric_chart(
                dates=rolling_mae_dates,
                values=rolling_mape_vals,
                threshold=mape_alert,
                metric_name="MAPE (%)",
                title=f"Rolling MAPE (window={roll_window})",
                threshold_label=f"Alert ({mape_alert}%)",
            )
        else:
            legacy_m = (summary.get("rolling_series") or {}).get("mape") or []
            if legacy_m:
                render_rolling_metric_chart(
                    dates=[r["date"] for r in legacy_m],
                    values=[r["value"] for r in legacy_m],
                    threshold=mape_alert,
                    metric_name="MAPE (%)",
                    title="Rolling MAPE (monitoring summary)",
                    threshold_label=f"Alert ({mape_alert}%)",
                )
            else:
                render_empty("No rolling MAPE data yet — generate forecasts with evaluable dates.")

    # --- Section 2: Drift Monitoring ---
    st.markdown("---")
    st.subheader("2. Drift Monitoring")
    drift = summary.get("drift") or {}
    per_feature = drift.get("per_feature_scores") or {}
    indicators = drift.get("indicators") or []
    if not per_feature and indicators:
        per_feature = {ind.get("feature", f"f{i}"): ind.get("score", 0) for i, ind in enumerate(indicators)}

    if per_feature:
        features = list(per_feature.keys())
        scores = [per_feature[f] for f in features]
        render_drift_bar_chart(
            features=features,
            scores=scores,
            threshold=drift_threshold,
            title="Per-feature drift scores",
        )
    else:
        render_empty("No drift data available for features.")


main()
