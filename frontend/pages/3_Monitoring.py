"""
Monitoring & Drift — performance metrics, drift detection, and alerts.
Read-only; all data and charts from API. Uses mock when API unavailable.
"""

from datetime import datetime, timezone

import streamlit as st

from components.api import get_monitoring_summary
from components.charts import render_drift_bar_chart, render_rolling_metric_chart
from components.ui import chart_loading_placeholder, render_empty, with_loading


def _summary_for_copilot(summary: dict) -> dict:
    """Build summary-level context for Copilot; exclude raw time-series data."""
    out = {k: v for k, v in summary.items() if k != "rolling_series"}
    return out


def main() -> None:
    st.title("Monitoring & Drift")

    chart_ph = chart_loading_placeholder()
    summary = with_loading(get_monitoring_summary)
    chart_ph.empty()

    # Thresholds from API only (mock includes them when backend unavailable)
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
            alerts_list.append({
                "alert_type": "MAE",
                "current_value": perf.get("mae"),
                "threshold": mae_alert,
            })
        if mape_alerted:
            alerts_list.append({
                "alert_type": "MAPE",
                "current_value": perf.get("mape"),
                "threshold": mape_alert,
            })
        if drift_alerted:
            alerts_list.append({
                "alert_type": "Drift",
                "current_value": drift_data.get("overall_score"),
                "threshold": drift_threshold,
            })

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

    # --- Section 1: Performance Monitoring ---
    st.markdown("---")
    st.subheader("1. Performance Monitoring")
    rolling = summary.get("rolling_series") or {}
    rolling_mae = rolling.get("mae") or []
    rolling_mape = rolling.get("mape") or []

    col1, col2 = st.columns(2)
    with col1:
        if rolling_mae:
            dates = [r["date"] for r in rolling_mae]
            values = [r["value"] for r in rolling_mae]
            render_rolling_metric_chart(
                dates=dates,
                values=values,
                threshold=mae_alert,
                metric_name="MAE",
                title="Rolling MAE",
                threshold_label=f"Alert ({mae_alert})",
            )
        else:
            render_empty("No rolling MAE data available.")
    with col2:
        if rolling_mape:
            dates = [r["date"] for r in rolling_mape]
            values = [r["value"] for r in rolling_mape]
            render_rolling_metric_chart(
                dates=dates,
                values=values,
                threshold=mape_alert,
                metric_name="MAPE (%)",
                title="Rolling MAPE",
                threshold_label=f"Alert ({mape_alert}%)",
            )
        else:
            render_empty("No rolling MAPE data available.")

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
