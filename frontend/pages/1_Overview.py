"""
Overview page — system health and status.
"""

from datetime import datetime

import streamlit as st

from components.api import get_monitoring_summary
from components.metrics import format_float, format_mape


def _format_timestamp(iso_str: str) -> str:
    """Format ISO 8601 timestamp for display."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(iso_str)


def _status_color(status: str) -> str:
    """Return hex color for status: healthy=green, warning=yellow, critical=red."""
    s = (status or "").lower()
    if s in ("ok", "healthy"):
        return "#22c55e"
    if s == "warning":
        return "#f59e0b"
    return "#ef4444"


def _status_label(status: str) -> str:
    """Return display label for status."""
    s = (status or "").lower()
    if s in ("ok", "healthy"):
        return "Healthy"
    if s == "warning":
        return "Warning"
    return "Critical"


def main() -> None:
    st.title("System Overview")
    st.markdown("Current system health and model status.")

    summary = get_monitoring_summary()

    # Overall status
    pipeline = summary.get("pipeline") or {}
    status = pipeline.get("status", "ok")
    overall = summary.get("overall_status") or status
    color = _status_color(overall)
    label = _status_label(overall)

    st.markdown("---")
    st.subheader("Status")
    st.markdown(
        f'<span style="color: {color}; font-weight: bold; font-size: 1.1em;">● {label}</span>',
        unsafe_allow_html=True,
    )

    # Model name and version
    model_name = summary.get("model_name") or "—"
    model_version = summary.get("model_version") or "—"
    st.markdown("---")
    st.subheader("Model")
    st.markdown(f"**{model_name}** · `{model_version}`")

    # KPIs: MAE, MAPE, Drift
    perf = summary.get("performance") or {}
    drift = summary.get("drift") or {}
    mae = perf.get("mae")
    mape = perf.get("mape")
    drift_score = drift.get("overall_score")

    st.markdown("---")
    st.subheader("Key metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="MAE", value=format_float(mae))
    with col2:
        st.metric(label="MAPE", value=format_mape(mape))
    with col3:
        st.metric(label="Drift score", value=format_float(drift_score))

    # As of timestamp
    as_of = summary.get("as_of")
    if as_of:
        st.caption(f"As of {_format_timestamp(as_of)}")


main()
