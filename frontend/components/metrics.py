"""
Metrics display components for the Streamlit frontend.

Provides reusable UI for showing MAE, RMSE, MAPE and other evaluation metrics.
"""

import streamlit as st


def render_metrics_cards(metrics: dict, columns=None):
    """
    Render metrics as cards in columns.

    Args:
        metrics: Dict of metric name -> value (e.g. {"mae": 5.2, "rmse": 6.1}).
        columns: Optional list of column objects from st.columns(); creates default if None.
    """
    if not metrics:
        st.info("No metrics available.")
        return
    cols = columns or st.columns(len(metrics))
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            st.metric(label=name.upper(), value=f"{value:.4f}" if isinstance(value, (int, float)) else str(value))


def render_metric_single(label: str, value, delta=None):
    """Render a single metric with optional delta."""
    st.metric(label=label, value=value, delta=delta)
