"""
Metrics display components for the Streamlit frontend.

Provides reusable UI for showing MAE, RMSE, MAPE and other evaluation metrics.
"""

import streamlit as st

from components.ui import render_empty_state


def format_float(value: float | None, decimals: int = 4) -> str:
    """Format a float for display; return '—' if value is None."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def format_mape(value: float | None) -> str:
    """Format MAPE for display: decimal (e.g. 0.042) -> '4.20%'; already percentage -> 'X.XX%'; None -> '—'."""
    if value is None:
        return "—"
    if 0 <= value < 1:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}%"


def render_metrics_cards(metrics: dict, columns=None):
    """
    Render metrics as cards in columns.

    Args:
        metrics: Dict of metric name -> value (e.g. {"mae": 5.2, "rmse": 6.1}).
        columns: Optional list of column objects from st.columns(); creates default if None.
    """
    if not metrics:
        render_empty_state("No metrics available.")
        return
    cols = columns or st.columns(len(metrics))
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            if isinstance(value, (int, float)):
                display = format_float(float(value), decimals=4)
            else:
                display = str(value) if value is not None else "—"
            st.metric(label=name.upper(), value=display)


def render_metric_single(label: str, value, delta=None):
    """Render a single metric with optional delta."""
    st.metric(label=label, value=value, delta=delta)
