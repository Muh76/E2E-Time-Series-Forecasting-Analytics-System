"""
Chart components for the Streamlit frontend.

Provides reusable visualization components for time series and forecasts.
Uses Plotly for interactive charts.
"""

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_time_series_chart(data=None, x_col: str = "date", y_col: str = "value", title: str = ""):
    """
    Render a time series line chart with Plotly.

    Args:
        data: DataFrame with time series data (optional).
        x_col: Column name for x-axis (dates).
        y_col: Column name for y-axis (values).
        title: Chart title.
    """
    if data is None or data.empty:
        st.info("No data available.")
        return
    if title:
        st.subheader(title)
    df = data.copy()
    if x_col in df.columns and y_col in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines", name=y_col))
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df)


def render_forecast_vs_actual_plotly(
    dates: list[str],
    actual: list[float | None],
    forecast: list[float],
    title: str = "Forecast vs Actual",
) -> None:
    """
    Render actual vs forecast line chart with Plotly.

    Args:
        dates: List of date strings for x-axis.
        actual: List of actual values (None for future dates).
        forecast: List of forecast values.
        title: Chart title.
    """
    if not dates:
        st.info("No forecast data available.")
        return
    n = len(dates)
    actual = (list(actual) + [None] * n)[:n]
    forecast = (list(forecast) + [None] * n)[:n]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=actual, mode="lines", name="Actual", line=dict(color="#3b82f6"))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=forecast, mode="lines", name="Forecast", line=dict(color="#ef4444", dash="dash"))
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Target value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=60, l=60, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rolling_metric_chart(
    dates: list[str],
    values: list[float],
    threshold: float | None,
    metric_name: str,
    title: str = "",
    threshold_label: str = "Alert threshold",
) -> None:
    """
    Render rolling metric line chart with optional threshold annotation.

    Args:
        dates: Date strings for x-axis.
        values: Metric values.
        threshold: Optional threshold line value.
        metric_name: Label for y-axis (e.g. MAE, MAPE).
        title: Chart title.
        threshold_label: Legend label for threshold.
    """
    if not dates or not values:
        st.info(f"No {metric_name} data available.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=values, mode="lines+markers", name=metric_name, line=dict(color="#3b82f6"))
    )
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=threshold_label,
            annotation_position="right",
        )
    fig.update_layout(
        title=title or f"Rolling {metric_name}",
        xaxis_title="Date",
        yaxis_title=metric_name,
        hovermode="x unified",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drift_bar_chart(
    features: list[str],
    scores: list[float],
    threshold: float | None = None,
    title: str = "Drift score per feature",
) -> None:
    """
    Render drift score bar chart with optional threshold annotation.

    Args:
        features: Feature names.
        scores: Drift scores per feature.
        threshold: Optional threshold line.
        title: Chart title.
    """
    if not features or not scores:
        st.info("No drift data available.")
        return
    colors = ["#ef4444" if (threshold and s >= threshold) else "#3b82f6" for s in scores]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=features, y=scores, marker_color=colors, name="Drift score")
    )
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Threshold",
            annotation_position="right",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Drift score",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_chart(actual=None, predicted=None, x_col: str = "date", title: str = "Forecast vs Actual"):
    """
    Render actual vs predicted time series overlay (legacy; uses Plotly when possible).
    """
    if actual is None and predicted is None:
        st.info("No forecast data available.")
        return
    # Build unified dates and values for Plotly
    dates: list[str] = []
    actual_vals: list[float] = []
    pred_vals: list[float] = []
    if actual is not None and not actual.empty and x_col in actual.columns:
        dates = actual[x_col].astype(str).tolist()
        for c in ["actual", "value", "target"]:
            if c in actual.columns:
                actual_vals = actual[c].tolist()
                break
    if predicted is not None and not predicted.empty and x_col in predicted.columns:
        if not dates:
            dates = predicted[x_col].astype(str).tolist()
        for c in ["forecast", "y_pred", "predicted"]:
            if c in predicted.columns:
                pred_vals = predicted[c].tolist()
                break
    if dates or actual_vals or pred_vals:
        render_forecast_vs_actual_plotly(dates, actual_vals, pred_vals, title=title)
    else:
        st.info("No forecast data available.")
