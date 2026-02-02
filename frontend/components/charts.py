"""
Chart components for the Streamlit frontend.

Provides reusable visualization components for time series and forecasts.
"""

import streamlit as st


def render_time_series_chart(data=None, x_col: str = "date", y_col: str = "value", title: str = ""):
    """
    Render a time series line chart.

    Args:
        data: DataFrame with time series data (optional).
        x_col: Column name for x-axis (dates).
        y_col: Column name for y-axis (values).
        title: Chart title.
    """
    if data is None or data.empty:
        st.info("No data available for chart.")
        return
    if title:
        st.subheader(title)
    st.line_chart(data.set_index(x_col)[y_col] if x_col in data.columns and y_col in data.columns else data)


def render_forecast_chart(actual=None, predicted=None, x_col: str = "date", title: str = "Forecast vs Actual"):
    """
    Render actual vs predicted time series overlay.

    Args:
        actual: DataFrame with actual values.
        predicted: DataFrame with predicted values.
        x_col: Column name for x-axis.
        title: Chart title.
    """
    if actual is None and predicted is None:
        st.info("No forecast data available.")
        return
    if title:
        st.subheader(title)
    # Placeholder: combine and plot when data is provided
    if actual is not None and not actual.empty:
        st.line_chart(actual.set_index(x_col) if x_col in actual.columns else actual)
    if predicted is not None and not predicted.empty:
        st.line_chart(predicted.set_index(x_col) if x_col in predicted.columns else predicted)
