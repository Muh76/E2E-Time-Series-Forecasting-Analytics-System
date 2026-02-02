"""
Forecast page — generate and explore time series forecasts.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import get_api_base_url
from components.charts import render_forecast_chart
from components.metrics import render_metrics_cards

# Page config is set in app.py; Streamlit applies it app-wide

st.title("Forecast")
st.markdown("Generate and explore forecasts for your time series.")

api_base = get_api_base_url()
st.caption(f"API: `{api_base}`")

st.markdown("---")
st.subheader("Generate forecast")
st.info("Configure parameters and request a forecast from the backend.")

# Placeholder for forecast form / results
render_forecast_chart(actual=None, predicted=None, title="Forecast vs Actual")
render_metrics_cards({})
