"""
Monitoring page — performance metrics and drift detection.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import get_api_base_url
from components.metrics import render_metrics_cards

# Page config is set in app.py; Streamlit applies it app-wide

st.title("Monitoring")
st.markdown("View performance metrics, drift alerts, and model health.")

api_base = get_api_base_url()
st.caption(f"API: `{api_base}`")

st.markdown("---")
st.subheader("Monitoring summary")
st.info("Fetch monitoring summary from `GET /api/v1/monitoring/summary`.")

# Placeholder for monitoring summary
render_metrics_cards({})
