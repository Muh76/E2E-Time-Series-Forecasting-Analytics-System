"""
Copilot page — AI-assisted explanations and insights.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st

from components.api import get_api_base_url

# Page config is set in app.py; Streamlit applies it app-wide

st.title("Copilot")
st.markdown("AI-assisted explanations for forecasts and model behavior.")

api_base = get_api_base_url()
st.caption(f"API: `{api_base}`")

st.markdown("---")
st.subheader("Chat with Copilot")
st.info("Ask questions about your forecasts, metrics, or model behavior.")
