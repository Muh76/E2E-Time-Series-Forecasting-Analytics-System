"""
Reusable UI helpers for the Streamlit frontend.

Consistent styling: neutral, professional tone; no emojis in titles.
"""

import streamlit as st

# Default messages for loading states (no duplicated strings across pages)
LOADING_MESSAGE = "Loading…"
LOADING_COPIOT_MESSAGE = "Retrieving explanation…"
CHART_LOADING_CAPTION = "Loading chart…"


def with_loading(fn, *args, message: str | None = None, **kwargs):
    """Run fn with st.spinner. Returns fn result. Use for simple API calls."""
    msg = message or LOADING_MESSAGE
    with st.spinner(msg):
        return fn(*args, **kwargs)


def chart_loading_placeholder():
    """
    Create a placeholder that shows "Loading chart…" until data is ready.
    Returns an st.empty() placeholder. Call .empty() when ready to render the chart.
    """
    ph = st.empty()
    with ph.container():
        st.caption(CHART_LOADING_CAPTION)
    return ph


def render_error(message: str, *, sidebar: bool = False) -> None:
    """Display an error message with consistent styling. Uses ❌ icon and markdown emphasis."""
    target = st.sidebar if sidebar else st
    target.markdown(f"**❌** {message}")


def render_warning(message: str, *, sidebar: bool = False) -> None:
    """Display a warning message with consistent styling. Uses ⚠️ icon and markdown emphasis."""
    target = st.sidebar if sidebar else st
    target.markdown(f"**⚠️** {message}")


def render_empty_state(message: str, *, sidebar: bool = False) -> None:
    """Display an empty-state message when no data is available. Uses ℹ️ icon. Neutral, professional tone."""
    target = st.sidebar if sidebar else st
    target.markdown(f"**ℹ️** {message}")
