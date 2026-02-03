"""
Reusable UI helpers for the Streamlit frontend.

Consistent styling: neutral, professional tone; no emojis in titles.
"""

import streamlit as st

# Default messages for loading states (no duplicated strings across pages)
LOADING_MESSAGE = "Loading…"
LOADING_COPIOT_MESSAGE = "Retrieving explanation…"
CHART_LOADING_CAPTION = "Loading chart…"


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
    """Display an error message with consistent styling."""
    if sidebar:
        st.sidebar.error(message)
    else:
        st.error(message)


def render_warning(message: str, *, sidebar: bool = False) -> None:
    """Display a warning message with consistent styling."""
    if sidebar:
        st.sidebar.warning(message)
    else:
        st.warning(message)


def render_empty_state(message: str, *, sidebar: bool = False) -> None:
    """Display an empty-state message when no data is available. Neutral, professional tone."""
    if sidebar:
        st.sidebar.info(message)
    else:
        st.info(message)
