"""
Reusable UI helpers for the Streamlit frontend.

Consistent styling: neutral, enterprise tone; no emojis in titles.
Uses st.markdown with subtle emphasis for error, warning, and empty states.
"""

import streamlit as st

# Default messages for loading states (no duplicated strings across pages)
LOADING_MESSAGE = "Loading data…"
LOADING_COPIOT_MESSAGE = "Retrieving explanation…"
CHART_LOADING_CAPTION = "Loading data…"

# Icons: consistent across all helpers
ICON_ERROR = "❌"
ICON_WARNING = "⚠️"
ICON_EMPTY = "ℹ️"


def _render_message(
    message: str,
    icon: str,
    border_color: str,
    *,
    sidebar: bool = False,
) -> None:
    """Render a message with subtle emphasis: light background, left border."""
    target = st.sidebar if sidebar else st
    html = (
        f'<div style="padding: 0.75rem 1rem; background-color: #f8fafc; '
        f'border-left: 4px solid {border_color}; border-radius: 4px; '
        f'color: #374151; font-size: 0.95rem;">'
        f'<strong>{icon}</strong> {message}'
        "</div>"
    )
    target.markdown(html, unsafe_allow_html=True)


def render_error(message: str, *, sidebar: bool = False) -> None:
    """Display an error message with consistent styling. Uses ❌ icon and subtle emphasis."""
    _render_message(message, ICON_ERROR, "#ef4444", sidebar=sidebar)


def render_warning(message: str, *, sidebar: bool = False) -> None:
    """Display a warning message with consistent styling. Uses ⚠️ icon and subtle emphasis."""
    _render_message(message, ICON_WARNING, "#f59e0b", sidebar=sidebar)


def render_empty(message: str, *, sidebar: bool = False) -> None:
    """Display an empty-state message when no data is available. Uses ℹ️ icon. Neutral, enterprise tone."""
    _render_message(message, ICON_EMPTY, "#6b7280", sidebar=sidebar)


def with_loading(fn, *args, message: str | None = None, **kwargs):
    """Run fn with st.spinner. Returns fn result. Use for simple API calls."""
    msg = message or LOADING_MESSAGE
    with st.spinner(msg):
        return fn(*args, **kwargs)


def chart_loading_placeholder():
    """
    Create a placeholder that shows loading text until data is ready.
    Returns an st.empty() placeholder. Call .empty() when ready to render the chart.
    """
    ph = st.empty()
    with ph.container():
        st.caption(CHART_LOADING_CAPTION)
    return ph
