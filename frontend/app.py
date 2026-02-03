"""
Time Series Forecasting & Analytics — Streamlit app entry point.

Sets page config, loads backend API base URL from config, and routes to pages.
Streamlit's native pages/ folder provides automatic sidebar navigation.
"""

import sys
from pathlib import Path

# Ensure project root is on path so "import frontend" works when running frontend/app.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import frontend  # noqa: F401 — runs frontend/__init__.py, which adds frontend dir for components.*

import os
from pathlib import Path

import streamlit as st
import yaml

from components.api import check_api_health, get_api_base_url
from components.ui import render_empty_state, render_warning


def load_page_config():
    """Load page title from config and set Streamlit page config."""
    config = _load_config()
    page_title = config.get("frontend", {}).get("page_title", "Time Series Forecasting & Analytics")
    st.set_page_config(
        page_title=page_title,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _load_config() -> dict:
    """Load base config and merge with env-specific overrides."""
    root = Path(__file__).resolve().parent.parent  # project root (parent of frontend)
    config_dir = root / "config"
    base_path = config_dir / "base" / "default.yaml"
    if not base_path.exists():
        return {}

    with open(base_path) as f:
        config = yaml.safe_load(f) or {}

    env = os.environ.get("APP_ENV", "local")
    env_path = config_dir / env / "config.yaml"
    if env_path.exists():
        with open(env_path) as f:
            env_config = yaml.safe_load(f) or {}
        for key, val in env_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(val, dict):
                config[key] = {**config[key], **val}
            else:
                config[key] = val
    return config


def render_sidebar():
    """Render sidebar with app info and API status."""
    st.sidebar.title("Time Series Forecasting")
    st.sidebar.markdown("---")

    api_base = get_api_base_url()
    st.sidebar.caption(f"API: `{api_base}`")

    if check_api_health():
        st.sidebar.success("API reachable")
    else:
        render_warning("API unreachable — mock data in use.", sidebar=True)

    st.sidebar.markdown("---")

    # Streamlit auto-adds page links when using pages/ folder;
    # this section provides additional context
    render_empty_state("Select a page from the list above to navigate.", sidebar=True)


def main():
    """Main entry: set config, sidebar, and navigation hint."""
    load_page_config()
    render_sidebar()

    st.markdown(
        "**System Overview**\n\n"
        "Use the sidebar to navigate:\n\n"
        "- **Overview** – system health and status\n"
        "- **Forecast** – actual vs predicted values\n"
        "- **Monitoring** – performance metrics and drift detection\n"
        "- **Copilot** – natural language explanations"
    )


if __name__ == "__main__":
    main()
