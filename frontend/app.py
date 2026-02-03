"""
Time Series Forecasting & Analytics — Streamlit app entry point.

Sets page config, loads backend API base URL from config, and routes to pages.
Streamlit's native pages/ folder provides automatic sidebar navigation.
"""

import os
import sys
from pathlib import Path

# Frontend dir = parent of app.py; project root = parent of frontend
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st
import yaml

from components.api import check_api_health, get_api_base_url


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
    root = Path(__file__).resolve().parent.parent
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
    st.sidebar.title("📈 Forecasting Analytics")
    st.sidebar.markdown("---")

    api_base = get_api_base_url()
    st.sidebar.caption(f"API: `{api_base}`")

    if check_api_health():
        st.sidebar.success("API reachable")
    else:
        st.sidebar.warning("API unreachable — using mock data")

    st.sidebar.markdown("---")

    # Streamlit auto-adds page links when using pages/ folder;
    # this section provides additional context
    st.sidebar.info(
        "Use the pages above to explore Overview, Forecast, Monitoring, and Copilot."
    )


def main():
    """Main entry: set config, sidebar, and navigation hint."""
    load_page_config()
    render_sidebar()

    st.markdown(
        "Use the sidebar to navigate the application:\n\n"
        "- **Overview** – system health\n"
        "- **Forecast** – actual vs predicted\n"
        "- **Monitoring** – performance & drift\n"
        "- **Copilot** – natural language explanations"
    )


if __name__ == "__main__":
    main()
