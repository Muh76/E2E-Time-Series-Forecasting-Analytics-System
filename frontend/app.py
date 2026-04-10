"""
Time Series Forecasting & Analytics — Streamlit app entry point.

Sets page config and sidebar. Backend base URL: environment variable API_URL
(default http://127.0.0.1:8001 when unset), then config ``frontend.api_base_url``.
"""

import sys
from pathlib import Path

# Ensure project root is on path so "import frontend" works when running frontend/app.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os  # noqa: E402

import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from components.api import check_api_health, get_api_base_url  # noqa: E402
from components.ui import render_empty, render_warning  # noqa: E402

import frontend  # noqa: E402, F401 — runs frontend/__init__.py, which adds frontend dir for components.*


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
    st.sidebar.caption(f"API base URL: `{api_base}`")
    st.sidebar.caption("Override with environment variable **API_URL** (default `http://127.0.0.1:8001`).")

    if check_api_health():
        st.sidebar.success("API reachable")
    else:
        render_warning(
            "API unreachable. Start the FastAPI server and check API_URL.",
            sidebar=True,
        )

    st.sidebar.markdown("---")

    # Streamlit auto-adds page links when using pages/ folder;
    # this section provides additional context
    render_empty("Select a page from the list above to navigate.", sidebar=True)


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
