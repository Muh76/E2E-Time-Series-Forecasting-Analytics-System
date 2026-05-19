"""Load and render Copilot prompt templates from the prompts directory."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_system_prompt() -> str:
    """Return the LLM system prompt, loaded from system_prompt.txt."""
    return (_PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()


def render_user_prompt(
    query: str,
    forecast_json: str,
    metrics_json: str,
    drift_json: str,
) -> str:
    """Render explain_user.txt with the given values."""
    template = (_PROMPTS_DIR / "explain_user.txt").read_text(encoding="utf-8")
    return template.format(
        query=query,
        forecast_json=forecast_json,
        metrics_json=metrics_json,
        drift_json=drift_json,
    )
