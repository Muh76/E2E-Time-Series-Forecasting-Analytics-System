"""
Copilot prompt templates — load and render LLM prompt strings from flat text files.

Prompt engineering lives in the .txt files alongside this package so that
wording can be adjusted without touching Python code.

Public API:
    load_system_prompt   Return the LLM system-role prompt string.
    render_user_prompt   Render the user-role prompt with forecast, metrics, and drift JSON.
"""

from .loader import load_system_prompt, render_user_prompt

__all__ = [
    "load_system_prompt",
    "render_user_prompt",
]
