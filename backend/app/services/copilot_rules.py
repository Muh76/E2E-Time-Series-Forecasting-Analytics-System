"""
Rule-based Insight Copilot — no LLM; explains monitoring context in plain language.

Implementation lives in ``copilot_explain``; this module re-exports for stable imports.
"""

from __future__ import annotations

from backend.app.services.copilot_explain import build_rule_based_explanation, build_structured_copilot_response

__all__ = ["build_rule_based_explanation", "build_structured_copilot_response"]
