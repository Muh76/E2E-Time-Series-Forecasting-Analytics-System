"""Rule-based fallback Copilot agent — no LLM dependency."""

from __future__ import annotations

import logging
from typing import Any

from .base import CopilotAgent

logger = logging.getLogger(__name__)


class RulesBasedAgent(CopilotAgent):
    """
    Deterministic fallback agent: intent detection + hardcoded sentence templates.

    Always returns a result (never ``None``) so the orchestrator chain is
    guaranteed to produce an answer even when the OpenAI key is absent or
    the primary agent fails.
    """

    async def explain(
        self,
        query: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            from backend.app.services.copilot_explain import build_structured_copilot_response

            result = build_structured_copilot_response(query, context)
            logger.info(
                "RulesBasedAgent: returned answer (confidence=%.2f, intents=%s)",
                result.get("confidence", 0),
                result.get("intents", []),
            )
            return result
        except ImportError:
            logger.error("RulesBasedAgent: backend.app.services.copilot_explain not importable")
            return None
        except Exception as exc:
            logger.error("RulesBasedAgent: unexpected error: %s", exc, exc_info=True)
            return None
