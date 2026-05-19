"""OpenAI-backed Copilot agent — delegates to backend copilot_openai service."""

from __future__ import annotations

import logging
from typing import Any

from .base import CopilotAgent

logger = logging.getLogger(__name__)


class OpenAIExplainAgent(CopilotAgent):
    """
    Calls the OpenAI Chat Completions API via the backend copilot_openai service.

    Returns ``None`` when ``OPENAI_API_KEY`` is absent or the call fails,
    allowing the orchestrator to fall back to the rules-based agent.
    """

    async def explain(
        self,
        query: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            from backend.app.services.copilot_openai import explain_with_openai

            result = await explain_with_openai(query, context)
            if result is not None:
                logger.info("OpenAIExplainAgent: returned answer (confidence=%.2f)", result.get("confidence", 0))
            return result
        except ImportError:
            logger.warning("OpenAIExplainAgent: backend.app.services.copilot_openai not importable; skipping")
            return None
        except Exception as exc:
            logger.warning("OpenAIExplainAgent: unexpected error: %s", exc, exc_info=False)
            return None
