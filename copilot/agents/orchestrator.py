"""
Copilot agent orchestrator — runs a priority chain and returns the first success.

Default chain: OpenAIExplainAgent → RulesBasedAgent (guaranteed answer).
"""

from __future__ import annotations

import logging
from typing import Any

from .base import CopilotAgent
from .explain_agent import OpenAIExplainAgent
from .rules_agent import RulesBasedAgent

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE: dict[str, Any] = {
    "answer": "No agent in the chain produced a response. Check logs for errors.",
    "reasoning": "### Reasoning (signals used)\n\n- All agents returned None.",
    "confidence": 0.0,
    "intents": ["general"],
    "explanation": "## Answer\n\nNo agent in the chain produced a response.",
    "sources": [{"type": "orchestrator", "title": "Chain exhausted"}],
    "generated_at": None,
}


class CopilotOrchestrator:
    """
    Runs each agent in ``chain`` in order and returns the first non-None result.

    If all agents return ``None``, returns a safe fallback dict so callers
    always receive a well-shaped response.
    """

    def __init__(self, chain: list[CopilotAgent] | None = None) -> None:
        self._chain: list[CopilotAgent] = chain if chain is not None else [
            OpenAIExplainAgent(),
            RulesBasedAgent(),
        ]

    async def run(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent chain and return the first successful result."""
        for agent in self._chain:
            name = type(agent).__name__
            try:
                result = await agent.explain(query, context)
            except Exception as exc:
                logger.warning("CopilotOrchestrator: %s raised %s; continuing chain", name, exc, exc_info=False)
                result = None

            if result is not None:
                logger.info("CopilotOrchestrator: answered by %s", name)
                result.setdefault("agent", name)
                return result

            logger.debug("CopilotOrchestrator: %s returned None; trying next agent", name)

        logger.error("CopilotOrchestrator: all agents returned None; returning fallback")
        import datetime

        fb = dict(_FALLBACK_RESPONSE)
        fb["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return fb


def create_orchestrator(chain: list[CopilotAgent] | None = None) -> CopilotOrchestrator:
    """Factory: return a CopilotOrchestrator with the default or custom agent chain."""
    return CopilotOrchestrator(chain=chain)
