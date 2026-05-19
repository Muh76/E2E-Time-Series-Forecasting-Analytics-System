"""Abstract base class for all Copilot agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CopilotAgent(ABC):
    """
    Common interface for Copilot agents.

    Each agent receives the user query and the enriched monitoring context,
    and returns a structured response dict (same shape as
    ``build_structured_copilot_response``) or ``None`` when it cannot
    handle the request (e.g., missing API key, empty context).
    """

    @abstractmethod
    async def explain(
        self,
        query: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Produce a structured explanation for ``query`` given ``context``.

        Returns ``None`` to signal that the agent is unavailable or skipped,
        allowing the orchestrator to try the next agent in the chain.
        """
