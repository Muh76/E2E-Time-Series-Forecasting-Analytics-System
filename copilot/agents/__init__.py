"""
Copilot agent layer — orchestrates LLM and rule-based explanation agents.

Default chain: OpenAIExplainAgent (primary) → RulesBasedAgent (fallback).
The orchestrator always returns a well-shaped response dict.

Public API:
    CopilotAgent         Abstract base class for custom agents.
    OpenAIExplainAgent   GPT-backed explanation agent.
    RulesBasedAgent      Deterministic fallback (no LLM required).
    CopilotOrchestrator  Runs the agent chain; returns first success.
    create_orchestrator  Factory for CopilotOrchestrator.
"""

from .base import CopilotAgent
from .explain_agent import OpenAIExplainAgent
from .orchestrator import CopilotOrchestrator, create_orchestrator
from .rules_agent import RulesBasedAgent

__all__ = [
    "CopilotAgent",
    "CopilotOrchestrator",
    "OpenAIExplainAgent",
    "RulesBasedAgent",
    "create_orchestrator",
]
