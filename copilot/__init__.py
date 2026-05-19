"""
Copilot package — LLM-backed and rule-based forecast explanation system.

Sub-packages:
    copilot.agents    Agent orchestration (OpenAI → rules-based fallback chain).
    copilot.prompts   Prompt template loader (system + user prompt text files).

Typical usage
-------------
# Run the full agent chain (tries OpenAI, falls back to rules):
from copilot.agents import create_orchestrator

orchestrator = create_orchestrator()
response = await orchestrator.run(query="What is the trend?", context=ctx)

# Load / render prompt templates directly:
from copilot.prompts import load_system_prompt, render_user_prompt
"""

from . import agents, prompts
from .agents import CopilotAgent, CopilotOrchestrator, create_orchestrator
from .prompts import load_system_prompt, render_user_prompt

__all__ = [
    # sub-packages
    "agents",
    "prompts",
    # agent API
    "CopilotAgent",
    "CopilotOrchestrator",
    "create_orchestrator",
    # prompt API
    "load_system_prompt",
    "render_user_prompt",
]
