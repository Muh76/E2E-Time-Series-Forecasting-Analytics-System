"""
Copilot API: rule-based explanations for forecasts and metrics (no LLM).

POST /api/v1/copilot/explain — query + optional context; returns markdown explanation.
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.copilot_rules import build_rule_based_explanation

router = APIRouter(prefix="/copilot", tags=["copilot"])


class CopilotExplainRequest(BaseModel):
    query: str = Field(default="What is the current model health?", min_length=1, max_length=4000)
    context: dict | None = None
    options: dict | None = None


@router.post("/explain")
async def explain(body: CopilotExplainRequest | None = None) -> dict:
    """
    Generate explanation for a natural-language query using monitoring context.
    """
    body = body or CopilotExplainRequest()
    query = body.query or ""
    context = body.context or {}

    explanation, sources = build_rule_based_explanation(query, context)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not any(s.get("type") == "monitoring_summary" for s in sources):
        sources.insert(0, {"type": "monitoring_summary", "note": "rule_engine"})

    return {
        "explanation": explanation,
        "sources": sources,
        "generated_at": now,
    }
