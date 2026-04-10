"""
Copilot API: rule-based explanations for forecasts and metrics (no LLM).

POST /api/v1/copilot — forecast series + metrics → summary, insights, confidence.
POST /api/v1/copilot/explain — query + optional context; returns markdown explanation.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.copilot_forecast_insights import build_forecast_insights
from backend.app.services.copilot_rules import build_rule_based_explanation

router = APIRouter(prefix="/copilot", tags=["copilot"])


class CopilotInsightRequest(BaseModel):
    """Insight Copilot input: forecast points and optional evaluation metrics."""

    forecast: list[Any] = Field(
        ...,
        description="Forecast points: numbers or objects with forecast/value/y",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metrics (e.g. mae, rmse, mape) for narrative context",
    )


class CopilotInsightResponse(BaseModel):
    summary: str
    insights: str
    confidence: float = Field(ge=0.0, le=1.0)


@router.post("", response_model=CopilotInsightResponse)
async def copilot_forecast_insights(body: CopilotInsightRequest) -> CopilotInsightResponse:
    """
    Rule-based Insight Copilot: trend, volatility, step anomalies; deterministic thresholds.
    """
    out = build_forecast_insights(body.forecast, body.metrics)
    return CopilotInsightResponse(**out)


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
