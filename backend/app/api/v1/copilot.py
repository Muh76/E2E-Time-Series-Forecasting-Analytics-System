"""
Copilot API: rule-based explanations for forecasts and metrics (no LLM).

POST /api/v1/copilot — forecast series + metrics → summary, insights, confidence.
POST /api/v1/copilot/explain — query + optional context; returns markdown explanation.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.copilot_explain import build_structured_copilot_response
from backend.app.services.copilot_forecast_insights import build_forecast_insights

logger = logging.getLogger(__name__)

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
    t0 = time.perf_counter()
    out = build_forecast_insights(body.forecast, body.metrics)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "copilot_insights: forecast_items=%d metrics_keys=%s confidence=%.4f latency_ms=%s",
        len(body.forecast),
        sorted(str(k) for k in body.metrics.keys())[:24],
        float(out.get("confidence", 0.0)),
        latency_ms,
    )
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

    t0 = time.perf_counter()
    out = build_structured_copilot_response(query, context)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    sources = list(out["sources"])

    if not any(s.get("type") == "monitoring_summary" for s in sources):
        sources.insert(0, {"type": "monitoring_summary", "note": "rule_engine"})

    logger.info(
        "copilot_explain: query_len=%d context_keys=%s sources=%d latency_ms=%s",
        len(query),
        sorted(str(k) for k in (context or {}).keys())[:20],
        len(sources),
        latency_ms,
    )

    generated_at = out.get("generated_at") or datetime.now(timezone.utc).isoformat()
    if isinstance(generated_at, str) and generated_at.endswith("+00:00"):
        generated_at = generated_at.replace("+00:00", "Z")

    return {
        "answer": out["answer"],
        "reasoning": out["reasoning"],
        "confidence": out["confidence"],
        "intents": out["intents"],
        "explanation": out["explanation"],
        "sources": sources,
        "generated_at": generated_at,
    }
