"""
Copilot API: LLM-based explanations for forecasts and metrics.

POST /api/v1/copilot/explain — accepts query and context, returns explanation.
Uses monitoring summary and metrics from context when provided.
Stub implementation; extend with LLM integration.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(prefix="/copilot", tags=["copilot"])


@router.post("/explain")
async def explain(body: dict | None = None) -> dict:
    """
    Generate explanation for a natural-language query.
    Context may include monitoring_summary and metrics for grounding.
    """
    body = body or {}
    query = body.get("query", "")
    context = body.get("context") or {}
    options = body.get("options") or {}

    perf = context.get("monitoring_summary", {}).get("performance", {}) or {}
    mae = perf.get("mae", 0)
    mape = perf.get("mape", 0)
    mape_pct = mape * 100 if mape and mape < 1 else (mape or 0)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    explanation = (
        f"You asked: \"{query}\"\n\n"
        f"**Stub response:** Current performance — MAE: {mae:.2f}, MAPE: {mape_pct:.1f}%. "
        "The copilot explains forecasts and metrics using precomputed data; it does not perform prediction. "
        "Configure LLM integration for real explanations."
    )
    return {
        "explanation": explanation,
        "sources": [{"type": "monitoring_summary"}, {"type": "metrics"}],
        "generated_at": now,
    }
