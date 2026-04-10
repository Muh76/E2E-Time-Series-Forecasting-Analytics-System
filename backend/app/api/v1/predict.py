"""
Unified prediction pipeline: forecast → aligned metrics → optional copilot.

POST /predict — same body as forecast/store; query ``include_insights`` adds
rule-based copilot output in one response.
"""

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.app.api.v1.copilot import CopilotInsightResponse
from backend.app.api.v1.forecast import ForecastRequest
from backend.app.services.copilot_forecast_insights import build_forecast_insights
from backend.app.services.metrics import evaluate_last_forecast_vs_actuals
from backend.app.services.model_loader import get_model_metadata
from backend.app.services.prediction_pipeline import execute_store_forecast, metrics_context_for_copilot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictResponse(BaseModel):
    """Unified pipeline output."""

    forecast: list[dict[str, Any]]
    metrics: dict[str, Any]
    copilot: CopilotInsightResponse | None = None


@router.post("", response_model=PredictResponse)
async def post_predict(
    request: Request,
    body: ForecastRequest,
    include_insights: Annotated[
        bool,
        Query(
            description="When true, run rule-based Insight Copilot on the forecast and metrics.",
        ),
    ] = False,
) -> PredictResponse:
    """
    Generate forecast, evaluate vs ground truth when possible, and optionally copilot insights.

    Flow: (1) forecast with intervals and monitoring hooks, (2) same metrics as
    GET /api/v1/metrics for this store, (3) if ``include_insights=true``, deterministic
    copilot summary on the series + metrics context.
    """
    model = request.app.state.primary_model
    feature_columns = request.app.state.feature_columns

    t_start = time.perf_counter()
    try:
        forecasts = execute_store_forecast(body.store_id, body.horizon, model, feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    try:
        metadata = get_model_metadata()
        model_version = metadata.get("model_version", "unknown")
    except (RuntimeError, TypeError, ValueError):
        model_version = "unknown"

    metrics = evaluate_last_forecast_vs_actuals(store_id=body.store_id)

    copilot_out: CopilotInsightResponse | None = None
    if include_insights:
        ctx = metrics_context_for_copilot(metrics)
        built = build_forecast_insights(forecasts, ctx)
        copilot_out = CopilotInsightResponse(**built)

    logger.info(
        "predict_pipeline: store_id=%d horizon=%d forecast_ms=%.1f metrics_status=%s "
        "include_insights=%s model_version=%s",
        body.store_id,
        body.horizon,
        latency_ms,
        metrics.get("status"),
        include_insights,
        model_version,
    )

    return PredictResponse(
        forecast=forecasts,
        metrics=metrics,
        copilot=copilot_out,
    )
