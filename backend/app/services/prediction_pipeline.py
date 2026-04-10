"""
Shared store-level forecast execution: inference, intervals, monitoring hooks.

Used by POST /forecast/store and POST /predict so behavior stays consistent.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.app.services.forecasting_service import forecast_store
from backend.app.services.metrics import record_forecast_for_evaluation
from backend.app.services.model_loader import get_model_metadata
from backend.app.services.monitoring_service import record_forecast_activity

logger = logging.getLogger(__name__)


def execute_store_forecast(
    store_id: int,
    horizon: int,
    model: Any,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Run forecast_store, attach residual-based intervals, record for metrics/copilot.

    Returns the same forecast row shape as the public forecast API.
    """
    forecasts = forecast_store(store_id, horizon, model, feature_columns)

    try:
        metadata = get_model_metadata()
        residual_std = float(metadata.get("residual_std", 0))
    except (RuntimeError, TypeError, ValueError):
        residual_std = 0.0

    if residual_std > 0:
        z = 1.96
        for f in forecasts:
            point = f["forecast"]
            f["confidence_low"] = round(point - z * residual_std, 2)
            f["confidence_high"] = round(point + z * residual_std, 2)
    else:
        for f in forecasts:
            f["confidence_low"] = None
            f["confidence_high"] = None

    record_forecast_for_evaluation(store_id, horizon, forecasts)
    record_forecast_activity(store_id, horizon)

    return forecasts


def metrics_context_for_copilot(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Subset of evaluation payload useful for rule-based copilot narrative."""
    if eval_result.get("status") == "ok":
        out: dict[str, Any] = {}
        for k in ("mae", "rmse", "mape", "n_samples", "evaluated_dates"):
            if k in eval_result:
                out[k] = eval_result[k]
        return out
    return {
        "status": eval_result.get("status"),
        "message": eval_result.get("message"),
    }
