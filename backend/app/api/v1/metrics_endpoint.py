"""
GET /api/v1/metrics — last forecast vs ground truth plus rolling error history.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from backend.app.services.metrics import evaluate_last_forecast_vs_actuals
from backend.app.services.rolling_performance import compute_rolling_series

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_forecast_metrics(
    store_id: Annotated[int | None, Query(description="Optional; must match last forecast store")] = None,
    window: Annotated[
        int,
        Query(ge=2, le=90, description="Rolling window size for MAE/MAPE series"),
    ] = 7,
) -> dict:
    """
    After ``POST /api/v1/forecast/store`` or ``POST /api/v1/predict``, ``current`` compares
    the last forecast to processed actuals. ``rolling`` returns windowed mean absolute error
    and mean absolute percentage error (as percent 0–100) over stored per-date error history.

    Rolling series is filtered by ``store_id`` when provided; otherwise all stores are included
    (sorted by date).
    """
    current = evaluate_last_forecast_vs_actuals(store_id=store_id)
    rolling = compute_rolling_series(window=window, store_id=store_id)
    return {"current": current, "rolling": rolling}
