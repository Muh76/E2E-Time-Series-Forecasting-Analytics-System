"""
GET /api/v1/metrics — evaluation metrics for the last forecast vs ground truth.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from backend.app.services.metrics import evaluate_last_forecast_vs_actuals

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_forecast_metrics(
    store_id: Annotated[int | None, Query(description="Optional; must match last forecast store")] = None,
) -> dict:
    """
    After ``POST /api/v1/forecast/store``, compares returned forecast dates to
    processed data actuals. Returns MAE, RMSE, MAPE when overlapping ground
    truth exists; otherwise ``no_ground_truth`` with a fixed message.
    """
    return evaluate_last_forecast_vs_actuals(store_id=store_id)
