"""
Backtest API: rolling-origin evaluation for store-level forecasts.

POST /backtest/store runs rolling-origin backtesting for a given store,
returning per-split and average RMSE, MAE, MAPE.
Uses the primary model preloaded at startup (app.state.primary_model).
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.services.backtest_service import backtest_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    store_id: int = Field(..., description="Store identifier")
    horizon: int = Field(..., ge=1, description="Forecast horizon per split")
    n_splits: int = Field(..., ge=1, le=20, description="Number of rolling-origin splits")


class SplitMetrics(BaseModel):
    split: int
    cutoff_date: str
    horizon: int
    rmse: float
    mae: float
    mape: float


class AverageMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float


class BacktestResponse(BaseModel):
    store_id: int
    n_splits: int
    horizon: int
    splits: list[SplitMetrics]
    average: AverageMetrics


@router.post("/store", response_model=BacktestResponse)
async def post_backtest_store(request: Request, body: BacktestRequest) -> BacktestResponse:
    """
    Run rolling-origin backtesting for a single store.

    Creates n_splits cutoff points across the store's history, forecasts
    horizon steps ahead from each cutoff using the pre-trained model,
    and computes RMSE, MAE, MAPE per split plus averages.
    """
    model = request.app.state.primary_model

    try:
        result = backtest_store(
            store_id=body.store_id,
            horizon=body.horizon,
            n_splits=body.n_splits,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BacktestResponse(**result)
