"""
Backtest API: rolling-origin evaluation for store-level forecasts.

POST /backtest/store runs rolling-origin backtesting for a given store,
returning per-split and average RMSE, MAE, MAPE.
Uses the primary model preloaded at startup (app.state.primary_model).
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.api.v1.validators import (
    HORIZON_MAX,
    HORIZON_MIN,
    N_SPLITS_MAX,
    N_SPLITS_MIN,
    get_valid_store_ids,
)
from backend.app.services.backtest_service import backtest_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    store_id: int = Field(..., description="Store identifier (must exist in dataset)")
    horizon: int = Field(
        ...,
        ge=HORIZON_MIN,
        le=HORIZON_MAX,
        description=f"Forecast horizon per split ({HORIZON_MIN}–{HORIZON_MAX})",
    )
    n_splits: int = Field(
        ...,
        ge=N_SPLITS_MIN,
        le=N_SPLITS_MAX,
        description=f"Number of rolling-origin splits ({N_SPLITS_MIN}–{N_SPLITS_MAX})",
    )

    @field_validator("store_id")
    @classmethod
    def store_must_exist(cls, v: int) -> int:
        valid = get_valid_store_ids()
        if valid and v not in valid:
            raise ValueError(
                f"store_id={v} does not exist in the dataset. "
                f"Valid range: {min(valid)}–{max(valid)} ({len(valid)} stores)."
            )
        return v


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
