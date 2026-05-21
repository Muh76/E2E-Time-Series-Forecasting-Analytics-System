"""
Backtest API: rolling-origin evaluation for store-level forecasts.

POST /backtest/store runs rolling-origin backtesting for a given store,
returning per-split and average RMSE, MAE, MAPE.
Uses the primary model preloaded at startup (app.state.primary_model).
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.api.v1.validators import HORIZON_MAX, HORIZON_MIN, N_SPLITS_MAX, N_SPLITS_MIN, validate_store_id
from backend.app.services.backtest_service import backtest_store
from backend.app.services.model_loader import get_model_metadata, load_primary_model
from backend.app.services.monitoring_service import update_monitoring_from_backtest

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
        return validate_store_id(v)


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
    model = getattr(request.app.state, "primary_model", None)
    try:
        if model is None:
            logger.warning("backtest_store: primary model missing from app.state; attempting lazy load")
            model = load_primary_model()
            request.app.state.primary_model = model
    except RuntimeError as exc:
        logger.error("backtest_store: model artifact unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable. Run training artifacts setup and retry.",
        ) from exc

    try:
        result = backtest_store(
            store_id=body.store_id,
            horizon=body.horizon,
            n_splits=body.n_splits,
            model=model,
        )
    except ValueError as exc:
        logger.warning(
            "backtest_store: validation/inference error store_id=%s horizon=%s n_splits=%s error=%s",
            body.store_id,
            body.horizon,
            body.n_splits,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(
            "backtest_store: runtime dependency error store_id=%s horizon=%s n_splits=%s error=%s",
            body.store_id,
            body.horizon,
            body.n_splits,
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        logger.error(
            "backtest_store: missing dependency store_id=%s horizon=%s n_splits=%s error=%s",
            body.store_id,
            body.horizon,
            body.n_splits,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Required data dependency is missing (parquet engine). " "Install pyarrow or fastparquet and retry."
            ),
        ) from exc
    except Exception as exc:
        logger.exception(
            "backtest_store: unexpected failure store_id=%s horizon=%s n_splits=%s",
            body.store_id,
            body.horizon,
            body.n_splits,
        )
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while running backtest. Check server logs for details.",
        ) from exc

    try:
        metadata = get_model_metadata()
    except (RuntimeError, TypeError, ValueError):
        metadata = None
    update_monitoring_from_backtest(result, model_metadata=metadata)

    return BacktestResponse(**result)
