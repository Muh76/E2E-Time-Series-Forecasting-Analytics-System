"""
Forecast API: store-level sales forecasting.

POST /forecast/store returns horizon-step forecasts for a given store_id.
POST /forecast/store/debug returns debug metadata without running inference.
Uses the primary model preloaded at startup (app.state.primary_model).
Inference only; no retraining.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.api.v1.validators import HORIZON_MAX, HORIZON_MIN, get_valid_store_ids
from backend.app.services.forecasting_service import forecast_store, get_store_last_date
from backend.app.services.model_loader import get_model_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])


class ForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store identifier (must exist in dataset)")
    horizon: int = Field(
        ...,
        ge=HORIZON_MIN,
        le=HORIZON_MAX,
        description=f"Number of steps to forecast ({HORIZON_MIN}–{HORIZON_MAX})",
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


class ForecastResponse(BaseModel):
    store_id: int
    horizon: int
    forecasts: list[dict]


@router.post("/store", response_model=ForecastResponse)
async def post_forecast_store(request: Request, body: ForecastRequest) -> ForecastResponse:
    """
    Generate horizon-step sales forecast for a single store.

    Uses the primary LightGBM model preloaded into app.state at startup.
    Returns a list of {date, forecast} dicts in chronological order.
    """
    model = request.app.state.primary_model
    feature_columns = request.app.state.feature_columns

    try:
        forecasts = forecast_store(body.store_id, body.horizon, model, feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Add 95% confidence interval using residual_std from training metadata
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

    return ForecastResponse(
        store_id=body.store_id,
        horizon=body.horizon,
        forecasts=forecasts,
    )


@router.post("/store/debug")
async def post_forecast_store_debug(body: ForecastRequest) -> dict:
    """
    Debug endpoint: return forecast metadata for a store without running inference.

    Validates the store exists and returns model version, feature columns,
    lookback window, and the last observed date for the requested store.
    """
    try:
        last_date = get_store_last_date(body.store_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        metadata = get_model_metadata()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = {
        "store_id": body.store_id,
        "last_observed_date": last_date,
        "model_version": metadata.get("model_version"),
        "feature_columns_used": metadata.get("feature_columns"),
        "max_lag_used": metadata.get("max_lag"),
        "lookback_window": metadata.get("lookback_window"),
        "recursive_steps": body.horizon,
    }

    logger.info(
        "Debug forecast info: store_id=%d, last_date=%s, horizon=%d",
        body.store_id, last_date, body.horizon,
    )

    return result
