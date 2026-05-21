"""
Forecast API: store-level sales forecasting.

POST /forecast/store returns horizon-step forecasts for a given store_id.
POST /forecast/store/debug returns debug metadata without running inference.
Uses the primary model preloaded at startup (app.state.primary_model).
Inference only; no retraining.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.api.v1.validators import HORIZON_MAX, HORIZON_MIN, validate_store_id
from backend.app.services.forecasting_service import get_store_last_date
from backend.app.services.model_loader import get_model_metadata, load_feature_columns, load_primary_model
from backend.app.services.prediction_pipeline import execute_store_forecast

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
        return validate_store_id(v)


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
    model = getattr(request.app.state, "primary_model", None)
    feature_columns = getattr(request.app.state, "feature_columns", None)
    try:
        if model is None:
            logger.warning("forecast_store: primary model missing from app.state; attempting lazy load")
            model = load_primary_model()
            request.app.state.primary_model = model
        if not isinstance(feature_columns, list) or not feature_columns:
            logger.warning("forecast_store: feature_columns missing from app.state; attempting lazy load")
            feature_columns = load_feature_columns()
            request.app.state.feature_columns = feature_columns
    except RuntimeError as exc:
        logger.error("forecast_store: model artifacts unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable. Run training artifacts setup and retry.",
        ) from exc

    t_start = time.perf_counter()
    try:
        forecasts = execute_store_forecast(body.store_id, body.horizon, model, feature_columns)
    except ValueError as exc:
        logger.warning(
            "forecast_store: validation/inference error store_id=%s horizon=%s error=%s",
            body.store_id,
            body.horizon,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(
            "forecast_store: runtime dependency error store_id=%s horizon=%s error=%s",
            body.store_id,
            body.horizon,
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        logger.error(
            "forecast_store: missing dependency during forecast store_id=%s horizon=%s error=%s",
            body.store_id,
            body.horizon,
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
            "forecast_store: unexpected failure store_id=%s horizon=%s",
            body.store_id,
            body.horizon,
        )
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while generating forecast. Check server logs for details.",
        ) from exc
    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    try:
        metadata = get_model_metadata()
        model_version = metadata.get("model_version", "unknown")
    except (RuntimeError, TypeError, ValueError):
        model_version = "unknown"

    logger.info(
        "Forecast completed: store_id=%d, horizon=%d, latency_ms=%.1f, model_version=%s",
        body.store_id,
        body.horizon,
        latency_ms,
        model_version,
    )

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
    except ImportError as exc:
        logger.error(
            "forecast_store_debug: missing dependency store_id=%s horizon=%s error=%s",
            body.store_id,
            body.horizon,
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
            "forecast_store_debug: unexpected failure store_id=%s horizon=%s",
            body.store_id,
            body.horizon,
        )
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while preparing debug forecast info. Check server logs.",
        ) from exc

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
        body.store_id,
        last_date,
        body.horizon,
    )

    return result
