"""
Forecast API: store-level sales forecasting.

POST /forecast/store returns horizon-step forecasts for a given store_id.
Inference only; no retraining.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.forecasting_service import forecast_store

router = APIRouter(prefix="/forecast", tags=["forecast"])


class ForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store identifier")
    horizon: int = Field(..., ge=1, description="Number of steps to forecast")


class ForecastResponse(BaseModel):
    store_id: int
    horizon: int
    forecasts: list[dict]


@router.post("/store", response_model=ForecastResponse)
async def post_forecast_store(request: ForecastRequest) -> ForecastResponse:
    """
    Generate horizon-step sales forecast for a single store.

    Returns a list of {date, forecast} dicts in chronological order.
    """
    try:
        forecasts = forecast_store(request.store_id, request.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ForecastResponse(
        store_id=request.store_id,
        horizon=request.horizon,
        forecasts=forecasts,
    )
