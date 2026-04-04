"""
Forecast API: store-level sales forecasting.

POST /forecast/store returns horizon-step forecasts for a given store_id.
Uses the primary model preloaded at startup (app.state.primary_model).
Inference only; no retraining.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.services.forecasting_service import forecast_store

router = APIRouter(prefix="/forecast", tags=["forecast"])


class ForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store identifier")
    horizon: int = Field(..., ge=1, description="Number of steps to forecast")


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

    return ForecastResponse(
        store_id=body.store_id,
        horizon=body.horizon,
        forecasts=forecasts,
    )
