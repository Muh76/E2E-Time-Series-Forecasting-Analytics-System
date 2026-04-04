# FastAPI main
import logging

from fastapi import FastAPI

from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.copilot import router as copilot_router
from backend.app.api.v1.forecast import router as forecast_router
from backend.app.api.v1.monitoring import router as monitoring_router
from backend.app.services.model_loader import load_baseline_model, load_primary_model

logger = logging.getLogger(__name__)

app = FastAPI(title="E2E Time Series Forecasting API")


@app.on_event("startup")
async def startup_load_models() -> None:
    """Load trained models at startup and store in app.state for request-time access."""
    app.state.primary_model = load_primary_model()
    app.state.baseline_model = load_baseline_model()
    logger.info("All models loaded and attached to app.state.")


@app.get("/health/live")
def health_live() -> dict:
    """Liveness probe per API contract."""
    return {"status": "ok"}


app.include_router(chat_router, prefix="/api/v1")
app.include_router(copilot_router, prefix="/api/v1")
app.include_router(forecast_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
