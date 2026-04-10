# FastAPI main
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.api.v1.backtest import router as backtest_router
from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.copilot import router as copilot_router
from backend.app.api.v1.forecast import router as forecast_router
from backend.app.api.v1.model_info import router as model_info_router
from backend.app.api.v1.monitoring import router as monitoring_router
from backend.app.services.model_loader import load_baseline_model, load_feature_columns, load_primary_model
from backend.app.services.monitoring_service import initialize_monitoring_state

logger = logging.getLogger(__name__)

app = FastAPI(title="E2E Time Series Forecasting API")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return structured 422 errors with per-field messages."""
    errors = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = ".".join(str(part) for part in loc if part != "body")
        errors.append(
            {
                "field": field or "unknown",
                "message": err.get("msg", "Validation error"),
                "type": err.get("type", "value_error"),
                "input": err.get("input"),
            }
        )
    logger.warning("Request validation failed: %s", errors)
    return JSONResponse(status_code=422, content={"detail": errors})


@app.on_event("startup")
async def startup_load_models() -> None:
    """Load trained models at startup and store in app.state for request-time access."""
    app.state.primary_model = load_primary_model()
    app.state.baseline_model = load_baseline_model()
    app.state.feature_columns = load_feature_columns()
    logger.info("All models and feature columns loaded and attached to app.state.")
    try:
        initialize_monitoring_state()
    except Exception as exc:
        logger.warning("Monitoring initialization failed (non-fatal): %s", exc)


@app.get("/health/live")
def health_live() -> dict:
    """Liveness probe per API contract."""
    return {"status": "ok"}


app.include_router(backtest_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(copilot_router, prefix="/api/v1")
app.include_router(forecast_router, prefix="/api/v1")
app.include_router(model_info_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
