# FastAPI main
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import backend.app.bootstrap_env  # noqa: F401 — loads `.env` from project root first
from backend.app.api.v1.backtest import router as backtest_router
from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.copilot import router as copilot_router
from backend.app.api.v1.forecast import router as forecast_router
from backend.app.api.v1.metrics_endpoint import router as forecast_metrics_router
from backend.app.api.v1.model_info import router as model_info_router
from backend.app.api.v1.monitoring import router as monitoring_router
from backend.app.api.v1.predict import router as predict_router
from backend.app.runtime_paths import chunk_metadata_file, faiss_index_file
from backend.app.services.model_loader import load_baseline_model, load_feature_columns, load_primary_model
from backend.app.services.monitoring_service import initialize_monitoring_state
from backend.app.services.rag_service import _get_embed_model, get_faiss_and_metadata

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
    app.state.primary_model = None
    app.state.baseline_model = None
    app.state.feature_columns = None
    try:
        app.state.primary_model = load_primary_model()
        app.state.baseline_model = load_baseline_model()
        app.state.feature_columns = load_feature_columns()
        logger.info("All models and feature columns loaded and attached to app.state.")
    except Exception as exc:
        logger.warning(
            "Model artifacts not found (non-fatal): %s. "
            "Forecast/predict endpoints will be unavailable until training is run. "
            "Copilot and monitoring endpoints are unaffected.",
            exc,
        )
    try:
        initialize_monitoring_state()
    except Exception as exc:
        logger.warning("Monitoring initialization failed (non-fatal): %s", exc)
    try:
        get_faiss_and_metadata(
            index_path=faiss_index_file(),
            metadata_path=chunk_metadata_file(),
        )
        # Load the embedding model synchronously — blocking the event loop briefly
        # during startup is acceptable (no requests are served yet) and avoids
        # PyTorch threading issues that arise when loading inside asyncio.to_thread.
        _get_embed_model()
    except Exception as exc:
        logger.warning("RAG index initialization failed (non-fatal): %s", exc)


@app.get("/health/live")
def health_live() -> dict:
    """Liveness probe per API contract."""
    return {"status": "ok"}


app.include_router(backtest_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(copilot_router, prefix="/api/v1")
app.include_router(forecast_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(model_info_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(forecast_metrics_router, prefix="/api/v1")
