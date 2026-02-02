# FastAPI main
from fastapi import FastAPI

from app.api.v1.chat import router as chat_router
from app.api.v1.copilot import router as copilot_router
from app.api.v1.monitoring import router as monitoring_router

app = FastAPI(title="E2E Time Series Forecasting API")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(copilot_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
