"""
Pydantic schema for model_metadata.json.

Shared by the training pipeline (validation at write time) and the
/api/v1/model/info endpoint (validation at read time). Kept in its own
module to avoid heavy transitive imports (joblib, numpy) when only the
schema is needed.
"""

from typing import Any

from pydantic import BaseModel, Field


class TrainingDateRange(BaseModel):
    start: str
    end: str


class ValidationMetrics(BaseModel):
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None


class ModelMetadataResponse(BaseModel):
    model_version: str
    trained_at: str
    training_date_range: TrainingDateRange
    feature_columns: list[str]
    feature_count: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    hyperparameters: dict[str, Any]
    residual_std: float = Field(ge=0)
    validation_metrics: ValidationMetrics
    max_lag: int = Field(ge=0)
    lookback_window: int = Field(ge=0)
