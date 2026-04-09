"""
Unit tests for model_metadata.json schema validation.

Uses the same Pydantic model that the /api/v1/model/info endpoint uses,
ensuring the training script output and the API response contract stay in sync.
"""

import copy

import pytest
from pydantic import ValidationError

from backend.app.schemas.model_metadata import ModelMetadataResponse


VALID_METADATA: dict = {
    "model_version": "v5",
    "trained_at": "2026-01-30T12:00:00Z",
    "training_date_range": {"start": "2013-01-01", "end": "2015-06-19"},
    "feature_columns": ["lag_1", "lag_7", "rolling_mean_7", "day_of_week"],
    "feature_count": 4,
    "sample_size": 800000,
    "hyperparameters": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "objective": "regression",
        "metric": "mae",
        "deterministic": True,
    },
    "residual_std": 576.34,
    "validation_metrics": {"rmse": 620.12, "mae": 410.55, "mape": 14.32},
    "max_lag": 14,
    "lookback_window": 14,
}


class TestModelMetadataSchema:

    def test_valid_metadata_passes(self):
        m = ModelMetadataResponse(**VALID_METADATA)
        assert m.model_version == "v5"
        assert m.feature_count == 4
        assert m.sample_size == 800000
        assert m.training_date_range.start == "2013-01-01"
        assert m.training_date_range.end == "2015-06-19"

    def test_hyperparameters_preserved(self):
        m = ModelMetadataResponse(**VALID_METADATA)
        assert m.hyperparameters["n_estimators"] == 100
        assert m.hyperparameters["learning_rate"] == 0.05
        assert m.hyperparameters["deterministic"] is True

    def test_validation_metrics_with_values(self):
        m = ModelMetadataResponse(**VALID_METADATA)
        assert m.validation_metrics.rmse == pytest.approx(620.12)
        assert m.validation_metrics.mae == pytest.approx(410.55)
        assert m.validation_metrics.mape == pytest.approx(14.32)

    def test_validation_metrics_nullable(self):
        data = copy.deepcopy(VALID_METADATA)
        data["validation_metrics"] = {"rmse": None, "mae": None, "mape": None}
        m = ModelMetadataResponse(**data)
        assert m.validation_metrics.rmse is None
        assert m.validation_metrics.mae is None
        assert m.validation_metrics.mape is None

    def test_missing_model_version_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        del data["model_version"]
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_missing_training_date_range_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        del data["training_date_range"]
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_missing_hyperparameters_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        del data["hyperparameters"]
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_missing_validation_metrics_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        del data["validation_metrics"]
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_negative_residual_std_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        data["residual_std"] = -1.0
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_negative_sample_size_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        data["sample_size"] = -100
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_negative_feature_count_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        data["feature_count"] = -1
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)

    def test_extra_fields_tolerated(self):
        data = copy.deepcopy(VALID_METADATA)
        data["custom_field"] = "should not break"
        m = ModelMetadataResponse(**data)
        assert m.model_version == "v5"

    def test_training_date_range_missing_end_fails(self):
        data = copy.deepcopy(VALID_METADATA)
        data["training_date_range"] = {"start": "2013-01-01"}
        with pytest.raises(ValidationError):
            ModelMetadataResponse(**data)
