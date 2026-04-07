"""
Loader for trained LightGBM and baseline models, plus evaluation metrics.

Provides explicit load functions (called at startup) and get functions
(for backward compatibility or direct access). Module-level singletons
are populated by the startup event; lazy loading is not used in production.

Expected artifact paths:
    <project_root>/artifacts/models/primary_lightgbm.joblib
    <project_root>/artifacts/models/baseline_seasonal_naive.joblib
    <project_root>/artifacts/models/feature_columns.json
    <project_root>/artifacts/models/metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "models"

_primary_model: Any = None
_baseline_model: Any = None
_feature_columns: list[str] | None = None
_metrics: dict[str, Any] | None = None


def load_primary_model() -> Any:
    """
    Load the primary LightGBM model from disk, cache in module singleton, and return it.

    Raises:
        RuntimeError: If the artifact file does not exist.
    """
    global _primary_model
    artifact_path = _ARTIFACTS_DIR / "primary_lightgbm.joblib"
    if not artifact_path.exists():
        raise RuntimeError(
            f"Primary model artifact not found: {artifact_path}. "
            "Run training (scripts/train.py) to generate it."
        )
    _primary_model = joblib.load(artifact_path)
    logger.info("Primary model loaded successfully from %s", artifact_path)
    return _primary_model


def load_baseline_model() -> Any:
    """
    Load the baseline seasonal-naive model from disk, cache in module singleton, and return it.

    Raises:
        RuntimeError: If the artifact file does not exist.
    """
    global _baseline_model
    artifact_path = _ARTIFACTS_DIR / "baseline_seasonal_naive.joblib"
    if not artifact_path.exists():
        raise RuntimeError(
            f"Baseline model artifact not found: {artifact_path}. "
            "Run training (scripts/train.py) to generate it."
        )
    _baseline_model = joblib.load(artifact_path)
    logger.info("Baseline model loaded successfully from %s", artifact_path)
    return _baseline_model


def load_feature_columns() -> list[str]:
    """
    Load the ordered feature column list from artifacts/models/feature_columns.json.

    This file is written by scripts/train.py from primary._feature_cols, so it
    reflects the exact columns and order the model was trained on. Loading at
    startup allows the inference layer to enforce strict column alignment.

    Raises:
        RuntimeError: If the artifact file does not exist.
    """
    global _feature_columns
    artifact_path = _ARTIFACTS_DIR / "feature_columns.json"
    if not artifact_path.exists():
        raise RuntimeError(
            f"Feature columns artifact not found: {artifact_path}. "
            "Run training (scripts/train.py) to generate it."
        )
    with artifact_path.open() as f:
        _feature_columns = json.load(f)
    logger.info(
        "Feature columns loaded successfully (%d columns) from %s",
        len(_feature_columns), artifact_path,
    )
    return _feature_columns


def get_primary_model() -> Any:
    """
    Return the cached primary model.

    Raises:
        RuntimeError: If load_primary_model() has not been called yet.
    """
    if _primary_model is None:
        raise RuntimeError(
            "Primary model has not been loaded. "
            "Ensure load_primary_model() is called at application startup."
        )
    return _primary_model


def get_baseline_model() -> Any:
    """
    Return the cached baseline model.

    Raises:
        RuntimeError: If load_baseline_model() has not been called yet.
    """
    if _baseline_model is None:
        raise RuntimeError(
            "Baseline model has not been loaded. "
            "Ensure load_baseline_model() is called at application startup."
        )
    return _baseline_model


def get_model_metadata() -> dict[str, Any]:
    """
    Load and return model metadata from artifacts/models/model_metadata.json.

    Raises:
        RuntimeError: If the metadata artifact does not exist.
    """
    metadata_path = _ARTIFACTS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"Model metadata not found: {metadata_path}. "
            "Run training (scripts/train.py) to generate it."
        )
    with metadata_path.open() as f:
        metadata = json.load(f)
    logger.info("Model metadata loaded from %s", metadata_path)
    return metadata


def get_metrics() -> dict[str, Any]:
    """
    Return evaluation metrics, loading from disk on first call.

    Raises:
        RuntimeError: If the metrics artifact does not exist.
    """
    global _metrics
    if _metrics is None:
        metrics_path = _ARTIFACTS_DIR / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(
                f"Metrics artifact not found: {metrics_path}. "
                "Run training (scripts/train.py) to generate it."
            )
        with metrics_path.open() as f:
            _metrics = json.load(f)
        logger.info("Metrics loaded successfully from %s", metrics_path)
    return _metrics
