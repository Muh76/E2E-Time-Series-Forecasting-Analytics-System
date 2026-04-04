"""
Singleton loader for trained LightGBM model and evaluation metrics.

Artifacts are loaded once on first access and cached in module-level variables.
Project root is resolved relative to this file's location.

Expected artifact paths:
    <project_root>/artifacts/models/primary_lightgbm.joblib
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
_metrics: dict[str, Any] | None = None


def get_primary_model() -> object:
    """
    Return the trained LightGBM model, loading from disk on first call.

    Raises:
        RuntimeError: If the model artifact does not exist.
    """
    global _primary_model
    if _primary_model is None:
        artifact_path = _ARTIFACTS_DIR / "primary_lightgbm.joblib"
        if not artifact_path.exists():
            raise RuntimeError(
                f"Primary model artifact not found: {artifact_path}. "
                "Run training (scripts/train.py) to generate it."
            )
        _primary_model = joblib.load(artifact_path)
        logger.info("Primary model loaded successfully from %s", artifact_path)
    return _primary_model


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
