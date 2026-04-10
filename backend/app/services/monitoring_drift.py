"""
Compute feature drift from processed parquet (reference vs recent windows).

Uses models.monitoring.drift.DataDriftDetector — no SciPy; deterministic.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.monitoring.drift import DataDriftDetector  # noqa: E402

# Numeric columns commonly present after ETL / Rossmann-style pipelines
_PREFERRED_DRIFT_COLS = (
    "target_cleaned",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "SchoolHoliday",
)


def _merged_config() -> dict[str, Any]:
    base_path = _PROJECT_ROOT / "config" / "base" / "default.yaml"
    if not base_path.exists():
        return {}
    with open(base_path) as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}
    env = os.environ.get("APP_ENV", "local")
    env_path = _PROJECT_ROOT / "config" / env / "config.yaml"
    if env_path.exists():
        with open(env_path) as f:
            env_cfg = yaml.safe_load(f) or {}
        for key, val in env_cfg.items():
            if key in config and isinstance(config[key], dict) and isinstance(val, dict):
                config[key] = {**config[key], **val}
            else:
                config[key] = val
    return config


def processed_parquet_path(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data") or {}
    rel = data_cfg.get("processed_path", "data/processed")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path / "etl_output.parquet"


def compute_feature_drift(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Compare early vs late rows in processed data for numeric feature drift.

    Returns a dict compatible with monitoring_service internal drift state:
    drift_detected, overall_score, threshold, per_feature_scores, ref/current sizes.
    On failure (missing file, empty data), returns a neutral no-drift result.
    """
    cfg = config or _merged_config()
    drift_cfg = (cfg.get("monitoring") or {}).get("drift") or {}
    if drift_cfg.get("enabled") is False:
        return {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": float(drift_cfg.get("threshold", 0.25)),
            "per_feature_scores": {},
            "ref_sample_size": 0,
            "current_sample_size": 0,
            "note": "drift checks disabled in config",
        }

    path = processed_parquet_path(cfg)
    if not path.exists():
        logger.warning("Drift skipped: processed parquet not found at %s", path)
        return {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": float(drift_cfg.get("threshold", 0.25)),
            "per_feature_scores": {},
            "ref_sample_size": 0,
            "current_sample_size": 0,
            "note": "processed data not available",
        }

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Drift skipped: could not read parquet: %s", exc)
        return {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": float(drift_cfg.get("threshold", 0.25)),
            "per_feature_scores": {},
            "ref_sample_size": 0,
            "current_sample_size": 0,
            "note": "parquet read failed",
        }

    if len(df) < 50:
        logger.warning("Drift skipped: insufficient rows (%d)", len(df))
        return {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": float(drift_cfg.get("threshold", 0.25)),
            "per_feature_scores": {},
            "ref_sample_size": len(df),
            "current_sample_size": 0,
            "note": "insufficient rows for drift",
        }

    date_col = (cfg.get("feature_engineering") or {}).get("date_column", "date")
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    split = max(n // 2, 1)
    ref_df = df.iloc[:split]
    cur_df = df.iloc[split:]

    feature_cols: list[str] = []
    for c in _PREFERRED_DRIFT_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    if not feature_cols:
        feature_cols = [
            c
            for c in df.columns
            if c not in (date_col, "store_id", "Date", "Store") and pd.api.types.is_numeric_dtype(df[c])
        ][:12]

    if not feature_cols:
        return {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": float(drift_cfg.get("threshold", 0.25)),
            "per_feature_scores": {},
            "ref_sample_size": len(ref_df),
            "current_sample_size": len(cur_df),
            "note": "no numeric columns for drift",
        }

    detector_cfg = {
        "monitoring": {
            "drift": {
                "threshold": float(drift_cfg.get("threshold", 0.25)),
                "min_sample_size": int(drift_cfg.get("min_sample_size", 100)),
                "n_bins": int(drift_cfg.get("n_bins", 10)),
            }
        }
    }
    detector = DataDriftDetector(detector_cfg)
    detector.fit_reference(ref_df, feature_cols=feature_cols)
    result = detector.detect_drift(cur_df, feature_cols=feature_cols)

    return {
        "drift_detected": bool(result["drift_detected"]),
        "overall_score": float(result["overall_score"]),
        "threshold": float(result["threshold"]),
        "per_feature_scores": result["per_feature_scores"],
        "ref_sample_size": int(result["ref_sample_size"]),
        "current_sample_size": int(result["current_sample_size"]),
    }
