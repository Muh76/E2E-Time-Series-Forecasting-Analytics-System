"""
In-memory monitoring state service.

Seeds performance from model validation metrics at startup, runs basic feature
drift on processed data, and refreshes metrics after each backtest. Exposes a
stable summary shape for GET /api/v1/monitoring/summary.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def compute_feature_drift(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run drift scan; lazy-imports pandas only when called."""
    from backend.app.services.monitoring_drift import compute_feature_drift as _run

    return _run(config)


_ROOT = Path(__file__).resolve().parents[3]


def _load_merged_config() -> dict[str, Any]:
    base_path = _ROOT / "config" / "base" / "default.yaml"
    if not base_path.exists():
        return {}
    with open(base_path) as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}
    env = os.environ.get("APP_ENV", "local")
    env_path = _ROOT / "config" / env / "config.yaml"
    if env_path.exists():
        with open(env_path) as f:
            env_cfg = yaml.safe_load(f) or {}
        for key, val in env_cfg.items():
            if key in config and isinstance(config[key], dict) and isinstance(val, dict):
                config[key] = {**config[key], **val}
            else:
                config[key] = val
    return config


# In-memory store: last computed monitoring state (or None until initialized)
_monitoring_state: dict[str, Any] | None = None


def _thresholds_from_config(cfg: dict[str, Any]) -> dict[str, float]:
    perf = (cfg.get("monitoring") or {}).get("performance") or {}
    th = perf.get("thresholds") or {}
    mae_alert = float(th.get("mae_alert", 15.0))
    mape_alert = float(th.get("mape_alert", 0.20))
    drift_cfg = (cfg.get("monitoring") or {}).get("drift") or {}
    drift_th = float(drift_cfg.get("threshold", 0.25))
    return {
        "mae_alert": mae_alert,
        "mape_alert": mape_alert,
        "drift_threshold": drift_th,
    }


def _compute_alerts(
    mae: float | None,
    mape: float | None,
    drift: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, bool]:
    """MAPE from API is percentage (e.g. 14.5); threshold mape_alert is decimal (0.20 = 20%)."""
    mae_alert = thresholds["mae_alert"]
    mape_limit_pct = thresholds["mape_alert"] * 100 if thresholds["mape_alert"] <= 1 else thresholds["mape_alert"]
    drift_th = thresholds["drift_threshold"]

    mae_bad = mae is not None and not math.isnan(mae) and mae > mae_alert
    mape_bad = False
    if mape is not None and not math.isnan(mape):
        mape_bad = mape > mape_limit_pct

    drift_bad = bool(drift.get("drift_detected")) or (float(drift.get("overall_score") or 0) > drift_th)

    return {"mae": mae_bad, "mape": mape_bad, "drift": drift_bad}


def get_stubbed_summary() -> dict[str, Any]:
    """Minimal shape when no artifacts exist (e.g. before training)."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cfg = _load_merged_config()
    th = _thresholds_from_config(cfg)
    return {
        "model_info": {"model_name": "", "model_version": "unknown"},
        "model_version": "unknown",
        "as_of": now,
        "performance": {
            "mae": None,
            "rmse": None,
            "mape": None,
            "sample_size": 0,
        },
        "drift": {
            "status": "ok",
            "last_checked": now,
            "indicators": [],
            "overall_score": None,
            "threshold": th["drift_threshold"],
            "per_feature_scores": {},
        },
        "pipeline": {
            "last_training": None,
            "last_etl": None,
            "status": "unknown",
        },
        "rolling_series": {"mae": [], "mape": []},
        "alerts": {"mae": False, "mape": False, "drift": False},
        "thresholds": th,
        "overall_status": "unknown",
    }


def set_monitoring_state(summary: dict[str, Any]) -> None:
    global _monitoring_state
    _monitoring_state = summary


def initialize_monitoring_state() -> None:
    """
    Seed monitoring from model_metadata.json + drift scan. Idempotent per process.
    """
    if _monitoring_state is not None:
        return

    from backend.app.services.model_loader import get_model_metadata

    try:
        metadata = get_model_metadata()
    except RuntimeError as exc:
        logger.warning("Monitoring not seeded: %s", exc)
        set_monitoring_state(_build_empty_state_from_error(str(exc)))
        return

    cfg = _load_merged_config()
    thresholds = _thresholds_from_config(cfg)
    drift = compute_feature_drift(cfg)

    vm = metadata.get("validation_metrics") or {}
    mae = vm.get("mae")
    rmse = vm.get("rmse")
    mape = vm.get("mape")
    sample_size = int(metadata.get("sample_size") or 0)

    mae_f = float(mae) if mae is not None else None
    rmse_f = float(rmse) if rmse is not None else None
    mape_f = float(mape) if mape is not None else None

    alerts = _compute_alerts(mae_f, mape_f, drift, thresholds)
    overall = "healthy"
    if alerts["mae"] or alerts["mape"]:
        overall = "warning"
    if alerts["drift"]:
        overall = "warning"

    version = metadata.get("model_version", "unknown")
    state: dict[str, Any] = {
        "model_info": {"model_name": "lightgbm", "version": version},
        "performance": {
            "current_metrics": {
                "mae": mae_f if mae_f is not None else 0.0,
                "rmse": rmse_f if rmse_f is not None else 0.0,
                "mape": mape_f if mape_f is not None else 0.0,
            },
            "evaluated_points": sample_size,
            "source": "validation_holdout",
            "rolling_series": {"mae": [], "mape": []},
        },
        "drift": drift,
        "overall_status": overall,
        "alerts": alerts,
        "thresholds": thresholds,
        "pipeline": {
            "last_training": metadata.get("trained_at"),
            "last_etl": None,
            "status": "ok",
        },
        "recent_activity": {},
    }
    set_monitoring_state(state)
    logger.info(
        "Monitoring seeded from metadata: version=%s mae=%s mape=%s drift=%s",
        version,
        mae_f,
        mape_f,
        drift.get("drift_detected"),
    )


def _build_empty_state_from_error(message: str) -> dict[str, Any]:
    cfg = _load_merged_config()
    th = _thresholds_from_config(cfg)
    return {
        "model_info": {"model_name": "", "version": "unknown"},
        "performance": {
            "current_metrics": {"mae": 0.0, "rmse": 0.0, "mape": 0.0},
            "evaluated_points": 0,
            "source": "none",
            "rolling_series": {"mae": [], "mape": []},
        },
        "drift": {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": th["drift_threshold"],
            "per_feature_scores": {},
            "note": message,
        },
        "overall_status": "unknown",
        "alerts": {"mae": False, "mape": False, "drift": False},
        "thresholds": th,
        "pipeline": {"last_training": None, "last_etl": None, "status": "unknown"},
        "recent_activity": {},
    }


def record_forecast_activity(store_id: int, horizon: int) -> None:
    """Record last forecast for Copilot / pipeline context."""
    global _monitoring_state
    if _monitoring_state is None:
        initialize_monitoring_state()
    if _monitoring_state is None:
        return
    state = copy.deepcopy(_monitoring_state)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state.setdefault("recent_activity", {})["last_forecast"] = {
        "store_id": store_id,
        "horizon": horizon,
        "at": now,
    }
    _monitoring_state = state


def update_monitoring_from_backtest(
    backtest_result: dict[str, Any],
    model_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Persist backtest averages and per-split series into monitoring state.
    Refreshes drift from processed data.
    """
    avg = backtest_result.get("average") or {}
    n_splits = backtest_result.get("n_splits", 0)
    store_id = backtest_result.get("store_id")
    horizon = backtest_result.get("horizon")
    splits = backtest_result.get("splits") or []

    version = "unknown"
    if model_metadata:
        version = model_metadata.get("model_version", "unknown")

    total_samples = sum(s.get("horizon", 0) for s in splits)

    rolling_mae = [{"date": s["cutoff_date"], "value": float(s["mae"])} for s in splits]
    rolling_mape = [{"date": s["cutoff_date"], "value": float(s["mape"])} for s in splits]

    cfg = _load_merged_config()
    drift = compute_feature_drift(cfg)
    thresholds = _thresholds_from_config(cfg)

    mae = float(avg.get("mae", 0.0))
    mape = avg.get("mape")
    mape_f = float(mape) if mape is not None and not (isinstance(mape, float) and math.isnan(mape)) else None
    rmse = float(avg.get("rmse", 0.0))

    alerts = _compute_alerts(mae, mape_f, drift, thresholds)
    overall = "healthy"
    if alerts["mae"] or alerts["mape"] or alerts["drift"]:
        overall = "warning"

    state = {
        "model_info": {"model_name": "lightgbm", "version": version},
        "performance": {
            "current_metrics": {"mae": mae, "rmse": rmse, "mape": mape_f if mape_f is not None else 0.0},
            "evaluated_points": total_samples,
            "source": "rolling_backtest",
            "backtest": {
                "store_id": store_id,
                "horizon": horizon,
                "n_splits": n_splits,
                "avg_rmse": rmse,
                "avg_mae": mae,
                "avg_mape": mape_f if mape_f is not None else 0.0,
            },
            "rolling_series": {"mae": rolling_mae, "mape": rolling_mape},
        },
        "drift": drift,
        "overall_status": overall,
        "alerts": alerts,
        "thresholds": thresholds,
        "pipeline": {
            "last_training": (model_metadata or {}).get("trained_at"),
            "last_etl": None,
            "status": "ok",
        },
        "recent_activity": (_monitoring_state or {}).get("recent_activity") or {},
    }

    set_monitoring_state(state)
    logger.info(
        "Monitoring updated from backtest: store_id=%s n_splits=%d avg_rmse=%.4f avg_mae=%.4f",
        store_id,
        n_splits,
        rmse,
        mae,
    )


def get_monitoring_summary() -> dict[str, Any]:
    """
    Return monitoring summary for API + Streamlit (normalized shape).
    """
    if _monitoring_state is None:
        initialize_monitoring_state()
    if _monitoring_state is None:
        return get_stubbed_summary()

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state = _monitoring_state
    model_info = state.get("model_info") or {}
    perf = state.get("performance") or {}
    drift = state.get("drift") or {}
    overall = state.get("overall_status", "healthy")
    thresholds = state.get("thresholds") or _thresholds_from_config(_load_merged_config())
    alerts = state.get("alerts") or _compute_alerts(
        perf.get("current_metrics", {}).get("mae"),
        perf.get("current_metrics", {}).get("mape"),
        drift,
        thresholds,
    )

    current = perf.get("current_metrics") or {}
    mae = current.get("mae", 0.0)
    mape = current.get("mape", 0.0)
    rmse = current.get("rmse", 0.0)
    sample_size = int(perf.get("evaluated_points", 0))

    drift_status = "drift_detected" if drift.get("drift_detected") else "ok"
    per_scores = drift.get("per_feature_scores") or {}
    th_drift = float(drift.get("threshold") or thresholds["drift_threshold"])
    indicators = [{"feature": k, "score": float(v)} for k, v in per_scores.items() if float(v) > th_drift]

    rolling = perf.get("rolling_series") or {}

    pipeline = state.get("pipeline") or {}

    version = model_info.get("version", "v1.0.0")

    return {
        "model_info": {
            "model_name": model_info.get("model_name", "lightgbm"),
            "model_version": version,
        },
        "model_version": version,
        "as_of": now,
        "performance": {
            "mae": float(mae) if mae is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
            "mape": float(mape) if mape is not None else None,
            "sample_size": sample_size,
            "source": perf.get("source", "unknown"),
        },
        "drift": {
            "status": drift_status,
            "last_checked": now,
            "indicators": indicators,
            "overall_score": drift.get("overall_score"),
            "threshold": th_drift,
            "per_feature_scores": per_scores,
        },
        "pipeline": {
            "last_training": pipeline.get("last_training"),
            "last_etl": pipeline.get("last_etl"),
            "status": pipeline.get("status", "ok" if overall == "healthy" else overall),
        },
        "rolling_series": rolling,
        "alerts": alerts,
        "thresholds": thresholds,
        "overall_status": overall,
        "recent_activity": state.get("recent_activity") or {},
    }


def get_evaluation_snapshot() -> dict[str, Any]:
    """Structured evaluation block for GET /monitoring/metrics."""
    from backend.app.services.model_loader import get_model_metadata

    summary = get_monitoring_summary()
    perf = summary.get("performance") or {}
    try:
        meta = get_model_metadata()
        vm = meta.get("validation_metrics") or {}
    except RuntimeError:
        vm = {}

    source = perf.get("source", "unknown")
    return {
        "model_version": summary.get("model_version"),
        "primary_metrics": {
            "mae": perf.get("mae"),
            "rmse": perf.get("rmse"),
            "mape": perf.get("mape"),
            "sample_size": perf.get("sample_size"),
        },
        "validation_holdout": {
            "mae": vm.get("mae"),
            "rmse": vm.get("rmse"),
            "mape": vm.get("mape"),
        },
        "source": source,
        "as_of": summary.get("as_of"),
    }
