"""
In-memory monitoring state service.

Holds latest performance and drift outputs; returns monitoring summary in
API contract format. No database; no background jobs.
"""

from datetime import datetime, timezone
from typing import Any

# In-memory store: last computed monitoring state (or None for stubbed)
_monitoring_state: dict[str, Any] | None = None


def get_stubbed_summary() -> dict[str, Any]:
    """Return stubbed monitoring summary when no computed state exists."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "model_version": "v1.0.0",
        "as_of": now,
        "performance": {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "sample_size": 0,
        },
        "drift": {
            "status": "ok",
            "last_checked": now,
            "indicators": [],
        },
        "pipeline": {
            "last_training": now,
            "last_etl": now,
            "status": "ok",
        },
    }


def set_monitoring_state(summary: dict[str, Any]) -> None:
    """Store computed monitoring summary (from build_monitoring_summary)."""
    global _monitoring_state
    _monitoring_state = summary


def get_monitoring_summary() -> dict[str, Any]:
    """
    Return monitoring summary in API contract format.

    Uses in-memory state if set; otherwise returns stubbed summary.
    No database; no file I/O.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if _monitoring_state is None:
        return get_stubbed_summary()

    model_info = _monitoring_state.get("model_info") or {}
    perf = _monitoring_state.get("performance") or {}
    drift = _monitoring_state.get("drift") or {}
    overall = _monitoring_state.get("overall_status", "healthy")

    # Map to API contract structure
    rolling = perf.get("rolling_metrics") or {}
    current = perf.get("current_metrics") or {}
    mae = rolling.get("mae", current.get("mae", 0.0))
    mape = rolling.get("mape", current.get("mape", 0.0))
    rmse = current.get("rmse", 0.0)
    sample_size = perf.get("evaluated_points", 0)

    drift_status = "drift_detected" if drift.get("drift_detected") else "ok"
    indicators = [
        {"feature": k, "score": v}
        for k, v in (drift.get("per_feature_scores") or {}).items()
        if drift.get("drift_detected") and v > (drift.get("threshold") or 0)
    ]

    return {
        "model_version": model_info.get("version", "v1.0.0"),
        "as_of": now,
        "performance": {
            "mae": float(mae) if mae is not None else 0.0,
            "rmse": float(rmse) if rmse is not None else 0.0,
            "mape": float(mape) if mape is not None else 0.0,
            "sample_size": int(sample_size),
        },
        "drift": {
            "status": drift_status,
            "last_checked": now,
            "indicators": indicators,
        },
        "pipeline": {
            "last_training": now,
            "last_etl": now,
            "status": "ok" if overall == "healthy" else overall,
        },
    }
