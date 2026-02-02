"""
Monitoring summary builder.

Combines performance monitor output, drift detector output, and model metadata
into a single structured dictionary with overall status. No file I/O.
"""

from typing import Any


def build_monitoring_summary(
    performance_output: dict[str, Any],
    drift_output: dict[str, Any],
    model_metadata: dict[str, Any],
    *,
    drift_warning_ratio: float = 0.8,
) -> dict[str, Any]:
    """
    Build a single monitoring summary from performance, drift, and model metadata.

    Args:
        performance_output: Output from PerformanceMonitor.evaluate (alerts, metrics).
        drift_output: Output from DataDriftDetector.detect_drift (drift_detected, overall_score).
        model_metadata: Dict with model_name and version (e.g. {"model_name": "lightgbm", "version": "v1.0"}).
        drift_warning_ratio: Consider drift "close to threshold" when score >= threshold * this
            (default 0.8). Used for warning status.

    Returns:
        Dictionary with:
            - model_info: from model_metadata (model_name, version)
            - performance: performance_output
            - drift: drift_output
            - overall_status: "healthy" | "warning" | "critical"

    Rules:
        - critical: any performance alert (mae or mape) OR drift_detected
        - warning: drift score close to threshold (score >= threshold * drift_warning_ratio) but not critical
        - healthy: otherwise

    No file I/O.
    """
    model_info = {
        "model_name": model_metadata.get("model_name", ""),
        "version": model_metadata.get("version", ""),
    }

    alerts = performance_output.get("alerts") or {}
    any_perf_alert = any(alerts.get(k, False) for k in ("mae", "mape"))
    drift_detected = drift_output.get("drift_detected", False)
    drift_score = drift_output.get("overall_score", 0.0)
    drift_threshold = drift_output.get("threshold", 1.0)
    drift_close = drift_score >= drift_threshold * drift_warning_ratio and not drift_detected

    if any_perf_alert or drift_detected:
        overall_status = "critical"
    elif drift_close:
        overall_status = "warning"
    else:
        overall_status = "healthy"

    return {
        "model_info": model_info,
        "performance": performance_output,
        "drift": drift_output,
        "overall_status": overall_status,
    }
