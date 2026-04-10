"""
Rule-based Insight Copilot for forecast series and metrics (no LLM).

Deterministic trend, volatility, and step-change detection with fixed thresholds.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1e-9

# Trend: normalized OLS slope vs series std (effect-size style)
_TREND_SLOPE_STRONG = 0.12
_TREND_SLOPE_WEAK = 0.04

# Volatility: coefficient of variation std/|mean|
_CV_HIGH = 0.22

# Step anomalies: |first difference| vs robust spread of diffs
_ANOMALY_Z = 2.5


def _extract_values(forecast: list[Any]) -> list[float]:
    out: list[float] = []
    for item in forecast:
        if item is None:
            continue
        if isinstance(item, (int, float)):
            if math.isfinite(float(item)):
                out.append(float(item))
            continue
        if isinstance(item, dict):
            v = item.get("forecast")
            if v is None:
                v = item.get("value")
            if v is None:
                v = item.get("y")
            if v is None:
                continue
            f = float(v)
            if math.isfinite(f):
                out.append(f)
    return out


def _ols_slope_normalized(y: np.ndarray) -> float:
    """Slope of y vs index, divided by std(y); ~0 means flat."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x < _EPS:
        return 0.0
    cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = cov_xy / var_x
    y_std = float(np.std(y, ddof=1)) if n > 1 else abs(y_mean) + _EPS
    denom = y_std + _EPS
    return float(slope / denom)


def _coefficient_of_variation(y: np.ndarray) -> float:
    m = float(np.mean(np.abs(y)) + _EPS)
    s = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    return s / m


def _detect_step_anomalies(y: np.ndarray) -> tuple[int, list[int]]:
    """
    Flag indices i where |y[i]-y[i-1]| is large vs scaled MAD of first differences.

    Returns (count, list of indices i where anomaly at step from i-1 to i).
    """
    if len(y) < 3:
        return 0, []
    d = np.diff(y)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + _EPS
    scale = 1.4826 * mad
    if scale < _EPS:
        scale = float(np.std(d, ddof=1)) + _EPS
    anoms: list[int] = []
    for i in range(len(d)):
        z = abs(float(d[i]) - med) / scale
        if z > _ANOMALY_Z:
            anoms.append(i + 1)
    return len(anoms), anoms


def _confidence_score(
    n: int,
    slope_norm: float,
    cv: float,
    n_anomalies: int,
) -> float:
    """Higher with more points, clearer trend, moderate volatility signal."""
    if n < 2:
        return 0.2
    coverage = min(1.0, n / 24.0)
    trend_strength = min(1.0, abs(slope_norm) / 0.25)
    vol_signal = min(1.0, cv / 0.35) if cv > 0.08 else 0.35
    anomaly_signal = min(1.0, 0.4 + 0.15 * min(n_anomalies, 4))
    raw = 0.35 * coverage + 0.35 * trend_strength + 0.15 * vol_signal + 0.15 * anomaly_signal
    return float(round(min(0.92, max(0.18, raw)), 3))


def build_forecast_insights(forecast: list[Any], metrics: dict[str, Any] | None) -> dict[str, Any]:
    """
    Produce summary, insights text, and confidence from forecast + optional metrics.

    Returns keys: summary, insights, confidence (float).
    """
    metrics = metrics or {}
    values = _extract_values(forecast)
    y = np.asarray(values, dtype=float)

    lines_insight: list[str] = []
    summary_parts: list[str] = []

    if len(y) < 2:
        summary = (
            "Not enough forecast points to characterize trend or volatility " "(need at least two numeric values)."
        )
        insights = (
            "- **Data**: Fewer than two valid forecast values after parsing.\n"
            "- **Metrics context**: " + _metrics_line(metrics)
        )
        return {"summary": summary, "insights": insights, "confidence": 0.15}

    slope_n = _ols_slope_normalized(y)
    cv = _coefficient_of_variation(y)

    # Low dispersion + modest slope → stable (distinguish noise from real drift)
    if cv < 0.02 and abs(slope_n) < 0.18:
        lines_insight.append(
            f"- **Trend**: **Stable** — near-constant level (CV {cv:.4f}, normalized slope {slope_n:.3f})."
        )
        summary_parts.append("a **stable pattern**")
    elif slope_n > _TREND_SLOPE_STRONG:
        lines_insight.append(
            f"- **Trend**: **Upward** — normalized slope vs dispersion is {slope_n:.3f} "
            f"(above strong threshold {_TREND_SLOPE_STRONG})."
        )
        summary_parts.append("an **upward trend**")
    elif slope_n < -_TREND_SLOPE_STRONG:
        lines_insight.append(
            f"- **Trend**: **Downward** — normalized slope vs dispersion is {slope_n:.3f} "
            f"(below -{_TREND_SLOPE_STRONG})."
        )
        summary_parts.append("a **downward trend**")
    elif abs(slope_n) < _TREND_SLOPE_WEAK:
        lines_insight.append(
            f"- **Trend**: **Stable** — normalized slope {slope_n:.3f} is within "
            f"±{_TREND_SLOPE_WEAK} (flat pattern)."
        )
        summary_parts.append("a **stable pattern**")
    else:
        direction = "slightly upward" if slope_n > 0 else "slightly downward"
        lines_insight.append(f"- **Trend**: **Moderate** — {direction} drift (normalized slope {slope_n:.3f}).")
        summary_parts.append(f"a **{direction}** trajectory")

    if cv >= _CV_HIGH:
        lines_insight.append(
            f"- **Volatility**: **High** — coefficient of variation is {cv:.3f} "
            f"(≥ {_CV_HIGH}: values swing strongly relative to typical level)."
        )
        summary_parts.append("**high volatility**")
    else:
        lines_insight.append(
            f"- **Volatility**: **Moderate or low** — CV {cv:.3f} is below the high-volatility " f"cutoff ({_CV_HIGH})."
        )

    n_anom, idx = _detect_step_anomalies(y)
    if n_anom > 0:
        preview = idx[:5]
        extra = f" (+{len(idx) - 5} more)" if len(idx) > 5 else ""
        lines_insight.append(
            "- **Anomalies**: **Sudden step changes** — "
            f"{n_anom} step(s) exceed robust z-score {_ANOMALY_Z} on first differences "
            f"(indices {preview}{extra})."
        )
        summary_parts.append("**sudden spikes or drops** between consecutive steps")
    else:
        lines_insight.append("- **Anomalies**: No sharp step changes detected vs robust spread of first differences.")

    lines_insight.append("- **Metrics context**: " + _metrics_line(metrics))

    confidence = _confidence_score(len(y), slope_n, cv, n_anom)

    if summary_parts:
        core = ", ".join(summary_parts[:2])
        if len(summary_parts) > 2:
            core += ", with " + summary_parts[2]
        summary = (
            f"The forecast series ({len(y)} points) shows {core}. "
            f"Analysis is rule-based (no LLM); confidence reflects series length and signal strength."
        )
    else:
        summary = f"Analyzed {len(y)} forecast points with rule-based detectors."

    insights = "\n".join(lines_insight)

    return {
        "summary": summary,
        "insights": insights,
        "confidence": confidence,
    }


def _metrics_line(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "none supplied."
    parts: list[str] = []
    for k in ("mae", "rmse", "mape", "mae_val", "rmse_val"):
        if k in metrics and metrics[k] is not None:
            try:
                parts.append(f"{k}={float(metrics[k]):.4g}")
            except (TypeError, ValueError):
                parts.append(f"{k}={metrics[k]}")
    if not parts:
        return "provided but no standard keys (mae, rmse, mape) found."
    return "; ".join(parts) + "."
