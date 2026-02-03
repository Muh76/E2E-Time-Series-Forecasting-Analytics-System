"""
API client utilities for the Streamlit frontend.

Loads backend API base URL from config and provides helpers for API calls.
Uses requests; returns mock data when backend is unavailable.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# frontend/components/api.py -> parent.parent = frontend, parent.parent.parent = project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_TIMEOUT = 10

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load base config and merge with env-specific overrides."""
    config_dir = _PROJECT_ROOT / "config"
    base_path = config_dir / "base" / "default.yaml"
    if not base_path.exists():
        return {}

    with open(base_path) as f:
        config = yaml.safe_load(f) or {}

    env = os.environ.get("APP_ENV", "local")
    env_path = config_dir / env / "config.yaml"
    if env_path.exists():
        with open(env_path) as f:
            env_config = yaml.safe_load(f) or {}
        for key, val in env_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(val, dict):
                config[key] = {**config[key], **val}
            else:
                config[key] = val
    return config


def get_api_base_url() -> str:
    """
    Return backend API base URL. Single source of truth.
    Priority: 1) config frontend.api_base_url, 2) env API_BASE_URL, 3) fallback http://localhost:8000.
    """
    config = _load_config()
    url = config.get("frontend", {}).get("api_base_url")
    if url:
        return url
    url = os.environ.get("API_BASE_URL")
    if url:
        return url
    return "http://localhost:8000"


def api_url(path: str) -> str:
    """Build full API URL from a path (e.g. /api/v1/health/live)."""
    base = get_api_base_url().rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{base}{path}"


def check_api_health() -> bool:
    """
    Check if the backend API is reachable.
    Calls GET /health/live. Returns True if status 200, else False.
    Handles timeout and connection errors safely; never raises.
    """
    try:
        resp = requests.get(api_url("/health/live"), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _is_backend_unavailable(exc: BaseException, include_404: bool = False) -> bool:
    """Return True if the exception indicates backend is unreachable."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.RequestException):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code >= 500:
            return True
        if include_404 and resp is not None and resp.status_code == 404:
            return True
    return False


def get_forecasts(
    series_ids: list[str] | None = None,
    horizon_steps: int = 12,
    frequency: str = "D",
    model_version: str | None = None,
    return_interval: bool = False,
) -> dict[str, Any]:
    """
    Request forecasts from the backend.
    POST /api/v1/forecasts/generate. Returns mock data if backend is unavailable.
    """
    series_ids = series_ids or ["series_001"]
    payload: dict[str, Any] = {
        "series_ids": series_ids,
        "horizon_steps": horizon_steps,
        "frequency": frequency,
        "options": {"return_interval": return_interval},
    }
    if model_version:
        payload["model_version"] = model_version

    url = api_url("/api/v1/forecasts/generate")
    try:
        resp = requests.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if _is_backend_unavailable(e):
            logger.warning("Backend unavailable, returning mock forecasts: %s", e)
            return _mock_forecasts(series_ids, horizon_steps, frequency)
        raise


def _mock_forecasts(
    series_ids: list[str],
    horizon_steps: int,
    frequency: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    forecasts = []
    for sid in series_ids:
        base = 100.0
        forecasts.append({
            "series_id": sid,
            "model_version": "v1.0.0",
            "frequency": frequency,
            "point_forecast": [base + i * 0.5 for i in range(horizon_steps)],
            "steps": list(range(1, horizon_steps + 1)),
        })
    return {
        "job_id": "mock_gen_001",
        "status": "completed",
        "forecasts": forecasts,
        "generated_at": now,
    }


def get_monitoring_summary(
    model_version: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Fetch monitoring summary. Returns mock data if backend is unavailable."""
    params: dict[str, str] = {}
    if model_version:
        params["model_version"] = model_version
    if since:
        params["since"] = since

    url = api_url("/api/v1/monitoring/summary")
    try:
        resp = requests.get(url, params=params or None, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if _is_backend_unavailable(e):
            logger.warning("Backend unavailable, returning mock monitoring summary: %s", e)
            return _mock_monitoring_summary()
        raise


def _mock_monitoring_summary() -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    today = datetime.now(timezone.utc).date()
    # Rolling series for charts (last 7 days)
    rolling_mae = [
        {"date": (today - timedelta(days=i)).isoformat(), "value": 2.1 + (i % 3) * 0.3}
        for i in range(6, -1, -1)
    ]
    rolling_mape = [
        {"date": (today - timedelta(days=i)).isoformat(), "value": 3.5 + (i % 3) * 0.5}
        for i in range(6, -1, -1)
    ]
    per_feature = {"lag_1": 0.12, "lag_7": 0.18, "rolling_7": 0.22, "day_of_week": 0.08}
    return {
        "model_name": "LightGBM",
        "model_version": "v1.0.0",
        "as_of": now,
        "performance": {"mae": 2.34, "rmse": 3.01, "mape": 0.042, "sample_size": 120},
        "drift": {
            "status": "ok",
            "last_checked": now,
            "indicators": [],
            "per_feature_scores": per_feature,
            "overall_score": 0.18,
            "threshold": 0.25,
        },
        "pipeline": {"last_training": now, "last_etl": now, "status": "ok"},
        "rolling_series": {"mae": rolling_mae, "mape": rolling_mape},
        "thresholds": {"mae_alert": 15.0, "mape_alert": 0.20, "drift_threshold": 0.25},
        "alerts": {"mae": False, "mape": False, "drift": False},
    }


def copilot_explain(
    query: str,
    context: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Request explanation from copilot. POST /api/v1/copilot/explain.
    Pass monitoring_summary and metrics in context for grounding.
    Returns mock when backend unavailable or endpoint missing (404).
    """
    payload: dict[str, Any] = {"query": query or "What is the current model health?"}
    if context:
        payload["context"] = context
    if options:
        payload["options"] = options

    url = api_url("/api/v1/copilot/explain")
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if _is_backend_unavailable(e, include_404=True):
            logger.warning("Copilot unavailable, returning mock explanation: %s", e)
            return _mock_copilot_explain(query, context)
        raise


def _mock_copilot_explain(query: str, context: dict[str, Any] | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    perf = (context or {}).get("monitoring_summary", {}).get("performance", {}) or {}
    mae = perf.get("mae", 0)
    mape = perf.get("mape", 0)
    mape_pct = mape * 100 if mape and mape < 1 else (mape or 0)
    return {
        "explanation": (
            f"You asked: \"{query}\"\n\n"
            "**Mock response** (Copilot endpoint not available): "
            f"Current performance: MAE={mae:.2f}, MAPE={mape_pct:.1f}%. "
            "The copilot explains forecasts and metrics using precomputed data; it does not perform prediction. "
            "Connect the backend with LLM to get real explanations."
        ),
        "sources": [{"type": "mock", "note": "Backend copilot unavailable"}],
        "generated_at": now,
    }


def get_monitoring_series(
    metric: str,
    start_date: str,
    end_date: str,
    model_version: str | None = None,
    granularity: str = "daily",
) -> dict[str, Any]:
    """
    Fetch monitoring time series (rolling MAE, MAPE, etc).
    GET /api/v1/monitoring/series. Returns mock data if backend unavailable.
    """
    params: dict[str, str] = {
        "metric": metric,
        "start_date": start_date,
        "end_date": end_date,
        "granularity": granularity,
    }
    if model_version:
        params["model_version"] = model_version

    url = api_url("/api/v1/monitoring/series")
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if _is_backend_unavailable(e):
            logger.warning("Backend unavailable, returning mock monitoring series: %s", e)
            return _mock_monitoring_series(metric, start_date, end_date)
        raise


def _mock_monitoring_series(metric: str, start_date: str, end_date: str) -> dict[str, Any]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    delta = (end - start).days + 1
    base = 2.3 if metric == "mae" else 0.04
    data = [
        {"date": (start + timedelta(days=i)).isoformat(), "value": base + (i % 5) * 0.1}
        for i in range(min(delta, 14))
    ]
    return {"metric": metric, "model_version": "v1.0.0", "granularity": "daily", "data": data}


def get_metrics(
    series_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metrics: list[str] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """
    Fetch historical metrics. GET /api/v1/metrics/historical.
    Returns mock data if backend is unavailable.
    """
    series_ids = series_ids or ["series_001"]
    today = datetime.now(timezone.utc).date()
    if not start_date:
        start_date = (today - timedelta(days=7)).isoformat()
    if not end_date:
        end_date = today.isoformat()

    params: dict[str, str] = {
        "series_ids": ",".join(series_ids),
        "start_date": start_date,
        "end_date": end_date,
    }
    if metrics:
        params["metrics"] = ",".join(metrics) if isinstance(metrics, list) else metrics
    if model_version:
        params["model_version"] = model_version

    url = api_url("/api/v1/metrics/historical")
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if _is_backend_unavailable(e):
            logger.warning("Backend unavailable, returning mock metrics: %s", e)
            return _mock_metrics(series_ids, start_date, end_date)
        raise


def _mock_metrics(
    series_ids: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    delta = (end - start).days + 1
    data = []
    for sid in series_ids:
        base = 100.0
        for i in range(delta):
            d = (start + timedelta(days=i)).isoformat()
            actual = base + (i % 7) * 2.0
            forecast = actual + 0.5
            data.append({
                "series_id": sid,
                "date": d,
                "actual": round(actual, 2),
                "forecast": round(forecast, 2),
                "error": round(forecast - actual, 2),
                "model_version": "v1.0.0",
            })
    return {
        "data": data,
        "meta": {"series_ids": series_ids, "start_date": start_date, "end_date": end_date, "count": len(data)},
    }


def get_forecast_vs_actual(
    entity_id: str | None = None,
    horizon: int = 14,
) -> dict[str, Any]:
    """
    Fetch forecast vs actual data ready for plotting.
    Returns actual and forecast time series plus precomputed metrics (MAE, RMSE, MAPE).
    GET /api/v1/forecasts/vs-actual. Returns mock data if backend unavailable.
    """
    params: dict[str, str | int] = {"horizon": horizon}
    if entity_id:
        params["entity_id"] = entity_id

    url = api_url("/api/v1/forecasts/vs-actual")
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if _is_backend_unavailable(e):
            logger.warning("Backend unavailable, returning mock forecast vs actual: %s", e)
            return _mock_forecast_vs_actual(entity_id, horizon)
        raise


def _mock_forecast_vs_actual(entity_id: str | None, horizon: int) -> dict[str, Any]:
    """Mock forecast vs actual with ready-to-plot data and metrics."""
    today = datetime.now(timezone.utc).date()
    entity_ids = ["series_001", "series_002", "series_003"]
    eid = entity_id or entity_ids[0]
    if eid not in entity_ids:
        entity_ids = [eid] + entity_ids

    # Historical actual + forecast overlap (last 14 days)
    hist_days = 14
    actual_dates: list[str] = []
    actual_values: list[float] = []
    forecast_dates: list[str] = []
    forecast_values: list[float] = []

    base = 100.0
    for i in range(hist_days):
        d = (today - timedelta(days=hist_days - 1 - i)).isoformat()
        actual_dates.append(d)
        val = base + (i % 7) * 2.0
        actual_values.append(round(val, 2))
        forecast_values.append(round(val + 0.3 * (i % 3 - 1), 2))
        forecast_dates.append(d)

    # Future forecast only
    for i in range(1, horizon + 1):
        d = (today + timedelta(days=i)).isoformat()
        forecast_dates.append(d)
        forecast_values.append(round(base + (i % 5) * 1.5, 2))

    # Precomputed metrics (API returns these; no frontend computation)
    metrics = {"mae": 2.34, "rmse": 3.01, "mape": 0.042}

    return {
        "entity_id": eid,
        "entity_ids": entity_ids,
        "actual": [{"date": d, "value": v} for d, v in zip(actual_dates, actual_values)],
        "forecast": [{"date": d, "value": v} for d, v in zip(forecast_dates, forecast_values)],
        "metrics": metrics,
        "horizon": horizon,
    }
