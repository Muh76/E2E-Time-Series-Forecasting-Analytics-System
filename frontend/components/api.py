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

# Project root = frontend/components/ -> frontend/ -> project root
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
    """Return backend API base URL from config (e.g. http://localhost:8000)."""
    config = _load_config()
    return config.get("frontend", {}).get("api_base_url", "http://localhost:8000")


def api_url(path: str) -> str:
    """Build full API URL from a path (e.g. /api/v1/health/live)."""
    base = get_api_base_url().rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{base}{path}"


def _is_backend_unavailable(exc: BaseException) -> bool:
    """Return True if the exception indicates backend is unreachable."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.RequestException):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code >= 500:
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

    Args:
        series_ids: List of series identifiers. Default: ["series_001"].
        horizon_steps: Number of steps to forecast.
        frequency: Series frequency (D, H, W).
        model_version: Model version; None uses backend default.
        return_interval: Whether to request prediction intervals.

    Returns:
        Dict with job_id, status, forecasts, generated_at.
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
    """Return mock forecast data when backend is unavailable."""
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
    """
    Fetch monitoring summary from the backend.

    GET /api/v1/monitoring/summary. Returns mock data if backend is unavailable.

    Args:
        model_version: Optional filter by model version.
        since: Optional ISO 8601 datetime; only metrics after this time.

    Returns:
        Dict with model_version, as_of, performance, drift, pipeline.
    """
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
    """Return mock monitoring summary when backend is unavailable."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "model_version": "v1.0.0",
        "as_of": now,
        "performance": {
            "mae": 2.34,
            "rmse": 3.01,
            "mape": 0.042,
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


def get_metrics(
    series_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metrics: list[str] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """
    Fetch historical metrics from the backend.

    GET /api/v1/metrics/historical. Returns mock data if backend is unavailable.

    Args:
        series_ids: List of series identifiers. Default: ["series_001"].
        start_date: Start of range (ISO 8601). Default: 7 days ago.
        end_date: End of range (ISO 8601). Default: today.
        metrics: Comma-separated or list: actual, forecast, error. Default: all.
        model_version: Optional filter by model version.

    Returns:
        Dict with data (list of {series_id, date, actual, forecast, error}) and meta.
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
    """Return mock historical metrics when backend is unavailable."""
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
        "meta": {
            "series_ids": series_ids,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(data),
        },
    }
