"""
API client utilities for the Streamlit frontend.

Base URL: ``API_URL`` environment variable (default ``http://127.0.0.1:8001``),
then ``frontend.api_base_url`` from config, then ``API_BASE_URL``.
Calls the real FastAPI backend; failures raise ``requests.RequestException``
(callers should catch and show UI errors — no silent mock data).
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import yaml

# frontend/components/api.py -> parent.parent = frontend, parent.parent.parent = project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_TIMEOUT = 10
_FORECAST_TIMEOUT = 120

# Backend base URL when ``API_URL`` is unset (see ``get_api_base_url`` for full resolution order).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")


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
    Return backend API base URL.

    Priority: ``API_URL`` env, ``frontend.api_base_url`` in config, ``API_BASE_URL`` env,
    then default ``http://127.0.0.1:8001``.
    """
    env_url = os.environ.get("API_URL")
    if env_url:
        return str(env_url).rstrip("/")
    config = _load_config()
    url = config.get("frontend", {}).get("api_base_url")
    if url:
        return str(url).rstrip("/")
    legacy = os.environ.get("API_BASE_URL")
    if legacy:
        return str(legacy).rstrip("/")
    return str(API_URL).rstrip("/")


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
    resp = requests.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def normalize_monitoring_summary(raw_summary: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize monitoring summary so frontend always receives a consistent shape.
    - Extracts model_name and model_version from raw_summary["model_info"].
    - Flattens alerts and thresholds into top-level keys with fallback defaults.
    - Preserves performance, drift, rolling_series, overall_status.
    """
    if not raw_summary:
        raw_summary = {}

    model_info = raw_summary.get("model_info") or {}
    model_name = model_info.get("model_name") or raw_summary.get("model_name") or "lightgbm"
    model_version = (
        model_info.get("model_version") or model_info.get("version") or raw_summary.get("model_version") or ""
    )

    # Flatten alerts to top-level with defaults
    alerts = raw_summary.get("alerts")
    if alerts is None:
        alerts = (raw_summary.get("monitoring") or {}).get("alerts") or {}
    if not isinstance(alerts, dict):
        alerts = {}
    alerts = {
        "mae": alerts.get("mae", False),
        "mape": alerts.get("mape", False),
        "drift": alerts.get("drift", False),
    }

    # Flatten thresholds to top-level with defaults
    thresholds = raw_summary.get("thresholds")
    if thresholds is None:
        thresholds = (raw_summary.get("monitoring") or {}).get("thresholds") or {}
    if not isinstance(thresholds, dict):
        thresholds = {}
    thresholds = {
        "mae_alert": thresholds.get("mae_alert", 15.0),
        "mape_alert": thresholds.get("mape_alert", 0.20),
        "drift_threshold": thresholds.get("drift_threshold", 0.25),
    }

    # Preserve these with fallbacks (API returns flat performance.*)
    performance = dict(raw_summary.get("performance") or {})
    drift = dict(raw_summary.get("drift") or {})
    rolling_series = raw_summary.get("rolling_series") or {}
    overall_status = raw_summary.get("overall_status") or raw_summary.get("status") or ""
    recent_activity = raw_summary.get("recent_activity") or {}

    # Build normalized summary; preserve other keys (e.g. as_of, pipeline) for compatibility
    normalized: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "performance": performance,
        "drift": drift,
        "rolling_series": rolling_series,
        "overall_status": overall_status,
        "alerts": alerts,
        "thresholds": thresholds,
        "recent_activity": recent_activity,
    }
    skip = set(normalized.keys()) | {"model_info", "status"}
    for k, v in raw_summary.items():
        if k not in skip:
            normalized[k] = v
    return normalized


@st.cache_data(ttl=15)
def _cached_monitoring_summary_api(
    model_version: str | None,
    since: str | None,
) -> dict[str, Any]:
    """Cached API call; only successful responses are cached."""
    params: dict[str, str] = {}
    if model_version:
        params["model_version"] = model_version
    if since:
        params["since"] = since
    url = api_url("/api/v1/monitoring/summary")
    resp = requests.get(url, params=params or None, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return normalize_monitoring_summary(resp.json())


def get_monitoring_summary(
    model_version: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Fetch monitoring summary from the API. Raises ``requests.RequestException`` on failure."""
    return _cached_monitoring_summary_api(model_version, since)


def forecast_store(store_id: int, horizon: int) -> dict[str, Any]:
    """
    POST /api/v1/forecast/store — store-level forecast.

    Returns the raw JSON response on success.
    Raises requests.HTTPError with the response attached on failure
    (caller should inspect response.status_code and response.json()).
    """
    url = api_url("/api/v1/forecast/store")
    resp = requests.post(
        url,
        json={"store_id": store_id, "horizon": horizon},
        timeout=_FORECAST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_forecast_evaluation_metrics(store_id: int | None = None) -> dict[str, Any]:
    """
    GET /api/v1/metrics — last forecast vs ground truth for the given store (optional).

    Call after ``forecast_store`` so the backend has recorded the forecast.
    """
    url = api_url("/api/v1/metrics")
    params: dict[str, int] = {}
    if store_id is not None:
        params["store_id"] = int(store_id)
    resp = requests.get(url, params=params or None, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def copilot_forecast_insights(
    forecast: list[Any],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST /api/v1/copilot — rule-based summary, insights, confidence."""
    url = api_url("/api/v1/copilot")
    payload = {"forecast": forecast, "metrics": metrics or {}}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def predict_store(store_id: int, horizon: int, *, include_insights: bool = False) -> dict[str, Any]:
    """
    POST /api/v1/predict — forecast, aligned metrics, and optional rule-based copilot.

    Returns ``forecast`` (list), ``metrics`` (dict), and ``copilot`` (dict or null).
    """
    url = api_url("/api/v1/predict")
    resp = requests.post(
        url,
        params={"include_insights": str(include_insights).lower()},
        json={"store_id": store_id, "horizon": horizon},
        timeout=_FORECAST_TIMEOUT if not include_insights else max(_FORECAST_TIMEOUT, 60),
    )
    resp.raise_for_status()
    return resp.json()


def backtest_store(store_id: int, horizon: int, n_splits: int) -> dict[str, Any]:
    """
    POST /api/v1/backtest/store — rolling-origin backtesting.

    Returns the raw JSON response on success.
    Raises requests.HTTPError with the response attached on failure.
    """
    url = api_url("/api/v1/backtest/store")
    resp = requests.post(
        url,
        json={"store_id": store_id, "horizon": horizon, "n_splits": n_splits},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_model_info() -> dict[str, Any]:
    """GET /api/v1/model/info — model metadata."""
    url = api_url("/api/v1/model/info")
    resp = requests.get(url, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_api_error(exc: requests.HTTPError) -> list[dict[str, str]]:
    """
    Parse structured validation errors from a 422 response.

    Returns a list of {"field": ..., "message": ...} dicts.
    Falls back to a single generic error if the response body is not structured.
    """
    resp = exc.response
    if resp is None:
        return [{"field": "unknown", "message": str(exc)}]
    try:
        body = resp.json()
    except Exception:
        return [{"field": "unknown", "message": resp.text or str(exc)}]

    detail = body.get("detail")
    if isinstance(detail, list):
        return [{"field": item.get("field", "unknown"), "message": item.get("message", str(item))} for item in detail]
    if isinstance(detail, str):
        return [{"field": "unknown", "message": detail}]
    return [{"field": "unknown", "message": str(body)}]


def describe_request_error(exc: BaseException) -> str:
    """Single user-facing message for connection, timeout, or HTTP errors."""
    if isinstance(exc, requests.Timeout):
        return "Request timed out. Check that the API is running and reachable."
    if isinstance(exc, requests.ConnectionError):
        return (
            "Cannot connect to the API. Set environment variable API_URL "
            "(default http://127.0.0.1:8001) and ensure the FastAPI server is running."
        )
    if isinstance(exc, requests.HTTPError):
        parts = parse_api_error(exc)
        if parts:
            return "; ".join(p.get("message", "") for p in parts)
    return str(exc) or "Request failed."


def _context_hash(context: dict[str, Any] | None) -> str:
    """Stable string for cache key from context dict."""
    if not context:
        return ""
    return json.dumps(context, sort_keys=True, default=str)


@st.cache_data(ttl=120)
def _cached_copilot_explain_api(
    query: str,
    context_serialized: str,
    options_serialized: str,
) -> dict[str, Any]:
    """Cached API call; only successful responses are cached. Cache key includes query + context hash."""
    context: dict[str, Any] | None = json.loads(context_serialized) if context_serialized else None
    options: dict[str, Any] | None = json.loads(options_serialized) if options_serialized else None
    payload: dict[str, Any] = {"query": query or "What is the current model health?"}
    if context:
        payload["context"] = context
    if options:
        payload["options"] = options
    url = api_url("/api/v1/copilot/explain")
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def copilot_explain(
    query: str,
    context: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Request explanation from copilot. POST /api/v1/copilot/explain.
    Raises ``requests.RequestException`` on failure.
    """
    context_serialized = _context_hash(context)
    options_serialized = _context_hash(options) if options else ""
    return _cached_copilot_explain_api(query, context_serialized, options_serialized)


def get_monitoring_series(
    metric: str,
    start_date: str,
    end_date: str,
    model_version: str | None = None,
    granularity: str = "daily",
) -> dict[str, Any]:
    """
    Fetch monitoring time series (rolling MAE, MAPE, etc).
    GET /api/v1/monitoring/series. Raises if the route is unavailable.
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
    resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_metrics(
    series_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metrics: list[str] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """
    Fetch historical metrics. GET /api/v1/metrics/historical.
    Raises if the route is unavailable.
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
    resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30)
def _cached_forecast_vs_actual_api(
    entity_id: str | None,
    horizon: int,
    include_baseline: bool,
) -> dict[str, Any]:
    """Cached API call; only successful responses are cached."""
    params: dict[str, str | int | bool] = {"horizon": horizon, "include_baseline": include_baseline}
    if entity_id:
        params["entity_id"] = entity_id
    url = api_url("/api/v1/forecasts/vs-actual")
    resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_forecast_vs_actual(
    entity_id: str | None = None,
    horizon: int = 14,
    include_baseline: bool = False,
) -> dict[str, Any]:
    """
    Fetch forecast vs actual data ready for plotting.
    GET /api/v1/forecasts/vs-actual. Raises if the route is unavailable.
    """
    return _cached_forecast_vs_actual_api(entity_id, horizon, include_baseline)
