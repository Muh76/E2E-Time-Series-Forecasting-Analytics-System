"""
Tests for API endpoints via FastAPI TestClient.

Covers:
- GET /health/live: liveness probe always returns {"status": "ok"}.
- GET /api/v1/monitoring/summary: returns 200 with expected top-level keys.
- GET /api/v1/monitoring/metrics: returns 200 with model_version key.
- POST /api/v1/forecast/store: horizon out of range (0, 61) returns 422.
- POST /api/v1/forecast/store: missing required field returns 422.
- POST /api/v1/forecast/store: model not loaded returns 503.
- POST /api/v1/backtest/store: n_splits out of range (0, 21) returns 422.
- POST /api/v1/backtest/store: invalid horizon returns 422.
- POST /api/v1/backtest/store: model not loaded returns 503.
- Structured 422 error shape: detail is a list; each item has field/message/type.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


_MONITORING_STUB = {
    "model_version": "v-test",
    "as_of": "2025-01-01T00:00:00Z",
    "performance": {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "sample_size": 0},
    "drift": {
        "status": "ok",
        "last_checked": "2025-01-01T00:00:00Z",
        "indicators": [],
        "overall_score": None,
        "threshold": 0.25,
        "per_feature_scores": {},
    },
    "pipeline": {"last_training": None, "last_etl": None, "status": "unknown"},
    "rolling_series": {"mae": [], "mape": []},
    "alerts": {"mae": False, "mape": False, "drift": False},
    "thresholds": {"mae_alert": 15.0, "mape_alert": 0.20, "drift_threshold": 0.25},
    "overall_status": "unknown",
    "recent_activity": {},
}

_METRICS_STUB = {
    "model_version": "v-test",
    "primary_metrics": {"mae": None, "rmse": None, "mape": None, "sample_size": 0},
    "validation_holdout": {"mae": None, "rmse": None, "mape": None},
    "source": "none",
    "as_of": "2025-01-01T00:00:00Z",
}


@pytest.fixture(autouse=True)
def _patch_validator(monkeypatch):
    noop = lambda v: v
    monkeypatch.setattr("backend.app.api.v1.validators.validate_store_id", noop)
    monkeypatch.setattr("backend.app.api.v1.forecast.validate_store_id", noop)
    monkeypatch.setattr("backend.app.api.v1.backtest.validate_store_id", noop)


@pytest.fixture(autouse=True)
def _patch_monitoring(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.v1.monitoring.get_monitoring_summary",
        lambda: _MONITORING_STUB,
    )
    monkeypatch.setattr(
        "backend.app.api.v1.monitoring.get_evaluation_snapshot",
        lambda: _METRICS_STUB,
    )


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_live_returns_ok(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_monitoring_summary_returns_200(client):
    resp = client.get("/api/v1/monitoring/summary")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_version", "performance", "drift", "pipeline", "alerts"):
        assert key in body, f"missing key: {key}"


def test_monitoring_metrics_returns_200(client):
    resp = client.get("/api/v1/monitoring/metrics")
    assert resp.status_code == 200
    assert "model_version" in resp.json()


def test_forecast_horizon_zero_returns_422(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 0})
    assert resp.status_code == 422


def test_forecast_horizon_61_returns_422(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 61})
    assert resp.status_code == 422


def test_forecast_horizon_boundary_low_valid(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 1})
    assert resp.status_code in (200, 503)


def test_forecast_horizon_boundary_high_valid(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 60})
    assert resp.status_code in (200, 503)


def test_forecast_missing_horizon_returns_422(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1})
    assert resp.status_code == 422


def test_forecast_missing_store_id_returns_422(client):
    resp = client.post("/api/v1/forecast/store", json={"horizon": 7})
    assert resp.status_code == 422


def test_forecast_model_not_loaded_returns_503(client):
    app.state.primary_model = None
    with patch("backend.app.api.v1.forecast.load_primary_model",
               side_effect=RuntimeError("artifact not found")):
        resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 7})
    assert resp.status_code == 503


def test_backtest_n_splits_zero_returns_422(client):
    resp = client.post(
        "/api/v1/backtest/store",
        json={"store_id": 1, "horizon": 7, "n_splits": 0},
    )
    assert resp.status_code == 422


def test_backtest_n_splits_21_returns_422(client):
    resp = client.post(
        "/api/v1/backtest/store",
        json={"store_id": 1, "horizon": 7, "n_splits": 21},
    )
    assert resp.status_code == 422


def test_backtest_horizon_zero_returns_422(client):
    resp = client.post(
        "/api/v1/backtest/store",
        json={"store_id": 1, "horizon": 0, "n_splits": 3},
    )
    assert resp.status_code == 422


def test_backtest_boundary_valid(client):
    for n in (1, 20):
        resp = client.post(
            "/api/v1/backtest/store",
            json={"store_id": 1, "horizon": 7, "n_splits": n},
        )
        assert resp.status_code in (200, 503), f"n_splits={n}: got {resp.status_code}"


def test_backtest_model_not_loaded_returns_503(client):
    app.state.primary_model = None
    with patch("backend.app.api.v1.backtest.load_primary_model",
               side_effect=RuntimeError("artifact not found")):
        resp = client.post(
            "/api/v1/backtest/store",
            json={"store_id": 1, "horizon": 7, "n_splits": 3},
        )
    assert resp.status_code == 503


def test_422_detail_is_list(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 0})
    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], list)
    assert len(resp.json()["detail"]) > 0


def test_422_detail_item_has_field_message_type(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 0})
    item = resp.json()["detail"][0]
    assert "field" in item
    assert "message" in item
    assert "type" in item


def test_422_input_field_present(client):
    resp = client.post("/api/v1/forecast/store", json={"store_id": 1, "horizon": 0})
    item = resp.json()["detail"][0]
    assert "input" in item
