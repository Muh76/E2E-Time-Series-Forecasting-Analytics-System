"""
Tests for monitoring state disk persistence and restart recovery.

Covers:
- set_monitoring_state() writes to disk.
- initialize_monitoring_state() restores rolling_series from a previous snapshot.
- initialize_monitoring_state() restores last_dispatched from a previous snapshot.
- Missing or corrupt state file is handled gracefully.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.services import monitoring_service
from backend.app.services.monitoring_service import (
    get_monitoring_summary,
    initialize_monitoring_state,
    set_monitoring_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_monitoring_state():
    monitoring_service._monitoring_state = None
    yield
    monitoring_service._monitoring_state = None


@pytest.fixture()
def tmp_state_file(tmp_path):
    """Return a temporary path for monitoring_state.json."""
    return tmp_path / "monitoring_state.json"


@pytest.fixture()
def neutral_drift(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.monitoring_service.compute_feature_drift",
        lambda cfg=None: {
            "drift_detected": False,
            "overall_score": 0.0,
            "threshold": 0.25,
            "per_feature_scores": {},
        },
    )


@pytest.fixture()
def stub_metadata(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.monitoring_service.initialize_monitoring_state.__globals__"
        if False else "backend.app.services.model_loader.get_model_metadata",
        lambda: {
            "model_version": "v1",
            "trained_at": "2025-01-01T00:00:00Z",
            "validation_metrics": {"mae": 5.0, "rmse": 7.0, "mape": 8.0},
            "sample_size": 100,
        },
    )


# ---------------------------------------------------------------------------
# Write to disk
# ---------------------------------------------------------------------------

def test_set_monitoring_state_writes_to_disk(tmp_state_file):
    with patch("backend.app.services.monitoring_service.monitoring_state_json_path", return_value=tmp_state_file):
        state = {"performance": {"rolling_series": {"mae": [], "mape": []}}, "test": True}
        set_monitoring_state(state)
    assert tmp_state_file.exists()
    loaded = json.loads(tmp_state_file.read_text())
    assert loaded["test"] is True


# ---------------------------------------------------------------------------
# Round-trip: rolling_series restored after restart
# ---------------------------------------------------------------------------

def test_rolling_series_restored_after_restart(tmp_state_file, neutral_drift, monkeypatch):
    rolling_mae = [{"date": "2025-01-01", "value": 4.5}, {"date": "2025-01-02", "value": 5.1}]
    saved = {
        "performance": {"rolling_series": {"mae": rolling_mae, "mape": []}},
        "last_dispatched": {},
    }
    tmp_state_file.write_text(json.dumps(saved))

    monkeypatch.setattr(
        "backend.app.services.monitoring_service.monitoring_state_json_path",
        lambda: tmp_state_file,
    )
    monkeypatch.setattr(
        "backend.app.services.model_loader.get_model_metadata",
        lambda: {
            "model_version": "v1",
            "trained_at": "2025-01-01T00:00:00Z",
            "validation_metrics": {"mae": 5.0, "rmse": 7.0, "mape": 8.0},
            "sample_size": 100,
        },
    )

    initialize_monitoring_state()

    summary = get_monitoring_summary()
    restored_mae = summary["rolling_series"].get("mae", [])
    assert restored_mae == rolling_mae


# ---------------------------------------------------------------------------
# Round-trip: last_dispatched restored
# ---------------------------------------------------------------------------

def test_last_dispatched_restored_after_restart(tmp_state_file, neutral_drift, monkeypatch):
    last_dispatched = {"slack": {"mae": "2025-01-01T10:00:00Z"}}
    saved = {
        "performance": {"rolling_series": {"mae": [], "mape": []}},
        "last_dispatched": last_dispatched,
    }
    tmp_state_file.write_text(json.dumps(saved))

    monkeypatch.setattr(
        "backend.app.services.monitoring_service.monitoring_state_json_path",
        lambda: tmp_state_file,
    )
    monkeypatch.setattr(
        "backend.app.services.model_loader.get_model_metadata",
        lambda: {
            "model_version": "v1",
            "trained_at": "2025-01-01T00:00:00Z",
            "validation_metrics": {"mae": 5.0, "rmse": 7.0, "mape": 8.0},
            "sample_size": 100,
        },
    )

    initialize_monitoring_state()

    assert monitoring_service._monitoring_state is not None
    stored = monitoring_service._monitoring_state.get("last_dispatched") or {}
    assert stored.get("slack", {}).get("mae") == "2025-01-01T10:00:00Z"


# ---------------------------------------------------------------------------
# Missing / corrupt state file — graceful degradation
# ---------------------------------------------------------------------------

def test_missing_state_file_does_not_raise(tmp_state_file, neutral_drift, monkeypatch):
    assert not tmp_state_file.exists()

    monkeypatch.setattr(
        "backend.app.services.monitoring_service.monitoring_state_json_path",
        lambda: tmp_state_file,
    )
    monkeypatch.setattr(
        "backend.app.services.model_loader.get_model_metadata",
        lambda: {
            "model_version": "v1",
            "trained_at": "2025-01-01T00:00:00Z",
            "validation_metrics": {"mae": 5.0, "rmse": 7.0, "mape": 8.0},
            "sample_size": 100,
        },
    )

    initialize_monitoring_state()  # must not raise
    assert monitoring_service._monitoring_state is not None


def test_corrupt_state_file_does_not_raise(tmp_state_file, neutral_drift, monkeypatch):
    tmp_state_file.write_text("this is not valid json {{{{")

    monkeypatch.setattr(
        "backend.app.services.monitoring_service.monitoring_state_json_path",
        lambda: tmp_state_file,
    )
    monkeypatch.setattr(
        "backend.app.services.model_loader.get_model_metadata",
        lambda: {
            "model_version": "v1",
            "trained_at": "2025-01-01T00:00:00Z",
            "validation_metrics": {"mae": 5.0, "rmse": 7.0, "mape": 8.0},
            "sample_size": 100,
        },
    )

    initialize_monitoring_state()  # must not raise
    assert monitoring_service._monitoring_state is not None
