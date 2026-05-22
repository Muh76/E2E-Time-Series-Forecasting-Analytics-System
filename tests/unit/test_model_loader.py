"""
Tests for backend.app.services.model_loader.

Uses tmp_path and monkeypatching to exercise the full public surface without
requiring real .joblib or JSON artifacts on disk.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.app.services import model_loader as _ml


# ---------------------------------------------------------------------------
# Reset singletons between tests so load / get pairs are tested in isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    _ml._primary_model = None
    _ml._baseline_model = None
    _ml._feature_columns = None
    _ml._metrics = None
    yield
    _ml._primary_model = None
    _ml._baseline_model = None
    _ml._feature_columns = None
    _ml._metrics = None


@pytest.fixture()
def artifacts_dir(tmp_path, monkeypatch):
    """Point artifacts_models_dir() at a temp directory for each test."""
    monkeypatch.setattr("backend.app.services.model_loader.artifacts_models_dir", lambda: tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# load_primary_model
# ---------------------------------------------------------------------------

def test_load_primary_model_missing_raises(artifacts_dir):
    from backend.app.services.model_loader import load_primary_model
    with pytest.raises(RuntimeError, match="Primary model artifact not found"):
        load_primary_model()


def test_load_primary_model_loads_and_caches(artifacts_dir):
    from backend.app.services.model_loader import load_primary_model
    (artifacts_dir / "primary_lightgbm.joblib").write_bytes(b"dummy")
    fake = MagicMock()
    with patch("backend.app.services.model_loader.joblib.load", return_value=fake):
        result = load_primary_model()
    assert result is fake
    assert _ml._primary_model is fake


# ---------------------------------------------------------------------------
# load_baseline_model
# ---------------------------------------------------------------------------

def test_load_baseline_model_missing_raises(artifacts_dir):
    from backend.app.services.model_loader import load_baseline_model
    with pytest.raises(RuntimeError, match="Baseline model artifact not found"):
        load_baseline_model()


def test_load_baseline_model_loads_and_caches(artifacts_dir):
    from backend.app.services.model_loader import load_baseline_model
    (artifacts_dir / "baseline_seasonal_naive.joblib").write_bytes(b"dummy")
    fake = MagicMock()
    with patch("backend.app.services.model_loader.joblib.load", return_value=fake):
        result = load_baseline_model()
    assert result is fake
    assert _ml._baseline_model is fake


# ---------------------------------------------------------------------------
# load_feature_columns
# ---------------------------------------------------------------------------

def test_load_feature_columns_missing_raises(artifacts_dir):
    from backend.app.services.model_loader import load_feature_columns
    with pytest.raises(RuntimeError, match="Feature columns artifact not found"):
        load_feature_columns()


def test_load_feature_columns_reads_json_and_caches(artifacts_dir):
    from backend.app.services.model_loader import load_feature_columns
    cols = ["lag_1", "lag_7", "rolling_mean_7", "day_of_week"]
    (artifacts_dir / "feature_columns.json").write_text(json.dumps(cols))
    result = load_feature_columns()
    assert result == cols
    assert _ml._feature_columns == cols


def test_load_feature_columns_preserves_order(artifacts_dir):
    from backend.app.services.model_loader import load_feature_columns
    cols = ["z_feature", "a_feature", "m_feature"]
    (artifacts_dir / "feature_columns.json").write_text(json.dumps(cols))
    assert load_feature_columns() == cols


# ---------------------------------------------------------------------------
# get_primary_model / get_baseline_model — error when not yet loaded
# ---------------------------------------------------------------------------

def test_get_primary_model_raises_when_not_loaded():
    from backend.app.services.model_loader import get_primary_model
    with pytest.raises(RuntimeError, match="Primary model has not been loaded"):
        get_primary_model()


def test_get_primary_model_returns_cached():
    fake = MagicMock()
    _ml._primary_model = fake
    from backend.app.services.model_loader import get_primary_model
    assert get_primary_model() is fake


def test_get_baseline_model_raises_when_not_loaded():
    from backend.app.services.model_loader import get_baseline_model
    with pytest.raises(RuntimeError, match="Baseline model has not been loaded"):
        get_baseline_model()


def test_get_baseline_model_returns_cached():
    fake = MagicMock()
    _ml._baseline_model = fake
    from backend.app.services.model_loader import get_baseline_model
    assert get_baseline_model() is fake


# ---------------------------------------------------------------------------
# get_model_metadata
# ---------------------------------------------------------------------------

def test_get_model_metadata_missing_raises(artifacts_dir):
    from backend.app.services.model_loader import get_model_metadata
    with pytest.raises(RuntimeError, match="Model metadata not found"):
        get_model_metadata()


def test_get_model_metadata_reads_json(artifacts_dir):
    from backend.app.services.model_loader import get_model_metadata
    meta = {
        "model_version": "v3",
        "trained_at": "2025-06-01T00:00:00Z",
        "max_lag": 14,
        "residual_std": 320.5,
    }
    (artifacts_dir / "model_metadata.json").write_text(json.dumps(meta))
    result = get_model_metadata()
    assert result["model_version"] == "v3"
    assert result["residual_std"] == 320.5


def test_get_model_metadata_re_reads_on_each_call(artifacts_dir):
    """Metadata is not cached — each call reads from disk for freshness."""
    from backend.app.services.model_loader import get_model_metadata
    path = artifacts_dir / "model_metadata.json"
    path.write_text(json.dumps({"model_version": "v1"}))
    assert get_model_metadata()["model_version"] == "v1"
    path.write_text(json.dumps({"model_version": "v2"}))
    assert get_model_metadata()["model_version"] == "v2"


# ---------------------------------------------------------------------------
# get_metrics — lazy load with caching
# ---------------------------------------------------------------------------

def test_get_metrics_missing_raises(artifacts_dir):
    from backend.app.services.model_loader import get_metrics
    with pytest.raises(RuntimeError, match="Metrics artifact not found"):
        get_metrics()


def test_get_metrics_reads_and_caches(artifacts_dir):
    from backend.app.services.model_loader import get_metrics
    m = {"primary": {"mae": 3.5, "rmse": 5.1}, "baseline": {"mae": 9.0}}
    (artifacts_dir / "metrics.json").write_text(json.dumps(m))
    result = get_metrics()
    assert result["primary"]["mae"] == 3.5
    # Overwrite the file — cached result should still be returned
    (artifacts_dir / "metrics.json").write_text(json.dumps({"primary": {"mae": 999}}))
    assert get_metrics()["primary"]["mae"] == 3.5
