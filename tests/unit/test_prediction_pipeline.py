"""
Tests for backend.app.services.prediction_pipeline.

Covers:
- execute_store_forecast: attaches 95% CI (z=1.96) when residual_std > 0.
- execute_store_forecast: sets CI to None when residual_std == 0.
- execute_store_forecast: sets CI to None when model metadata is unavailable.
- Exact CI formula and rounding (2 decimal places).
- Multiple forecast points each get independent CI values.
- Return shape preserves date and forecast value.
- metrics_context_for_copilot: extracts known fields for "ok" status.
- metrics_context_for_copilot: unwraps nested "current" key.
- metrics_context_for_copilot: returns status + message for non-ok results.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def _make_forecasts(*values):
    return [{"date": f"2025-01-{i+1:02d}", "forecast": v} for i, v in enumerate(values)]


@contextmanager
def _patch_pipeline(forecasts, residual_std):
    def fake_metadata():
        if residual_std is None:
            raise RuntimeError("no metadata available")
        return {"residual_std": residual_std, "model_version": "v1"}

    with patch("backend.app.services.prediction_pipeline.forecast_store",
               return_value=[dict(f) for f in forecasts]):
        with patch("backend.app.services.prediction_pipeline.get_model_metadata",
                   side_effect=fake_metadata):
            with patch("backend.app.services.prediction_pipeline.record_forecast_for_evaluation"):
                with patch("backend.app.services.prediction_pipeline.record_forecast_activity"):
                    from backend.app.services.prediction_pipeline import execute_store_forecast
                    yield execute_store_forecast


def _run(residual_std, values=(1000.0, 1200.0)):
    forecasts = _make_forecasts(*values)
    with _patch_pipeline(forecasts, residual_std) as fn:
        return fn(store_id=1, horizon=len(forecasts), model=MagicMock(), feature_columns=[])


def test_ci_attached_when_residual_std_positive():
    result = _run(500.0)
    for row in result:
        assert row["confidence_low"] is not None
        assert row["confidence_high"] is not None


def test_ci_is_none_when_residual_std_zero():
    result = _run(0.0)
    for row in result:
        assert row["confidence_low"] is None
        assert row["confidence_high"] is None


def test_ci_is_none_when_metadata_unavailable():
    result = _run(None)
    for row in result:
        assert row["confidence_low"] is None
        assert row["confidence_high"] is None


def test_ci_formula_single_point():
    result = _run(200.0, (1000.0,))
    row = result[0]
    assert row["confidence_low"] == round(1000.0 - 1.96 * 200.0, 2)
    assert row["confidence_high"] == round(1000.0 + 1.96 * 200.0, 2)


def test_ci_formula_multiple_points_each_correct():
    values = (500.0, 750.0, 1100.0)
    result = _run(100.0, values)
    z, std = 1.96, 100.0
    for row, v in zip(result, values):
        assert row["confidence_low"] == round(v - z * std, 2)
        assert row["confidence_high"] == round(v + z * std, 2)


def test_ci_low_below_forecast_and_high_above():
    result = _run(300.0, (800.0,))
    row = result[0]
    assert row["confidence_low"] < row["forecast"]
    assert row["confidence_high"] > row["forecast"]


def test_ci_symmetry_around_forecast():
    result = _run(150.0, (1000.0,))
    row = result[0]
    assert abs(row["forecast"] - row["confidence_low"]) == pytest.approx(
        abs(row["confidence_high"] - row["forecast"]), abs=0.01
    )


def test_returns_list_of_dicts():
    result = _run(300.0)
    assert isinstance(result, list)
    assert all(isinstance(r, dict) for r in result)


def test_preserves_date_field():
    result = _run(300.0)
    assert result[0]["date"] == "2025-01-01"
    assert result[1]["date"] == "2025-01-02"


def test_preserves_forecast_value():
    result = _run(300.0, (999.9,))
    assert result[0]["forecast"] == 999.9


def test_length_matches_input():
    result = _run(100.0, (1.0, 2.0, 3.0, 4.0, 5.0))
    assert len(result) == 5


def test_metrics_context_ok_status_extracts_known_fields():
    from backend.app.services.prediction_pipeline import metrics_context_for_copilot
    eval_result = {
        "status": "ok",
        "mae": 4.5,
        "rmse": 6.1,
        "mape": 8.2,
        "n_samples": 10,
        "evaluated_dates": ["2025-01-01"],
        "extra_field": "should_not_appear",
    }
    out = metrics_context_for_copilot(eval_result)
    assert out["mae"] == 4.5
    assert out["rmse"] == 6.1
    assert out["mape"] == 8.2
    assert out["n_samples"] == 10
    assert "extra_field" not in out
    assert "status" not in out


def test_metrics_context_unwraps_nested_current_key():
    from backend.app.services.prediction_pipeline import metrics_context_for_copilot
    eval_result = {
        "current": {
            "status": "ok",
            "mae": 2.0,
            "rmse": 3.0,
            "mape": 4.0,
            "n_samples": 5,
            "evaluated_dates": [],
        }
    }
    out = metrics_context_for_copilot(eval_result)
    assert out["mae"] == 2.0
    assert out["rmse"] == 3.0


def test_metrics_context_non_ok_returns_status_and_message():
    from backend.app.services.prediction_pipeline import metrics_context_for_copilot
    eval_result = {"status": "insufficient_data", "message": "need more points"}
    out = metrics_context_for_copilot(eval_result)
    assert out["status"] == "insufficient_data"
    assert out["message"] == "need more points"


def test_metrics_context_missing_optional_fields_are_absent():
    from backend.app.services.prediction_pipeline import metrics_context_for_copilot
    eval_result = {"status": "ok", "mae": 1.0}
    out = metrics_context_for_copilot(eval_result)
    assert "rmse" not in out
    assert "mape" not in out
