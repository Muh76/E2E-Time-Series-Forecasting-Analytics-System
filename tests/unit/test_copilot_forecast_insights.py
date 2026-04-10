"""Unit tests for rule-based forecast Insight Copilot."""

from backend.app.services.copilot_forecast_insights import build_forecast_insights


def test_upward_trend_detected():
    y = [100.0, 102.0, 105.0, 110.0, 118.0, 130.0]
    out = build_forecast_insights([{"forecast": v} for v in y], {"mae": 1.2})
    assert "upward" in out["summary"].lower() or "upward" in out["insights"].lower()
    assert 0.0 <= out["confidence"] <= 1.0


def test_downward_trend():
    y = [200.0, 195.0, 188.0, 175.0, 160.0]
    out = build_forecast_insights(y, {})
    assert "downward" in out["insights"].lower()


def test_stable_flat_series():
    y = [50.0, 50.5, 49.8, 50.2, 50.0, 50.1]
    out = build_forecast_insights(y, {})
    assert "stable" in out["insights"].lower()


def test_high_volatility():
    y = [100.0, 180.0, 90.0, 200.0, 85.0, 190.0]
    out = build_forecast_insights(y, {})
    assert "high" in out["insights"].lower() and "volatility" in out["insights"].lower()


def test_step_anomaly():
    y = [10.0, 10.1, 10.0, 10.2, 50.0, 10.1, 10.0]
    out = build_forecast_insights(y, {})
    assert "anomal" in out["insights"].lower() or "step" in out["insights"].lower()


def test_too_few_points():
    out = build_forecast_insights([42.0], {"mape": 5.0})
    assert out["confidence"] < 0.5
    assert "two" in out["summary"].lower() or "enough" in out["summary"].lower()
