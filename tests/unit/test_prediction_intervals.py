"""
Unit tests for residual-based prediction interval computation.

Covers:
- Correct confidence bounds at 95% level (z=1.96)
- Zero residual_std produces None bounds
- Missing / negative residual_std produces None bounds
- Multiple forecast points computed independently
"""

import math

import pytest

Z_95 = 1.96


def compute_intervals(
    forecasts: list[dict],
    residual_std: float | None,
) -> list[dict]:
    """
    Replicate the interval logic from backend/app/api/v1/forecast.py.

    Mutates each forecast dict in-place (same as the endpoint) and returns
    the list for assertion convenience.
    """
    if residual_std is not None and residual_std > 0:
        for f in forecasts:
            point = f["forecast"]
            f["confidence_low"] = round(point - Z_95 * residual_std, 2)
            f["confidence_high"] = round(point + Z_95 * residual_std, 2)
    else:
        for f in forecasts:
            f["confidence_low"] = None
            f["confidence_high"] = None
    return forecasts


class TestPredictionIntervals:
    """Suite for the ±1.96·σ prediction interval formula."""

    def test_basic_interval(self):
        forecasts = [{"date": "2025-01-01", "forecast": 1000.0}]
        result = compute_intervals(forecasts, residual_std=100.0)

        assert result[0]["confidence_low"] == pytest.approx(1000.0 - 1.96 * 100.0, abs=0.01)
        assert result[0]["confidence_high"] == pytest.approx(1000.0 + 1.96 * 100.0, abs=0.01)

    def test_interval_symmetry(self):
        forecasts = [{"date": "2025-01-01", "forecast": 500.0}]
        compute_intervals(forecasts, residual_std=50.0)

        mid = (forecasts[0]["confidence_low"] + forecasts[0]["confidence_high"]) / 2
        assert mid == pytest.approx(500.0, abs=0.01)

    def test_interval_width_scales_with_std(self):
        f1 = [{"date": "2025-01-01", "forecast": 500.0}]
        f2 = [{"date": "2025-01-01", "forecast": 500.0}]
        compute_intervals(f1, residual_std=10.0)
        compute_intervals(f2, residual_std=100.0)

        width_narrow = f1[0]["confidence_high"] - f1[0]["confidence_low"]
        width_wide = f2[0]["confidence_high"] - f2[0]["confidence_low"]
        assert width_wide == pytest.approx(width_narrow * 10.0, abs=0.1)

    def test_multiple_forecasts(self):
        forecasts = [
            {"date": "2025-01-01", "forecast": 100.0},
            {"date": "2025-01-02", "forecast": 200.0},
            {"date": "2025-01-03", "forecast": 300.0},
        ]
        compute_intervals(forecasts, residual_std=50.0)

        for f in forecasts:
            expected_low = round(f["forecast"] - 1.96 * 50.0, 2)
            expected_high = round(f["forecast"] + 1.96 * 50.0, 2)
            assert f["confidence_low"] == expected_low
            assert f["confidence_high"] == expected_high

    def test_zero_residual_std_returns_none(self):
        forecasts = [{"date": "2025-01-01", "forecast": 1000.0}]
        compute_intervals(forecasts, residual_std=0.0)

        assert forecasts[0]["confidence_low"] is None
        assert forecasts[0]["confidence_high"] is None

    def test_none_residual_std_returns_none(self):
        forecasts = [{"date": "2025-01-01", "forecast": 1000.0}]
        compute_intervals(forecasts, residual_std=None)

        assert forecasts[0]["confidence_low"] is None
        assert forecasts[0]["confidence_high"] is None

    def test_negative_residual_std_returns_none(self):
        forecasts = [{"date": "2025-01-01", "forecast": 1000.0}]
        compute_intervals(forecasts, residual_std=-5.0)

        assert forecasts[0]["confidence_low"] is None
        assert forecasts[0]["confidence_high"] is None

    def test_small_residual_std(self):
        forecasts = [{"date": "2025-01-01", "forecast": 1000.0}]
        compute_intervals(forecasts, residual_std=0.01)

        assert forecasts[0]["confidence_low"] is not None
        assert forecasts[0]["confidence_high"] is not None
        assert math.isclose(
            forecasts[0]["confidence_high"] - forecasts[0]["confidence_low"],
            2 * 1.96 * 0.01,
            abs_tol=0.01,
        )

    def test_large_forecast_value(self):
        forecasts = [{"date": "2025-01-01", "forecast": 1_000_000.0}]
        compute_intervals(forecasts, residual_std=500.0)

        assert forecasts[0]["confidence_low"] == pytest.approx(1_000_000.0 - 1.96 * 500.0, abs=0.01)
        assert forecasts[0]["confidence_high"] == pytest.approx(1_000_000.0 + 1.96 * 500.0, abs=0.01)

    def test_negative_forecast_value(self):
        forecasts = [{"date": "2025-01-01", "forecast": -200.0}]
        compute_intervals(forecasts, residual_std=50.0)

        assert forecasts[0]["confidence_low"] == pytest.approx(-200.0 - 1.96 * 50.0, abs=0.01)
        assert forecasts[0]["confidence_high"] == pytest.approx(-200.0 + 1.96 * 50.0, abs=0.01)
        assert forecasts[0]["confidence_low"] < forecasts[0]["confidence_high"]
