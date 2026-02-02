"""
Unit tests for PerformanceMonitor.

- Rolling window computation correctness
- Alert triggers when MAE/MAPE exceed thresholds
- No alert when metrics are within bounds
- Deterministic output for same inputs
- Works with and without entity_ids
"""

import numpy as np
import pandas as pd
import pytest

from models.monitoring.performance import PerformanceMonitor


# ---------------------------------------------------------------------------
# Fixtures: synthetic arrays with known outcomes
# ---------------------------------------------------------------------------


@pytest.fixture
def ten_days_perfect():
    """10 days, perfect predictions. Rolling MAE=0, MAPE=0 for last 3 days."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    y_pred = y_true.copy()
    return y_true, y_pred, dates


@pytest.fixture
def ten_days_known_errors():
    """10 days; last 3 days have errors: true=[8,9,10], pred=[8,9,12] -> MAE=1, MAPE=10/9 %."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0])  # error 2 on last day
    return y_true, y_pred, dates


@pytest.fixture
def ten_days_high_mae():
    """Last 3 days: large errors so rolling MAE and MAPE are high."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    y_true = np.array([1.0] * 10)
    y_pred = np.array([1.0] * 7 + [10.0, 10.0, 10.0])  # last 3: pred=10, true=1 -> MAE=9, MAPE high
    return y_true, y_pred, dates


@pytest.fixture
def two_entities_five_days():
    """Two entities, 5 days each. A: 1..5, B: 10..50. Interleaved by date."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    for i in range(5):
        rows.append({"date": dates[i], "entity": "A", "true": float(i + 1), "pred": float(i + 1)})
        rows.append({"date": dates[i], "entity": "B", "true": float((i + 1) * 10), "pred": float((i + 1) * 10)})
    df = pd.DataFrame(rows)
    return (
        df["true"].values,
        df["pred"].values,
        df["date"].values,
        df["entity"].values,
    )


# ---------------------------------------------------------------------------
# Rolling window computation correctness
# ---------------------------------------------------------------------------


def test_rolling_window_correctness_perfect_predictions(ten_days_perfect):
    """Last 3 days with perfect predictions -> rolling MAE=0, MAPE=0."""
    y_true, y_pred, dates = ten_days_perfect
    config = {
        "monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 10.0, "mape_alert": 20.0}}}
    }
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    assert result["rolling_metrics"]["mae"] == 0.0
    assert result["rolling_metrics"]["mape"] == 0.0
    assert result["evaluated_points"] == 3
    assert result["window_size"] == 3


def test_rolling_window_correctness_known_errors(ten_days_known_errors):
    """Last 3 days: true=[8,9,10], pred=[8,9,12]. MAE=2/3, MAPE from one error."""
    y_true, y_pred, dates = ten_days_known_errors
    config = {
        "monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 10.0, "mape_alert": 50.0}}}
    }
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    # Last 3: errors 0, 0, 2 -> MAE = 2/3
    assert result["rolling_metrics"]["mae"] == pytest.approx(2.0 / 3.0)
    # MAPE: only day 10 has error; |10-12|/10 = 20%
    assert result["rolling_metrics"]["mape"] == pytest.approx(20.0)
    assert result["evaluated_points"] == 3


# ---------------------------------------------------------------------------
# Alert triggers when MAE/MAPE exceed thresholds
# ---------------------------------------------------------------------------


def test_alert_triggers_when_mae_exceeds_threshold(ten_days_high_mae):
    """Rolling MAE=9 > mae_alert=5 -> mae alert true."""
    y_true, y_pred, dates = ten_days_high_mae
    config = {
        "monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 5.0, "mape_alert": 1000.0}}}
    }
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    assert result["alerts"]["mae"] is True
    assert result["rolling_metrics"]["mae"] == pytest.approx(9.0)


def test_alert_triggers_when_mape_exceeds_threshold(ten_days_high_mae):
    """Rolling MAPE high (900% since pred=10, true=1) > mape_alert=100 -> mape alert true."""
    y_true, y_pred, dates = ten_days_high_mae
    config = {
        "monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 1000.0, "mape_alert": 100.0}}}
    }
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    assert result["alerts"]["mape"] is True
    # MAPE: |1-10|/1 * 100 = 900% for each of 3 points -> 900%
    assert result["rolling_metrics"]["mape"] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# No alert when metrics are within bounds
# ---------------------------------------------------------------------------


def test_no_alert_when_metrics_within_bounds(ten_days_known_errors):
    """Rolling MAE=2/3, MAPE=20; thresholds 10 and 50 -> no alerts."""
    y_true, y_pred, dates = ten_days_known_errors
    config = {
        "monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 10.0, "mape_alert": 50.0}}}
    }
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    assert result["alerts"]["mae"] is False
    assert result["alerts"]["mape"] is False


def test_no_alert_perfect_predictions(ten_days_perfect):
    """Perfect predictions -> no alerts regardless of thresholds."""
    y_true, y_pred, dates = ten_days_perfect
    config = {
        "monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 0.1, "mape_alert": 0.1}}}
    }
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    assert result["alerts"]["mae"] is False
    assert result["alerts"]["mape"] is False


# ---------------------------------------------------------------------------
# Deterministic output for same inputs
# ---------------------------------------------------------------------------


def test_deterministic_same_inputs_twice(ten_days_known_errors):
    """Same inputs -> identical output."""
    y_true, y_pred, dates = ten_days_known_errors
    config = {"monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 5.0, "mape_alert": 30.0}}}}
    monitor = PerformanceMonitor(config)
    r1 = monitor.evaluate(y_true, y_pred, dates)
    r2 = monitor.evaluate(y_true, y_pred, dates)
    assert r1["rolling_metrics"] == r2["rolling_metrics"]
    assert r1["alerts"] == r2["alerts"]
    assert r1["evaluated_points"] == r2["evaluated_points"]


def test_deterministic_unchanged_by_input_order(ten_days_known_errors):
    """Shuffled input order -> same result (internally sorted by timestamp)."""
    y_true, y_pred, dates = ten_days_known_errors
    order = np.random.RandomState(42).permutation(len(dates))
    y_true_shuf = y_true[order]
    y_pred_shuf = y_pred[order]
    dates_shuf = dates[order]
    config = {"monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 5.0, "mape_alert": 30.0}}}}
    monitor = PerformanceMonitor(config)
    r_orig = monitor.evaluate(y_true, y_pred, dates)
    r_shuf = monitor.evaluate(y_true_shuf, y_pred_shuf, dates_shuf)
    assert r_orig["rolling_metrics"]["mae"] == r_shuf["rolling_metrics"]["mae"]
    assert r_orig["rolling_metrics"]["mape"] == r_shuf["rolling_metrics"]["mape"]
    assert r_orig["alerts"] == r_shuf["alerts"]


# ---------------------------------------------------------------------------
# Works with and without entity_ids
# ---------------------------------------------------------------------------


def test_works_without_entity_ids(ten_days_perfect):
    """Evaluate without entity_ids returns valid structure."""
    y_true, y_pred, dates = ten_days_perfect
    config = {"monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 10.0, "mape_alert": 20.0}}}}
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates)
    assert "current_metrics" in result
    assert "rolling_metrics" in result
    assert "alerts" in result
    assert "window_size" in result
    assert "evaluated_points" in result
    assert result["current_metrics"]["mae"] == 0.0
    assert result["current_metrics"]["mape"] == 0.0


def test_works_with_entity_ids(two_entities_five_days):
    """Evaluate with entity_ids returns valid structure; per-entity aggregation."""
    y_true, y_pred, dates, entity_ids = two_entities_five_days
    config = {"monitoring": {"performance": {"window_days": 3, "thresholds": {"mae_alert": 10.0, "mape_alert": 20.0}}}}
    monitor = PerformanceMonitor(config)
    result = monitor.evaluate(y_true, y_pred, dates, entity_ids=entity_ids)
    assert "current_metrics" in result
    assert "rolling_metrics" in result
    assert "alerts" in result
    assert result["evaluated_points"] == 6  # 3 days * 2 entities
    assert result["rolling_metrics"]["mae"] == 0.0  # perfect preds
    assert result["alerts"]["mae"] is False


def test_entity_ids_same_length_required():
    """entity_ids must have same length as y_true."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    monitor = PerformanceMonitor({})
    with pytest.raises(ValueError, match="same length"):
        monitor.evaluate(y_true, y_pred, dates, entity_ids=np.array(["A"]))  # length 1
