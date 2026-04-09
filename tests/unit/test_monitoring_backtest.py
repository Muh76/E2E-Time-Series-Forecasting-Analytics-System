"""
Tests that monitoring summary reflects real backtest metrics.

Verifies:
- Before any backtest, monitoring returns stubbed zeros.
- After update_monitoring_from_backtest(), monitoring returns real metrics.
- Successive backtests overwrite previous metrics.
- Missing model metadata is handled gracefully.
"""

import pytest

from backend.app.services import monitoring_service
from backend.app.services.monitoring_service import (
    get_monitoring_summary,
    update_monitoring_from_backtest,
)


@pytest.fixture(autouse=True)
def _reset_monitoring_state():
    """Reset in-memory monitoring state before each test."""
    monitoring_service._monitoring_state = None
    yield
    monitoring_service._monitoring_state = None


def _make_backtest_result(
    avg_rmse: float = 100.0,
    avg_mae: float = 80.0,
    avg_mape: float = 12.5,
    n_splits: int = 3,
    store_id: int = 1,
    horizon: int = 7,
) -> dict:
    splits = [
        {
            "split": i + 1,
            "cutoff_date": f"2024-01-{10 + i:02d}",
            "horizon": horizon,
            "rmse": avg_rmse + i,
            "mae": avg_mae + i,
            "mape": avg_mape + i,
        }
        for i in range(n_splits)
    ]
    return {
        "store_id": store_id,
        "n_splits": n_splits,
        "horizon": horizon,
        "splits": splits,
        "average": {
            "rmse": avg_rmse,
            "mae": avg_mae,
            "mape": avg_mape,
        },
    }


class TestMonitoringBacktestWiring:

    def test_stub_before_backtest(self):
        summary = get_monitoring_summary()
        perf = summary["performance"]
        assert perf["rmse"] == 0.0
        assert perf["mae"] == 0.0
        assert perf["mape"] == 0.0
        assert perf["sample_size"] == 0

    def test_summary_reflects_backtest_metrics(self):
        result = _make_backtest_result(avg_rmse=150.0, avg_mae=120.0, avg_mape=18.5)
        update_monitoring_from_backtest(result, model_metadata={"model_version": "v3"})

        summary = get_monitoring_summary()
        perf = summary["performance"]
        assert perf["rmse"] == pytest.approx(150.0, abs=0.01)
        assert perf["mae"] == pytest.approx(120.0, abs=0.01)
        assert perf["mape"] == pytest.approx(18.5, abs=0.01)
        assert summary["model_version"] == "v3"

    def test_sample_size_sums_split_horizons(self):
        result = _make_backtest_result(n_splits=3, horizon=7)
        update_monitoring_from_backtest(result)

        summary = get_monitoring_summary()
        assert summary["performance"]["sample_size"] == 3 * 7

    def test_successive_backtests_overwrite(self):
        first = _make_backtest_result(avg_rmse=200.0, avg_mae=180.0, avg_mape=25.0)
        update_monitoring_from_backtest(first)

        second = _make_backtest_result(avg_rmse=50.0, avg_mae=40.0, avg_mape=5.0)
        update_monitoring_from_backtest(second)

        summary = get_monitoring_summary()
        perf = summary["performance"]
        assert perf["rmse"] == pytest.approx(50.0, abs=0.01)
        assert perf["mae"] == pytest.approx(40.0, abs=0.01)
        assert perf["mape"] == pytest.approx(5.0, abs=0.01)

    def test_missing_model_metadata(self):
        result = _make_backtest_result()
        update_monitoring_from_backtest(result, model_metadata=None)

        summary = get_monitoring_summary()
        assert summary["model_version"] == "unknown"
        assert summary["performance"]["rmse"] > 0

    def test_overall_status_healthy_after_backtest(self):
        result = _make_backtest_result()
        update_monitoring_from_backtest(result)

        summary = get_monitoring_summary()
        assert summary["pipeline"]["status"] == "ok"

    def test_drift_status_ok_after_backtest(self):
        result = _make_backtest_result()
        update_monitoring_from_backtest(result)

        summary = get_monitoring_summary()
        assert summary["drift"]["status"] == "ok"
        assert summary["drift"]["indicators"] == []

    def test_backtest_detail_persisted(self):
        result = _make_backtest_result(store_id=42, horizon=14, n_splits=5)
        update_monitoring_from_backtest(result)

        state = monitoring_service._monitoring_state
        backtest = state["performance"]["backtest"]
        assert backtest["store_id"] == 42
        assert backtest["horizon"] == 14
        assert backtest["n_splits"] == 5
