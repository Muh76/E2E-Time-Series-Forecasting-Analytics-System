"""Unit tests for forecast evaluation metrics service."""

import numpy as np
import pandas as pd
import pytest

from backend.app.services import metrics as metrics_service


@pytest.fixture(autouse=True)
def _clear_last_forecast():
    metrics_service._last_forecast_record = None
    yield
    metrics_service._last_forecast_record = None


def test_compute_aligned_metrics_perfect():
    y = np.array([10.0, 20.0, 30.0])
    out = metrics_service.compute_aligned_metrics(y, y.copy())
    assert out is not None
    assert out["mae"] == 0.0
    assert out["rmse"] == 0.0
    assert out["mape"] == 0.0


def test_compute_aligned_metrics_known_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 220.0])
    out = metrics_service.compute_aligned_metrics(y_true, y_pred)
    assert out is not None
    assert out["mae"] == pytest.approx(15.0)
    assert out["rmse"] == pytest.approx(np.sqrt(250.0))


def test_compute_aligned_metrics_empty():
    assert metrics_service.compute_aligned_metrics(np.array([]), np.array([])) is None


def test_no_record_returns_no_ground_truth():
    r = metrics_service.evaluate_last_forecast_vs_actuals()
    assert r["status"] == "no_ground_truth"
    assert r["reason"] == "no_forecast_record"
    assert "forecast" in r["message"].lower() or "record" in r["message"].lower()


def test_evaluate_with_mocked_parquet(tmp_path, monkeypatch):
    metrics_service.record_forecast_for_evaluation(
        1,
        2,
        [
            {"date": "2024-06-01", "forecast": 100.0},
            {"date": "2024-06-02", "forecast": 110.0},
        ],
    )
    df = pd.DataFrame(
        {
            "store_id": [1, 1],
            "date": pd.to_datetime(["2024-06-01", "2024-06-02"]),
            "target_cleaned": [105.0, 108.0],
        }
    )
    pq = tmp_path / "etl_output.parquet"
    df.to_parquet(pq)

    monkeypatch.setattr(metrics_service, "_parquet_path", lambda: pq)
    monkeypatch.setattr(metrics_service, "_load_target_column", lambda: "target_cleaned")

    r = metrics_service.evaluate_last_forecast_vs_actuals()
    assert r["status"] == "ok"
    assert r["n_samples"] == 2
    assert r["mae"] is not None
    assert r["rmse"] is not None
    assert r["store_id"] == 1
    assert len(r["evaluated_dates"]) == 2


def test_wrong_store_id_returns_no_ground_truth(tmp_path, monkeypatch):
    metrics_service.record_forecast_for_evaluation(1, 1, [{"date": "2024-06-01", "forecast": 1.0}])
    df = pd.DataFrame(
        {
            "store_id": [1],
            "date": pd.to_datetime(["2024-06-01"]),
            "target_cleaned": [1.0],
        }
    )
    pq = tmp_path / "etl_output.parquet"
    df.to_parquet(pq)
    monkeypatch.setattr(metrics_service, "_parquet_path", lambda: pq)
    monkeypatch.setattr(metrics_service, "_load_target_column", lambda: "target_cleaned")

    r = metrics_service.evaluate_last_forecast_vs_actuals(store_id=999)
    assert r["status"] == "no_ground_truth"
    assert r["reason"] == "store_mismatch"
