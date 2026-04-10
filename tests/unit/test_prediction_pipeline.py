"""Unit tests for prediction pipeline helpers."""

from backend.app.services.prediction_pipeline import metrics_context_for_copilot


def test_metrics_context_for_copilot_ok():
    ev = {
        "status": "ok",
        "store_id": 3,
        "mae": 1.5,
        "rmse": 2.0,
        "mape": 0.05,
        "n_samples": 7,
        "evaluated_dates": ["2024-01-01"],
        "horizon_requested": 7,
    }
    ctx = metrics_context_for_copilot(ev)
    assert ctx["mae"] == 1.5
    assert ctx["rmse"] == 2.0
    assert ctx["mape"] == 0.05
    assert ctx["n_samples"] == 7
    assert "horizon_requested" not in ctx


def test_metrics_context_for_copilot_no_ground_truth():
    ev = {"status": "no_ground_truth", "message": "Metrics unavailable"}
    ctx = metrics_context_for_copilot(ev)
    assert ctx["status"] == "no_ground_truth"
    assert ctx["message"] == "Metrics unavailable"
