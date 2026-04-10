"""Tests for merging latest forecast into Copilot context."""

from unittest.mock import patch

from backend.app.services.copilot_context import enrich_context_with_latest_forecast


def test_enrich_no_record_returns_context_unchanged_structure():
    with patch("backend.app.services.copilot_context.get_last_forecast_record", return_value=None):
        ctx = {"monitoring_summary": {"performance": {"mae": 1.0}}}
        out = enrich_context_with_latest_forecast(ctx)
    assert "forecast" not in out
    assert out["monitoring_summary"]["performance"]["mae"] == 1.0


def test_enrich_merges_forecast_and_ok_metrics():
    rec = {
        "store_id": 3,
        "horizon": 5,
        "recorded_at": "2026-01-01T00:00:00Z",
        "forecasts": [{"date": "2026-01-02", "forecast": 10.0}, {"date": "2026-01-03", "forecast": 11.0}],
    }
    eval_ok = {
        "status": "ok",
        "mae": 2.5,
        "rmse": 3.0,
        "mape": 0.08,
        "n_samples": 2,
    }
    with (
        patch("backend.app.services.copilot_context.get_last_forecast_record", return_value=rec),
        patch(
            "backend.app.services.copilot_context.evaluate_last_forecast_vs_actuals",
            return_value=eval_ok,
        ),
    ):
        out = enrich_context_with_latest_forecast({"monitoring_summary": {}})

    assert len(out["forecast"]) == 2
    assert out["latest_forecast_meta"]["store_id"] == 3
    assert out["latest_forecast_meta"]["n_points"] == 2
    perf = out["monitoring_summary"]["performance"]
    assert perf["mae"] == 2.5
    assert perf["source"] == "last_forecast_vs_actuals"
    assert out["monitoring_summary"]["latest_forecast_evaluation"]["status"] == "ok"
