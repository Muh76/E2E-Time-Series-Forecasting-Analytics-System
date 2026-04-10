"""Structured Copilot explain: intents, forecast analysis, answer shape."""

from backend.app.services.copilot_explain import build_structured_copilot_response


def test_trend_intent_with_upward_forecast():
    ctx = {
        "monitoring_summary": {
            "performance": {"mae": 1.0, "mape": 0.05, "sample_size": 100},
            "drift": {"status": "ok", "overall_score": 0.05, "threshold": 0.25},
            "overall_status": "healthy",
        },
        "forecast": [10.0, 11.0, 12.5, 14.0, 15.0, 16.0, 17.5],
    }
    out = build_structured_copilot_response("What is the forecast trend?", ctx)
    assert "trend" in out["intents"]
    assert "upward" in out["answer"].lower() or "increase" in out["answer"].lower()
    assert out["confidence"] >= 0.35
    assert "reasoning" in out and "Query focus" in out["reasoning"]


def test_high_mape_wording():
    ctx = {
        "monitoring_summary": {
            "performance": {"mape": 22.0, "mae": 5.0},
            "drift": {"overall_score": 0.1, "threshold": 0.25},
        }
    }
    out = build_structured_copilot_response("How accurate is the model?", ctx)
    assert "error" in out["intents"] or "general" in out["intents"]
    assert "high" in out["answer"].lower() or "noisy" in out["answer"].lower()
