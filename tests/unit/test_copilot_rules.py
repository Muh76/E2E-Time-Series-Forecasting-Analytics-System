"""Unit tests for rule-based Insight Copilot."""

from backend.app.services.copilot_rules import build_rule_based_explanation


def test_explain_performance_keywords():
    ctx = {
        "monitoring_summary": {
            "performance": {
                "mae": 10.5,
                "rmse": 12.0,
                "mape": 8.2,
                "sample_size": 1000,
                "source": "validation_holdout",
            },
            "drift": {"status": "ok", "overall_score": 0.1, "threshold": 0.25},
            "overall_status": "healthy",
        }
    }
    text, sources = build_rule_based_explanation("What is our MAE?", ctx)
    assert "MAE" in text
    assert "10.5" in text
    assert any("metrics" in s.get("type", "") for s in sources)


def test_explain_default_snapshot():
    text, sources = build_rule_based_explanation("", {})
    assert "Insight Copilot" in text or "MAE" in text
    assert sources
