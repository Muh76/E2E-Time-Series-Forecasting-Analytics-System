"""OpenAI Copilot helper tests (no live API calls)."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.copilot_openai import _pack_llm_payload, explain_with_openai


def test_pack_llm_truncates_long_forecast():
    ctx = {
        "forecast": [{"date": "2026-01-01", "forecast": float(i)} for i in range(150)],
        "latest_forecast_meta": {"store_id": 1, "horizon": 7},
        "monitoring_summary": {
            "performance": {"mae": 1.0},
            "drift": {"overall_score": 0.1, "status": "ok"},
        },
    }
    fc, metrics, drift = _pack_llm_payload(ctx)
    assert fc["n_points_total"] == 150
    assert fc["truncated_from_older_values"] is True
    assert len(fc["points"]) == 100
    assert metrics["performance"]["mae"] == 1.0
    assert drift.get("overall_score") == 0.1


def test_explain_openai_returns_none_without_api_key():
    async def _run() -> None:
        out = await explain_with_openai("What is the trend?", {"monitoring_summary": {}})
        assert out is None

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        asyncio.run(_run())


def test_explain_openai_parses_successful_response():
    async def _run() -> None:
        resp_mock = MagicMock()
        resp_mock.choices = [
            MagicMock(
                message=MagicMock(content='{"answer": "It rises.", "reasoning": "- Points go up", "confidence": 0.8}')
            )
        ]
        mock_create = AsyncMock(return_value=resp_mock)
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        ctx = {
            "forecast": [{"date": "2026-01-01", "forecast": 1.0}],
            "monitoring_summary": {"drift": {}, "performance": {}},
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("openai.AsyncOpenAI", return_value=mock_client):
                out = await explain_with_openai("Trend?", ctx)

        assert out is not None
        assert "rises" in out["answer"].lower()
        assert out["confidence"] == 0.8
        assert any(s.get("type") == "openai" for s in out["sources"])
        mock_create.assert_awaited_once()

    asyncio.run(_run())
