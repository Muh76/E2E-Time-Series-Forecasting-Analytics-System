"""
OpenAI-backed Copilot explanations: natural language from forecast, metrics, and drift.

Requires ``OPENAI_API_KEY``. Optional ``OPENAI_COPILOT_MODEL`` (default ``gpt-4o-mini``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from backend.app.services.copilot_explain import detect_query_intents

logger = logging.getLogger(__name__)

_MAX_FORECAST_POINTS = 100
_MAX_JSON_CHARS = 100_000


def _clip_json(data: Any, max_chars: int = _MAX_JSON_CHARS) -> str:
    raw = json.dumps(data, indent=2, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 40] + '\n  "...": "[truncated]"'


def _pack_llm_payload(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build forecast, metrics, and drift dicts for the prompt."""
    ms = context.get("monitoring_summary") or {}
    raw_fc = context.get("forecast")
    if not isinstance(raw_fc, list):
        raw_fc = []

    n_total = len(raw_fc)
    truncated = n_total > _MAX_FORECAST_POINTS
    series = raw_fc if not truncated else raw_fc[-_MAX_FORECAST_POINTS:]

    forecast_block: dict[str, Any] = {
        "points": series,
        "n_points_total": n_total,
        "n_points_included": len(series),
        "truncated_from_older_values": truncated,
    }
    if context.get("latest_forecast_meta"):
        forecast_block["server_record"] = context["latest_forecast_meta"]

    metrics_block: dict[str, Any] = {
        "performance": ms.get("performance") or {},
        "latest_forecast_evaluation": ms.get("latest_forecast_evaluation"),
        "overall_status": ms.get("overall_status"),
        "alerts": ms.get("alerts"),
        "thresholds": ms.get("thresholds"),
    }

    drift_block: dict[str, Any] = dict(ms.get("drift") or {})
    return forecast_block, metrics_block, drift_block


async def explain_with_openai(query: str, context: dict[str, Any]) -> dict[str, Any] | None:
    """
    Call OpenAI Chat Completions with forecast + metrics + drift; return the same
    shape as ``build_structured_copilot_response``, or ``None`` if skipped/failed.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    model = (os.getenv("OPENAI_COPILOT_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"

    fc_json, metrics_json, drift_json = _pack_llm_payload(context)

    user_prompt = f"""Explain this forecast based on the following data.

Write as a helpful analyst for a business stakeholder: clear, natural, and human.
Avoid jargon dumps unless the user asked for them.

User question:
{query}

## Forecast (series and record metadata)
{_clip_json(fc_json)}

## Metrics (holdout / last-run performance and evaluation status)
{_clip_json(metrics_json)}

## Drift (distribution shift vs training / reference window)
{_clip_json(drift_json)}

Respond with a single JSON object only (no markdown code fences), with exactly these keys:
- "answer": string, 2–6 sentences in plain language; tie forecast path, errors, and drift to the question
- "reasoning": string, short markdown bullets citing fields you used (MAE, MAPE, overall_score, points, etc.)
- "confidence": number from 0 to 1 for how well the supplied data supports the answer

If forecast, metrics, or drift blocks are empty or sparse, say so honestly instead of inventing values."""

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior forecasting analyst helping a teammate understand model output. "
                        "Sound natural and conversational while staying faithful to the JSON context only — "
                        "never invent metrics, dates, or drift scores. "
                        "If the data is insufficient for a strong claim, soften your language. "
                        "Output valid JSON only, matching the user's requested schema."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.45,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("copilot_openai: request or parse failed: %s", exc, exc_info=False)
        return None

    answer = str(data.get("answer", "")).strip()
    reasoning = str(data.get("reasoning", "")).strip()
    if not answer:
        return None

    try:
        conf_f = float(data.get("confidence", 0.75))
    except (TypeError, ValueError):
        conf_f = 0.75
    conf_f = max(0.0, min(1.0, conf_f))

    if reasoning and not reasoning.lstrip().startswith("#"):
        reasoning_body = reasoning
        reasoning = "### Reasoning (signals used)\n\n" + reasoning_body
    elif not reasoning:
        reasoning = (
            "### Reasoning (signals used)\n\n"
            "- Generated from the supplied forecast series, metrics block, and drift block via OpenAI."
        )

    intents = detect_query_intents(query)
    now = datetime.now(timezone.utc).isoformat()
    explanation = f"## Answer\n\n{answer}\n\n{reasoning}"

    return {
        "answer": answer,
        "reasoning": reasoning,
        "confidence": round(conf_f, 2),
        "intents": intents,
        "explanation": explanation,
        "sources": [{"type": "openai", "model": model, "title": "OpenAI"}],
        "generated_at": now,
    }
