"""
Structured Insight Copilot: intent detection, forecast analysis, and natural-language answers.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from backend.app.services.copilot_forecast_insights import analyze_forecast_structure


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def detect_query_intents(query: str) -> list[str]:
    """Ordered intent tags from user text (trend, error, drift, anomaly, forecast, health, general)."""
    q = (query or "").lower().strip()
    intents: list[str] = []

    def has(*words: str) -> bool:
        return any(w in q for w in words)

    if has("anomaly", "spike", "jump", "sudden", "outlier", "change point", "changepoint", "break"):
        intents.append("anomaly")
    if has("drift", "distribution shift", "covariate", "data shift", "population shift"):
        intents.append("drift")
    # Prefer alert when user asks about triggered alerts (often also mentions MAE/MAPE).
    if has("alert", "warning", "critical", "breach", "triggered", "firing"):
        intents.append("alert")
    if has("mae", "mape", "rmse", "mse", "error", "accuracy", "mistake", "wrong", "how good", "how bad"):
        intents.append("error")
    if has(
        "trend",
        "increas",
        "decreas",
        "upward",
        "downward",
        "direction",
        "going up",
        "going down",
        "flat",
        "slope",
    ):
        intents.append("trend")
    if has("forecast", "predict", "prediction", "projection"):
        intents.append("forecast")
    if has("health", "status", "overview", "summary", "how is", "how are", "overall"):
        intents.append("health")

    if not intents:
        intents.append("general")
    # De-duplicate preserving order
    seen: set[str] = set()
    return [i for i in intents if not (i in seen or seen.add(i))]


def _mape_interpretation(mape: float | None) -> tuple[str | None, str]:
    """Return (sentence for answer, bullet for reasoning)."""
    if mape is None:
        return None, ""
    # Heuristic: values often stored as percent (e.g. 12.5) vs fraction (0.125)
    pct = mape * 100.0 if mape <= 1.0 else mape
    if pct >= 20:
        return (
            "Model error is relatively high, suggesting instability or noisy data.",
            f"MAPE is about **{pct:.1f}%**, above a typical 15–20% watch band.",
        )
    if pct >= 15:
        return (
            "Error is moderately elevated; treat point forecasts with extra caution.",
            f"MAPE is about **{pct:.1f}%** — worth monitoring against business tolerance.",
        )
    return (
        "Point forecast error looks reasonable for many operational uses.",
        f"MAPE is about **{pct:.1f}%**.",
    )


def _drift_interpretation(score: float | None, threshold: float | None) -> tuple[str | None, str]:
    if score is None:
        return None, ""
    thr = threshold if threshold is not None and threshold > 0 else 0.25
    if score < thr * 0.5:
        return (
            "No significant data drift detected; model assumptions remain broadly valid.",
            f"Drift score **{score:.3f}** is well below threshold **{thr:.3f}**.",
        )
    if score < thr:
        return (
            "Drift is present but still below the alert threshold.",
            f"Drift score **{score:.3f}** is under threshold **{thr:.3f}**.",
        )
    return (
        "Data drift is elevated relative to the configured threshold; consider retraining or feature review.",
        f"Drift score **{score:.3f}** meets or exceeds threshold **{thr:.3f}**.",
    )


def _trend_sentence(fa: dict[str, Any]) -> tuple[str | None, str]:
    t = fa.get("trend")
    n = fa.get("n_points", 0)
    slope = fa.get("slope_normalized")
    if t == "insufficient" or n < 2:
        return None, "Forecast series too short for a reliable trend read."
    if t in ("upward", "moderate_up"):
        return (
            "The forecast shows an upward trend, likely reflecting recent momentum in the historical pattern.",
            f"Normalized slope **{slope}** over **{n}** points indicates upward movement.",
        )
    if t in ("downward", "moderate_down"):
        return (
            "The forecast trends downward, consistent with a declining pattern in the series.",
            f"Normalized slope **{slope}** over **{n}** points indicates downward movement.",
        )
    return (
        "The forecast is relatively flat with limited directional movement.",
        f"Normalized slope **{slope}** and **{n}** points suggest a stable near-term path.",
    )


def _volatility_sentence(fa: dict[str, Any]) -> tuple[str | None, str]:
    vol = fa.get("volatility")
    cv = fa.get("coefficient_of_variation")
    if vol == "high":
        return (
            "Volatility in the forecast path is high (large relative swings step-to-step).",
            f"Coefficient of variation **{cv}** flags elevated relative variability.",
        )
    if vol == "low_or_moderate":
        return (
            "Step-to-step volatility looks moderate or low compared with typical series noise.",
            f"Coefficient of variation **{cv}**.",
        )
    return None, ""


def _anomaly_sentence(fa: dict[str, Any]) -> tuple[str | None, str]:
    n = int(fa.get("n_change_points") or 0)
    idx = fa.get("change_point_indices") or []
    if n <= 0:
        return (
            "No sudden step jumps stand out in the forecast window.",
            "No large step changes detected versus local baseline.",
        )
    preview = ", ".join(str(i) for i in idx[:5])
    more = f" (+{n - len(idx[:5])} more)" if n > 5 else ""
    return (
        f"The forecast shows **{n}** notable step change(s), which can indicate regime shifts or noisy inputs.",
        f"Indices with large jumps: **{preview}**{more}.",
    )


def _confidence_score(
    intents: list[str],
    fa: dict[str, Any] | None,
    has_perf: bool,
    has_drift: bool,
) -> float:
    base = 0.45
    if fa and (fa.get("n_points") or 0) >= 7:
        base += 0.12
    if has_perf:
        base += 0.1
    if has_drift:
        base += 0.08
    base += min(0.15, 0.03 * len(intents))
    return round(min(0.92, max(0.35, base)), 2)


def build_structured_copilot_response(query: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Return ``answer``, ``reasoning`` (markdown), ``confidence``, ``intents``, ``explanation`` (legacy markdown),
    ``sources``, ``generated_at``.
    """
    q = (query or "").strip()
    intents = detect_query_intents(q)
    ms = context.get("monitoring_summary") or {}
    perf = ms.get("performance") or {}
    drift = ms.get("drift") or {}
    alerts = ms.get("alerts") or []
    alert_ctx = context.get("alert_context")

    raw_forecast = context.get("forecast")
    if raw_forecast is None and isinstance(ms.get("last_forecast"), list):
        raw_forecast = ms["last_forecast"]
    fa = analyze_forecast_structure(raw_forecast if isinstance(raw_forecast, list) else None)

    mae = _safe_float(perf.get("mae"))
    mape = _safe_float(perf.get("mape"))
    rmse = _safe_float(perf.get("rmse"))
    drift_score = _safe_float(drift.get("overall_score"))
    drift_thr = _safe_float(drift.get("threshold"))

    answer_parts: list[str] = []
    reasoning_lines: list[str] = []
    sources: list[dict[str, Any]] = []

    def add_source(stype: str, title: str, detail: str = "") -> None:
        entry: dict[str, Any] = {"type": stype, "title": title}
        if detail:
            entry["detail"] = detail
        sources.append(entry)

    # Lead with what the user asked (intent-aware ordering)
    primary = intents[0] if intents else "general"

    if primary == "anomaly" and fa:
        s, r = _anomaly_sentence(fa)
        if s:
            answer_parts.append(s)
        if r:
            reasoning_lines.append(f"- **Anomaly / change points**: {r}")
        add_source("forecast_analysis", "Forecast step changes", r or s or "")

    if primary == "trend" and fa:
        s, r = _trend_sentence(fa)
        if s:
            answer_parts.append(s)
        if r:
            reasoning_lines.append(f"- **Trend**: {r}")
        add_source("forecast_analysis", "Forecast trend", r or s or "")

    if primary == "drift":
        ds, dr = _drift_interpretation(drift_score, drift_thr)
        if ds:
            answer_parts.append(ds)
        if dr:
            reasoning_lines.append(f"- **Drift**: {dr}")
        add_source("drift", "Population drift", dr or ds or "")

    if primary == "error":
        if "mae" in q.lower() and mae is not None:
            answer_parts.append(f"The holdout MAE is **{mae:.4g}** (mean absolute error vs actuals).")
        es, er = _mape_interpretation(mape)
        if es:
            answer_parts.append(es)
        if er:
            reasoning_lines.append(f"- **Error (MAPE)**: {er}")
        if mae is not None:
            reasoning_lines.append(f"- **MAE**: **{mae:.4g}**.")
        if rmse is not None:
            reasoning_lines.append(f"- **RMSE**: **{rmse:.4g}**.")
        add_source("metrics", "Model error metrics", er or es or "")

    if primary == "forecast" and fa:
        s, r = _trend_sentence(fa)
        if s and s not in answer_parts:
            answer_parts.append(s)
        vs, vr = _volatility_sentence(fa)
        if vs:
            answer_parts.append(vs)
        if r:
            reasoning_lines.append(f"- **Trend**: {r}")
        if vr:
            reasoning_lines.append(f"- **Volatility**: {vr}")
        add_source("forecast_analysis", "Forecast structure", "trend + volatility")

    if primary == "health":
        status = ms.get("overall_status") or "unknown"
        answer_parts.append(f"Overall monitoring status is **{status}**.")
        reasoning_lines.append(f"- **Overall status**: `{status}`.")
        add_source("monitoring_summary", "Monitoring snapshot", status)

    if primary == "alert":
        ac = alert_ctx if isinstance(alert_ctx, dict) else {}
        alist = ac.get("alerts") if isinstance(ac.get("alerts"), list) else []
        if not alist and isinstance(context.get("alerts"), list):
            alist = context["alerts"]
        if alist:
            bits: list[str] = []
            for a in alist:
                if not isinstance(a, dict):
                    continue
                at = a.get("alert_type") or "Alert"
                cv = a.get("current_value")
                th = a.get("threshold")
                bits.append(f"**{at}** (value `{cv}` vs threshold `{th}`)")
            answer_parts.append(
                "These alerts fire when a live metric crosses its configured threshold. "
                + ("; ".join(bits) if bits else "Review the monitoring cards for details.")
            )
            reasoning_lines.append("- **Alerts**: " + "; ".join(bits) if bits else "- **Alerts**: active.")
            add_source("alert", "Threshold alerts", str(len(alist)))
        elif alert_ctx:
            answer_parts.append(
                "An alert context was passed, but no alert list was found; check Monitoring for current thresholds."
            )
            add_source("alert", "Alert context", "empty list")

    # Secondary signals: fill gaps for general / multi-intent
    intent_set = set(intents)

    if fa and "trend" in intent_set and primary != "trend" and primary != "forecast":
        s, r = _trend_sentence(fa)
        if s and s not in answer_parts:
            answer_parts.append(s)
        if r:
            reasoning_lines.append(f"- **Trend**: {r}")
        add_source("forecast_analysis", "Forecast trend", r or "")

    if fa and "anomaly" in intent_set and primary != "anomaly":
        s, r = _anomaly_sentence(fa)
        if s and s not in answer_parts:
            answer_parts.append(s)
        if r:
            reasoning_lines.append(f"- **Change points**: {r}")

    if ("error" in intent_set or primary == "general") and mape is not None:
        es, er = _mape_interpretation(mape)
        if es and es not in " ".join(answer_parts):
            answer_parts.append(es)
        if er and not any("MAPE" in line for line in reasoning_lines):
            reasoning_lines.append(f"- **MAPE**: {er}")
        add_source("metrics", "MAPE", er or "")

    if ("drift" in intent_set or primary == "general") and drift_score is not None:
        ds, dr = _drift_interpretation(drift_score, drift_thr)
        if ds and ds not in " ".join(answer_parts):
            answer_parts.append(ds)
        if dr and not any("Drift score" in line for line in reasoning_lines):
            reasoning_lines.append(f"- **Drift**: {dr}")
        add_source("drift", "Drift score", dr or "")

    if fa and primary == "general":
        vs, vr = _volatility_sentence(fa)
        if vs and vs not in " ".join(answer_parts):
            answer_parts.append(vs)
        if vr:
            reasoning_lines.append(f"- **Volatility**: {vr}")

    # Performance one-liner if still thin
    if len(answer_parts) < 2 and mae is not None:
        answer_parts.append(f"Holdout MAE is **{mae:.4g}**.")
        reasoning_lines.append(f"- **MAE**: **{mae:.4g}**.")
        add_source("metrics", "MAE", str(mae))

    if not answer_parts:
        answer_parts.append(
            "Here is a quick read from the latest monitoring snapshot. "
            "Ask about trend, error, drift, or anomalies for a focused answer."
        )

    # Alerts summary
    if isinstance(alerts, list) and alerts:
        crit = sum(1 for a in alerts if isinstance(a, dict) and str(a.get("severity", "")).lower() == "critical")
        if crit:
            answer_parts.append(f"There are **{crit}** critical alert(s) in the monitoring feed.")
        reasoning_lines.append(f"- **Open alerts**: **{len(alerts)}** total.")
        add_source("alerts", "Monitoring alerts", f"{len(alerts)} items")

    meta = context.get("latest_forecast_meta")
    if meta:
        reasoning_lines.append(
            "- **Latest forecast (API)**: "
            f"store `{meta.get('store_id')}`, horizon `{meta.get('horizon')}`, "
            f"`{meta.get('n_points')}` points, recorded `{meta.get('recorded_at')}`."
        )
        add_source("latest_forecast", "Server forecast snapshot", str(meta.get("recorded_at") or ""))

    lfe = (context.get("monitoring_summary") or {}).get("latest_forecast_evaluation") or {}
    if lfe.get("status") and str(lfe.get("status")) != "ok":
        reason = str(lfe.get("reason") or "")
        if reason != "no_forecast_record":
            msg = str(lfe.get("message") or "")[:240]
            reasoning_lines.append(f"- **Forecast vs actuals**: `{reason}` — {msg}")

    reasoning_lines.insert(0, f"- **Query focus**: {', '.join(intents)}.")

    if not sources:
        add_source(
            "monitoring_summary",
            "Insight Copilot",
            "rule-based; add monitoring_summary or forecast for richer signals",
        )

    answer = " ".join(answer_parts)
    reasoning = "### Reasoning (signals used)\n\n" + "\n".join(reasoning_lines)

    has_perf = mae is not None or mape is not None
    has_drift = drift_score is not None
    confidence = _confidence_score(intents, fa, has_perf, has_drift)

    explanation = f"## Answer\n\n{answer}\n\n{reasoning}"

    return {
        "answer": answer,
        "reasoning": reasoning,
        "confidence": confidence,
        "intents": intents,
        "explanation": explanation,
        "sources": sources,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_rule_based_explanation(query: str, context: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    """Backward-compatible: returns (markdown explanation, sources)."""
    out = build_structured_copilot_response(query, context or {})
    return out["explanation"], out["sources"]
