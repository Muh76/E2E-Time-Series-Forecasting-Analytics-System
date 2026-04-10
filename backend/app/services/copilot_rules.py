"""
Rule-based Insight Copilot — no LLM; explains monitoring context in plain language.
"""

from __future__ import annotations

from typing import Any


def _perf(context: dict[str, Any]) -> dict[str, Any]:
    ms = context.get("monitoring_summary") or {}
    return ms.get("performance") or context.get("performance") or {}


def _drift(context: dict[str, Any]) -> dict[str, Any]:
    ms = context.get("monitoring_summary") or {}
    return ms.get("drift") or context.get("drift") or {}


def _alerts(context: dict[str, Any]) -> dict[str, Any]:
    ms = context.get("monitoring_summary") or {}
    return ms.get("alerts") or {}


def build_rule_based_explanation(query: str, context: dict[str, Any] | None) -> tuple[str, list[dict[str, str]]]:
    """
    Return (markdown_explanation, sources).
    """
    ctx = context or {}
    q = (query or "").strip().lower()
    sources: list[dict[str, str]] = []

    perf = _perf(ctx)
    drift = _drift(ctx)
    alerts = _alerts(ctx)
    recent = (ctx.get("monitoring_summary") or {}).get("recent_activity") or ctx.get("recent_activity") or {}
    last_fc = recent.get("last_forecast") if isinstance(recent, dict) else None

    mae = perf.get("mae")
    rmse = perf.get("rmse")
    mape = perf.get("mape")
    sample = perf.get("sample_size")
    src = perf.get("source", "unknown")

    overall = (ctx.get("monitoring_summary") or {}).get("overall_status") or ctx.get("overall_status") or ""
    drift_score = drift.get("overall_score")
    drift_ok = drift.get("status") == "ok" and not drift.get("drift_detected", False)

    lines: list[str] = []

    # --- Alert-specific context (from Monitoring page) ---
    alert_ctx = ctx.get("alerts")
    if not alert_ctx:
        ac = ctx.get("alert_context") or {}
        alert_ctx = ac.get("alerts") if isinstance(ac, dict) else None
    if isinstance(alert_ctx, list) and alert_ctx:
        lines.append("### Active alerts")
        for a in alert_ctx:
            lines.append(
                f"- **{a.get('alert_type', 'Alert')}**: current value "
                f"`{a.get('current_value')}` vs threshold `{a.get('threshold')}`."
            )
        lines.append("")
        sources.append({"type": "alert_context"})

    # --- Keyword-driven sections ---
    if any(k in q for k in ("health", "status", "overview", "summary", "how is")):
        lines.append("### System status")
        lines.append(
            f"- Overall status: **{overall or 'unknown'}**.\n"
            f"- Performance metrics source: **{src}** "
            f"(validation holdout until you run a backtest; backtest refreshes rolling splits)."
        )
        sources.append({"type": "monitoring_summary"})

    if any(k in q for k in ("mae", "error", "accuracy", "performance", "metric")):
        lines.append("### Performance metrics")
        if mae is not None:
            lines.append(
                f"- **MAE** (mean absolute error): `{mae:.4f}` — average absolute gap between "
                "forecast and actual in target units."
            )
        if rmse is not None:
            lines.append(f"- **RMSE**: `{rmse:.4f}` — penalizes larger errors more than MAE.")
        if mape is not None:
            lines.append(f"- **MAPE**: `{mape:.2f}%` — mean absolute percentage error vs non-zero actuals.")
        if sample is not None:
            lines.append(f"- **Sample size** used for the displayed aggregate: `{sample}`.")
        lines.append("")
        sources.append({"type": "metrics"})

    if any(k in q for k in ("drift", "distribution", "data quality", "shift")):
        lines.append("### Data drift")
        if drift_score is not None:
            lines.append(
                f"- Overall drift score: `{float(drift_score):.4f}` "
                f"(threshold `{drift.get('threshold', 'n/a')}`). "
                f"Computed by comparing early vs late windows in processed features."
            )
        lines.append(f"- Drift flag: **{'no significant drift' if drift_ok else 'review recommended'}**.")
        pf = drift.get("per_feature_scores") or {}
        if pf:
            top = sorted(pf.items(), key=lambda x: -x[1])[:5]
            lines.append("- Highest-shift features: " + ", ".join(f"`{k}` ({v:.3f})" for k, v in top))
        lines.append("")
        sources.append({"type": "drift"})

    if any(k in q for k in ("alert", "warning", "threshold")):
        lines.append("### Alerts")
        lines.append(
            f"- MAE alert: **{'firing' if alerts.get('mae') else 'clear'}**.\n"
            f"- MAPE alert: **{'firing' if alerts.get('mape') else 'clear'}**.\n"
            f"- Drift alert: **{'firing' if alerts.get('drift') else 'clear'}**."
        )
        lines.append("")
        sources.append({"type": "alerts"})

    if any(k in q for k in ("forecast", "predict", "last run")):
        lines.append("### Forecast activity")
        if isinstance(last_fc, dict):
            lines.append(
                f"- Last API forecast: store `{last_fc.get('store_id')}`, "
                f"horizon `{last_fc.get('horizon')}` at `{last_fc.get('at')}`."
            )
        else:
            lines.append("- No recent forecast recorded in this API process yet.")
        lines.append("\n*The copilot does not run or change predictions; it only interprets supplied context.*")
        sources.append({"type": "recent_activity"})

    # Default if nothing matched
    if not lines:
        lines.append("### Insight Copilot")
        lines.append(
            "Ask about **performance** (MAE, RMSE, MAPE), **drift**, **alerts**, **health**, or **forecasts**. "
            "Below is a snapshot of the context you provided:\n"
        )
        if mae is not None or mape is not None:
            lines.append(
                f"- MAE `{mae}`, RMSE `{rmse}`, MAPE `{mape}%`, sample_size `{sample}`.\n"
                f"- Drift score `{drift_score}`, status `{drift.get('status', 'n/a')}`."
            )
        else:
            lines.append("- No performance numbers in context; open Monitoring or run a backtest first.")
        sources.append({"type": "monitoring_summary"})

    explanation = "\n".join(lines).strip()
    return explanation, sources
