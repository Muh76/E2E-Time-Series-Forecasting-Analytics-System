"""
Merge server-side latest forecast + evaluation metrics into Copilot request context.

The API records the last forecast on POST /forecast/store and POST /predict via
``record_forecast_for_evaluation``. Explanations should use that snapshot unless
the client opts out (``options.skip_latest_forecast_merge``).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from backend.app.services.metrics import evaluate_last_forecast_vs_actuals, get_last_forecast_record

logger = logging.getLogger(__name__)


def enrich_context_with_latest_forecast(context: dict[str, Any] | None) -> dict[str, Any]:
    """
    Deep-copy ``context``, attach ``forecast`` from the last recorded run, refresh
    ``monitoring_summary.performance`` when evaluation returns ``status=ok``,
    and set ``latest_forecast_meta`` / ``latest_forecast_evaluation`` for reasoning.
    """
    ctx = copy.deepcopy(context) if context else {}
    ms = dict(ctx.get("monitoring_summary") or {})
    perf_existing = dict(ms.get("performance") or {})

    rec = get_last_forecast_record()
    if rec is None:
        ctx["monitoring_summary"] = ms
        return ctx

    fc_rows = list(rec.get("forecasts") or [])
    ctx["forecast"] = fc_rows
    ctx["latest_forecast_meta"] = {
        "store_id": int(rec["store_id"]),
        "horizon": int(rec["horizon"]),
        "recorded_at": rec.get("recorded_at"),
        "n_points": len(fc_rows),
    }

    eval_res = evaluate_last_forecast_vs_actuals(store_id=int(rec["store_id"]))
    ms["latest_forecast_evaluation"] = {
        "status": eval_res.get("status"),
        "reason": eval_res.get("reason"),
        "message": eval_res.get("message"),
    }

    perf = dict(perf_existing)
    if eval_res.get("status") == "ok":
        perf["mae"] = eval_res.get("mae")
        perf["rmse"] = eval_res.get("rmse")
        perf["mape"] = eval_res.get("mape")
        perf["sample_size"] = eval_res.get("n_samples")
        perf["source"] = "last_forecast_vs_actuals"
    elif not perf.get("source"):
        perf["source"] = perf_existing.get("source") or "monitoring_or_unavailable"

    ms["performance"] = perf
    ctx["monitoring_summary"] = ms

    logger.info(
        "copilot_context: merged latest forecast store_id=%s points=%s eval_status=%s",
        rec["store_id"],
        len(fc_rows),
        eval_res.get("status"),
    )
    return ctx
