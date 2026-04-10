"""
Forecast evaluation metrics after prediction.

Aligns the most recent API forecast dates with processed ground truth when
available. Does not fabricate zeros for missing metrics — uses null for
undefined MAPE (e.g. all-zero denominators) and omits numeric metrics when
no evaluation is possible.
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import yaml

from models.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PARQUET_PATH = _PROJECT_ROOT / "data" / "processed" / "etl_output.parquet"
_CONFIG_PATH = _PROJECT_ROOT / "config" / "base" / "default.yaml"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Last forecast from POST /forecast/store (for evaluation vs actuals)
_last_forecast_record: dict[str, Any] | None = None

NO_GROUND_TRUTH: dict[str, str] = {
    "status": "no_ground_truth",
    "message": "Metrics unavailable without actual values",
}


class MetricsOkResponse(TypedDict, total=False):
    status: str
    store_id: int
    horizon_requested: int
    n_samples: int
    mae: float | None
    rmse: float | None
    mape: float | None
    evaluated_dates: list[str]


def _load_target_column() -> str:
    if not _CONFIG_PATH.exists():
        return "target_cleaned"
    with _CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    fe = cfg.get("feature_engineering") or {}
    return str(fe.get("target_column", "target_cleaned"))


def _normalize_date_key(d: Any) -> str:
    if d is None:
        return ""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    s = str(d)
    return s[:10] if len(s) >= 10 else s


def compute_aligned_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None] | None:
    """
    Compute MAE, RMSE, MAPE for aligned finite pairs.

    Returns None if there are no valid pairs. MAPE is None when undefined
    (all |y_true| below epsilon). Never returns 0 as a placeholder for unknown.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return None

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return None

    yt = y_true[valid]
    yp = y_pred[valid]

    raw = compute_metrics(yt, yp)
    mae_v = float(raw["mae"])
    rmse_v = float(raw["rmse"])
    mape_v = float(raw["mape"])

    out: dict[str, float | None] = {
        "mae": None if math.isnan(mae_v) else mae_v,
        "rmse": None if math.isnan(rmse_v) else rmse_v,
        "mape": None if math.isnan(mape_v) else mape_v,
    }
    return out


def record_forecast_for_evaluation(
    store_id: int,
    horizon: int,
    forecasts: list[dict[str, Any]],
) -> None:
    """
    Store the latest forecast points (date + forecast) for GET /metrics evaluation.
    """
    global _last_forecast_record
    clean: list[dict[str, Any]] = []
    for row in forecasts:
        clean.append(
            {
                "date": _normalize_date_key(row.get("date")),
                "forecast": float(row["forecast"]),
            }
        )
    _last_forecast_record = {
        "store_id": int(store_id),
        "horizon": int(horizon),
        "forecasts": clean,
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    logger.info(
        "Recorded forecast for evaluation: store_id=%d horizon=%d points=%d",
        store_id,
        horizon,
        len(clean),
    )


def get_last_forecast_record() -> dict[str, Any] | None:
    """Return a copy of the last forecast record, if any."""
    if _last_forecast_record is None:
        return None
    return dict(_last_forecast_record)


def evaluate_last_forecast_vs_actuals(
    store_id: int | None = None,
) -> dict[str, Any]:
    """
    Align last recorded forecast with processed parquet ground truth.

    If ``store_id`` is set, it must match the recorded forecast's store.

    Returns:
        MetricsOkResponse with status \"ok\" and real metrics, or
        NO_GROUND_TRUTH when no forecast, wrong store, missing data, or no
        overlapping dates with finite actuals.
    """
    rec = _last_forecast_record
    if rec is None:
        return dict(NO_GROUND_TRUTH)

    sid = int(rec["store_id"])
    if store_id is not None and int(store_id) != sid:
        return dict(NO_GROUND_TRUTH)

    if not _PARQUET_PATH.exists():
        logger.warning("Metrics: processed parquet missing at %s", _PARQUET_PATH)
        return dict(NO_GROUND_TRUTH)

    target_col = _load_target_column()
    try:
        df = pd.read_parquet(_PARQUET_PATH, columns=["store_id", "date", target_col])
    except Exception as exc:
        logger.warning("Metrics: could not read parquet: %s", exc)
        return dict(NO_GROUND_TRUTH)

    store_df = df[df["store_id"] == sid].copy()
    if store_df.empty:
        return dict(NO_GROUND_TRUTH)

    store_df["_d"] = store_df["date"].map(_normalize_date_key)
    fmap = {row["date"]: row["forecast"] for row in rec["forecasts"]}

    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    dates_out: list[str] = []
    for dkey, yhat in fmap.items():
        if not dkey:
            continue
        match = store_df[store_df["_d"] == dkey]
        if match.empty:
            continue
        y = float(match.iloc[0][target_col])
        if not math.isfinite(y):
            continue
        if not math.isfinite(yhat):
            continue
        y_true_list.append(y)
        y_pred_list.append(yhat)
        dates_out.append(dkey)

    if not y_true_list:
        return dict(NO_GROUND_TRUTH)

    computed = compute_aligned_metrics(np.array(y_true_list), np.array(y_pred_list))
    if computed is None:
        return dict(NO_GROUND_TRUTH)

    result: MetricsOkResponse = {
        "status": "ok",
        "store_id": sid,
        "horizon_requested": int(rec["horizon"]),
        "n_samples": len(y_true_list),
        "mae": computed["mae"],
        "rmse": computed["rmse"],
        "mape": computed["mape"],
        "evaluated_dates": dates_out,
    }
    return result
