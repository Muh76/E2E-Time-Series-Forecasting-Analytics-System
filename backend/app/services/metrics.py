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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import yaml

from backend.app.runtime_paths import base_default_config_path, ensure_project_on_sys_path, processed_parquet_path
from backend.services.drift import compute_distribution_drift
from models.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

ensure_project_on_sys_path()


def _parquet_path() -> Path:
    return processed_parquet_path()


def _config_path() -> Path:
    return base_default_config_path()


# Last forecast from POST /forecast/store or POST /predict (for evaluation vs actuals)
_last_forecast_record: dict[str, Any] | None = None


def _no_ground_truth(reason: str, message: str) -> dict[str, str]:
    return {"status": "no_ground_truth", "reason": reason, "message": message}


class DriftPayload(TypedDict):
    drift_score: float
    status: str


class MetricsOkResponse(TypedDict, total=False):
    status: str
    store_id: int
    horizon_requested: int
    n_samples: int
    mae: float | None
    rmse: float | None
    mape: float | None
    evaluated_dates: list[str]
    drift: DriftPayload
    message: str


def _load_target_column() -> str:
    cfg_path = _config_path()
    if not cfg_path.exists():
        return "target_cleaned"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    fe = cfg.get("feature_engineering") or {}
    return str(fe.get("target_column", "target_cleaned"))


def _pack_drift_payload(raw: dict[str, Any] | None) -> DriftPayload | None:
    """Expose only drift_score and status on the API."""
    if raw is None:
        return None
    return {
        "drift_score": float(raw["drift_score"]),
        "status": str(raw["status"]),
    }


def _drift_for_metrics_request(store_id_query: int | None, rec: dict[str, Any] | None) -> DriftPayload | None:
    """
    Choose store slice for drift: explicit query wins, else last forecast store, else all rows.
    """
    sid: int | None
    if store_id_query is not None:
        sid = int(store_id_query)
    elif rec is not None:
        sid = int(rec["store_id"])
    else:
        sid = None
    return _pack_drift_payload(compute_distribution_drift(store_id=sid))


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
        MetricsOkResponse with status ``ok`` and real metrics, or
        ``no_ground_truth`` with a ``reason`` code and ``message`` when evaluation
        cannot run.
    """
    rec = _last_forecast_record
    drift_part = _drift_for_metrics_request(store_id, rec)
    pq = _parquet_path()

    if rec is None:
        out = _no_ground_truth(
            "no_forecast_record",
            "No forecast has been recorded yet; run POST /api/v1/forecast/store or POST /api/v1/predict first.",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info("metrics_evaluation: status=no_ground_truth reason=no_forecast_record")
        return out

    sid = int(rec["store_id"])
    if store_id is not None and int(store_id) != sid:
        out = _no_ground_truth(
            "store_mismatch",
            f"Requested store_id={store_id} does not match the last forecast store_id={sid}.",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info(
            "metrics_evaluation: status=no_ground_truth reason=store_mismatch requested=%s recorded=%s",
            store_id,
            sid,
        )
        return out

    if not pq.exists():
        logger.warning("Metrics: processed parquet missing at %s", pq)
        out = _no_ground_truth(
            "processed_data_missing",
            f"Processed dataset not found at {pq}. Run ETL or set E2E_PROCESSED_PARQUET_PATH.",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info("metrics_evaluation: status=no_ground_truth reason=processed_data_missing path=%s", pq)
        return out

    target_col = _load_target_column()
    try:
        df = pd.read_parquet(pq, columns=["store_id", "date", target_col])
    except Exception as exc:
        logger.warning("Metrics: could not read parquet: %s", exc)
        out = _no_ground_truth(
            "parquet_read_error",
            f"Could not read processed data: {exc}",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info("metrics_evaluation: status=no_ground_truth reason=parquet_read_error")
        return out

    store_df = df[df["store_id"] == sid].copy()
    if store_df.empty:
        out = _no_ground_truth(
            "no_rows_for_store",
            f"No rows in processed data for store_id={sid}.",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info("metrics_evaluation: status=no_ground_truth reason=no_rows_for_store store_id=%s", sid)
        return out

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
        out = _no_ground_truth(
            "no_overlapping_dates",
            "Forecast dates do not overlap finite actuals in processed data for this store.",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info(
            "metrics_evaluation: status=no_ground_truth reason=no_overlapping_dates store_id=%s",
            sid,
        )
        return out

    computed = compute_aligned_metrics(np.array(y_true_list), np.array(y_pred_list))
    if computed is None:
        out = _no_ground_truth(
            "metrics_computation_failed",
            "Could not compute MAE/RMSE/MAPE from aligned pairs (no valid samples).",
        )
        if drift_part is not None:
            out["drift"] = drift_part
        logger.info("metrics_evaluation: status=no_ground_truth reason=metrics_computation_failed store_id=%s", sid)
        return out

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
    if drift_part is not None:
        result["drift"] = drift_part
    logger.info(
        "metrics_evaluation: status=ok store_id=%s n_samples=%s mae=%s rmse=%s mape=%s",
        sid,
        len(y_true_list),
        computed["mae"],
        computed["rmse"],
        computed["mape"],
    )
    return result
