"""
Rolling-origin backtesting service for store-level forecast evaluation.

For a given store, creates n_splits rolling cutoff points, predicts
horizon steps ahead from each cutoff using the pre-trained model,
and compares with actuals to produce per-split and average metrics.

Reuses feature engineering pipeline and inference config from
forecasting_service. The model is NOT retrained per split — this
measures how the fixed production model generalises across time.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from backend.app.runtime_paths import processed_parquet_path
from backend.app.services.model_loader import get_model_metadata

from .forecasting_service import _MIN_HISTORY_ROWS, _get_inference_config, _run_feature_pipeline

logger = logging.getLogger(__name__)


def backtest_store(
    store_id: int,
    horizon: int,
    n_splits: int,
    model: Any,
) -> dict[str, Any]:
    """
    Rolling-origin backtesting for one store.

    Algorithm:
      1. Load store data and run the feature pipeline once on the full series.
      2. Compute n_splits cutoff dates spaced evenly across the last portion
         of the series (each cutoff has at least `horizon` actual dates ahead).
      3. For each split, pass history up to the cutoff into model.predict()
         and align forecasts with actuals.
      4. Compute RMSE, MAE, MAPE per split and averages.

    Args:
        store_id:  Integer store identifier.
        horizon:   Number of steps to forecast per split.
        n_splits:  Number of rolling-origin splits.
        model:     Pre-trained forecasting model (from app.state).

    Returns:
        Dict with "splits" (per-split metrics) and "average" (mean metrics).

    Raises:
        ValueError: If inputs are invalid or store/data constraints not met.
        RuntimeError: If dataset or config is missing.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")

    pq = processed_parquet_path()
    if not pq.exists():
        raise RuntimeError(
            f"Processed dataset not found: {pq}. " "Run the ETL pipeline or set E2E_PROCESSED_PARQUET_PATH."
        )

    df = pd.read_parquet(pq)
    store_df = df[df["store_id"] == store_id].copy()
    if store_df.empty:
        raise ValueError(f"store_id={store_id} not found in dataset.")

    store_df = store_df.sort_values("date").reset_index(drop=True)

    try:
        metadata = get_model_metadata()
        max_lag = int(metadata.get("max_lag", 0))
    except (RuntimeError, TypeError, ValueError):
        max_lag = 0
    min_required = max(max_lag, _MIN_HISTORY_ROWS) + horizon

    if len(store_df) < min_required:
        raise ValueError(
            f"Insufficient history for backtesting. "
            f"store_id={store_id} has {len(store_df)} observations; "
            f"required minimum {min_required} observations "
            f"(max_lag={max_lag} + horizon={horizon})."
        )

    config = _get_inference_config()
    fe_cfg = config.get("feature_engineering") or {}
    target_col = fe_cfg.get("target_column", "target_cleaned")

    logger.info(
        "Running feature pipeline for backtest: store_id=%d, rows=%d",
        store_id,
        len(store_df),
    )
    featured_df = _run_feature_pipeline(store_df, config)

    # Determine cutoff indices: each cutoff must leave at least `horizon` dates
    # ahead for evaluation, and at least _MIN_HISTORY_ROWS behind for context.
    unique_dates = featured_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)
    earliest_cutoff_idx = _MIN_HISTORY_ROWS
    latest_cutoff_idx = n_dates - horizon

    if earliest_cutoff_idx >= latest_cutoff_idx:
        raise ValueError(
            f"Not enough dates for backtesting with horizon={horizon} and "
            f"n_splits={n_splits}. Series has {n_dates} unique dates."
        )

    available_range = latest_cutoff_idx - earliest_cutoff_idx
    actual_splits = min(n_splits, available_range)

    if actual_splits < n_splits:
        logger.warning(
            "Reduced n_splits from %d to %d (limited by available date range).",
            n_splits,
            actual_splits,
        )

    # Space cutoffs evenly across the available range
    if actual_splits == 1:
        cutoff_indices = [latest_cutoff_idx]
    else:
        step = available_range / (actual_splits - 1)
        cutoff_indices = [earliest_cutoff_idx + round(i * step) for i in range(actual_splits)]

    cutoff_dates = [unique_dates.iloc[idx] for idx in cutoff_indices]

    splits: list[dict[str, Any]] = []

    for i, cutoff_date in enumerate(cutoff_dates):
        history = featured_df[featured_df["date"] <= cutoff_date]
        actuals = featured_df[featured_df["date"] > cutoff_date].head(
            horizon * featured_df["store_id"].nunique()  # single store, but safe
        )

        if len(history) < _MIN_HISTORY_ROWS:
            logger.warning("Split %d: insufficient history (%d rows), skipping.", i, len(history))
            continue

        predictions = model.predict(history, horizon, config)

        # Align forecasts with actuals on date
        actual_dates = actuals[["date", target_col]].drop_duplicates(subset="date")
        pred_dates = predictions[["date", "y_pred"]].drop_duplicates(subset="date")
        merged = pred_dates.merge(actual_dates, on="date", how="inner")

        if merged.empty:
            logger.warning("Split %d: no overlapping dates between forecast and actuals.", i)
            continue

        y_true = np.array(merged[target_col], dtype=float)
        y_pred = np.array(merged["y_pred"], dtype=float)

        residuals = y_true - y_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))

        nonzero = y_true != 0
        if nonzero.any():
            mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100)
        else:
            mape = float("nan")

        splits.append(
            {
                "split": i + 1,
                "cutoff_date": str(cutoff_date)[:10],
                "horizon": int(len(merged)),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mape": round(mape, 2),
            }
        )

        logger.info(
            "Backtest split %d/%d: cutoff=%s, RMSE=%.2f, MAE=%.2f, MAPE=%.2f%%",
            i + 1,
            actual_splits,
            str(cutoff_date)[:10],
            rmse,
            mae,
            mape,
        )

    if not splits:
        raise ValueError("No valid backtest splits could be computed.")

    avg_rmse = round(float(np.mean([s["rmse"] for s in splits])), 4)
    avg_mae = round(float(np.mean([s["mae"] for s in splits])), 4)
    mape_vals = [s["mape"] for s in splits if not math.isnan(s["mape"])]
    avg_mape = round(float(np.mean(mape_vals)), 2) if mape_vals else float("nan")

    logger.info(
        "Backtest complete: store_id=%d, splits=%d, avg_RMSE=%.2f, avg_MAE=%.2f, avg_MAPE=%.2f%%",
        store_id,
        len(splits),
        avg_rmse,
        avg_mae,
        avg_mape,
    )

    return {
        "store_id": store_id,
        "n_splits": len(splits),
        "horizon": horizon,
        "splits": splits,
        "average": {
            "rmse": avg_rmse,
            "mae": avg_mae,
            "mape": avg_mape,
        },
    }
