"""
Performance monitoring for forecasting models.

Computes rolling MAE and MAPE, compares against thresholds, returns structured
results. No file I/O; deterministic; pure Python + numpy/pandas.
"""

from typing import Any

import numpy as np
import pandas as pd

from models.evaluation import compute_metrics


class PerformanceMonitor:
    """
    Monitor forecasting performance: rolling metrics and threshold-based alerts.

    Accepts y_true, y_pred, timestamps, optional entity_ids. Computes rolling
    MAE and rolling MAPE over a configurable window (days). Compares rolling
    metrics against thresholds from config. Returns a structured dictionary.
    No file I/O; deterministic; uses compute_metrics from models.evaluation.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize with monitoring config.

        Config shape (from monitoring.performance):
            window_days: int, rolling window in days (default 7)
            thresholds:
                mae_alert: float, alert if rolling MAE > this (default 15.0)
                mape_alert: float, alert if rolling MAPE > this in percent (default 20.0)
        """
        cfg = (config or {}).get("monitoring", {}).get("performance", {}) or config or {}
        self._window_days = int(cfg.get("window_days", 7))
        thresh = cfg.get("thresholds") or {}
        self._mae_alert = float(thresh.get("mae_alert", 15.0))
        mape_alert = thresh.get("mape_alert", 20.0)
        self._mape_alert = float(mape_alert)
        # If mape_alert < 1, treat as fraction (0.20 -> 20%)
        if self._mape_alert < 1 and self._mape_alert > 0:
            self._mape_alert = self._mape_alert * 100.0

    def evaluate(
        self,
        y_true: np.ndarray | list[float],
        y_pred: np.ndarray | list[float],
        timestamps: np.ndarray | list[Any],
        entity_ids: np.ndarray | list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Compute current metrics, rolling metrics, and alerts.

        Args:
            y_true: Ground truth values (1D).
            y_pred: Predicted values (1D), same length as y_true.
            timestamps: Datetime-like values (1D), same length. Used to determine
                the rolling window (last window_days of unique dates).
            entity_ids: Optional 1D array same length as y_true; for multi-entity
                aggregation in compute_metrics.

        Returns:
            Dictionary with:
                - current_metrics: {"mae": float, "rmse": float, "mape": float} over all points
                - rolling_metrics: {"mae": float, "mape": float} over last window_days
                - alerts: {"mae": bool, "mape": bool} true if rolling metric exceeds threshold
                - window_size: int (window_days)
                - evaluated_points: int (number of points in rolling window)
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        timestamps = np.asarray(timestamps)
        if len(timestamps.shape) == 0:
            timestamps = np.atleast_1d(timestamps)
        timestamps = timestamps.ravel()
        if len(y_true) != len(y_pred) or len(y_true) != len(timestamps):
            raise ValueError("y_true, y_pred, and timestamps must have the same length.")
        if entity_ids is not None:
            entity_ids = np.asarray(entity_ids).ravel()
            if len(entity_ids) != len(y_true):
                raise ValueError("entity_ids must have the same length as y_true.")

        # Sort by timestamp for deterministic ordering
        order = np.argsort(pd.to_datetime(timestamps))
        y_true = y_true[order]
        y_pred = y_pred[order]
        timestamps = timestamps[order]
        if entity_ids is not None:
            entity_ids = entity_ids[order]

        # Current metrics over all points
        current = compute_metrics(y_true, y_pred, entity_ids=entity_ids)

        # Rolling window: last window_days of unique dates (inclusive)
        dates = pd.to_datetime(timestamps).normalize()
        unique_dates = np.unique(dates)
        if len(unique_dates) == 0:
            rolling_mae = float("nan")
            rolling_mape = float("nan")
            evaluated_points = 0
        else:
            cutoff_date = unique_dates[-1] - pd.Timedelta(days=self._window_days - 1)
            mask = dates >= cutoff_date
            y_true_roll = y_true[mask]
            y_pred_roll = y_pred[mask]
            evaluated_points = int(np.sum(mask))
            if entity_ids is not None:
                entity_ids_roll = entity_ids[mask]
            else:
                entity_ids_roll = None
            roll_metrics = compute_metrics(y_true_roll, y_pred_roll, entity_ids=entity_ids_roll)
            rolling_mae = roll_metrics["mae"]
            rolling_mape = roll_metrics["mape"]

        rolling_metrics = {"mae": rolling_mae, "mape": rolling_mape}

        # Alerts: true if rolling metric exceeds threshold
        mae_alert = bool(evaluated_points > 0 and not np.isnan(rolling_mae) and rolling_mae > self._mae_alert)
        mape_alert = bool(evaluated_points > 0 and not np.isnan(rolling_mape) and rolling_mape > self._mape_alert)
        alerts = {"mae": mae_alert, "mape": mape_alert}

        return {
            "current_metrics": current,
            "rolling_metrics": rolling_metrics,
            "alerts": alerts,
            "window_size": self._window_days,
            "evaluated_points": evaluated_points,
        }
