"""
Evaluation metrics for time series forecasting.

Operate on y_true and y_pred arrays. Support multi-entity aggregation.
No plotting. Return a clean metrics dictionary.

Interpretation (forecasting context):
- MAE: Average absolute error in the same units as the target. Robust to outliers;
  use for typical magnitude of error. Lower is better.
- RMSE: Root mean squared error; penalizes large errors more than MAE. Same units
  as target. Use when large errors are costlier. Lower is better.
- MAPE: Mean absolute percentage error; scale-invariant (%). Useful for comparing
  across series with different scales. Unstable when y_true is near zero; zeros are
  excluded from MAPE. Lower is better. Interpret as "on average, forecasts are X% off."
"""

from typing import Any

import numpy as np


def mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Mean Absolute Error: mean(|y_true - y_pred|).

    Interpretation: Average absolute error in target units. Robust to outliers.
    Lower is better. Typical "average error size" for business reporting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Root Mean Squared Error: sqrt(mean((y_true - y_pred)^2)).

    Interpretation: Same units as target; penalizes large errors more than MAE.
    Use when large errors are costlier. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    *,
    epsilon: float = 1e-10,
) -> float:
    """
    Mean Absolute Percentage Error: mean(|y_true - y_pred| / |y_true|) * 100 (%).

    Interpretation: Scale-invariant; "on average, forecasts are X% off." Lower is
    better. Useful for comparing across series with different scales. Where y_true
    is zero or near zero, those points are excluded (MAPE is undefined at zero).
    epsilon is used to exclude near-zero denominators.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    entity_ids: np.ndarray | list[Any] | None = None,
    *,
    mape_epsilon: float = 1e-10,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE. Return a clean metrics dictionary.

    Args:
        y_true: Ground truth values (1D).
        y_pred: Predicted values (1D), same length as y_true.
        entity_ids: Optional 1D array same length as y_true. When provided, each
            metric is computed per entity then averaged (each entity weighted
            equally). When None, metrics are computed globally over all points.
        mape_epsilon: Values with |y_true| <= epsilon are excluded from MAPE.

    Returns:
        Dictionary with keys "mae", "rmse", "mape" and float values. MAPE is in
        percent (0–100). If all y_true are near zero, "mape" may be nan.
        Interpretation of each metric: see module docstring.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if entity_ids is not None:
        entity_ids = np.asarray(entity_ids).ravel()
        if len(entity_ids) != len(y_true):
            raise ValueError("entity_ids must have the same length as y_true.")

    if entity_ids is None:
        return {
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred, epsilon=mape_epsilon),
        }

    # Multi-entity: compute per entity then take mean (each entity weighted equally)
    entities = np.unique(entity_ids)
    mae_vals: list[float] = []
    rmse_vals: list[float] = []
    mape_vals: list[float] = []
    for e in entities:
        mask = entity_ids == e
        if not np.any(mask):
            continue
        mae_vals.append(mae(y_true[mask], y_pred[mask]))
        rmse_vals.append(rmse(y_true[mask], y_pred[mask]))
        m = mape(y_true[mask], y_pred[mask], epsilon=mape_epsilon)
        if not np.isnan(m):
            mape_vals.append(m)
    mape_agg = float(np.mean(mape_vals)) if mape_vals else float("nan")
    return {
        "mae": float(np.mean(mae_vals)) if mae_vals else float("nan"),
        "rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
        "mape": mape_agg,
    }
