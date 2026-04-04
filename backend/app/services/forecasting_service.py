"""
Inference service for store-level sales forecasting.

Loads the processed dataset from disk, filters to the requested store,
and calls the supplied model to produce horizon-step forecasts.
The model must be passed explicitly (preloaded at application startup);
this service does not perform lazy model loading.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PARQUET_PATH = _PROJECT_ROOT / "data" / "processed" / "etl_output.parquet"

# Minimum rows required for reliable feature computation
# (covers the longest lag/rolling window: 14 days + a small buffer)
_MIN_HISTORY_ROWS = 15


def forecast_store(store_id: int, horizon: int, model: Any) -> list[dict[str, Any]]:
    """
    Generate a horizon-step sales forecast for a single store.

    Loads the processed parquet dataset, filters to `store_id`, and calls
    model.predict() using the preloaded model passed in from app.state.

    Args:
        store_id: Integer store identifier matching the `store_id` column.
        horizon:  Number of future steps to forecast (must be >= 1).
        model:    Fitted forecasting model (preloaded at startup via app.state).

    Returns:
        List of {"date": str, "forecast": float} dicts in chronological order.

    Raises:
        ValueError: If horizon < 1, store_id not found, or insufficient history.
        RuntimeError: If the processed dataset is missing.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    if not _PARQUET_PATH.exists():
        raise RuntimeError(
            f"Processed dataset not found: {_PARQUET_PATH}. "
            "Run the ETL pipeline (scripts/run_etl.py) to generate it."
        )

    df = pd.read_parquet(_PARQUET_PATH)

    store_col = "store_id"
    if store_col not in df.columns:
        raise ValueError(f"Column '{store_col}' not found in dataset.")

    store_df = df[df[store_col] == store_id].copy()

    if store_df.empty:
        raise ValueError(f"store_id={store_id} not found in dataset.")

    store_df = store_df.sort_values("date").reset_index(drop=True)

    if len(store_df) < _MIN_HISTORY_ROWS:
        raise ValueError(
            f"store_id={store_id} has only {len(store_df)} rows of history; "
            f"at least {_MIN_HISTORY_ROWS} required for reliable feature computation."
        )

    predictions: pd.DataFrame = model.predict(store_df, horizon)

    result = [
        {"date": str(row["date"])[:10], "forecast": float(row["y_pred"])}
        for _, row in predictions.iterrows()
    ]

    logger.info(
        "Forecast generated for store_id=%d, horizon=%d, steps=%d",
        store_id, horizon, len(result),
    )

    return result
