"""
Inference service for store-level sales forecasting.

Loads the processed dataset from disk, applies the same feature engineering
pipeline used during training, and calls the supplied model to produce
horizon-step forecasts.

Feature engineering is always run at inference time to ensure lag, rolling,
and calendar features match the schema the model was trained on.
No data leakage: features are computed from historical data only.
The model is passed explicitly (preloaded at application startup).
"""

import logging
from typing import Any

import pandas as pd
import yaml

from backend.app.runtime_paths import base_default_config_path, ensure_project_on_sys_path, processed_parquet_path
from backend.app.services.model_loader import get_model_metadata

logger = logging.getLogger(__name__)

ensure_project_on_sys_path()

# Minimum rows required for reliable feature computation
# (covers the longest lag/rolling window: 14 days + a small buffer)
_MIN_HISTORY_ROWS = 15

# Cached inference config (loaded once, matches training config)
_inference_config: dict[str, Any] | None = None


def _get_inference_config() -> dict[str, Any]:
    """
    Load and cache the base YAML config used during training.

    The feature_engineering section drives lag/rolling/calendar parameters;
    loading the same config at inference ensures feature consistency with training.
    """
    global _inference_config
    if _inference_config is None:
        cfg_path = base_default_config_path()
        if not cfg_path.exists():
            raise RuntimeError(
                f"Base config not found: {cfg_path}. " "Ensure config exists or set E2E_BASE_CONFIG_PATH."
            )
        with cfg_path.open() as f:
            _inference_config = yaml.safe_load(f) or {}
        logger.info("Inference config loaded from %s", cfg_path)
    return _inference_config


def _run_feature_pipeline(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Apply lag, rolling, and calendar feature engineering to the input DataFrame.

    Imports from data.feature_engineering (project-root package). The project
    root is added to sys.path at module load time to ensure importability.
    """
    from data.feature_engineering import run_feature_pipeline  # noqa: PLC0415

    return run_feature_pipeline(df, config)


def get_store_last_date(store_id: int) -> str:
    """
    Return the last observed date (YYYY-MM-DD) for a store in the processed dataset.

    Validates store_id exists and the dataset is loadable.

    Raises:
        ValueError: If store_id not found or dataset missing required columns.
        RuntimeError: If processed parquet does not exist.
    """
    pq = processed_parquet_path()
    if not pq.exists():
        raise RuntimeError(
            f"Processed dataset not found: {pq}. " "Run the ETL pipeline or set E2E_PROCESSED_PARQUET_PATH."
        )

    df = pd.read_parquet(pq, columns=["store_id", "date"])

    store_df = df[df["store_id"] == store_id]
    if store_df.empty:
        raise ValueError(f"store_id={store_id} not found in dataset.")

    last_date = store_df["date"].max()
    return str(last_date)[:10]


def _enforce_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Validate that all expected feature columns are present in df and return
    df with columns reordered to exactly match the training order.

    Raises:
        ValueError: If any expected feature column is missing from df,
                    listing all missing columns clearly.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature mismatch: {len(missing)} column(s) expected by the model "
            f"are missing after feature engineering.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {sorted(df.columns.tolist())}\n"
            "Ensure the feature pipeline config matches the config used during training."
        )
    # Return a view with columns in exact training order (extra columns are silently excluded)
    return df[feature_columns]


def forecast_store(store_id: int, horizon: int, model: Any, feature_columns: list[str]) -> list[dict[str, Any]]:
    """
    Generate a horizon-step sales forecast for a single store.

    Steps:
      1. Load processed parquet and filter to store_id.
      2. Apply the same feature engineering pipeline used during training
         (lag, rolling, calendar) so the feature schema matches what the model
         was trained on.
      3. Enforce strict column alignment: validate that all columns in
         feature_columns are present and reorder to the exact training order.
         Raises ValueError with a clear message if any column is missing.
      4. Call model.predict() with the aligned DataFrame and config (config
         enables recursive multi-step to re-run the pipeline per step).
      5. Return a list of {"date": str, "forecast": float} dicts.

    No data leakage: all features at time t are computed from data at t or earlier.

    Args:
        store_id:        Integer store identifier matching the `store_id` column.
        horizon:         Number of future steps to forecast (must be >= 1).
        model:           Fitted forecasting model (preloaded at startup via app.state).
        feature_columns: Ordered list of feature column names from training
                         (loaded from artifacts/models/feature_columns.json).

    Returns:
        List of {"date": str, "forecast": float} dicts in chronological order.

    Raises:
        ValueError: If horizon < 1, store_id not found, insufficient history,
                    or feature column mismatch detected.
        RuntimeError: If the processed dataset or config is missing.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    pq = processed_parquet_path()
    if not pq.exists():
        raise RuntimeError(
            f"Processed dataset not found: {pq}. " "Run the ETL pipeline or set E2E_PROCESSED_PARQUET_PATH."
        )

    df = pd.read_parquet(pq)

    store_col = "store_id"
    if store_col not in df.columns:
        raise ValueError(f"Column '{store_col}' not found in dataset.")

    store_df = df[df[store_col] == store_id].copy()

    if store_df.empty:
        raise ValueError(f"store_id={store_id} not found in dataset.")

    store_df = store_df.sort_values("date").reset_index(drop=True)

    try:
        metadata = get_model_metadata()
        max_lag = int(metadata.get("max_lag", 0))
    except (RuntimeError, TypeError, ValueError):
        max_lag = 0
    min_required = max(max_lag, _MIN_HISTORY_ROWS)

    if len(store_df) < min_required:
        raise ValueError(
            f"Insufficient history for forecasting. "
            f"store_id={store_id} has {len(store_df)} observations; "
            f"required minimum {min_required} observations "
            f"(max_lag={max_lag})."
        )

    # Load config (same as used during training) and run feature pipeline
    config = _get_inference_config()
    logger.info("Running feature pipeline for store_id=%d (%d rows)", store_id, len(store_df))
    featured_df = _run_feature_pipeline(store_df, config)

    # Enforce strict column alignment: validate presence and reorder to training order
    logger.info("Enforcing feature column alignment: expecting %d columns", len(feature_columns))
    _enforce_feature_columns(featured_df, feature_columns)
    logger.info("Feature column alignment verified for store_id=%d", store_id)

    # Pass config so model.predict() can re-run feature pipeline for recursive
    # multi-step forecasting (horizon > 1). The full featured_df (not aligned_df)
    # is passed so the model has access to date, target, and entity columns needed
    # for recursive extension; the model internally selects self._feature_cols.
    predictions: pd.DataFrame = model.predict(featured_df, horizon, config)

    result = [{"date": str(row["date"])[:10], "forecast": float(row["y_pred"])} for _, row in predictions.iterrows()]

    logger.info(
        "Forecast generated for store_id=%d, horizon=%d, steps=%d",
        store_id,
        horizon,
        len(result),
    )

    return result
