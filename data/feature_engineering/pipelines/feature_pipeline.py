"""
Feature engineering pipeline for time-series data.

Accepts processed ETL output and config; applies lag, rolling, and calendar
transformers; concatenates features into a single DataFrame. Preserves index
and entity identifiers. Does not drop rows. No file I/O; no model logic.
"""

import logging
from typing import Any

import pandas as pd

from ..transformers import CalendarTransformer, LagTransformer, RollingTransformer

logger = logging.getLogger(__name__)


def run_feature_pipeline(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build features from processed ETL output: lag, rolling, calendar.

    Fits and applies LagTransformer, RollingTransformer, and CalendarTransformer
    in order; concatenates all feature columns with the input DataFrame.
    Preserves index and all original columns (including entity identifiers).
    Does not drop any rows; row dropping is left to the modeling layer.

    Config shape (all optional; nested under feature_engineering):
        feature_engineering:
          date_column: "date"
          target_column: "target_cleaned"
          entity_column: "store_id"
          lag:
            lags: [1, 7, 14]
          rolling:
            windows: [7, 14]
            min_periods: 1
          calendar: {}

    Each transformer receives:
      - date_column, target_column, entity_column at top level (shared)
      - plus its own slice (lag, rolling, calendar) merged in.

    Args:
        df: Processed ETL output (e.g. with date, entity, target_cleaned).
        config: Full config; feature_engineering slice used. If None, defaults
            are used (date, target_cleaned, no entity; lag [1,7,14];
            rolling [7,14]; calendar from date).

    Returns:
        DataFrame with original columns plus lag_*, rolling_*, and calendar
        feature columns. Same index and row count as input. Input is not
        mutated.
    """
    cfg = (config or {}).get("feature_engineering", {})
    date_col = cfg.get("date_column", "date")
    target_col = cfg.get("target_column", "target_cleaned")
    entity_col = cfg.get("entity_column")

    # Shared transformer config (column names)
    shared = {
        "date_column": date_col,
        "target_column": target_col,
        "entity_column": entity_col,
    }

    # Base: copy of input so we preserve index and identifiers; no mutation of df
    base = df.copy()
    feature_dfs: list[pd.DataFrame] = []

    # 1. Lag features
    lag_cfg = {**shared, **(cfg.get("lag") or {})}
    logger.info("Feature step: lag (lags=%s)", lag_cfg.get("lags", [1, 7, 14]))
    lag_transformer = LagTransformer()
    lag_features = lag_transformer.fit_transform(base, lag_cfg)
    feature_dfs.append(lag_features)
    logger.info("Feature step: lag done (columns=%s)", list(lag_features.columns))

    # 2. Rolling features
    roll_cfg = {**shared, **(cfg.get("rolling") or {})}
    logger.info("Feature step: rolling (windows=%s)", roll_cfg.get("windows", [7, 14]))
    rolling_transformer = RollingTransformer()
    rolling_features = rolling_transformer.fit_transform(base, roll_cfg)
    feature_dfs.append(rolling_features)
    logger.info("Feature step: rolling done (columns=%s)", list(rolling_features.columns))

    # 3. Calendar features
    cal_cfg = {**shared, **(cfg.get("calendar") or {})}
    logger.info("Feature step: calendar (date_column=%s)", cal_cfg.get("date_column", "date"))
    calendar_transformer = CalendarTransformer()
    calendar_features = calendar_transformer.fit_transform(base, cal_cfg)
    feature_dfs.append(calendar_features)
    logger.info("Feature step: calendar done (columns=%s)", list(calendar_features.columns))

    # 4. Concatenate: base + all feature blocks (same index; no row drop)
    out = pd.concat([base] + feature_dfs, axis=1)
    n_features = sum(f.shape[1] for f in feature_dfs)
    logger.info("Feature step: concatenate %d feature columns (total cols=%d, rows=%d)", n_features, out.shape[1], len(out))

    return out
