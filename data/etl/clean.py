"""
Cleaning and normalization logic for daily time series.

Handles missing values, duplicates, date normalization (frequency, timezone),
and optional outlier clipping. Config-driven; deterministic and testable.
No model logic.
"""

from typing import Any

import pandas as pd


def clean_dates(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Normalize date column: ensure daily frequency and optional timezone.

    Sorts by date (and series_id if present) for deterministic ordering.
    Does not fill gaps; that can be a separate step if required.

    Args:
        df: DataFrame with a date column.
        config: Optional clean config. Expected keys (all optional):
            - date_column: Name of date column (default "date").
            - frequency: Target frequency; only "D" (daily) is applied here
              (normalize to date, drop time).
            - timezone: If set, localize or convert to this zone (e.g. "UTC").
            - sort_by: List of columns for sort order (default [date_column, ...]).

    Returns:
        New DataFrame with normalized date column and sorted rows.
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    tz = cfg.get("timezone")
    sort_by = cfg.get("sort_by")

    out = df.copy()

    if date_col not in out.columns:
        return out

    # Normalize to date (daily) — drop time component
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()

    if tz:
        if out[date_col].dt.tz is None:
            out[date_col] = out[date_col].dt.tz_localize(tz)
        else:
            out[date_col] = out[date_col].dt.tz_convert(tz)

    sort_cols = sort_by if sort_by is not None else [date_col]
    sort_cols = [c for c in sort_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def clean_values(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Clean value column: drop or fill nulls, clip outliers, deduplicate.

    Args:
        df: DataFrame with at least a value column.
        config: Optional clean config. Expected keys (all optional):
            - value_column: Name of value column (default "value").
            - drop_null_value: If True, drop rows where value is NA (default True).
            - clip_lower: If set, clip value to this lower bound.
            - clip_upper: If set, clip value to this upper bound.
            - series_column: If set, deduplicate by (date_column, series_column),
              keeping first occurrence.
            - date_column: Used with series_column for dedup (default "date").

    Returns:
        New DataFrame with cleaned values and optional deduplication applied.
    """
    cfg = config or {}
    value_col = cfg.get("value_column", "value")
    date_col = cfg.get("date_column", "date")
    drop_null = cfg.get("drop_null_value", True)
    clip_lower = cfg.get("clip_lower")
    clip_upper = cfg.get("clip_upper")
    series_col = cfg.get("series_column")

    out = df.copy()

    if value_col not in out.columns:
        return out

    if drop_null:
        out = out.dropna(subset=[value_col])

    if clip_lower is not None:
        out[value_col] = out[value_col].clip(lower=clip_lower)
    if clip_upper is not None:
        out[value_col] = out[value_col].clip(upper=clip_upper)

    if series_col and series_col in out.columns:
        key_cols = [date_col, series_col]
        if all(c in out.columns for c in key_cols):
            out = out.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

    return out


def clean(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run full cleaning: dates first, then values.

    Order is deterministic: clean_dates then clean_values. Config can
    contain separate keys for date vs value behavior; pass the same config
    and both steps will read their keys.

    Args:
        df: Raw or validated DataFrame.
        config: Optional config for clean_dates and clean_values (see their
            docstrings). Can be flat or nested, e.g. config["clean"]["dates"].

    Returns:
        Cleaned DataFrame, sorted and deduplicated per config.
    """
    cfg = config or {}
    # Support nested config for pipeline
    if "dates" in cfg and isinstance(cfg.get("dates"), dict):
        date_cfg = cfg["dates"]
    else:
        date_cfg = cfg
    if "values" in cfg and isinstance(cfg.get("values"), dict):
        value_cfg = cfg["values"]
    else:
        value_cfg = cfg

    out = clean_dates(df, date_cfg)
    out = clean_values(out, value_cfg)
    return out


# ---------------------------------------------------------------------------
# Retail daily time series cleaning (reindex, fill, clip; preserve original)
# ---------------------------------------------------------------------------


def clean_retail(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Clean daily retail time series: reindex to daily, fill missing values, optional clip.

    - Handles missing dates by reindexing to a full daily range (per store or global).
    - Applies a configurable missing-value strategy (forward-fill or zero) for added dates.
    - Optionally clips outliers on the cleaned series (absolute bounds or IQR).
    - Preserves the original sales column and adds a cleaned column (e.g. sales_cleaned).

    Args:
        df: DataFrame with columns date, store_id, sales (or config overrides).
            Must have unique (date, store_id). Call validate_retail first if needed.
        config: Optional. Keys (all optional):
            - date_column: default "date".
            - store_id_column: default "store_id".
            - sales_column: default "sales".
            - filled_column: name of cleaned column (default "sales_cleaned").
            - missing_dates: "reindex" (fill to daily frequency) or "none" (no reindex).
            - date_range: "per_store" (min–max per store) or "global" (min–max over all).
            - missing_value_strategy: "forward_fill" or "zero".
            - outlier_clip: dict. If present and enabled:
                - enabled: bool.
                - lower: optional absolute lower bound.
                - upper: optional absolute upper bound.
                - method: "iqr" to use IQR-based bounds (optional).
                - iqr_multiplier: used when method="iqr" (default 1.5).
                Clipping is applied to the cleaned column only.

    Returns:
        DataFrame with date, store_id, original sales column, and filled_column
        (sales_cleaned by default). Sorted by date, store_id.
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    store_col = cfg.get("store_id_column", "store_id")
    sales_col = cfg.get("sales_column", "sales")
    filled_col = cfg.get("filled_column", "sales_cleaned")
    missing_dates = cfg.get("missing_dates", "reindex")
    date_range = cfg.get("date_range", "per_store")
    fill_strategy = cfg.get("missing_value_strategy", "forward_fill")
    clip_cfg = cfg.get("outlier_clip") or {}

    for col in (date_col, store_col, sales_col):
        if col not in df.columns:
            raise ValueError(f"clean_retail requires column '{col}'. Found: {list(df.columns)}.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()

    if missing_dates != "reindex":
        out[filled_col] = out[sales_col].astype(float)
        if clip_cfg.get("enabled"):
            out[filled_col] = _clip_series(out[filled_col], clip_cfg)
        return out.sort_values([date_col, store_col]).reset_index(drop=True)

    # Reindex to full daily range per store (or global)
    pieces: list[pd.DataFrame] = []
    for store_id, group in out.groupby(store_col, sort=False):
        grp = group[[date_col, sales_col]].copy()
        grp = grp.set_index(date_col).sort_index()
        min_d, max_d = grp.index.min(), grp.index.max()
        if date_range == "global":
            # Use global range from full df so all stores share same calendar
            min_d = out[date_col].min()
            max_d = out[date_col].max()
        full_idx = pd.date_range(min_d, max_d, freq="D")
        reindexed = grp.reindex(full_idx)
        reindexed.index.name = date_col
        reindexed = reindexed.reset_index()
        reindexed[store_col] = store_id
        # Preserve original sales (NaN for filled-in dates); set cleaned column from fill strategy
        if fill_strategy == "forward_fill":
            reindexed[filled_col] = reindexed[sales_col].ffill()
        else:
            reindexed[filled_col] = reindexed[sales_col].fillna(0.0)
        pieces.append(reindexed)

    out = pd.concat(pieces, ignore_index=True)
    if clip_cfg.get("enabled"):
        out[filled_col] = _clip_series(out[filled_col], clip_cfg)

    return out.sort_values([date_col, store_col]).reset_index(drop=True)


def _clip_series(
    series: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """Apply config-driven clipping: absolute bounds or IQR. In-place on copy."""
    lower = config.get("lower")
    upper = config.get("upper")
    method = config.get("method")

    if method == "iqr":
        mult = config.get("iqr_multiplier", 1.5)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if lower is None:
            lower = q1 - mult * iqr
        if upper is None:
            upper = q3 + mult * iqr

    if lower is not None:
        series = series.clip(lower=lower)
    if upper is not None:
        series = series.clip(upper=upper)
    return series
