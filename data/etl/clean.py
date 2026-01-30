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
