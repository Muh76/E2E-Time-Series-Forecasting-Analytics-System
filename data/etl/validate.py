"""
Schema and basic data validation for daily time series.

Validates presence and types of required columns, date range, missing values,
and duplicate (date, series) keys. Config-driven rules; deterministic.
No model logic.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ValidationResult:
    """Result of schema or data validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        return self.valid


def _default_schema() -> dict[str, str]:
    """Default expected column names and dtypes for daily series."""
    return {
        "date": "datetime64[ns]",
        "target": "float64",
    }


def validate_schema(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Validate that the DataFrame has required columns and compatible dtypes.

    Args:
        df: Raw or intermediate DataFrame.
        config: Optional validation config. Expected keys (all optional):
            - required_columns: List of column names that must exist.
            - date_column: Name of date column (default "date").
            - value_column: Name of value column (default "target").
            - allow_extra_columns: If True, extra columns are allowed (default True).

    Returns:
        ValidationResult with valid=True if schema passes; errors list otherwise.
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    value_col = cfg.get("value_column", "target")
    required = cfg.get("required_columns") or [date_col, value_col]
    allow_extra = cfg.get("allow_extra_columns", True)

    errors: list[str] = []
    warnings: list[str] = []

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if not allow_extra:
        extra = set(df.columns) - set(required)
        if extra:
            warnings.append(f"Unexpected columns (allowed by config): {extra}")

    # Type checks: date column datetime-like, value numeric
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        errors.append(f"Column '{date_col}' must be datetime-like.")
    if date_col in df.columns and df[date_col].dt.tz is not None and not cfg.get("allow_tz", True):
        warnings.append("Date column is timezone-aware; downstream may assume UTC or local.")

    if value_col in df.columns and not pd.api.types.is_numeric_dtype(df[value_col]):
        errors.append(f"Column '{value_col}' must be numeric.")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_data(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Validate data quality: missing values, duplicates, date range.

    Assumes schema has already been validated (required columns exist).

    Args:
        df: DataFrame with at least date and value columns.
        config: Optional validation config. Expected keys (all optional):
            - date_column: Name of date column (default "date").
            - value_column: Name of value column (default "target").
            - series_column: If set, (date, series_column) must be unique.
            - allow_missing_value: If True, missing values in value column are
              allowed (default False).
            - min_rows: Minimum number of rows (default 1).
            - max_duplicate_ratio: Max allowed ratio of duplicate rows by
              (date, series) if series_column set (default 0.0).

    Returns:
        ValidationResult with valid=True if all checks pass.
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    value_col = cfg.get("value_column", "target")
    series_col = cfg.get("series_column")
    allow_missing = cfg.get("allow_missing_value", False)
    min_rows = cfg.get("min_rows", 1)
    max_duplicate_ratio = cfg.get("max_duplicate_ratio", 0.0)

    errors: list[str] = []
    warnings: list[str] = []

    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} < {min_rows}.")

    if not allow_missing and value_col in df.columns:
        null_count = df[value_col].isna().sum()
        if null_count > 0:
            errors.append(f"Missing values in '{value_col}': {null_count}.")

    key_cols = [date_col] if series_col is None else [date_col, series_col]
    if series_col is not None and series_col not in df.columns:
        errors.append(f"series_column '{series_col}' not in DataFrame.")
    elif all(c in df.columns for c in key_cols):
        n_before = len(df)
        n_unique = df[key_cols].drop_duplicates().shape[0]
        if n_before > 0:
            ratio = 1.0 - (n_unique / n_before)
            if ratio > max_duplicate_ratio:
                errors.append(
                    f"Duplicate (date, series) ratio {ratio:.4f} exceeds max {max_duplicate_ratio}."
                )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Retail time series validation (raises on failure)
# ---------------------------------------------------------------------------

REQUIRED_RETAIL_COLUMNS = ("date", "store_id", "target")


def validate_retail(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Validate retail time series data: schema and data rules.

    Checks:
    - Required columns exist: date, store_id, target (or config overrides).
    - Date column is datetime.
    - No duplicate (date, store_id) pairs.
    - Target values are non-negative.
    - Time index is strictly monotonic per store (dates increasing per store_id).

    Args:
        df: DataFrame to validate (e.g. after load_retail_sales_csv).
        config: Optional. Keys: date_column (default "date"), store_id_column
            (default "store_id"), target_column (default "target").

    Returns:
        None on success.

    Raises:
        ValueError: With a single message listing all failed checks (one or more
            of: missing columns, date not datetime, duplicates, negative target,
            non-monotonic dates per store).
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    store_col = cfg.get("store_id_column", "store_id")
    target_col = cfg.get("target_column", "target")
    required = (date_col, store_col, target_col)

    errors: list[str] = []

    # 1. Required columns exist
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(
            f"Missing required columns: {missing}. "
            f"Expected: date, store_id, target (or config overrides). Found: {list(df.columns)}."
        )
    if errors:
        raise ValueError("\n".join(errors))

    # 2. Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        errors.append(
            f"Column '{date_col}' must be datetime. "
            f"Got dtype: {df[date_col].dtype}. Use pd.to_datetime() before validation."
        )
    if errors:
        raise ValueError("\n".join(errors))

    # 3. No duplicate (date, store_id) pairs
    key = [date_col, store_col]
    n_rows = len(df)
    n_unique = df[key].drop_duplicates().shape[0]
    if n_rows != n_unique:
        n_dup = n_rows - n_unique
        dup_example = df[df.duplicated(subset=key, keep=False)].head(2)
        errors.append(
            f"Duplicate (date, store_id) pairs: {n_dup} duplicate row(s). "
            f"Expected unique (date, store_id). Example duplicates:\n{dup_example.to_string()}"
        )
    if errors:
        raise ValueError("\n".join(errors))

    # 4. Target non-negative
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        errors.append(
            f"Column '{target_col}' must be numeric. Got dtype: {df[target_col].dtype}."
        )
    else:
        neg = (df[target_col] < 0).sum()
        if neg > 0:
            min_target = df[target_col].min()
            errors.append(
                f"Target must be non-negative. Found {int(neg)} negative value(s). "
                f"Min value: {min_target}. Column: '{target_col}'."
            )
    if errors:
        raise ValueError("\n".join(errors))

    # 5. Time index strictly monotonic per store (no duplicate dates per store)
    non_monotonic_stores: list[Any] = []
    for store_id, group in df.groupby(store_col, sort=False):
        dates = group[date_col].sort_values()
        diffs = dates.diff().dropna()
        if (diffs <= pd.Timedelta(0)).any():
            non_monotonic_stores.append(store_id)
    if non_monotonic_stores:
        sample = non_monotonic_stores[:5]
        errors.append(
            f"Time index must be strictly monotonic per store. "
            f"Stores with non-monotonic or duplicate dates: {sample}"
            + (f" (and {len(non_monotonic_stores) - 5} more)" if len(non_monotonic_stores) > 5 else "")
            + f". Total: {len(non_monotonic_stores)} store(s)."
        )
    if errors:
        raise ValueError("\n".join(errors))
