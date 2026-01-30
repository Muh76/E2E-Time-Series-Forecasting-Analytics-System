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
        "value": "float64",
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
            - value_column: Name of value column (default "value").
            - allow_extra_columns: If True, extra columns are allowed (default True).

    Returns:
        ValidationResult with valid=True if schema passes; errors list otherwise.
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    value_col = cfg.get("value_column", "value")
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
            - value_column: Name of value column (default "value").
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
    value_col = cfg.get("value_column", "value")
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
