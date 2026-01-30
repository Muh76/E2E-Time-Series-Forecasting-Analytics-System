"""
Ingest raw CSV data for daily time series.

Loads CSV files from disk with config-driven encoding, date parsing,
and column selection. No model logic; deterministic and testable.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_csv(
    path: str | Path,
    config: dict[str, Any] | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load raw time series data from a CSV file.

    Expects daily series with at least a date column and one or more value
    columns. Config drives encoding, date column name, and dtype hints.
    Extra keyword arguments are passed through to pandas.read_csv.

    Args:
        path: Path to the CSV file (local path).
        config: Optional ingest config. Expected keys (all optional):
            - date_column: Name of the date/datetime column (default "date").
            - encoding: File encoding (default "utf-8").
            - parse_dates: If True, parse date_column to datetime (default True).
            - value_columns: List of value column names; if None, all non-date
              columns are treated as values.
        **read_csv_kwargs: Passed to pd.read_csv (e.g. sep, skiprows).

    Returns:
        DataFrame with raw data. Date column is parsed to datetime if
        config["parse_dates"] is True.

    Raises:
        FileNotFoundError: If path does not exist.
        pd.errors.EmptyDataError: If CSV is empty (pandas behavior).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    cfg = config or {}
    encoding = cfg.get("encoding", "utf-8")
    date_column = cfg.get("date_column", "date")
    parse_dates = cfg.get("parse_dates", True)

    df = pd.read_csv(path, encoding=encoding, **read_csv_kwargs)

    if df.empty:
        return df

    if parse_dates and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], utc=False)

    return df


def load_retail_sales_csv(
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load daily retail sales data from a CSV file.

    Loads from a configurable path, parses the date column safely (invalid
    dates raise), and sorts by date and entity (e.g. store_id). No other
    transformations. Raises clear errors on missing file, empty data,
    missing columns, or date parse failure.

    Args:
        path: Path to the CSV file. If None, config["path"] is used (required).
        config: Optional config. Expected keys:
            - path: Path to CSV (required if path arg is None).
            - date_column: Name of date column (default "date").
            - entity_column: Name of entity column for sorting, e.g. store_id
              (default "store_id").
            - encoding: File encoding (default "utf-8").
            - date_errors: Passed to pd.to_datetime: "raise" (default) or "coerce".
        **read_csv_kwargs: Passed to pd.read_csv (e.g. sep, skiprows).

    Returns:
        DataFrame with parsed dates, sorted by date then entity_column.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If path is missing, file is empty, required columns are
            missing, or date parsing fails (when date_errors="raise").
    """
    cfg = config or {}
    file_path = path if path is not None else cfg.get("path")
    if file_path is None:
        raise ValueError("Path is required: pass path= or config['path']")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Retail sales CSV not found: {file_path}")

    date_column = cfg.get("date_column", "date")
    entity_column = cfg.get("entity_column", "store_id")
    encoding = cfg.get("encoding", "utf-8")
    date_errors = cfg.get("date_errors", "raise")

    try:
        df = pd.read_csv(file_path, encoding=encoding, **read_csv_kwargs)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Retail sales CSV is empty: {file_path}") from None
    except Exception as e:
        raise type(e)(f"Failed to read CSV {file_path}: {e}") from e

    if df.empty:
        raise ValueError(f"Retail sales CSV has no rows: {file_path}")

    missing = [c for c in (date_column, entity_column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Retail sales CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    try:
        df[date_column] = pd.to_datetime(df[date_column], utc=False, errors=date_errors)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to parse date column '{date_column}' in {file_path}: {e}"
        ) from e

    if date_errors == "coerce" and df[date_column].isna().any():
        bad_count = df[date_column].isna().sum()
        raise ValueError(
            f"Date column '{date_column}' has {bad_count} unparseable value(s) in {file_path}"
        )

    df = df.sort_values([date_column, entity_column]).reset_index(drop=True)
    return df
