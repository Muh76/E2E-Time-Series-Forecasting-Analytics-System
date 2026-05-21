"""
Parquet loader — persist and reload processed ETL output.

Writes the final ETL DataFrame to Parquet with an enforced, explicit schema
so that downstream feature engineering and model training always read the same
column set, dtypes, and sort order.  Deterministic: same input → same file.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present in every processed ETL output.
# The loader verifies these before writing to catch upstream bugs early.
REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "store_id",
    "target_cleaned",
)


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Persist processed ETL output to a Parquet file.

    Validates required columns, sorts by (date, store_id) for deterministic
    row order, and writes with the snappy compression codec. Creates parent
    directories automatically.

    Args:
        df: Processed DataFrame produced by the ETL pipeline.
        path: Destination Parquet file path.
        config: Optional loader config. Supported keys:
            - compression: Parquet compression codec (default "snappy").
            - required_columns: Override the required column check (default
              REQUIRED_OUTPUT_COLUMNS).
            - date_column: Name of date column used for sort (default "date").
            - entity_column: Name of entity column used for sort (default "store_id").
        overwrite: If False, raise FileExistsError when the file already exists.

    Returns:
        Resolved Path to the written Parquet file.

    Raises:
        ValueError: If required columns are missing from df.
        FileExistsError: If overwrite=False and path already exists.
        OSError: If the parent directory cannot be created.
    """
    cfg = config or {}
    compression = cfg.get("compression", "snappy")
    required = cfg.get("required_columns", REQUIRED_OUTPUT_COLUMNS)
    date_col = cfg.get("date_column", "date")
    entity_col = cfg.get("entity_column", "store_id")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"ETL output is missing required columns: {missing}. "
            f"Found: {list(df.columns)}. "
            "Ensure the ETL pipeline ran successfully before saving."
        )

    out_path = Path(path).resolve()

    if not overwrite and out_path.exists():
        raise FileExistsError(f"Parquet file already exists: {out_path}. " "Pass overwrite=True to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sort_cols = [c for c in (date_col, entity_col) if c in df.columns]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df

    df_sorted.to_parquet(out_path, index=False, compression=compression)

    logger.info(
        "ETL output saved: path=%s rows=%d cols=%d compression=%s",
        out_path,
        len(df_sorted),
        len(df_sorted.columns),
        compression,
    )
    return out_path


def load_parquet(
    path: str | Path,
    config: dict[str, Any] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load processed ETL output from a Parquet file.

    Validates that the file exists and contains the required columns.
    Parses the date column to datetime if not already.

    Args:
        path: Path to the Parquet file written by save_parquet.
        config: Optional loader config. Supported keys:
            - required_columns: Columns to verify after loading (default
              REQUIRED_OUTPUT_COLUMNS).
            - date_column: Name of date column to parse (default "date").
        columns: If provided, load only these columns (passed to pd.read_parquet).

    Returns:
        DataFrame with at least the required columns, sorted by (date, store_id).

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        ValueError: If required columns are missing from the loaded DataFrame.
    """
    cfg = config or {}
    required = cfg.get("required_columns", REQUIRED_OUTPUT_COLUMNS)
    date_col = cfg.get("date_column", "date")

    in_path = Path(path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(
            f"Processed ETL Parquet not found: {in_path}. " "Run the ETL pipeline (scripts/run_etl.py) to generate it."
        )

    df = pd.read_parquet(in_path, columns=columns)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Loaded Parquet is missing required columns: {missing}. "
            f"Found: {list(df.columns)}. "
            "The file may have been written by an older pipeline version."
        )

    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    logger.info(
        "ETL output loaded: path=%s rows=%d cols=%d",
        in_path,
        len(df),
        len(df.columns),
    )
    return df


def parquet_metadata(path: str | Path) -> dict[str, Any]:
    """
    Return lightweight metadata about a saved Parquet file without loading all rows.

    Args:
        path: Path to the Parquet file.

    Returns:
        Dict with keys: path, row_count, column_count, columns, size_bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import pyarrow.parquet as pq

    in_path = Path(path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {in_path}")

    pf = pq.read_metadata(in_path)
    schema = pq.read_schema(in_path)

    return {
        "path": str(in_path),
        "row_count": pf.num_rows,
        "column_count": pf.num_columns,
        "columns": schema.names,
        "size_bytes": in_path.stat().st_size,
    }
