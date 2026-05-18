"""
ETL loaders — persist and reload processed ETL output.

Public API:
    save_parquet            Write a processed ETL DataFrame to Parquet.
    load_parquet            Load a previously saved Parquet file.
    parquet_metadata        Inspect row/column counts without loading all data.
    REQUIRED_OUTPUT_COLUMNS Columns every ETL output must contain.
"""

from .parquet import REQUIRED_OUTPUT_COLUMNS, load_parquet, parquet_metadata, save_parquet

__all__ = [
    "REQUIRED_OUTPUT_COLUMNS",
    "load_parquet",
    "parquet_metadata",
    "save_parquet",
]
