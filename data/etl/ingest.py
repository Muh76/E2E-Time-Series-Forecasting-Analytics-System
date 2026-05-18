"""
Backward-compatibility shim — ingest logic lives in data.etl.extractors.csv.

Import from the canonical location instead:
    from data.etl.extractors import load_raw_csv, load_retail_sales_csv
"""

from .extractors.csv import load_raw_csv, load_retail_sales_csv

__all__ = [
    "load_raw_csv",
    "load_retail_sales_csv",
]
