"""
ETL package: ingest → validate → clean → augment → save.

Subdirectory structure (canonical locations):
    data.etl.extractors     CSV loaders and source-specific extractors
    data.etl.transformers   Validation, cleaning, and augmentation
    data.etl.loaders        Parquet persistence (save / load)

Flat-module names (ingest, validate, clean, augment) are kept as
backward-compatibility shims so existing imports continue to work.

Typical usage
-------------
# High-level pipeline (recommended)
from data.etl.pipeline import run_pipeline

# Subdirectory API (canonical)
from data.etl.extractors import load_retail_sales_csv, RossmannETL
from data.etl.transformers import validate_retail, clean_retail, augment_timeseries
from data.etl.loaders import save_parquet, load_parquet

# Legacy flat imports (still work via shims)
from data.etl import clean_retail, augment_timeseries, validate_retail
"""

# Sub-packages — expose as data.etl.extractors, .transformers, .loaders
from . import augment, clean, extractors, ingest, loaders, pipeline, transformers, validate

# Flat re-exports for backward compatibility
from .augment import augment_timeseries
from .clean import clean_retail
from .rossmann_etl import RossmannETL
from .validate import REQUIRED_RETAIL_COLUMNS, ValidationResult, validate_retail

__all__ = [
    # sub-packages (new canonical API)
    "extractors",
    "transformers",
    "loaders",
    "pipeline",
    # flat shim modules (backward compat)
    "augment",
    "clean",
    "ingest",
    "validate",
    # commonly-used names re-exported at top level
    "REQUIRED_RETAIL_COLUMNS",
    "RossmannETL",
    "ValidationResult",
    "augment_timeseries",
    "clean_retail",
    "validate_retail",
]
