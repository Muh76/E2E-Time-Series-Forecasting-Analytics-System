"""
ETL transformers — validate, clean, and augment daily time series DataFrames.

Public API:
  Validation:
    ValidationResult        Dataclass returned by validation functions.
    validate_schema         Check column presence and dtype compatibility.
    validate_data           Check nulls, duplicates, and row count.
    validate_retail         Strict retail-specific validation (raises on failure).
    REQUIRED_RETAIL_COLUMNS Tuple of columns required by the retail contract.

  Cleaning:
    clean_dates             Normalize date column to daily frequency.
    clean_values            Fill nulls, clip outliers, deduplicate.
    clean                   Run clean_dates then clean_values.
    clean_retail            Full retail cleaning: reindex, fill, optional clip.

  Augmentation:
    add_gaussian_noise      Add Gaussian noise to the value column.
    augment                 Apply config-driven augmentation (noise).
    augment_timeseries      Deterministic augmentation: missing blocks, noise
                            regime shifts, trend changes.
"""

from .augment import add_gaussian_noise, augment, augment_timeseries
from .clean import clean, clean_dates, clean_retail, clean_values
from .validate import (
    REQUIRED_RETAIL_COLUMNS,
    ValidationResult,
    validate_data,
    validate_retail,
    validate_schema,
)

__all__ = [
    # validation
    "REQUIRED_RETAIL_COLUMNS",
    "ValidationResult",
    "validate_data",
    "validate_retail",
    "validate_schema",
    # cleaning
    "clean",
    "clean_dates",
    "clean_retail",
    "clean_values",
    # augmentation
    "add_gaussian_noise",
    "augment",
    "augment_timeseries",
]
