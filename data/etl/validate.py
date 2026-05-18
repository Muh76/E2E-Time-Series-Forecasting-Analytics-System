"""
Backward-compatibility shim — validation logic lives in data.etl.transformers.validate.

Import from the canonical location instead:
    from data.etl.transformers import validate_schema, validate_data, validate_retail
"""

from .transformers.validate import (
    REQUIRED_RETAIL_COLUMNS,
    ValidationResult,
    validate_data,
    validate_retail,
    validate_schema,
)

__all__ = [
    "REQUIRED_RETAIL_COLUMNS",
    "ValidationResult",
    "validate_data",
    "validate_retail",
    "validate_schema",
]
