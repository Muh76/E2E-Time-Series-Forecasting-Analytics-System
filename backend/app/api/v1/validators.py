"""
Shared request validators for the V1 API layer.

Provides a cached set of valid store IDs loaded from the processed dataset
and reusable Pydantic field validators for store_id and horizon.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_PARQUET_PATH = _PROJECT_ROOT / "data" / "processed" / "etl_output.parquet"

_valid_store_ids: set[int] | None = None


def get_valid_store_ids() -> set[int]:
    """
    Return the set of store IDs present in the processed dataset.

    Loaded once and cached for the process lifetime.
    """
    global _valid_store_ids
    if _valid_store_ids is None:
        if not _PARQUET_PATH.exists():
            logger.warning(
                "Cannot load valid store IDs: %s not found. "
                "Store ID validation will be skipped until file is available.",
                _PARQUET_PATH,
            )
            return set()
        ids = pd.read_parquet(_PARQUET_PATH, columns=["store_id"])["store_id"]
        _valid_store_ids = set(ids.unique().tolist())
        logger.info(
            "Loaded %d valid store IDs from %s", len(_valid_store_ids), _PARQUET_PATH
        )
    return _valid_store_ids


HORIZON_MIN = 1
HORIZON_MAX = 60
N_SPLITS_MIN = 1
N_SPLITS_MAX = 20
