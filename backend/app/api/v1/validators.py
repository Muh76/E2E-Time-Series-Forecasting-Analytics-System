"""
Shared request validators for the V1 API layer.

Provides a cached set of valid store IDs loaded from the processed dataset
and reusable Pydantic field validators for store_id and horizon.
"""

import logging

import pandas as pd

from backend.app.runtime_paths import processed_parquet_path

logger = logging.getLogger(__name__)

_valid_store_ids: set[int] | None = None


def get_valid_store_ids() -> set[int]:
    """
    Return the set of store IDs present in the processed dataset.

    Loaded once and cached for the process lifetime.
    """
    global _valid_store_ids
    if _valid_store_ids is None:
        pq = processed_parquet_path()
        if not pq.exists():
            logger.warning(
                "Cannot load valid store IDs: %s not found. "
                "Store ID validation will be skipped until file is available.",
                pq,
            )
            return set()
        try:
            ids = pd.read_parquet(pq, columns=["store_id"])["store_id"]
            _valid_store_ids = set(ids.unique().tolist())
            logger.info("Loaded %d valid store IDs from %s", len(_valid_store_ids), pq)
        except ImportError as exc:
            # Keep API alive when parquet engines are missing (e.g., pyarrow).
            logger.error(
                "Cannot load valid store IDs from %s due to missing parquet engine: %s. "
                "Install pyarrow/fastparquet to enable strict store_id validation.",
                pq,
                exc,
            )
            _valid_store_ids = set()
        except (KeyError, ValueError, OSError) as exc:
            logger.error(
                "Cannot load valid store IDs from %s: %s. " "Store ID validation will be skipped for this process.",
                pq,
                exc,
            )
            _valid_store_ids = set()
    return _valid_store_ids


HORIZON_MIN = 1
HORIZON_MAX = 60
N_SPLITS_MIN = 1
N_SPLITS_MAX = 20
