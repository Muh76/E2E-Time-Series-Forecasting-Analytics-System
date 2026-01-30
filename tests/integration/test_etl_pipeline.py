"""
Lightweight integration test for the ETL pipeline.

Runs pipeline with test config, writes parquet, loads and asserts on output.
No model logic. Low runtime (small fixture).
"""

import pytest
import pandas as pd
from pathlib import Path

from data.etl.pipeline import run_pipeline


# Minimal retail CSV: two stores, few days, no gaps (monotonic per entity)
_MINIMAL_CSV = """date,store_id,target
2020-01-01,A,10.0
2020-01-02,A,11.0
2020-01-03,A,12.0
2020-01-01,B,20.0
2020-01-02,B,21.0
2020-01-03,B,22.0
"""


@pytest.fixture
def raw_csv_path(tmp_path: Path) -> Path:
    """Write minimal retail CSV to a temp file."""
    p = tmp_path / "raw.csv"
    p.write_text(_MINIMAL_CSV)
    return p


@pytest.fixture
def etl_config(raw_csv_path: Path) -> dict:
    """Test config: retail pipeline, reindex + zero fill, timeseries augmentation with noise."""
    return {
        "ingest": {"path": str(raw_csv_path)},
        "validate": {},
        "clean": {
            "missing_dates": "reindex",
            "date_range": "per_store",
            "missing_value_strategy": "zero",
        },
        "augment": {
            "timeseries_enabled": True,
            "seed": 42,
            "noise_regime_shift": {"enabled": True, "n_shifts": 1, "scale_before": 0.0, "scale_after": 0.5},
        },
    }


def test_etl_pipeline_run_and_parquet_assertions(
    raw_csv_path: Path, etl_config: dict, tmp_path: Path
) -> None:
    """
    Run ETL pipeline with test config, write parquet, load and assert:
    - target_cleaned has no NaNs
    - dates are monotonic per entity
    - target_augmented exists
    - augmentation flags consistent with config (noise_shift enabled -> appears in augmentation_type)
    """
    df = run_pipeline(raw_csv_path, config=etl_config, run_augment=True, pipeline_mode="retail")
    out_parquet = tmp_path / "processed.parquet"
    df.to_parquet(out_parquet, index=False)
    loaded = pd.read_parquet(out_parquet)

    # 1. target_cleaned has no NaNs (zero-fill used)
    assert "target_cleaned" in loaded.columns
    assert loaded["target_cleaned"].notna().all(), "target_cleaned must have no NaNs"

    # 2. Dates strictly monotonic per entity
    date_col, entity_col = "date", "store_id"
    for _entity, group in loaded.groupby(entity_col, sort=False):
        dates = pd.to_datetime(group[date_col]).sort_values()
        diffs = dates.diff().dropna()
        assert (diffs > pd.Timedelta(0)).all(), f"Dates must be strictly monotonic per {entity_col}"

    # 3. target_augmented exists
    assert "target_augmented" in loaded.columns
    assert loaded["target_augmented"].notna().any(), "target_augmented must be present"

    # 4. Augmentation flags consistent with config (noise_regime_shift enabled)
    assert "augmentation_type" in loaded.columns
    types = loaded["augmentation_type"].astype(str)
    assert types.str.contains("noise_shift", na=False).any(), (
        "augmentation_type must contain noise_shift when noise_regime_shift is enabled"
    )
