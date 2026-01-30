"""
Unit tests for the ETL pipeline: validation, cleaning, augmentation.

Run from project root: pytest tests/unit/test_etl.py -v
"""

import pytest
import pandas as pd

from data.etl.validate import validate_retail, validate_schema, ValidationResult
from data.etl.clean import clean_retail
from data.etl.augment import augment_timeseries


# ---------------------------------------------------------------------------
# Fixtures: retail-style DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def df_retail_valid() -> pd.DataFrame:
    """Valid retail series: date, store_id, target; unique (date, store_id); monotonic per store."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02"]),
        "store_id": ["A", "A", "A", "B", "B"],
        "target": [100.0, 102.0, 98.0, 200.0, 205.0],
    })


@pytest.fixture
def df_with_gaps() -> pd.DataFrame:
    """Retail series with missing dates: store A has 2020-01-01 and 2020-01-03 but not 2020-01-02."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"]),
        "store_id": ["A", "A", "B", "B", "B"],
        "target": [10.0, 14.0, 20.0, 22.0, 24.0],
    })


@pytest.fixture
def df_for_augment() -> pd.DataFrame:
    """Simple series for augmentation: date + target (and optional store_id)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        "target": [100.0, 101.0, 102.0, 103.0, 104.0],
    })


# ---------------------------------------------------------------------------
# 1. Schema validation catches missing columns
# ---------------------------------------------------------------------------


def test_validate_retail_raises_on_missing_columns(df_retail_valid: pd.DataFrame) -> None:
    """Schema validation catches missing required columns (date, store_id, target)."""
    # Drop 'target' -> should raise
    df = df_retail_valid[["date", "store_id"]].copy()
    with pytest.raises(ValueError) as exc_info:
        validate_retail(df)
    assert "Missing required columns" in str(exc_info.value)
    assert "target" in str(exc_info.value)


def test_validate_retail_raises_on_missing_date_column(df_retail_valid: pd.DataFrame) -> None:
    """Missing date column is reported."""
    df = df_retail_valid.rename(columns={"date": "dt"})
    with pytest.raises(ValueError) as exc_info:
        validate_retail(df)
    assert "Missing required columns" in str(exc_info.value)
    assert "date" in str(exc_info.value)


def test_validate_schema_returns_errors_for_missing_columns() -> None:
    """Generic validate_schema returns ValidationResult with errors for missing columns."""
    df = pd.DataFrame({"target": [1.0, 2.0]})  # no date column
    result = validate_schema(df, config={"required_columns": ["date", "target"]})
    assert isinstance(result, ValidationResult)
    assert result.valid is False
    assert any("date" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 2. Duplicate rows are detected
# ---------------------------------------------------------------------------


def test_validate_retail_raises_on_duplicate_date_store_id(df_retail_valid: pd.DataFrame) -> None:
    """Duplicate (date, store_id) pairs are detected."""
    df = pd.concat([df_retail_valid, df_retail_valid.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError) as exc_info:
        validate_retail(df)
    assert "Duplicate" in str(exc_info.value)
    assert "date, store_id" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()


def test_validate_retail_passes_when_no_duplicates(df_retail_valid: pd.DataFrame) -> None:
    """Valid retail data with unique (date, store_id) passes."""
    validate_retail(df_retail_valid)  # no raise


# ---------------------------------------------------------------------------
# 3. Missing dates are handled correctly
# ---------------------------------------------------------------------------


def test_clean_retail_reindex_fills_missing_dates_forward_fill(df_with_gaps: pd.DataFrame) -> None:
    """Missing dates are filled: reindex to daily and forward-fill target_cleaned."""
    config = {
        "missing_dates": "reindex",
        "date_range": "per_store",
        "missing_value_strategy": "forward_fill",
    }
    out = clean_retail(df_with_gaps, config=config)
    # Store A: had 2020-01-01 and 2020-01-03; reindex adds 2020-01-02. Forward-fill: 2020-01-02 gets 10.0
    store_a = out[out["store_id"] == "A"].sort_values("date")
    assert len(store_a) == 3
    jan2 = pd.Timestamp("2020-01-02")
    row_jan2 = store_a[store_a["date"] == jan2].iloc[0]
    assert row_jan2["target_cleaned"] == 10.0  # forward-filled from 2020-01-01


def test_clean_retail_reindex_fills_missing_dates_zero(df_with_gaps: pd.DataFrame) -> None:
    """Missing dates filled with zero when missing_value_strategy is zero."""
    config = {
        "missing_dates": "reindex",
        "date_range": "per_store",
        "missing_value_strategy": "zero",
    }
    out = clean_retail(df_with_gaps, config=config)
    store_a = out[out["store_id"] == "A"].sort_values("date")
    row_jan2 = store_a[store_a["date"] == pd.Timestamp("2020-01-02")].iloc[0]
    assert row_jan2["target_cleaned"] == 0.0
    # Original target column should be NaN for the inserted row
    assert pd.isna(row_jan2["target"])


def test_clean_retail_preserves_original_target_column(df_with_gaps: pd.DataFrame) -> None:
    """Original target column is preserved; filled-in dates have NaN in target."""
    config = {"missing_dates": "reindex", "date_range": "per_store", "missing_value_strategy": "zero"}
    out = clean_retail(df_with_gaps, config=config)
    assert "target" in out.columns
    assert "target_cleaned" in out.columns
    # Rows that were in the original data should have non-NaN target
    orig_dates_stores = df_with_gaps.set_index(["date", "store_id"]).index
    for _, row in out.iterrows():
        if (row["date"], row["store_id"]) in orig_dates_stores:
            assert not pd.isna(row["target"])


# ---------------------------------------------------------------------------
# 4. Augmentation is reproducible with a fixed seed
# ---------------------------------------------------------------------------


def test_augment_timeseries_reproducible_with_fixed_seed(df_for_augment: pd.DataFrame) -> None:
    """Same config and seed produce identical target_augmented and augmentation_type."""
    config = {
        "missing_blocks": {"enabled": True, "n_blocks": 1, "block_size_min": 1, "block_size_max": 2},
        "noise_regime_shift": {"enabled": True, "n_shifts": 1, "scale_before": 0.0, "scale_after": 0.5},
        "trend_change": {"enabled": True, "n_windows": 1, "window_length_min": 2, "window_length_max": 4, "slope_min": -0.1, "slope_max": 0.1},
    }
    seed = 42
    out1 = augment_timeseries(df_for_augment.copy(), config=config, seed=seed)
    out2 = augment_timeseries(df_for_augment.copy(), config=config, seed=seed)
    pd.testing.assert_series_equal(out1["target_augmented"], out2["target_augmented"])
    pd.testing.assert_series_equal(out1["augmentation_type"], out2["augmentation_type"])


def test_augment_timeseries_different_seeds_differ(df_for_augment: pd.DataFrame) -> None:
    """Different seeds can produce different augmented values (when augmentation is enabled)."""
    config = {
        "noise_regime_shift": {"enabled": True, "n_shifts": 1, "scale_before": 0.0, "scale_after": 1.0},
    }
    out1 = augment_timeseries(df_for_augment.copy(), config=config, seed=1)
    out2 = augment_timeseries(df_for_augment.copy(), config=config, seed=2)
    # At least one value should differ (noise is added)
    assert not out1["target_augmented"].equals(out2["target_augmented"])


def test_augment_timeseries_preserves_original_column(df_for_augment: pd.DataFrame) -> None:
    """Original target column is preserved; target_original and target_augmented are added."""
    config = {"noise_regime_shift": {"enabled": True, "n_shifts": 1, "scale_before": 0.0, "scale_after": 0.1}}
    out = augment_timeseries(df_for_augment.copy(), config=config, seed=99)
    assert "target" in out.columns
    assert "target_original" in out.columns
    assert "target_augmented" in out.columns
    assert "augmentation_type" in out.columns
    pd.testing.assert_series_equal(out["target"], out["target_original"])
