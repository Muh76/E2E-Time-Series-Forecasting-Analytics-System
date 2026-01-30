"""Unit tests for the feature engineering pipeline."""

import pandas as pd
import pytest

from data.feature_engineering import run_feature_pipeline


@pytest.fixture
def etl_like_df():
    """Minimal processed ETL output: date, entity, target_cleaned."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "store_id": ["A"] * 15 + ["B"] * 15,
            "target": [10.0] * 30,
            "target_cleaned": [10.0] * 30,
        }
    )


@pytest.fixture
def feature_config():
    """Config slice for feature_engineering."""
    return {
        "feature_engineering": {
            "date_column": "date",
            "target_column": "target_cleaned",
            "entity_column": "store_id",
            "lag": {"lags": [1, 7]},
            "rolling": {"windows": [7], "min_periods": 1},
            "calendar": {},
        }
    }


def test_run_feature_pipeline_preserves_rows_and_index(etl_like_df, feature_config):
    """Pipeline must not drop rows; index and row count preserved."""
    out = run_feature_pipeline(etl_like_df, feature_config)
    assert len(out) == len(etl_like_df)
    assert list(out.index) == list(etl_like_df.index)


def test_run_feature_pipeline_preserves_identifiers(etl_like_df, feature_config):
    """Entity and date columns must be preserved."""
    out = run_feature_pipeline(etl_like_df, feature_config)
    assert "store_id" in out.columns
    assert "date" in out.columns
    assert "target_cleaned" in out.columns
    assert out["store_id"].equals(etl_like_df["store_id"])


def test_run_feature_pipeline_adds_lag_rolling_calendar(etl_like_df, feature_config):
    """Output must contain lag, rolling, and calendar feature columns."""
    out = run_feature_pipeline(etl_like_df, feature_config)
    assert "lag_1" in out.columns and "lag_7" in out.columns
    assert "rolling_mean_7" in out.columns and "rolling_std_7" in out.columns
    assert "day_of_week" in out.columns and "month" in out.columns and "is_weekend" in out.columns


def test_run_feature_pipeline_no_mutation(etl_like_df, feature_config):
    """Input DataFrame must not be mutated."""
    original_cols = set(etl_like_df.columns)
    run_feature_pipeline(etl_like_df, feature_config)
    assert set(etl_like_df.columns) == original_cols


def test_run_feature_pipeline_default_config(etl_like_df):
    """Pipeline runs with None config (defaults)."""
    out = run_feature_pipeline(etl_like_df, None)
    assert len(out) == len(etl_like_df)
    assert "lag_1" in out.columns
    assert "rolling_mean_7" in out.columns or "rolling_mean_14" in out.columns
    assert "day_of_week" in out.columns
