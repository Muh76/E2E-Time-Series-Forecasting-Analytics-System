"""Tests for distribution drift (backend.services.drift)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.services import drift as drift_mod


@pytest.fixture
def drift_parquet_shifted(tmp_path):
    """Two windows with clearly different means (high drift signal)."""
    n = 80
    rng = np.random.default_rng(42)
    early = rng.normal(100.0, 5.0, n // 2)
    late = rng.normal(200.0, 5.0, n - n // 2)
    target = np.concatenate([early, late])
    df = pd.DataFrame(
        {
            "store_id": np.ones(n, dtype=int),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target_cleaned": target,
        }
    )
    path = tmp_path / "etl_output.parquet"
    df.to_parquet(path)
    return path


def test_drift_detects_shift(drift_parquet_shifted):
    out = drift_mod.compute_distribution_drift(
        parquet_path=drift_parquet_shifted,
        store_id=1,
    )
    assert out is not None
    assert 0.0 <= out["drift_score"] <= 1.0
    assert out["status"] in ("low", "medium", "high")
    assert out["drift_score"] >= 0.5
    assert out["status"] in ("medium", "high")


def test_drift_stable_data_low_score(tmp_path):
    n = 80
    rng = np.random.default_rng(0)
    v = rng.normal(50.0, 2.0, n)
    df = pd.DataFrame(
        {
            "store_id": np.ones(n, dtype=int),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target_cleaned": v,
        }
    )
    path = tmp_path / "p.parquet"
    df.to_parquet(path)
    out = drift_mod.compute_distribution_drift(parquet_path=path, store_id=1)
    assert out is not None
    assert out["drift_score"] < 0.5
    assert out["status"] == "low"


def test_drift_insufficient_rows(tmp_path):
    df = pd.DataFrame(
        {
            "store_id": [1, 1],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_cleaned": [1.0, 2.0],
        }
    )
    path = tmp_path / "small.parquet"
    df.to_parquet(path)
    assert drift_mod.compute_distribution_drift(parquet_path=path, store_id=1) is None


def test_drift_missing_file():
    assert drift_mod.compute_distribution_drift(parquet_path=Path("/nonexistent/drift.parquet")) is None
