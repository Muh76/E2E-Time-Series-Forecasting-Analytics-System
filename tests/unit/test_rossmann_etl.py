"""
Unit tests for RossmannETL: merge, normalize_schema, clean.

Uses small synthetic DataFrames only (no CSV loading).
Run from project root: pytest tests/unit/test_rossmann_etl.py -v
"""

import pytest
import pandas as pd

from data.etl.rossmann_etl import RossmannETL


# ---------------------------------------------------------------------------
# Fixtures: synthetic train and store DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Minimal train with required columns: 2 stores, 3 dates each."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2),
        "Store": [1, 1, 1, 2, 2, 2],
        "Sales": [100.0, 105.0, 98.0, 200.0, 210.0, 195.0],
        "Open": [1, 1, 1, 1, 0, 1],
        "Promo": [0, 1, 0, 0, 0, 1],
        "StateHoliday": ["a", "a", "a", "a", "a", "a"],
        "SchoolHoliday": [0, 0, 1, 0, 0, 0],
    })


@pytest.fixture
def store_df() -> pd.DataFrame:
    """Minimal store with required columns."""
    return pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "b"],
        "CompetitionDistance": [100.0, 250.0],
    })


@pytest.fixture
def store_df_with_missing_competition() -> pd.DataFrame:
    """Store with one missing CompetitionDistance."""
    return pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "b"],
        "CompetitionDistance": [100.0, float("nan")],
    })


# ---------------------------------------------------------------------------
# 1. Train + store merge produces expected columns
# ---------------------------------------------------------------------------


def test_merge_produces_expected_columns(train_df: pd.DataFrame, store_df: pd.DataFrame) -> None:
    """Merge produces expected columns; normalized output has date, store_id, target_raw, target_cleaned, etc."""
    etl = RossmannETL()
    merged = etl.merge(train_df, store_df)
    normalized = etl.normalize_schema(merged)
    cleaned = etl.clean(normalized)

    expected = {"date", "store_id", "target_raw", "target_cleaned", "open", "promo", "school_holiday", "state_holiday", "store_type", "assortment", "competition_distance"}
    assert expected.issubset(set(cleaned.columns)), f"Missing columns: {expected - set(cleaned.columns)}"


# ---------------------------------------------------------------------------
# 2. target_cleaned == 0 when Open == 0
# ---------------------------------------------------------------------------


def test_target_cleaned_zero_when_open_zero(train_df: pd.DataFrame, store_df: pd.DataFrame) -> None:
    """Rows with Open == 0 have target_cleaned = 0."""
    etl = RossmannETL()
    merged = etl.merge(train_df, store_df)
    normalized = etl.normalize_schema(merged)
    cleaned = etl.clean(normalized)

    closed_rows = cleaned[cleaned["open"] == 0]
    assert len(closed_rows) >= 1, "Fixture has at least one row with Open==0"
    assert (closed_rows["target_cleaned"] == 0.0).all()


# ---------------------------------------------------------------------------
# 3. No duplicate (store_id, date)
# ---------------------------------------------------------------------------


def test_no_duplicate_store_id_date(train_df: pd.DataFrame, store_df: pd.DataFrame) -> None:
    """Output has no duplicate (store_id, date) keys."""
    etl = RossmannETL()
    merged = etl.merge(train_df, store_df)
    normalized = etl.normalize_schema(merged)
    cleaned = etl.clean(normalized)

    dupes = cleaned.duplicated(subset=["store_id", "date"])
    assert not dupes.any(), f"duplicate (store_id, date): {cleaned[dupes]}"


# ---------------------------------------------------------------------------
# 4. Sorted by store_id then date
# ---------------------------------------------------------------------------


def test_sorted_by_store_id_then_date(train_df: pd.DataFrame, store_df: pd.DataFrame) -> None:
    """Output is sorted by (store_id, date)."""
    etl = RossmannETL()
    merged = etl.merge(train_df, store_df)
    normalized = etl.normalize_schema(merged)
    cleaned = etl.clean(normalized)

    sorted_df = cleaned.sort_values(["store_id", "date"], ignore_index=True)
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), sorted_df)


# ---------------------------------------------------------------------------
# 5. Missing CompetitionDistance is filled
# ---------------------------------------------------------------------------


def test_missing_competition_distance_filled() -> None:
    """Missing CompetitionDistance is filled with median."""
    train = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "Store": [1, 2],
        "Sales": [100.0, 200.0],
        "Open": [1, 1],
        "Promo": [0, 0],
        "StateHoliday": ["a", "a"],
        "SchoolHoliday": [0, 0],
    })
    store = pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "b"],
        "CompetitionDistance": [100.0, float("nan")],
    })
    etl = RossmannETL()
    merged = etl.merge(train, store)
    normalized = etl.normalize_schema(merged)
    cleaned = etl.clean(normalized)

    assert cleaned["competition_distance"].notna().all()
    # Store 2 had NaN; median of [100, nan] = 100, so store 2 gets 100
    assert cleaned[cleaned["store_id"] == 2]["competition_distance"].iloc[0] == 100.0


# ---------------------------------------------------------------------------
# 6. Raises ValueError if required columns missing
# ---------------------------------------------------------------------------


def test_merge_raises_when_train_missing_store() -> None:
    """merge raises ValueError when train_df missing Store."""
    etl = RossmannETL()
    train = pd.DataFrame({"Date": ["2020-01-01"], "Sales": [100.0]})  # no Store
    store = pd.DataFrame({"Store": [1], "StoreType": ["a"], "Assortment": ["a"], "CompetitionDistance": [100.0]})
    with pytest.raises(ValueError) as exc_info:
        etl.merge(train, store)
    assert "Store" in str(exc_info.value)
    assert "train" in str(exc_info.value).lower()


def test_merge_raises_when_store_missing_store() -> None:
    """merge raises ValueError when store_df missing Store."""
    etl = RossmannETL()
    train = pd.DataFrame({"Date": ["2020-01-01"], "Store": [1], "Sales": [100.0], "Open": [1], "Promo": [0], "StateHoliday": ["a"], "SchoolHoliday": [0]})
    store = pd.DataFrame({"StoreType": ["a"], "Assortment": ["a"], "CompetitionDistance": [100.0]})  # no Store
    with pytest.raises(ValueError) as exc_info:
        etl.merge(train, store)
    assert "Store" in str(exc_info.value)
    assert "store" in str(exc_info.value).lower()


def test_normalize_schema_raises_when_columns_missing() -> None:
    """normalize_schema raises ValueError when required columns missing."""
    etl = RossmannETL()
    df = pd.DataFrame({
        "Date": ["2020-01-01"],
        "Store": [1],
        "Sales": [100.0],
        # missing Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance
    })
    with pytest.raises(ValueError) as exc_info:
        etl.normalize_schema(df)
    assert "missing required columns" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


def test_clean_raises_when_columns_missing() -> None:
    """clean raises ValueError when required columns missing."""
    etl = RossmannETL()
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "store_id": [1],
        "target_raw": [100.0],
        "target_cleaned": [100.0],
        # missing open, promo, school_holiday, competition_distance, state_holiday, store_type, assortment
    })
    with pytest.raises(ValueError) as exc_info:
        etl.clean(df)
    assert "missing required columns" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()
