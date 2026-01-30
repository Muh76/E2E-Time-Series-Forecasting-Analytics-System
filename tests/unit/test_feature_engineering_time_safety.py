"""
Unit tests for time safety of feature engineering transformers.

- Lag: never use future values (only past).
- Rolling: window excludes current and future rows.
- No cross-entity leakage.
- Output row count equals input row count.
"""

import pandas as pd
import pytest

from data.feature_engineering.transformers import (
    CalendarTransformer,
    LagTransformer,
    RollingTransformer,
)


# ---------------------------------------------------------------------------
# Fixtures: small synthetic DataFrames with known values
# ---------------------------------------------------------------------------


@pytest.fixture
def single_entity_ordered():
    """One entity, 5 rows, strict date order. Values 100, 200, 300, 400, 500."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "target_cleaned": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def two_entities_distinct_values():
    """
    Two entities (A, B), 5 dates each. Distinct value ranges so we can detect leakage.
    A: 1, 2, 3, 4, 5  |  B: 100, 200, 300, 400, 500
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "store_id": ["A"] * 5 + ["B"] * 5,
            "target_cleaned": [1.0, 2.0, 3.0, 4.0, 5.0] + [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def two_entities_unsorted():
    """Same as two_entities_distinct_values but rows interleaved (A0, B0, A1, B1, ...)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    for i in range(5):
        rows.append({"date": dates[i], "store_id": "A", "target_cleaned": float(i + 1)})
        rows.append({"date": dates[i], "store_id": "B", "target_cleaned": float((i + 1) * 100)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Row count: output rows == input rows
# ---------------------------------------------------------------------------


def test_lag_output_row_count_equals_input(single_entity_ordered, two_entities_distinct_values):
    """LagTransformer output must have same number of rows as input."""
    cfg = {"date_column": "date", "target_column": "target_cleaned", "lags": [1, 2]}
    t = LagTransformer().fit(single_entity_ordered, cfg)
    out = t.transform(single_entity_ordered)
    assert len(out) == len(single_entity_ordered)

    cfg["entity_column"] = "store_id"
    t2 = LagTransformer().fit(two_entities_distinct_values, cfg)
    out2 = t2.transform(two_entities_distinct_values)
    assert len(out2) == len(two_entities_distinct_values)


def test_rolling_output_row_count_equals_input(single_entity_ordered, two_entities_distinct_values):
    """RollingTransformer output must have same number of rows as input."""
    cfg = {"date_column": "date", "target_column": "target_cleaned", "windows": [3], "min_periods": 1}
    t = RollingTransformer().fit(single_entity_ordered, cfg)
    out = t.transform(single_entity_ordered)
    assert len(out) == len(single_entity_ordered)

    cfg["entity_column"] = "store_id"
    t2 = RollingTransformer().fit(two_entities_distinct_values, cfg)
    out2 = t2.transform(two_entities_distinct_values)
    assert len(out2) == len(two_entities_distinct_values)


def test_calendar_output_row_count_equals_input(single_entity_ordered):
    """CalendarTransformer output must have same number of rows as input."""
    t = CalendarTransformer().fit(single_entity_ordered, {"date_column": "date"})
    out = t.transform(single_entity_ordered)
    assert len(out) == len(single_entity_ordered)


# ---------------------------------------------------------------------------
# Lag: never use future values
# ---------------------------------------------------------------------------


def test_lag_uses_only_past_values_single_entity(single_entity_ordered):
    """
    At time t, lag_k must equal value at t-k (past). With values [100,200,300,400,500]:
    - Row 0 (100): lag_1 NaN, lag_2 NaN
    - Row 1 (200): lag_1 = 100, lag_2 NaN
    - Row 2 (300): lag_1 = 200, lag_2 = 100
    - Row 3 (400): lag_1 = 300, lag_2 = 200
    - Row 4 (500): lag_1 = 400, lag_2 = 300
    So lag_1 never equals current row value; always previous or NaN.
    """
    cfg = {"date_column": "date", "target_column": "target_cleaned", "lags": [1, 2]}
    t = LagTransformer().fit(single_entity_ordered, cfg)
    out = t.transform(single_entity_ordered)

    # Align by index (input is already date-sorted)
    combined = single_entity_ordered.copy()
    combined["lag_1"] = out["lag_1"].values
    combined["lag_2"] = out["lag_2"].values

    # Explicit expected: shift(1) and shift(2) in date order
    expected_lag_1 = [float("nan"), 100.0, 200.0, 300.0, 400.0]
    expected_lag_2 = [float("nan"), float("nan"), 100.0, 200.0, 300.0]

    for i in range(5):
        if i == 0:
            assert pd.isna(combined["lag_1"].iloc[i])
            assert pd.isna(combined["lag_2"].iloc[i])
        else:
            assert combined["lag_1"].iloc[i] == expected_lag_1[i]
        if i >= 2:
            assert combined["lag_2"].iloc[i] == expected_lag_2[i]

    # Sanity: no row has lag_1 equal to its own target_cleaned (would indicate current-value leak)
    for i in range(5):
        if not pd.isna(combined["lag_1"].iloc[i]):
            assert combined["lag_1"].iloc[i] != combined["target_cleaned"].iloc[i]


def test_lag_last_row_does_not_see_future(single_entity_ordered):
    """Last row (value 500): lag_1 must be 400 (previous), never 500 or any future."""
    cfg = {"date_column": "date", "target_column": "target_cleaned", "lags": [1]}
    t = LagTransformer().fit(single_entity_ordered, cfg)
    out = t.transform(single_entity_ordered)
    # Last row index
    last_idx = single_entity_ordered.index[-1]
    assert out.loc[last_idx, "lag_1"] == 400.0


# ---------------------------------------------------------------------------
# Rolling: window excludes current and future rows
# ---------------------------------------------------------------------------


def test_rolling_excludes_current_row(single_entity_ordered):
    """
    Rolling uses shift(1).rolling(), so at row t the window is over t-1, t-2, ...
    Values [10, 20, 30, 40, 50]. At last row (50), rolling_mean_3 = mean(20, 30, 40) = 30.
    """
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "target_cleaned": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    cfg = {"date_column": "date", "target_column": "target_cleaned", "windows": [3], "min_periods": 1}
    t = RollingTransformer().fit(df, cfg)
    out = t.transform(df)

    # Last row: window = previous 3 values (20, 30, 40) -> mean 30
    last_idx = df.index[-1]
    assert out.loc[last_idx, "rolling_mean_3"] == 30.0
    # Current value 50 must not be in the window
    assert out.loc[last_idx, "rolling_mean_3"] != 50.0


def test_rolling_first_row_has_nan_or_min_periods(single_entity_ordered):
    """First row has no past; rolling over shift(1) gives NaN at position 0."""
    cfg = {"date_column": "date", "target_column": "target_cleaned", "windows": [3], "min_periods": 1}
    t = RollingTransformer().fit(single_entity_ordered, cfg)
    out = t.transform(single_entity_ordered)
    first_idx = single_entity_ordered.index[0]
    # shift(1) makes first row NaN for the series we roll; so rolling mean at first row is NaN
    assert pd.isna(out.loc[first_idx, "rolling_mean_3"])


# ---------------------------------------------------------------------------
# No cross-entity leakage
# ---------------------------------------------------------------------------


def test_lag_no_cross_entity_leakage(two_entities_distinct_values):
    """
    Entity A has values 1,2,3,4,5; entity B has 100,200,300,400,500.
    A's last row (value 5) must have lag_1 = 4 (A's previous), not 400 (B's).
    B's last row (value 500) must have lag_1 = 400 (B's previous), not 4.
    """
    cfg = {
        "date_column": "date",
        "target_column": "target_cleaned",
        "entity_column": "store_id",
        "lags": [1],
    }
    df = two_entities_distinct_values
    t = LagTransformer().fit(df, cfg)
    out = t.transform(df)

    merged = df.copy()
    merged["lag_1"] = out["lag_1"].values

    # Entity A: last row (by date) has target_cleaned=5, lag_1 must be 4
    a_rows = merged[merged["store_id"] == "A"].sort_values("date")
    a_last = a_rows.iloc[-1]
    assert a_last["target_cleaned"] == 5.0
    assert a_last["lag_1"] == 4.0

    # Entity B: last row has target_cleaned=500, lag_1 must be 400
    b_rows = merged[merged["store_id"] == "B"].sort_values("date")
    b_last = b_rows.iloc[-1]
    assert b_last["target_cleaned"] == 500.0
    assert b_last["lag_1"] == 400.0


def test_rolling_no_cross_entity_leakage(two_entities_distinct_values):
    """
    Rolling must be per-entity. Entity A values [1,2,3,4,5]: at last row
    rolling_mean_3 = mean(2,3,4) = 3. Entity B [100,200,300,400,500]: at last row
    rolling_mean_3 = mean(200,300,400) = 300. If we leaked, A could get 300 or B could get 3.
    """
    cfg = {
        "date_column": "date",
        "target_column": "target_cleaned",
        "entity_column": "store_id",
        "windows": [3],
        "min_periods": 1,
    }
    df = two_entities_distinct_values
    t = RollingTransformer().fit(df, cfg)
    out = t.transform(df)

    merged = df.copy()
    merged["rolling_mean_3"] = out["rolling_mean_3"].values

    a_last = merged[merged["store_id"] == "A"].sort_values("date").iloc[-1]
    b_last = merged[merged["store_id"] == "B"].sort_values("date").iloc[-1]

    # A last row: past 3 values are 2,3,4 -> mean 3
    assert a_last["rolling_mean_3"] == pytest.approx(3.0)
    # B last row: past 3 values are 200,300,400 -> mean 300
    assert b_last["rolling_mean_3"] == pytest.approx(300.0)


def test_lag_no_cross_entity_leakage_unsorted_order(two_entities_unsorted):
    """Row order (A,B interleaved) must not cause cross-entity leakage."""
    cfg = {
        "date_column": "date",
        "target_column": "target_cleaned",
        "entity_column": "store_id",
        "lags": [1],
    }
    df = two_entities_unsorted
    t = LagTransformer().fit(df, cfg)
    out = t.transform(df)

    merged = df.copy()
    merged["lag_1"] = out["lag_1"].values

    # Each A row: lag_1 must be previous A value or NaN (no B values)
    a_df = merged[merged["store_id"] == "A"].sort_values("date")
    assert pd.isna(a_df.iloc[0]["lag_1"]), "A's first row has no past; lag_1 must be NaN"
    assert a_df.iloc[1]["lag_1"] == 1.0
    assert a_df.iloc[-1]["lag_1"] == 4.0

    b_df = merged[merged["store_id"] == "B"].sort_values("date")
    assert b_df.iloc[-1]["lag_1"] == 400.0


# ---------------------------------------------------------------------------
# Calendar: deterministic, no target; row count already tested
# ---------------------------------------------------------------------------
# Calendar does not use target or future; it's deterministic from date.
# Row count test above is sufficient; optional sanity:


def test_calendar_same_date_same_features(single_entity_ordered):
    """Calendar features depend only on date; same date -> same features."""
    t = CalendarTransformer().fit(single_entity_ordered, {"date_column": "date"})
    out = t.transform(single_entity_ordered)
    # 2024-01-01 is Monday -> day_of_week 0
    first = out.iloc[0]
    assert first["day_of_week"] == 0
    assert first["month"] == 1
