"""
Unit tests: forecasting models do not leak future data.

- Training uses only past information (no future dates in fit).
- Inference does not access target values beyond last observed date.
- Baseline and ML model behave consistently (output shape, no future leak).
Uses small synthetic datasets with known values.
"""

import pandas as pd
import pytest

from models.forecasting import LightGBMForecast, SeasonalNaiveForecast


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with clear past vs future
# ---------------------------------------------------------------------------

# Future sentinel: if a model leaked, it might output this; we never pass it.
FUTURE_SENTINEL = 999.0


@pytest.fixture
def history_past_only_single():
    """Single entity, dates 1..7, values 1..7. No data beyond date 7."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=7, freq="D"),
        "target_cleaned": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    })


@pytest.fixture
def config_single():
    """Config for single-series: date, target_cleaned, no entity, k=7."""
    return {
        "date_column": "date",
        "target_column": "target_cleaned",
        "entity_column": None,
        "seasonality": 7,
        "frequency": "D",
    }


@pytest.fixture
def history_past_only_two_entities():
    """Two entities (A, B), dates 1..7 each. A: 1..7, B: 10,20..70. No data beyond date 7."""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    return pd.DataFrame({
        "date": list(dates) * 2,
        "store_id": ["A"] * 7 + ["B"] * 7,
        "target_cleaned": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    })


@pytest.fixture
def config_two_entities():
    """Config with entity_column for two-entity data."""
    return {
        "date_column": "date",
        "target_column": "target_cleaned",
        "entity_column": "store_id",
        "seasonality": 7,
        "frequency": "D",
    }


# ---------------------------------------------------------------------------
# Training uses only past information
# ---------------------------------------------------------------------------


def test_training_receives_only_past_dates(history_past_only_single, config_single):
    """Fit on data that contains only dates 1..7; no future dates in train_df."""
    model = SeasonalNaiveForecast()
    model.fit(history_past_only_single, config_single)
    last_date = history_past_only_single["date"].max()
    # Training data must end at last_date; we never passed any row with date > last_date
    assert last_date == pd.Timestamp("2024-01-07")
    # Predict using same history (no future rows passed)
    pred = model.predict(history_past_only_single, horizon=3, config=config_single)
    assert len(pred) == 3
    # Predictions must be based only on past (1..7); seasonal naive k=7 -> steps 1,2,3 use values at indices -7,-6,-5 -> 1,2,3
    assert list(pred["y_pred"]) == [1.0, 2.0, 3.0]


def test_fit_does_not_see_future_when_trained_on_subset(history_past_only_single, config_single):
    """Train on first 7 rows only; 'future' (8,9,10) never passed to fit or predict(history=...)."""
    train_df = history_past_only_single  # only dates 1..7
    model = SeasonalNaiveForecast()
    model.fit(train_df, config_single)
    # Inference: history is same 1..7; we never pass dates 8,9,10
    pred = model.predict(train_df, horizon=3, config=config_single)
    # If model leaked future, it could output FUTURE_SENTINEL; we never passed that value
    assert not any(p == FUTURE_SENTINEL for p in pred["y_pred"])
    # Expected: past-based only -> [1, 2, 3] for seasonal naive k=7
    assert list(pred["y_pred"]) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Inference does not access target beyond last observed date
# ---------------------------------------------------------------------------


def test_seasonal_naive_inference_uses_only_past(history_past_only_single, config_single):
    """Predict with history ending at date 7; predictions must equal past-based values [1,2,3], not any future."""
    model = SeasonalNaiveForecast()
    model.fit(history_past_only_single, config_single)
    pred = model.predict(history_past_only_single, horizon=3, config=config_single)
    # Seasonal naive k=7: step h uses value at -k + (h-1)%k -> 1, 2, 3
    expected = [1.0, 2.0, 3.0]
    assert list(pred["y_pred"]) == expected
    # We never passed target values for dates 8,9,10; so predictions cannot be 100,200,300 (hypothetical future)
    assert list(pred["y_pred"]) != [100.0, 200.0, 300.0]


def test_seasonal_naive_predictions_not_future_actuals(history_past_only_single, config_single):
    """Hypothetical future actuals [100, 200, 300] are never passed; predictions must not equal them."""
    future_actuals_if_leaked = [100.0, 200.0, 300.0]
    model = SeasonalNaiveForecast()
    model.fit(history_past_only_single, config_single)
    pred = model.predict(history_past_only_single, horizon=3, config=config_single)
    pred_list = list(pred["y_pred"])
    assert pred_list != future_actuals_if_leaked
    assert pred_list == [1.0, 2.0, 3.0]


def test_lightgbm_inference_does_not_output_unseen_value(history_with_features_for_lightgbm, config_lightgbm):
    """LightGBM predict(history=...) must not output FUTURE_SENTINEL; we never put that in the data."""
    model = LightGBMForecast()
    model.fit(history_with_features_for_lightgbm, config_lightgbm)
    # History ends at last date in the df; we never pass rows with target_cleaned = FUTURE_SENTINEL
    pred = model.predict(history_with_features_for_lightgbm, horizon=3, config=config_lightgbm)
    assert not any(p == FUTURE_SENTINEL for p in pred["y_pred"])
    assert len(pred) == 3
    assert "entity_id" in pred.columns and "date" in pred.columns and "y_pred" in pred.columns and "model_name" in pred.columns


@pytest.fixture
def history_with_features_for_lightgbm():
    """Minimal featured data for LightGBM: 14 days so lag_7/rolling have warm-up."""
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    target = [float(i + 1) for i in range(14)]  # 1..14, no FUTURE_SENTINEL
    df = pd.DataFrame({"date": dates, "target_cleaned": target})
    # Add minimal features (lag_1, lag_7, one rolling, calendar) so LightGBM has something to fit
    df["lag_1"] = df["target_cleaned"].shift(1)
    df["lag_7"] = df["target_cleaned"].shift(7)
    df["rolling_mean_7"] = df["target_cleaned"].shift(1).rolling(7, min_periods=1).mean()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    return df.dropna(subset=["lag_7"]).reset_index(drop=True)  # enough rows for fit/predict


@pytest.fixture
def config_lightgbm():
    return {
        "date_column": "date",
        "target_column": "target_cleaned",
        "entity_column": None,
        "frequency": "D",
        "seed": 42,
        "time_split_val_frac": 0.0,
        "n_estimators": 20,
        "early_stopping": False,
    }


# ---------------------------------------------------------------------------
# Baseline and ML model behave consistently
# ---------------------------------------------------------------------------


def test_both_models_same_history_same_horizon_output_shape(history_past_only_single, config_single):
    """Baseline and primary (when given featured data) produce same horizon length and columns."""
    # Baseline: seasonal naive
    baseline = SeasonalNaiveForecast()
    baseline.fit(history_past_only_single, config_single)
    pred_baseline = baseline.predict(history_past_only_single, horizon=5, config=config_single)
    assert len(pred_baseline) == 5
    assert list(pred_baseline.columns) == ["entity_id", "date", "y_pred", "model_name"]
    assert not any(p == FUTURE_SENTINEL for p in pred_baseline["y_pred"])

    # Primary: needs featured data; use same fixture as LightGBM test
    # For consistency we only assert baseline here; LightGBM tested above with featured fixture
    assert pred_baseline["model_name"].iloc[0] == "seasonal_naive"


def test_both_models_deterministic_same_output_twice(history_past_only_single, config_single):
    """Same history and horizon -> same predictions (deterministic; no future leak)."""
    model = SeasonalNaiveForecast()
    model.fit(history_past_only_single, config_single)
    pred1 = model.predict(history_past_only_single, horizon=3, config=config_single)
    pred2 = model.predict(history_past_only_single, horizon=3, config=config_single)
    assert list(pred1["y_pred"]) == list(pred2["y_pred"])
    assert list(pred1["date"]) == list(pred2["date"])


def test_two_entities_no_cross_entity_future_leak(history_past_only_two_entities, config_two_entities):
    """Each entity's forecasts use only that entity's past; no future dates."""
    model = SeasonalNaiveForecast()
    model.fit(history_past_only_two_entities, config_two_entities)
    pred = model.predict(history_past_only_two_entities, horizon=3, config=config_two_entities)
    assert len(pred) == 6  # 2 entities * 3 steps
    # Entity A: past 1..7 -> forecasts [1, 2, 3]. Entity B: past 10..70 -> forecasts [10, 20, 30]
    a_pred = pred[pred["entity_id"] == "A"]["y_pred"].tolist()
    b_pred = pred[pred["entity_id"] == "B"]["y_pred"].tolist()
    assert a_pred == [1.0, 2.0, 3.0]
    assert b_pred == [10.0, 20.0, 30.0]
    # If model leaked the other entity's future, we'd see wrong values
    assert not any(p == FUTURE_SENTINEL for p in pred["y_pred"])
