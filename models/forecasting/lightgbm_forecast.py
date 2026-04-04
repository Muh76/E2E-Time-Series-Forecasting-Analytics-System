"""
LightGBM-based time series forecasting model.

Uses engineered features (lags, rolling, calendar). Excludes target from feature
matrix. Supports multi-entity. fit() uses time-aware split; predict() supports
recursive multi-step. Deterministic (fixed seed).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecastingModel

logger = logging.getLogger(__name__)
MODEL_NAME = "lightgbm"


def _get_feature_columns(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    entity_col: str | None,
) -> list[str]:
    """Columns to use as features: all except date, target, and identifiers not used as features."""
    exclude = {date_col, target_col}
    if entity_col:
        exclude.add(entity_col)
    return [c for c in df.columns if c not in exclude]


def _time_aware_split(
    df: pd.DataFrame,
    date_col: str,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time: train = rows before cutoff date, val = rows at or after cutoff.

    Validation uses a time-based holdout: the last fraction of unique dates
    (e.g. last 20%) form the validation set; all earlier dates form the train set.
    This is a simplified approximation of forecasting validation: it avoids
    lookahead and respects temporal order, but uses a single train/val split.
    In more advanced setups, rolling-origin or backtesting (repeated refit and
    evaluate over multiple cutoffs) would be used to better approximate
    out-of-sample performance across time. For the current project scope, a
    single time-based holdout is acceptable: it is simple, reproducible, and
    sufficient to compare models and guard against overfitting without the
    complexity of full backtesting.
    """
    dates = df[date_col].drop_duplicates().sort_values()
    n = len(dates)
    if n < 2 or val_frac <= 0 or val_frac >= 1:
        return df, df.iloc[0:0]
    cutoff_idx = max(0, int(n * (1 - val_frac)))
    cutoff_date = dates.iloc[cutoff_idx]
    train = df[df[date_col] < cutoff_date]
    val = df[df[date_col] >= cutoff_date]
    return train, val


class LightGBMForecast(BaseForecastingModel):
    """
    LightGBM forecaster using engineered features (lags, rolling, calendar).

    - Excludes target from feature matrix; uses only lag_*, rolling_*, calendar, and optional entity.
    - Supports multi-entity: entity column is used as a categorical feature (one global model).
    - fit(): time-aware split (e.g. last 20% of dates as validation), trains LightGBM with fixed seed.
    - predict(): single-step from last row; multi-step recursive by extending series and re-running
      feature pipeline each step (requires config with feature_engineering for horizon > 1).
    - Deterministic: seed from config (default 42). No file I/O; no future data.
    """

    def __init__(self) -> None:
        self._date_col: str = "date"
        self._target_col: str = "target_cleaned"
        self._entity_col: str | None = None
        self._feature_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._category_levels: dict[str, Any] = {}
        self._frequency: Any = "D"
        self._model: Any = None
        self._fitted: bool = False

    def fit(self, train_df: pd.DataFrame, config: dict[str, Any] | None = None) -> "LightGBMForecast":
        """
        Train on time-aware split. Excludes target from X. Does not mutate train_df; no file I/O.

        Validation uses a time-based holdout (last fraction of dates via
        time_split_val_frac). This is a simplified approximation of forecasting
        validation; rolling-origin or backtesting would be used in more advanced
        setups. For the current project scope this approach is acceptable:
        simple, reproducible, and sufficient to compare models and guard
        against overfitting.
        """
        import lightgbm as lgb

        cfg = config or {}
        self._date_col = cfg.get("date_column", "date")
        self._target_col = cfg.get("target_column", "target_cleaned")
        self._entity_col = cfg.get("entity_column")
        self._frequency = cfg.get("frequency", "D")
        val_frac = float(cfg.get("time_split_val_frac", 0.2))
        seed = int(cfg.get("seed", 42))

        for col in [self._date_col, self._target_col]:
            if col not in train_df.columns:
                raise ValueError(f"LightGBMForecast requires column '{col}'. Found: {list(train_df.columns)}.")
        if self._entity_col is not None and self._entity_col not in train_df.columns:
            raise ValueError(f"entity_column '{self._entity_col}' not in DataFrame.")

        self._feature_cols = _get_feature_columns(
            train_df, self._date_col, self._target_col, self._entity_col
        )
        if not self._feature_cols:
            raise ValueError("No feature columns found; need lags, rolling, or calendar features.")
        self._cat_cols = [c for c in self._feature_cols if c == self._entity_col or train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]

        train_sub, val_sub = _time_aware_split(train_df, self._date_col, val_frac)
        logger.info(
            "Time-aware split: train_rows=%d, val_rows=%d, val_frac=%.2f",
            len(train_sub), len(val_sub), val_frac,
        )
        # Drop rows with NaN in features or target (e.g. warm-up from lags/rolling)
        X_train = train_sub[self._feature_cols].copy()
        y_train = train_sub[self._target_col]
        valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train.loc[valid_mask]
        y_train = y_train.loc[valid_mask]
        if len(X_train) == 0:
            raise ValueError("No valid training rows after dropping NaN in features/target.")

        # Convert object dtype columns to category for LightGBM (int, float, bool, category only)
        obj_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
        if obj_cols:
            logger.info("Converting object columns to category for LightGBM: %s", obj_cols)
            for c in obj_cols:
                X_train[c] = X_train[c].astype("category")

        # Store category levels for each categorical column (exact training categories)
        for c in self._cat_cols:
            if c in X_train.columns:
                self._category_levels[c] = X_train[c].astype("category").cat.categories.tolist()

        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "random_state": seed,
            "force_col_wise": True,
            "n_estimators": int(cfg.get("n_estimators", 100)),
            "learning_rate": float(cfg.get("learning_rate", 0.05)),
            "num_leaves": int(cfg.get("num_leaves", 31)),
        }
        if self._cat_cols:
            cat_idx = [self._feature_cols.index(c) for c in self._cat_cols if c in self._feature_cols]
            if cat_idx:
                lgb_params["categorical_feature"] = cat_idx

        self._model = lgb.LGBMRegressor(**lgb_params)
        if len(val_sub) > 0:
            X_val = val_sub[self._feature_cols].dropna(how="all")
            y_val = val_sub.loc[X_val.index, self._target_col]
            valid_val = y_val.notna()
            X_val = X_val[valid_val]
            y_val = y_val[valid_val]
            if len(X_val) > 0:
                # Apply same object->category conversion and align categories with training
                for c in obj_cols:
                    if c in X_val.columns:
                        X_val[c] = X_val[c].astype("category")
                for col in self._category_levels:
                    if col in X_val.columns:
                        X_val[col] = X_val[col].astype("category").cat.set_categories(self._category_levels[col])
                self._model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)] if cfg.get("early_stopping", True) else None,
                )
            else:
                self._model.fit(X_train, y_train)
        else:
            self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Generate horizon-step forecasts. Single-step from last row; multi-step recursive
        by extending series and re-running feature pipeline (requires feature_engineering in config).
        Returns DataFrame with entity_id, date, y_pred, model_name.
        """
        if not self._fitted:
            raise RuntimeError("LightGBMForecast must be fitted before predict.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self._date_col not in history_df.columns or self._target_col not in history_df.columns:
            raise ValueError(f"predict requires columns '{self._date_col}', '{self._target_col}'.")
        for c in self._feature_cols:
            if c not in history_df.columns:
                raise ValueError(f"predict requires feature column '{c}' in history_df.")

        cfg = config or {}
        freq = cfg.get("frequency", self._frequency)
        freq = pd.DateOffset(days=1) if freq == "D" else (pd.tseries.frequencies.to_offset(freq) if isinstance(freq, str) else freq)

        rows: list[dict[str, Any]] = []
        if self._entity_col is not None and self._entity_col in history_df.columns:
            entities = history_df[self._entity_col].drop_duplicates()
            for entity_id in entities:
                group = history_df[history_df[self._entity_col] == entity_id].sort_values(self._date_col)
                preds = self._predict_one_series(group, horizon, freq, cfg)
                for p in preds:
                    p["entity_id"] = entity_id
                    rows.append(p)
        else:
            group = history_df.sort_values(self._date_col)
            preds = self._predict_one_series(group, horizon, freq, cfg)
            for p in preds:
                p["entity_id"] = None
                rows.append(p)

        out = pd.DataFrame(rows)
        out["model_name"] = MODEL_NAME
        return out[["entity_id", "date", "y_pred", "model_name"]]

    def _predict_one_series(
        self,
        group: pd.DataFrame,
        horizon: int,
        freq: pd.DateOffset,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        True recursive autoregressive multi-step forecasting for one entity series.

        **Algorithm (per step h = 1 … horizon):**
          1. Append a placeholder row at the forecast date (target = NaN) to the
             working base series (date + target + static store features only).
          2. Re-run the feature pipeline on the last `lookback + 2` rows of the
             extended series so lag_*, rolling_*, and calendar features are freshly
             computed from actual or previously predicted targets — not stale values.
          3. Read the last row's features, enforce column order and category levels.
          4. Fill any remaining numeric NaNs with 0 (categorical: left as NaN for LightGBM).
          5. Call model.predict() for y_pred.
          6. Replace the placeholder's target with y_pred, append to the working
             series, and trim to `lookback + buffer` rows (performance guard).

        **Why this avoids the constant-prediction bug:**
          The working series (`current_base`) holds only *base* columns (no stale
          lag/rolling columns). Engineered features are fully recomputed each step
          from the updated target history, so lag_1 correctly reflects the previous
          prediction after the first step.

        **Performance optimisation:**
          The feature pipeline is re-run on a tail of `lookback + 2` rows rather
          than the full historical series. `lookback = max(max_lag, max_window)`
          (default: 14 days), so the tail is always small regardless of series length.

        **No data leakage:**
          The placeholder row carries NaN as target; lag features for this row are
          computed from rows at t-1, t-7, t-14 — all of which are observed history
          or prior predictions, never future actuals.

        **Error accumulation:**
          Each step conditions on prior predictions rather than true values; errors
          can propagate through lags and rolling features over longer horizons.
          Recursive autoregressive forecasting is standard production practice and
          is the appropriate default for the current system.
        """
        group = group.sort_values(self._date_col).reset_index(drop=True)
        last_date = group[self._date_col].iloc[-1]

        # Load feature pipeline for recursive recomputation
        fe_config = config.get("feature_engineering") or {}
        run_pipeline = None
        if fe_config:
            try:
                from data.feature_engineering import run_feature_pipeline  # noqa: PLC0415
                run_pipeline = run_feature_pipeline
            except ImportError:
                logger.warning("data.feature_engineering not importable; recursive features disabled.")

        # Determine lookback: largest lag or rolling window drives how many rows are needed
        lags: list[int] = list((fe_config.get("lag") or {}).get("lags", [1, 7, 14]))
        windows: list[int] = list((fe_config.get("rolling") or {}).get("windows", [7, 14]))
        lookback: int = max(max(lags, default=14), max(windows, default=14))
        _BUFFER = 5                            # extra rows as safety margin
        keep_rows = lookback + _BUFFER

        # Identify engineered columns by naming convention so they are excluded from
        # the working base series (they must be recomputed fresh each step).
        _calendar_feature_names = {"day_of_week", "day_of_month", "week_of_year", "month", "is_weekend"}
        _engineered = {
            c for c in group.columns
            if c.startswith("lag_") or c.startswith("rolling_") or c in _calendar_feature_names
        }
        base_cols = [c for c in group.columns if c not in _engineered]

        # Working base series: only pre-pipeline columns; trimmed to `keep_rows`
        current_base = group[base_cols].tail(keep_rows).reset_index(drop=True)

        result: list[dict[str, Any]] = []

        for h in range(1, horizon + 1):
            forecast_date = last_date + freq * h
            if hasattr(forecast_date, "normalize"):
                forecast_date = forecast_date.normalize()

            if run_pipeline is not None:
                # --- Step 1: extend with NaN-target placeholder for forecast_date ---
                placeholder = current_base.iloc[-1].copy()
                placeholder[self._date_col] = forecast_date
                placeholder[self._target_col] = float("nan")
                extended = pd.concat(
                    [current_base, pd.DataFrame([placeholder])], ignore_index=True
                )

                # --- Step 2: recompute features on a trimmed tail (performance) ---
                tail = extended.tail(lookback + 2)
                featured = run_pipeline(tail, config)
                # Drop any duplicate columns produced by re-running pipeline
                featured = featured.loc[:, ~featured.columns.duplicated()]

                # --- Step 3: build X in strict training column order ---
                missing_fc = [c for c in self._feature_cols if c not in featured.columns]
                if missing_fc:
                    logger.warning(
                        "Step h=%d: %d feature column(s) absent after pipeline recompute "
                        "(will be NaN, then filled): %s", h, len(missing_fc), missing_fc,
                    )
                X = featured.iloc[[-1]].reindex(columns=self._feature_cols).copy()
            else:
                # No pipeline: use stale last row (horizon > 1 will repeat predictions)
                logger.warning(
                    "Feature pipeline unavailable; step h=%d uses last observed features. "
                    "Recursive forecasting requires feature_engineering in config.", h,
                )
                X = group.reindex(columns=self._feature_cols).ffill().iloc[[-1]].copy()

            # --- Step 4: category alignment then NaN filling ---
            for col in self._category_levels:
                if col in X.columns:
                    X[col] = X[col].astype("category").cat.set_categories(
                        self._category_levels[col]
                    )
            numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            nan_count = int(X[numeric_cols].isna().sum().sum()) if numeric_cols else 0
            if nan_count > 0:
                logger.info(
                    "Step h=%d: filling %d numeric NaN(s) with 0.", h, nan_count,
                )
                X[numeric_cols] = X[numeric_cols].fillna(0)
            # Categorical NaNs left as-is: LightGBM handles missing categories natively

            # --- Step 5: predict ---
            y_pred = float(self._model.predict(X)[0])
            result.append({"date": forecast_date, "y_pred": y_pred})
            logger.debug("Step h=%d: date=%s, y_pred=%.4f", h, forecast_date, y_pred)

            # --- Step 6: append prediction to base series for next step ---
            predicted_row = current_base.iloc[-1].copy()
            predicted_row[self._date_col] = forecast_date
            predicted_row[self._target_col] = y_pred
            current_base = pd.concat(
                [current_base, pd.DataFrame([predicted_row])], ignore_index=True
            )
            # Trim to keep memory and pipeline cost bounded
            if len(current_base) > keep_rows + 1:
                current_base = current_base.iloc[-(keep_rows + 1):].reset_index(drop=True)

        return result
