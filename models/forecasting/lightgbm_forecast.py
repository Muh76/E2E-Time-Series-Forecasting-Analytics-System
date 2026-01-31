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
    """Split by time: train = rows before cutoff date, val = rows at or after cutoff."""
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
        self._frequency: Any = "D"
        self._model: Any = None
        self._fitted: bool = False

    def fit(self, train_df: pd.DataFrame, config: dict[str, Any] | None = None) -> "LightGBMForecast":
        """
        Train on time-aware split. Excludes target from X. Does not mutate train_df; no file I/O.
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
        # Drop rows with NaN in features or target (e.g. warm-up from lags/rolling)
        X_train = train_sub[self._feature_cols].copy()
        y_train = train_sub[self._target_col]
        valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train.loc[valid_mask]
        y_train = y_train.loc[valid_mask]
        if len(X_train) == 0:
            raise ValueError("No valid training rows after dropping NaN in features/target.")

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
        Predict for one series using recursive multi-step forecasting.

        **Recursive multi-step strategy:**
        Future target values are unknown at prediction time, so they are replaced
        by the model's own predictions. For step h=1 we use the last observed row's
        features and predict y_1. For step h+1 we need features at the next date;
        those features (e.g. lags, rolling stats) depend on the target at the
        previous step. We therefore append the predicted value from step h to the
        series, re-run the feature pipeline to compute features for the new row,
        then predict step h+1 from that row. So features for step h+1 are computed
        using the predicted value from step h (and earlier predictions), not true
        future targets.

        **Error accumulation:** Because each step conditions on prior predictions
        rather than true values, errors can accumulate over longer horizons:
        early over- or under-predictions affect lags and rolling features for
        later steps, which can amplify bias or variance further out.

        **Why this strategy is acceptable:** Recursive (autoregressive) multi-step
        forecasting is standard in production: it requires only one trained model,
        avoids training separate models per horizon, and matches the setting where
        future targets are never available at inference time. Alternatives (e.g.
        direct multi-step or true multi-output models) have different trade-offs;
        recursive remains the default for many time-series and ML forecasting
        systems.
        """
        group = group.sort_values(self._date_col).reset_index(drop=True)
        last_date = group[self._date_col].iloc[-1]
        fe_config = config.get("feature_engineering")
        run_pipeline = None
        if horizon > 1 and fe_config:
            try:
                from data.feature_engineering import run_feature_pipeline
                run_pipeline = run_feature_pipeline
            except ImportError:
                pass

        result: list[dict[str, Any]] = []
        # Base series for recursive: date, entity, target_cleaned only (minimal for pipeline)
        base_cols = [self._date_col, self._target_col]
        if self._entity_col and self._entity_col in group.columns:
            base_cols.append(self._entity_col)
        current = group.copy()
        entity_val = current[self._entity_col].iloc[0] if self._entity_col and self._entity_col in current.columns else None

        for h in range(1, horizon + 1):
            # Last row's features (may have been updated by pipeline in previous step)
            raw_last = current[self._feature_cols].iloc[[-1]]
            if raw_last.isna().any(axis=1).any():
                logger.warning(
                    "NaNs encountered at inference time in features; applying forward-fill (last observation carried forward), then zero fallback for any remaining."
                )
            X = current[self._feature_cols].ffill().iloc[[-1]]
            if X.isna().any(axis=1).any():
                X = X.fillna(0)
            y_pred = float(self._model.predict(X)[0])
            forecast_date = last_date + freq * h
            if hasattr(forecast_date, "normalize"):
                forecast_date = forecast_date.normalize()
            result.append({"date": forecast_date, "y_pred": y_pred})

            if h < horizon and run_pipeline is not None:
                # Extend series with prediction (base cols only); re-run feature pipeline
                new_row = {self._date_col: forecast_date, self._target_col: y_pred}
                if self._entity_col and entity_val is not None:
                    new_row[self._entity_col] = entity_val
                extended = pd.concat([current[base_cols], pd.DataFrame([new_row])], ignore_index=True)
                featured = run_pipeline(extended, config)
                current = featured

        return result
