"""
Seasonal naive baseline forecasting model.

Predicts value at t+h as the value from t+h-k (same season, previous cycle).
Default seasonality k=7 (weekly). Works per entity; requires at least k historical points.
"""

from typing import Any

import pandas as pd

from .base import BaseForecastingModel

MODEL_NAME = "seasonal_naive"


class SeasonalNaiveForecast(BaseForecastingModel):
    """
    Seasonal naive baseline: forecast at t+h = value at t+h-k.

    - Default k=7 (weekly seasonality). Config: date_column, target_column,
      entity_column (optional), seasonality (int, default 7), frequency (str or
      DateOffset for forecast dates, default "D").
    - Works per entity (e.g. store_id); single-series if entity_column is None.
    - Requires at least k historical points per entity.
    - fit(): stores configuration only (no learned parameters).
    - predict(): deterministic; uses only past data (value at last_date + h - k,
      or same season from previous cycle when h > k).
    - No file I/O; no future data.
    """

    def __init__(self) -> None:
        self._date_col: str = "date"
        self._target_col: str = "target_cleaned"
        self._entity_col: str | None = None
        self._k: int = 7
        self._frequency: Any = "D"
        self._fitted: bool = False

    def fit(self, train_df: pd.DataFrame, config: dict[str, Any] | None = None) -> "SeasonalNaiveForecast":
        """
        Store configuration only. Does not mutate train_df; no file I/O.
        """
        cfg = config or {}
        self._date_col = cfg.get("date_column", "date")
        self._target_col = cfg.get("target_column", "target_cleaned")
        self._entity_col = cfg.get("entity_column")
        self._k = int(cfg.get("seasonality", 7))
        self._frequency = cfg.get("frequency", "D")
        if self._k < 1:
            raise ValueError("seasonality k must be >= 1")
        for col in [self._date_col, self._target_col]:
            if col not in train_df.columns:
                raise ValueError(f"SeasonalNaiveForecast requires column '{col}'. Found: {list(train_df.columns)}.")
        if self._entity_col is not None and self._entity_col not in train_df.columns:
            raise ValueError(f"entity_column '{self._entity_col}' not in DataFrame.")
        self._fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Generate horizon-step forecasts per entity. Uses only past data.
        Returns DataFrame with entity_id, date, y_pred, model_name.
        """
        if not self._fitted:
            raise RuntimeError("SeasonalNaiveForecast must be fitted before predict.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self._date_col not in history_df.columns or self._target_col not in history_df.columns:
            raise ValueError(f"predict requires columns '{self._date_col}', '{self._target_col}'.")
        if self._entity_col is not None and self._entity_col not in history_df.columns:
            raise ValueError(f"entity_column '{self._entity_col}' not in DataFrame.")

        freq = pd.DateOffset(days=1) if self._frequency == "D" else self._frequency
        if isinstance(freq, str):
            freq = pd.tseries.frequencies.to_offset(freq)

        rows: list[dict[str, Any]] = []
        if self._entity_col is not None:
            for entity_id, group in history_df.groupby(self._entity_col, sort=False):
                entity_forecasts = self._forecast_one_series(
                    group.sort_values(self._date_col), horizon, freq
                )
                for r in entity_forecasts:
                    r["entity_id"] = entity_id
                    rows.append(r)
        else:
            entity_forecasts = self._forecast_one_series(
                history_df.sort_values(self._date_col), horizon, freq
            )
            for r in entity_forecasts:
                r["entity_id"] = None
                rows.append(r)

        out = pd.DataFrame(rows)
        out["model_name"] = MODEL_NAME
        return out[["entity_id", "date", "y_pred", "model_name"]]

    def _forecast_one_series(
        self, group: pd.DataFrame, horizon: int, freq: pd.DateOffset
    ) -> list[dict[str, Any]]:
        """Forecast for one entity/series. Requires at least k points. Uses only past data."""
        if len(group) < self._k:
            raise ValueError(
                f"SeasonalNaiveForecast requires at least k={self._k} historical points; got {len(group)}."
            )
        group = group.sort_values(self._date_col).reset_index(drop=True)
        last_date = group[self._date_col].iloc[-1]
        values = group[self._target_col]

        # Forecast at t+h = value at t+h-k (same season). For h>k use previous cycle: value at -k + (h-1)%k.
        result: list[dict[str, Any]] = []
        for h in range(1, horizon + 1):
            hist_idx = -self._k + (h - 1) % self._k
            val = values.iloc[hist_idx]
            forecast_date = last_date + freq * h
            if hasattr(forecast_date, "normalize"):
                forecast_date = forecast_date.normalize()
            result.append({"date": forecast_date, "y_pred": float(val)})
        return result
