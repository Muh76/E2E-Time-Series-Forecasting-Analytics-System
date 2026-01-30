"""
Lag-based feature transformer for time-series data.

Computes past values of the target column per entity. Time-safe; no leakage.
Returns a DataFrame of lag columns only (e.g. lag_1, lag_7, lag_14).
"""

from typing import Any

import pandas as pd

from .base import BaseTimeSeriesTransformer


class LagTransformer(BaseTimeSeriesTransformer):
    """
    Lag features from a target column, group-aware and time-safe.

    For each row at time t, produces target_cleaned at t-1, t-7, etc.
    Lags are computed within each entity (e.g. per store_id) so there is
    no leakage across entities. NaNs introduced by lagging are left as-is
    (no fill). Returns only the lag feature columns, same index as input.
    """

    def __init__(self) -> None:
        self._target_col: str = "target_cleaned"
        self._date_col: str = "date"
        self._entity_col: str | None = None
        self._lags: list[int] = [1, 7, 14]
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> "LagTransformer":
        """
        Read config and validate columns exist. Does not mutate df.

        Config: target_column (default "target_cleaned"), date_column
        (default "date"), entity_column (optional, e.g. "store_id"),
        lags (list of int, default [1, 7, 14]).
        """
        cfg = config or {}
        self._target_col = cfg.get("target_column", "target_cleaned")
        self._date_col = cfg.get("date_column", "date")
        self._entity_col = cfg.get("entity_column")
        self._lags = cfg.get("lags", [1, 7, 14])
        if isinstance(self._lags, (int, float)):
            self._lags = [int(self._lags)]
        self._lags = sorted(set(int(k) for k in self._lags if k > 0))

        for col in [self._date_col, self._target_col]:
            if col not in df.columns:
                raise ValueError(f"LagTransformer requires column '{col}'. Found: {list(df.columns)}.")
        if self._entity_col is not None and self._entity_col not in df.columns:
            raise ValueError(f"LagTransformer entity_column '{self._entity_col}' not in DataFrame.")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute lag features per entity. Does not mutate df.
        Returns a DataFrame of lag columns only (e.g. lag_1, lag_7);
        same index as input. NaNs from lagging are not filled.
        """
        if not self._fitted:
            raise RuntimeError("LagTransformer must be fitted before transform.")

        if self._target_col not in df.columns or self._date_col not in df.columns:
            raise ValueError(f"transform requires columns '{self._date_col}', '{self._target_col}'.")

        if self._entity_col is not None and self._entity_col not in df.columns:
            raise ValueError(f"transform requires entity_column '{self._entity_col}'.")

        parts: list[pd.DataFrame] = []
        if self._entity_col is not None:
            for _entity, group in df.groupby(self._entity_col, sort=False):
                lag_df = self._lags_for_group(group)
                parts.append(lag_df)
            out = pd.concat(parts)
        else:
            out = self._lags_for_group(df.sort_values(self._date_col))

        # Restore original row order (groupby/concat may reorder)
        out = out.reindex(df.index)
        return out

    def _lags_for_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute lag columns for a single entity (sorted by date). No fill."""
        g = group.sort_values(self._date_col)
        result = pd.DataFrame(index=g.index)
        for lag in self._lags:
            result[f"lag_{lag}"] = g[self._target_col].shift(lag)
        return result
