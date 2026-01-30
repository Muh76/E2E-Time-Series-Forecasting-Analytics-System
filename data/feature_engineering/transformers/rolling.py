"""
Rolling-window statistical features for time-series data.

Computes rolling mean and rolling std over target_cleaned, past-only and
group-aware. Returns a DataFrame of rolling feature columns only.
"""

from typing import Any

import pandas as pd

from .base import BaseTimeSeriesTransformer


class RollingTransformer(BaseTimeSeriesTransformer):
    """
    Rolling mean and rolling standard deviation over target_cleaned.

    Past-only: at time t, uses only values at t-1, t-2, ... (shift(1).rolling).
    Group-aware: computed within each entity to avoid cross-entity leakage.
    Feature names: rolling_mean_7, rolling_std_7, rolling_mean_14, etc.
    Returns only the rolling feature columns, same index as input.
    """

    def __init__(self) -> None:
        self._target_col: str = "target_cleaned"
        self._date_col: str = "date"
        self._entity_col: str | None = None
        self._windows: list[int] = [7, 14]
        self._min_periods: int = 1
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> "RollingTransformer":
        """
        Read config and validate columns exist. Does not mutate df.

        Config: target_column (default "target_cleaned"), date_column
        (default "date"), entity_column (optional), windows (list of int,
        default [7, 14]), min_periods (int, default 1).
        """
        cfg = config or {}
        self._target_col = cfg.get("target_column", "target_cleaned")
        self._date_col = cfg.get("date_column", "date")
        self._entity_col = cfg.get("entity_column")
        self._windows = cfg.get("windows", [7, 14])
        if isinstance(self._windows, (int, float)):
            self._windows = [int(self._windows)]
        self._windows = sorted(set(int(k) for k in self._windows if k > 0))
        self._min_periods = int(cfg.get("min_periods", 1))

        for col in [self._date_col, self._target_col]:
            if col not in df.columns:
                raise ValueError(f"RollingTransformer requires column '{col}'. Found: {list(df.columns)}.")
        if self._entity_col is not None and self._entity_col not in df.columns:
            raise ValueError(f"RollingTransformer entity_column '{self._entity_col}' not in DataFrame.")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling mean and rolling std per entity (past-only).
        Does not mutate df. Returns only rolling feature columns; same index.
        """
        if not self._fitted:
            raise RuntimeError("RollingTransformer must be fitted before transform.")

        if self._target_col not in df.columns or self._date_col not in df.columns:
            raise ValueError(f"transform requires columns '{self._date_col}', '{self._target_col}'.")
        if self._entity_col is not None and self._entity_col not in df.columns:
            raise ValueError(f"transform requires entity_column '{self._entity_col}'.")

        parts: list[pd.DataFrame] = []
        if self._entity_col is not None:
            for _entity, group in df.groupby(self._entity_col, sort=False):
                roll_df = self._rolling_for_group(group)
                parts.append(roll_df)
            out = pd.concat(parts)
        else:
            out = self._rolling_for_group(df.sort_values(self._date_col))

        out = out.reindex(df.index)
        return out

    def _rolling_for_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Past-only rolling: shift(1) then rolling so window excludes current row."""
        g = group.sort_values(self._date_col)
        past = g[self._target_col].shift(1)
        result = pd.DataFrame(index=g.index)
        for w in self._windows:
            result[f"rolling_mean_{w}"] = past.rolling(window=w, min_periods=self._min_periods).mean()
            result[f"rolling_std_{w}"] = past.rolling(window=w, min_periods=self._min_periods).std()
        return result
