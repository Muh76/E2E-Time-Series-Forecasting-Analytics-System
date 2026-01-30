"""
Calendar-based features from a datetime column.

Extracts day of week, day of month, week of year, month, is weekend.
Deterministic; no external calendars; no dependence on target values.
Returns numeric columns only.
"""

from typing import Any

import pandas as pd

from .base import BaseTimeSeriesTransformer


class CalendarTransformer(BaseTimeSeriesTransformer):
    """
    Calendar features from a datetime column.

    Features: day_of_week (0=Monday), day_of_month (1-31), week_of_year (1-53),
    month (1-12), is_weekend (0 or 1). All numeric. Deterministic; uses only
    the datetime column (no target). No external calendar libraries.
    """

    def __init__(self) -> None:
        self._date_col: str = "date"
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> "CalendarTransformer":
        """
        Read config and validate date column exists. Does not mutate df.

        Config: date_column (default "date").
        """
        cfg = config or {}
        self._date_col = cfg.get("date_column", "date")
        if self._date_col not in df.columns:
            raise ValueError(f"CalendarTransformer requires column '{self._date_col}'. Found: {list(df.columns)}.")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute calendar features from the datetime column. Does not mutate df.
        Returns only calendar feature columns (numeric); same index as input.
        """
        if not self._fitted:
            raise RuntimeError("CalendarTransformer must be fitted before transform.")
        if self._date_col not in df.columns:
            raise ValueError(f"transform requires column '{self._date_col}'.")

        dt = pd.to_datetime(df[self._date_col])
        out = pd.DataFrame(index=df.index)
        out["day_of_week"] = dt.dt.dayofweek.astype("int64")   # 0=Monday, 6=Sunday
        out["day_of_month"] = dt.dt.day.astype("int64")
        out["week_of_year"] = dt.dt.isocalendar().week.astype("int64")
        out["month"] = dt.dt.month.astype("int64")
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int64")  # 1 if Sat/Sun, else 0
        return out
