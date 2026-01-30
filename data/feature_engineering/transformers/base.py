"""
Base interface for time-series feature transformers.

Designed for use in both training and inference: fit() captures state from
training data; transform() applies using that state only. No lookahead;
no mutation of inputs.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseTimeSeriesTransformer(ABC):
    """
    Abstract base for feature transformers in a time-series system.

    **Training vs inference:** Fit once on training data (e.g. train split or
    full history). At inference, call only transform(df) using the fitted
    state. Do not refit on inference data. This ensures the same features are
    produced in training and serving.

    **Time safety:** Transformers must not use future information. For a row
    at time t, only data at t or earlier may be used. No lookahead, no
    leakage from future dates. Implementations should rely on past-only
    operations (e.g. lag, rolling with closed="left" or equivalent).

    **Immutability:** transform(df) must not mutate the input DataFrame.
    Return a new DataFrame or a copy with new columns. fit(df) may store
    aggregates or parameters derived from df but must not modify df.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> "BaseTimeSeriesTransformer":
        """
        Learn state from the given DataFrame (e.g. column names, scales, windows).

        Called during training only. May store parameters or aggregates;
        must not mutate df.

        Args:
            df: Training time-series DataFrame (e.g. with date and target).
            config: Optional transformer-specific config.

        Returns:
            self, for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation using fitted state. Does not mutate df.

        Safe to call at inference: uses only state from fit(), no future data.
        Must not modify the input DataFrame in place.

        Args:
            df: Time-series DataFrame (same grain as fit, or inference slice).

        Returns:
            New DataFrame with original columns plus transformed features.
            Same row count and order as input unless documented otherwise.
        """
        ...

    def fit_transform(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Fit on df then transform df. Convenience for training; equivalent to
        self.fit(df, config).transform(df). Does not mutate df.
        """
        return self.fit(df, config).transform(df)
