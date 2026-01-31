"""
Base interface for forecasting models.

Used by both baseline and ML models. No file I/O inside the model;
deterministic inference; no access to future data.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseForecastingModel(ABC):
    """
    Abstract base for time-series forecasting models.

    **Time safety:** The model must never use future data. At prediction time,
    only history_df (past observations) and fitted state from fit() are used.
    No lookahead; no leakage from future dates.

    **Deterministic inference:** For the same fitted model, same history_df,
    horizon, and config, predict() must return the same outputs. No random
    sampling or dropout at inference unless explicitly documented and
    controlled (e.g. via seed in config).

    **No file I/O:** fit() and predict() must not read or write files. Persistence
    (save/load) is the responsibility of the caller or a separate layer.

    **Assumptions (caller must ensure):**
    - train_df and history_df have a time column (e.g. date) sorted per entity.
    - No duplicate (time, entity) rows in the provided DataFrames.
    - horizon is a positive integer (number of steps to forecast).
    - config is a dict; model-specific keys are documented per implementation.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, config: dict[str, Any] | None = None) -> "BaseForecastingModel":
        """
        Fit the model on historical data. Does not use future data; does not perform file I/O.

        Args:
            train_df: Historical time-series data (e.g. date, target, optional entity).
                Must be past-only; no future dates. Sorted by time per entity if multi-entity.
            config: Optional model and training config (e.g. frequency, model params).
                Must not mutate train_df.

        Returns:
            self, for method chaining. Fitted state is stored on the instance;
            no files are written.
        """
        ...

    @abstractmethod
    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """
        Produce deterministic forecasts for the next horizon steps. Uses only past data.

        Args:
            history_df: Data available at prediction time (past observations only).
                Same schema as train_df. Must not contain future dates relative to
                the forecast origin.
            horizon: Number of steps to forecast (positive integer).
            config: Optional config (e.g. frequency, prediction interval settings).
                Must not mutate history_df.

        Returns:
            DataFrame or Series with forecasts. Typically one row per horizon step
            (e.g. date index and a forecast column). Same length as horizon.
            Deterministic: same inputs and fitted state yield same outputs.
        """
        ...

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """
        Fit on train_df then predict the next horizon steps using train_df as history.
        Convenience method; equivalent to self.fit(train_df, config).predict(train_df, horizon, config).
        No file I/O; deterministic; no future data.
        """
        return self.fit(train_df, config).predict(train_df, horizon, config)
