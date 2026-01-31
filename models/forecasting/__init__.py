# Forecasting models: base interface and implementations.

from .base import BaseForecastingModel
from .seasonal_naive import MODEL_NAME as SEASONAL_NAIVE_NAME, SeasonalNaiveForecast

__all__ = ["BaseForecastingModel", "SeasonalNaiveForecast", "SEASONAL_NAIVE_NAME"]
