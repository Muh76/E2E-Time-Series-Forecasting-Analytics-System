"""
Forecast evaluation utilities.

Implementation lives in ``backend.app.services.metrics``; this module re-exports
for imports such as ``from backend.services.metrics import ...``.
"""

from backend.app.services.metrics import (  # noqa: F401
    compute_aligned_metrics,
    evaluate_last_forecast_vs_actuals,
    get_last_forecast_record,
    record_forecast_for_evaluation,
)

__all__ = [
    "compute_aligned_metrics",
    "evaluate_last_forecast_vs_actuals",
    "get_last_forecast_record",
    "record_forecast_for_evaluation",
]
