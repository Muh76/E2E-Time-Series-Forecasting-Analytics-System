"""
Monitoring API: expose computed monitoring state and evaluation snapshot.

GET /monitoring/summary — full dashboard payload.
GET /monitoring/metrics — compact evaluation metrics for integrations.
"""

from fastapi import APIRouter

from backend.app.services.monitoring_service import get_evaluation_snapshot, get_monitoring_summary

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary")
async def get_monitoring() -> dict:
    """
    Monitoring summary: performance, drift, rolling series, alerts, thresholds.
    """
    return get_monitoring_summary()


@router.get("/metrics")
async def get_metrics() -> dict:
    """
    Evaluation-focused view: validation holdout vs live summary primary metrics.
    """
    return get_evaluation_snapshot()
