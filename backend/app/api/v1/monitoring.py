"""
Monitoring API: expose computed monitoring state.

GET /monitoring/summary returns monitoring summary JSON per API_CONTRACT.md.
No database; no background jobs; in-memory or stubbed service.
"""

from fastapi import APIRouter

from app.services.monitoring_service import get_monitoring_summary

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary")
async def get_monitoring() -> dict:
    """
    Get monitoring summary.

    Returns latest metrics from in-memory computation or stubbed service.
    Follows API contract: model_version, as_of, performance, drift, pipeline.
    """
    return get_monitoring_summary()
