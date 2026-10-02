"""Analytics and metrics endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.analytics import DashboardStatsResponse
from backend.app.services.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStatsResponse, summary="Get Dashboard Overview Metrics")
def get_dashboard_metrics(db: Session = Depends(get_db)):
    """Fetch aggregated platform metrics, active cameras, average FPS/latency, and recent sessions."""
    return AnalyticsService.get_dashboard_stats(db)


@router.get("/breakdown", summary="Get Detailed Telemetry Breakdown")
def get_analytics_breakdown(db: Session = Depends(get_db)):
    """Fetch detected class distribution, session types, and hardware utilization."""
    return AnalyticsService.get_analytics_breakdown(db)
