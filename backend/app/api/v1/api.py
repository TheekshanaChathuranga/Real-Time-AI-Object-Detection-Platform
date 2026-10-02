"""API v1 Router aggregation."""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import (
    analytics,
    cameras,
    detection,
    health,
    models,
    training,
    websocket,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(detection.router, prefix="/detection", tags=["Detection"])
api_router.include_router(websocket.router, tags=["Real-Time & WebSocket"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["Cameras & RTSP"])
api_router.include_router(models.router, prefix="/models", tags=["Models & Registry"])
api_router.include_router(training.router, prefix="/training", tags=["Custom Training"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics & Telemetry"])
