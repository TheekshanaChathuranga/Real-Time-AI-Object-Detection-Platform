"""Pydantic schemas for analytics and dashboard summary metrics."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class RecentSessionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    session_type: str
    model_name: str
    start_time: datetime
    duration_ms: float
    object_count: int
    fps: float
    latency_ms: float
    status: str


class DashboardStatsResponse(BaseModel):
    total_sessions: int
    total_detected_objects: int
    active_cameras: int
    available_models: int
    average_fps: float
    average_latency_ms: float
    hardware_device: str
    recent_sessions: List[RecentSessionSummary]


class AnalyticsSummaryResponse(BaseModel):
    total_sessions: int
    total_detected_objects: int
    average_fps: float
    average_latency_ms: float
    class_distribution: Dict[str, int]
    session_type_distribution: Dict[str, int]
