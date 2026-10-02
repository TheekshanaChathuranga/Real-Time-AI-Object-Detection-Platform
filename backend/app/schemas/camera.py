"""Pydantic schemas for RTSP cameras."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class CameraCreate(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Warehouse Camera 01"})
    rtsp_url: str = Field(..., json_schema_extra={"example": "rtsp://192.168.1.100:554/stream1"})
    resolution: str = Field("1280x720", json_schema_extra={"example": "1280x720"})
    target_fps: int = Field(30, ge=1, le=120)
    model_name: str = Field("yolov8n.pt", json_schema_extra={"example": "yolov8n.pt"})


class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    name: str
    rtsp_url: str
    resolution: str
    target_fps: int
    model_name: str
    status: str
    is_active: bool
    created_at: datetime


class CameraStreamTelemetry(BaseModel):
    camera_id: str
    name: str
    rtsp_url: str
    status: str
    fps: float
    resolution: str
    total_frames_received: int
    reconnect_count: int
    uptime_seconds: float
    error_message: Optional[str] = None
