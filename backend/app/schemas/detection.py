"""Pydantic schemas for image, video, webcam, and RTSP detection."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItemSchema(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BBoxSchema


class PerformanceMetricsSchema(BaseModel):
    preprocessing_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0
    number_of_detections: int = 0
    number_of_frames: int = 1


class ImageDetectionResponse(BaseModel):
    session_id: str
    model: str
    image_width: int
    image_height: int
    object_count: int
    detections: List[DetectionItemSchema]
    metrics: PerformanceMetricsSchema
    annotated_image_url: Optional[str] = None
    annotated_image_base64: Optional[str] = None


class VideoDetectionResponse(BaseModel):
    session_id: str
    model: str
    total_frames: int
    total_detections: int
    average_fps: float
    average_latency_ms: float
    class_distribution: Dict[str, int]
    processed_video_url: Optional[str] = None
    status: str = "COMPLETED"


class LiveFramePayload(BaseModel):
    frame_base64: str
    model_name: Optional[str] = None
    conf_threshold: Optional[float] = 0.25
    iou_threshold: Optional[float] = 0.45


class LiveDetectionResult(BaseModel):
    detections: List[DetectionItemSchema]
    object_count: int
    fps: float
    inference_latency_ms: float
    annotated_frame_base64: Optional[str] = None
