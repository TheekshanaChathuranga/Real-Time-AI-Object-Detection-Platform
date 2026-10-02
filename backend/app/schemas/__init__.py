"""Schemas package exports."""

from backend.app.schemas.detection import (
    BBoxSchema,
    DetectionItemSchema,
    PerformanceMetricsSchema,
    ImageDetectionResponse,
    VideoDetectionResponse,
    LiveFramePayload,
    LiveDetectionResult,
)
from backend.app.schemas.camera import (
    CameraCreate,
    CameraResponse,
    CameraStreamTelemetry,
)
from backend.app.schemas.model import (
    ModelResponse,
    ModelVersionResponse,
    ModelActivateRequest,
)
from backend.app.schemas.training import (
    TrainingJobCreate,
    TrainingJobResponse,
    DatasetValidationResponse,
)
from backend.app.schemas.analytics import (
    DashboardStatsResponse,
    AnalyticsSummaryResponse,
    RecentSessionSummary,
)

__all__ = [
    "BBoxSchema",
    "DetectionItemSchema",
    "PerformanceMetricsSchema",
    "ImageDetectionResponse",
    "VideoDetectionResponse",
    "LiveFramePayload",
    "LiveDetectionResult",
    "CameraCreate",
    "CameraResponse",
    "CameraStreamTelemetry",
    "ModelResponse",
    "ModelVersionResponse",
    "ModelActivateRequest",
    "TrainingJobCreate",
    "TrainingJobResponse",
    "DatasetValidationResponse",
    "DashboardStatsResponse",
    "AnalyticsSummaryResponse",
    "RecentSessionSummary",
]
