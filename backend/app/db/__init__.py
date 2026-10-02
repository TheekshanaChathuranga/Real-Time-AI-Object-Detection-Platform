"""Database package exports."""

from backend.app.db.base import Base
from backend.app.db.session import engine, SessionLocal, get_db, init_db
from backend.app.db.models import (
    ModelRecord,
    ModelVersionRecord,
    InferenceSessionRecord,
    DetectionResultRecord,
    TrainingJobRecord,
    CameraRecord,
    DatasetRecord,
    SystemMetricRecord
)
from backend.app.db.repository import (
    ModelRepository,
    InferenceRepository,
    CameraRepository,
    TrainingJobRepository
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "ModelRecord",
    "ModelVersionRecord",
    "InferenceSessionRecord",
    "DetectionResultRecord",
    "TrainingJobRecord",
    "CameraRecord",
    "DatasetRecord",
    "SystemMetricRecord",
    "ModelRepository",
    "InferenceRepository",
    "CameraRepository",
    "TrainingJobRepository"
]
