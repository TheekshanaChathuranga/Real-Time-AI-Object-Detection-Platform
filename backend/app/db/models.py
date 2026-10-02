"""SQLAlchemy ORM models for detection platform entities."""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
)
from sqlalchemy.orm import relationship
from backend.app.db.base import Base


class ModelRecord(Base):
    """Registered AI detection model entity."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(Text, default="")
    model_type = Column(String(50), default="YOLOv8")
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    versions = relationship("ModelVersionRecord", back_populates="model", cascade="all, delete-orphan")


class ModelVersionRecord(Base):
    """Specific trained or baseline version of a model."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    version = Column(String(50), nullable=False)  # e.g., "v1.0", "custom-v1"
    file_path = Column(String(255), nullable=False)
    dataset_name = Column(String(100), default="")
    training_date = Column(DateTime, default=datetime.utcnow)
    
    # Evaluation benchmarks
    map50 = Column(Float, default=0.0)
    map50_95 = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    
    status = Column(String(50), default="ready")  # ready, training, deprecated
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRecord", back_populates="versions")


class InferenceSessionRecord(Base):
    """Session log for every image, video, webcam, or RTSP inference run."""
    __tablename__ = "inference_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    session_type = Column(String(30), nullable=False)  # image, video, webcam, rtsp
    model_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Float, default=0.0)
    object_count = Column(Integer, default=0)
    fps = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    source_file = Column(String(255), nullable=True)
    result_file = Column(String(255), nullable=True)
    status = Column(String(30), default="COMPLETED")

    detections = relationship("DetectionResultRecord", back_populates="session", cascade="all, delete-orphan")


class DetectionResultRecord(Base):
    """Individual object detection bounding box record."""
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("inference_sessions.session_id"), nullable=False)
    frame_index = Column(Integer, default=0)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("InferenceSessionRecord", back_populates="detections")


class TrainingJobRecord(Base):
    """Custom YOLO training job lifecycle and telemetry."""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    model_name = Column(String(100), nullable=False)
    dataset_path = Column(String(255), nullable=False)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=16)
    image_size = Column(Integer, default=640)
    learning_rate = Column(Float, default=0.01)
    device = Column(String(30), default="auto")
    
    # Progress & metrics
    status = Column(String(30), default="PENDING")
    current_epoch = Column(Integer, default=0)
    train_loss = Column(Float, default=0.0)
    val_loss = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    map50 = Column(Float, default=0.0)
    map50_95 = Column(Float, default=0.0)
    
    best_model_path = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class CameraRecord(Base):
    """Configured RTSP or IP camera source."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    rtsp_url = Column(String(255), nullable=False)
    resolution = Column(String(50), default="1280x720")
    target_fps = Column(Integer, default=30)
    model_name = Column(String(100), default="yolov8n.pt")
    status = Column(String(30), default="STOPPED")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatasetRecord(Base):
    """Registered dataset for model training."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    yaml_path = Column(String(255), nullable=False)
    num_classes = Column(Integer, default=0)
    train_count = Column(Integer, default=0)
    val_count = Column(Integer, default=0)
    is_valid = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemMetricRecord(Base):
    """Periodic hardware and throughput monitoring record."""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_percent = Column(Float, default=0.0)
    memory_percent = Column(Float, default=0.0)
    active_cameras = Column(Integer, default=0)
    total_inferences = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)
