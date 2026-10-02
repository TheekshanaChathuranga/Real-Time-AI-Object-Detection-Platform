"""Pydantic schemas for training jobs and dataset validation."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DatasetValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    train_images_count: int = 0
    val_images_count: int = 0
    num_classes: int = 0
    class_names: List[str] = []
    total_annotations: int = 0
    class_distribution: Dict[str, int] = {}


class TrainingJobCreate(BaseModel):
    model_name: str = Field("yolov8n.pt", json_schema_extra={"example": "yolov8n.pt"})
    dataset_yaml: str = Field(..., json_schema_extra={"example": "data/datasets/coco8/data.yaml"})
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(16, ge=1, le=128)
    image_size: int = Field(640, ge=320, le=1280)
    learning_rate: float = Field(0.01, gt=0.0, le=1.0)
    device: str = Field("auto", json_schema_extra={"example": "auto"})


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    model_name: str
    dataset_path: str
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    device: str
    status: str
    current_epoch: int
    train_loss: float
    val_loss: float
    precision: float
    recall: float
    map50: float
    map50_95: float
    best_model_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
