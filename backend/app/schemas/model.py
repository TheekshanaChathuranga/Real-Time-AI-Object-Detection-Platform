"""Pydantic schemas for Model Management and Registry."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: str
    file_path: str
    dataset_name: str
    training_date: datetime
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1_score: float
    status: str
    notes: Optional[str] = ""


class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    model_type: str
    is_active: bool
    created_at: datetime
    versions: List[ModelVersionResponse] = []


class ModelActivateRequest(BaseModel):
    model_name: str
