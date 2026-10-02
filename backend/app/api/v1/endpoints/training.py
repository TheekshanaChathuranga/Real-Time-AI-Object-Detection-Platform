"""Custom model training endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.training import DatasetValidationResponse, TrainingJobCreate, TrainingJobResponse
from backend.app.services.training_service import TrainingService

router = APIRouter()


class DatasetValidateRequest(BaseModel):
    yaml_path: str


@router.post("", response_model=TrainingJobResponse, summary="Submit Custom Training Job")
def create_training_job(req: TrainingJobCreate, db: Session = Depends(get_db)):
    """Dispatch non-blocking custom YOLO model training job."""
    return TrainingService.start_job(db, req)


@router.get("", response_model=List[TrainingJobResponse], summary="List Training Jobs")
def list_training_jobs(db: Session = Depends(get_db)):
    """Retrieve history of all submitted training jobs and current statuses."""
    return TrainingService.list_jobs(db)


@router.get("/{job_id}", response_model=TrainingJobResponse, summary="Get Training Job Telemetry")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    """Retrieve live training progress, loss curves, precision, recall, and mAP."""
    job = TrainingService.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found.")
    return job


@router.post("/validate-dataset", response_model=DatasetValidationResponse, summary="Validate YOLO Dataset")
def validate_dataset(req: DatasetValidateRequest):
    """Validate dataset structure, images, annotations, and bounding box coordinates."""
    report = TrainingService.validate_dataset(req.yaml_path)
    return DatasetValidationResponse(
        is_valid=report.is_valid,
        errors=report.errors,
        warnings=report.warnings,
        train_images_count=report.train_images_count,
        val_images_count=report.val_images_count,
        num_classes=report.num_classes,
        class_names=report.class_names,
        total_annotations=report.total_annotations,
        class_distribution=report.class_distribution
    )
