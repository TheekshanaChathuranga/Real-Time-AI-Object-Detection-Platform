"""Model management and registry endpoints."""

from typing import List
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.model import ModelActivateRequest, ModelResponse
from backend.app.services.model_service import ModelService

router = APIRouter()


class ModelEvaluateRequest(BaseModel):
    model_name: str
    dataset_yaml: str


@router.get("", response_model=List[ModelResponse], summary="List Registered Models")
def get_models(db: Session = Depends(get_db)):
    """List all registered models, their active status, and evaluation versions."""
    return ModelService.list_models(db)


@router.post("/activate", response_model=ModelResponse, summary="Set Active Model")
def activate_model(req: ModelActivateRequest, db: Session = Depends(get_db)):
    """Set the specified model as globally active for default inference."""
    return ModelService.set_active_model(db, req.model_name)


@router.post("/upload", response_model=ModelResponse, summary="Upload Custom Weights")
def upload_model_weights(
    file: UploadFile = File(..., description="Custom model file (.pt, .onnx)"),
    version: str = Form("v1.0"),
    dataset_name: str = Form("custom"),
    notes: str = Form(""),
    db: Session = Depends(get_db)
):
    """Upload custom YOLO weights and register into the model version registry."""
    return ModelService.register_custom_model(
        db=db,
        file=file,
        version=version,
        dataset_name=dataset_name,
        notes=notes
    )


@router.post("/evaluate", summary="Run Model Evaluation Benchmark")
def evaluate_model(req: ModelEvaluateRequest, db: Session = Depends(get_db)):
    """Run validation benchmark on selected model and dataset to compute mAP, precision, and recall."""
    return ModelService.evaluate_model(
        db=db,
        model_name=req.model_name,
        dataset_yaml=req.dataset_yaml
    )
