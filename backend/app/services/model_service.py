"""Model management service handling registry, weights discovery, and active state."""

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.core.exceptions import InvalidFileError, ModelNotFoundError
from backend.app.core.security import ALLOWED_MODEL_EXTENSIONS, sanitize_filename, validate_file_extension
from backend.app.db.models import ModelRecord
from backend.app.db.repository import ModelRepository
from backend.app.services.detection_service import get_detection_engine
from backend.app.storage.storage_manager import LocalStorageManager
from ml.configs.model_configs import PRETRAINED_MODELS
from ml.evaluation.evaluator import ModelEvaluator

model_storage = LocalStorageManager(settings.MODEL_DIRECTORY)


class ModelService:
    @classmethod
    def sync_models_with_db(cls, db: Session) -> None:
        """Ensure standard pretrained and discovered custom models exist in DB."""
        for filename, info in PRETRAINED_MODELS.items():
            record = ModelRepository.get_by_name(db, filename)
            if not record:
                is_def = (filename == settings.DEFAULT_MODEL)
                ModelRepository.create_or_update(
                    db=db,
                    name=filename,
                    description=info.description,
                    model_type="YOLOv8",
                    is_active=is_def
                )
                ModelRepository.add_version(
                    db=db,
                    model_name=filename,
                    version="v1.0-official",
                    file_path=filename,
                    dataset_name="COCO",
                    map50=0.528 if "n" in filename else 0.65,
                    map50_95=0.373 if "n" in filename else 0.45,
                    precision=0.68,
                    recall=0.62,
                    f1_score=0.65,
                    status="ready",
                    notes="Pretrained COCO baseline"
                )

    @classmethod
    def list_models(cls, db: Session) -> List[ModelRecord]:
        cls.sync_models_with_db(db)
        return ModelRepository.list_all(db)

    @classmethod
    def set_active_model(cls, db: Session, model_name: str) -> ModelRecord:
        record = ModelRepository.set_active(db, model_name)
        if not record:
            raise ModelNotFoundError(model_name)
        # Hot reload model into detection engine singleton
        engine = get_detection_engine()
        engine.load_model(model_name)
        return record

    @classmethod
    def register_custom_model(
        cls,
        db: Session,
        file: UploadFile,
        version: str = "v1.0",
        dataset_name: str = "custom",
        notes: str = ""
    ) -> ModelRecord:
        """Upload and register a custom YOLO weights file."""
        filename = sanitize_filename(file.filename or "custom_model.pt")
        if not validate_file_extension(filename, ALLOWED_MODEL_EXTENSIONS):
            raise InvalidFileError(f"Unsupported model extension. Allowed: {ALLOWED_MODEL_EXTENSIONS}")

        file_bytes = file.file.read()
        saved_path = model_storage.save_file(file_bytes, filename)

        record = ModelRepository.create_or_update(
            db=db,
            name=filename,
            description=f"Custom YOLO model: {filename}",
            model_type="YOLO-Custom",
            is_active=False
        )

        ModelRepository.add_version(
            db=db,
            model_name=filename,
            version=version,
            file_path=saved_path,
            dataset_name=dataset_name,
            notes=notes
        )
        return record

    @classmethod
    def evaluate_model(
        cls,
        db: Session,
        model_name: str,
        dataset_yaml: str
    ) -> Dict:
        """Run validation and record benchmark results."""
        report = ModelEvaluator.evaluate(
            model_path=model_name,
            data_yaml=dataset_yaml,
            device=settings.DEVICE
        )

        # Update or record version
        ModelRepository.add_version(
            db=db,
            model_name=model_name,
            version=f"eval-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            file_path=model_name,
            dataset_name=dataset_yaml,
            map50=report.map50,
            map50_95=report.map50_95,
            precision=report.precision,
            recall=report.recall,
            f1_score=report.f1_score,
            status="evaluated",
            notes=f"Evaluation speed: {report.inference_speed_ms:.2f}ms"
        )
        return report.to_dict()
