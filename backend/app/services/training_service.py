"""Training service coordinating dataset validation, job dispatching, and model registration."""

from datetime import datetime
import logging
from typing import Dict, List, Optional
import uuid
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.models import TrainingJobRecord
from backend.app.db.repository import ModelRepository, TrainingJobRepository
from backend.app.db.session import SessionLocal
from backend.app.schemas.training import TrainingJobCreate
from ml.training.dataset_validator import DatasetValidationReport, DatasetValidator
from ml.training.trainer import JobStatus, TrainingJobConfig, TrainingManager, TrainingProgress

logger = logging.getLogger("backend.services.training")

_training_manager = TrainingManager()


class TrainingService:
    @staticmethod
    def validate_dataset(yaml_path: str) -> DatasetValidationReport:
        """Validate dataset structure, labels, coordinates, and splits."""
        return DatasetValidator.validate_yaml_dataset(yaml_path)

    @classmethod
    def start_job(cls, db: Session, req: TrainingJobCreate) -> TrainingJobRecord:
        """Create database record and launch training job in background."""
        job_id = f"job-{uuid.uuid4().hex[:8]}"

        record = TrainingJobRepository.create_job(
            db=db,
            job_id=job_id,
            model_name=req.model_name,
            dataset_path=req.dataset_yaml,
            epochs=req.epochs,
            batch_size=req.batch_size,
            image_size=req.image_size,
            learning_rate=req.learning_rate,
            device=req.device
        )

        config = TrainingJobConfig(
            job_id=job_id,
            model_name=req.model_name,
            dataset_yaml=req.dataset_yaml,
            epochs=req.epochs,
            batch_size=req.batch_size,
            image_size=req.image_size,
            learning_rate=req.learning_rate,
            device=req.device,
            output_dir="runs/train",
            project_name=job_id
        )

        def progress_callback(progress: TrainingProgress):
            # Dedicated background session for DB update
            with SessionLocal() as bg_db:
                TrainingJobRepository.update_progress(
                    db=bg_db,
                    job_id=progress.job_id,
                    status=progress.status.value,
                    current_epoch=progress.current_epoch,
                    train_loss=progress.train_loss,
                    val_loss=progress.val_loss,
                    precision=progress.precision,
                    recall=progress.recall,
                    map50=progress.map50,
                    map50_95=progress.map50_95,
                    best_model_path=progress.best_model_path,
                    error_message=progress.error_message
                )

                # If job completed with best weights, register automatically in ModelRepository
                if progress.status == JobStatus.COMPLETED and progress.best_model_path:
                    try:
                        custom_model_name = f"custom_{job_id}.pt"
                        ModelRepository.create_or_update(
                            db=bg_db,
                            name=custom_model_name,
                            description=f"Trained model from {job_id} on {req.dataset_yaml}",
                            model_type="YOLO-Custom",
                            is_active=False
                        )
                        ModelRepository.add_version(
                            db=bg_db,
                            model_name=custom_model_name,
                            version="v1.0",
                            file_path=progress.best_model_path,
                            dataset_name=req.dataset_yaml,
                            map50=progress.map50,
                            map50_95=progress.map50_95,
                            precision=progress.precision,
                            recall=progress.recall,
                            status="ready",
                            notes=f"Trained for {req.epochs} epochs"
                        )
                    except Exception as ex:
                        logger.error(f"Failed to register model after training: {ex}")

        _training_manager.start_training(config, progress_callback=progress_callback)
        return record

    @staticmethod
    def get_job(db: Session, job_id: str) -> Optional[TrainingJobRecord]:
        return TrainingJobRepository.get_job(db, job_id)

    @staticmethod
    def list_jobs(db: Session, limit: int = 20) -> List[TrainingJobRecord]:
        return TrainingJobRepository.list_jobs(db, limit)
