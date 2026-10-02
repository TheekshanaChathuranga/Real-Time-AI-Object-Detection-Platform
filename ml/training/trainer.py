"""Custom YOLO Training Manager supporting non-blocking background jobs and progress callbacks."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional
from ultralytics import YOLO

from ml.configs.model_configs import detect_device
from ml.training.dataset_validator import DatasetValidator

logger = logging.getLogger("ml.training.trainer")


class JobStatus(str, Enum):
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass
class TrainingJobConfig:
    """Configurable hyperparameters for a custom YOLO training job."""
    job_id: str
    model_name: str = "yolov8n.pt"
    dataset_yaml: str = ""
    epochs: int = 10
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    device: str = "auto"
    output_dir: str = "runs/train"
    project_name: str = "custom_yolo"


@dataclass
class TrainingProgress:
    """Live telemetry for an active or completed training job."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    best_model_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": round(self.train_loss, 4),
            "val_loss": round(self.val_loss, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "best_model_path": self.best_model_path,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class TrainingManager:
    """Manages training jobs, background thread isolation, and progress streaming."""

    def __init__(self):
        self._jobs: Dict[str, TrainingProgress] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def get_job_progress(self, job_id: str) -> Optional[TrainingProgress]:
        with self._lock:
            return self._jobs.get(job_id)

    def start_training(
        self,
        config: TrainingJobConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ) -> TrainingProgress:
        """Initiate training job in a separate background thread."""
        with self._lock:
            progress = TrainingProgress(
                job_id=config.job_id,
                status=JobStatus.VALIDATING,
                total_epochs=config.epochs,
                started_at=datetime.utcnow().isoformat()
            )
            self._jobs[config.job_id] = progress

        thread = threading.Thread(
            target=self._run_training_job,
            args=(config, progress, progress_callback),
            name=f"Train-{config.job_id}",
            daemon=True
        )
        self._threads[config.job_id] = thread
        thread.start()
        return progress

    def _run_training_job(
        self,
        config: TrainingJobConfig,
        progress: TrainingProgress,
        progress_callback: Optional[Callable[[TrainingProgress], None]]
    ) -> None:
        """Internal execution worker."""
        logger.info(f"Validating dataset for job {config.job_id}: {config.dataset_yaml}")

        # 1. Dataset validation step
        validation_report = DatasetValidator.validate_yaml_dataset(config.dataset_yaml)
        if not validation_report.is_valid:
            progress.status = JobStatus.FAILED
            progress.error_message = f"Dataset validation failed: {'; '.join(validation_report.errors)}"
            progress.completed_at = datetime.utcnow().isoformat()
            if progress_callback:
                progress_callback(progress)
            return

        progress.status = JobStatus.RUNNING
        device = detect_device(config.device)

        try:
            logger.info(f"Starting YOLO training: {config.model_name} on {device}")
            model = YOLO(config.model_name)

            # Custom epoch callback hook
            def on_train_epoch_end(trainer):
                progress.current_epoch = trainer.epoch + 1
                if hasattr(trainer, "loss"):
                    progress.train_loss = float(trainer.loss.item()) if hasattr(trainer.loss, "item") else float(trainer.loss)
                
                # Fetch validation metrics if computed
                if hasattr(trainer, "metrics"):
                    metrics = trainer.metrics
                    if metrics:
                        progress.precision = float(metrics.get("metrics/precision(B)", 0.0))
                        progress.recall = float(metrics.get("metrics/recall(B)", 0.0))
                        progress.map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
                        progress.map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))

                if progress_callback:
                    progress_callback(progress)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            results = model.train(
                data=config.dataset_yaml,
                epochs=config.epochs,
                batch=config.batch_size,
                imgsz=config.image_size,
                lr0=config.learning_rate,
                device=device,
                project=config.output_dir,
                name=config.project_name,
                verbose=False
            )

            # Locate best weights
            best_weights = Path(config.output_dir) / config.project_name / "weights" / "best.pt"
            if best_weights.exists():
                progress.best_model_path = str(best_weights.resolve())
            elif hasattr(results, "save_dir"):
                candidate = Path(results.save_dir) / "weights" / "best.pt"
                if candidate.exists():
                    progress.best_model_path = str(candidate.resolve())

            # Final metrics
            if hasattr(results, "results_dict"):
                r = results.results_dict
                progress.map50 = float(r.get("metrics/mAP50(B)", progress.map50))
                progress.map50_95 = float(r.get("metrics/mAP50-95(B)", progress.map50_95))
                progress.precision = float(r.get("metrics/precision(B)", progress.precision))
                progress.recall = float(r.get("metrics/recall(B)", progress.recall))

            progress.status = JobStatus.COMPLETED
            progress.completed_at = datetime.utcnow().isoformat()
            logger.info(f"Training job {config.job_id} successfully completed.")

        except Exception as e:
            logger.error(f"Training job {config.job_id} failed: {e}", exc_info=True)
            progress.status = JobStatus.FAILED
            progress.error_message = str(e)
            progress.completed_at = datetime.utcnow().isoformat()

        finally:
            if progress_callback:
                progress_callback(progress)
