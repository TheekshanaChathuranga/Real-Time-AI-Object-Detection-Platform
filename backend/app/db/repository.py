"""Repository pattern implementation for isolated database persistence."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.app.db.models import (
    CameraRecord,
    DatasetRecord,
    DetectionResultRecord,
    InferenceSessionRecord,
    ModelRecord,
    ModelVersionRecord,
    SystemMetricRecord,
    TrainingJobRecord,
)


class ModelRepository:
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ModelRecord]:
        return db.query(ModelRecord).filter(ModelRecord.name == name).first()

    @staticmethod
    def get_active(db: Session) -> Optional[ModelRecord]:
        return db.query(ModelRecord).filter(ModelRecord.is_active == True).first()

    @staticmethod
    def list_all(db: Session) -> List[ModelRecord]:
        return db.query(ModelRecord).all()

    @staticmethod
    def create_or_update(
        db: Session,
        name: str,
        description: str = "",
        model_type: str = "YOLOv8",
        is_active: bool = False
    ) -> ModelRecord:
        record = db.query(ModelRecord).filter(ModelRecord.name == name).first()
        if not record:
            record = ModelRecord(
                name=name,
                description=description,
                model_type=model_type,
                is_active=is_active
            )
            db.add(record)
        else:
            record.description = description
            record.model_type = model_type
            if is_active:
                record.is_active = True
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def set_active(db: Session, name: str) -> Optional[ModelRecord]:
        target = db.query(ModelRecord).filter(ModelRecord.name == name).first()
        if not target:
            return None
        # Deactivate others
        db.query(ModelRecord).update({ModelRecord.is_active: False})
        target.is_active = True
        db.commit()
        db.refresh(target)
        return target

    @staticmethod
    def add_version(
        db: Session,
        model_name: str,
        version: str,
        file_path: str,
        dataset_name: str = "",
        map50: float = 0.0,
        map50_95: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        status: str = "ready",
        notes: str = ""
    ) -> ModelVersionRecord:
        model = ModelRepository.get_by_name(db, model_name)
        if not model:
            model = ModelRepository.create_or_update(db, model_name)

        version_record = ModelVersionRecord(
            model_id=model.id,
            version=version,
            file_path=file_path,
            dataset_name=dataset_name,
            map50=map50,
            map50_95=map50_95,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            status=status,
            notes=notes
        )
        db.add(version_record)
        db.commit()
        db.refresh(version_record)
        return version_record


class InferenceRepository:
    @staticmethod
    def create_session(
        db: Session,
        session_id: str,
        session_type: str,
        model_name: str,
        duration_ms: float = 0.0,
        object_count: int = 0,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        source_file: Optional[str] = None,
        result_file: Optional[str] = None,
        status: str = "COMPLETED"
    ) -> InferenceSessionRecord:
        record = InferenceSessionRecord(
            session_id=session_id,
            session_type=session_type,
            model_name=model_name,
            duration_ms=duration_ms,
            object_count=object_count,
            fps=fps,
            latency_ms=latency_ms,
            source_file=source_file,
            result_file=result_file,
            status=status
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def add_detections(
        db: Session,
        session_id: str,
        detections: List[Dict[str, Any]],
        frame_index: int = 0
    ) -> None:
        records = [
            DetectionResultRecord(
                session_id=session_id,
                frame_index=frame_index,
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                x1=d["bbox"]["x1"],
                y1=d["bbox"]["y1"],
                x2=d["bbox"]["x2"],
                y2=d["bbox"]["y2"]
            )
            for d in detections
        ]
        db.bulk_save_objects(records)
        db.commit()

    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[InferenceSessionRecord]:
        return db.query(InferenceSessionRecord).filter(InferenceSessionRecord.session_id == session_id).first()

    @staticmethod
    def list_recent_sessions(db: Session, limit: int = 20) -> List[InferenceSessionRecord]:
        return db.query(InferenceSessionRecord).order_by(desc(InferenceSessionRecord.start_time)).limit(limit).all()

    @staticmethod
    def get_summary_metrics(db: Session) -> Dict[str, Any]:
        total_sessions = db.query(InferenceSessionRecord).count()
        total_objects = db.query(func.sum(InferenceSessionRecord.object_count)).scalar() or 0
        avg_fps = db.query(func.avg(InferenceSessionRecord.fps)).filter(InferenceSessionRecord.fps > 0).scalar() or 0.0
        avg_latency = db.query(func.avg(InferenceSessionRecord.latency_ms)).filter(InferenceSessionRecord.latency_ms > 0).scalar() or 0.0

        # Class distribution
        class_counts = (
            db.query(DetectionResultRecord.class_name, func.count(DetectionResultRecord.id))
            .group_by(DetectionResultRecord.class_name)
            .order_by(desc(func.count(DetectionResultRecord.id)))
            .limit(10)
            .all()
        )
        distribution = {name: count for name, count in class_counts}

        return {
            "total_sessions": total_sessions,
            "total_detected_objects": int(total_objects),
            "average_fps": round(float(avg_fps), 1),
            "average_latency_ms": round(float(avg_latency), 2),
            "class_distribution": distribution
        }


class CameraRepository:
    @staticmethod
    def get_by_id(db: Session, camera_id: str) -> Optional[CameraRecord]:
        return db.query(CameraRecord).filter(CameraRecord.camera_id == camera_id).first()

    @staticmethod
    def list_all(db: Session) -> List[CameraRecord]:
        return db.query(CameraRecord).all()

    @staticmethod
    def create(
        db: Session,
        camera_id: str,
        name: str,
        rtsp_url: str,
        resolution: str = "1280x720",
        target_fps: int = 30,
        model_name: str = "yolov8n.pt"
    ) -> CameraRecord:
        record = CameraRecord(
            camera_id=camera_id,
            name=name,
            rtsp_url=rtsp_url,
            resolution=resolution,
            target_fps=target_fps,
            model_name=model_name,
            status="STOPPED"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def update_status(db: Session, camera_id: str, status: str) -> Optional[CameraRecord]:
        record = CameraRepository.get_by_id(db, camera_id)
        if record:
            record.status = status
            db.commit()
            db.refresh(record)
        return record

    @staticmethod
    def delete(db: Session, camera_id: str) -> bool:
        record = CameraRepository.get_by_id(db, camera_id)
        if record:
            db.delete(record)
            db.commit()
            return True
        return False


class TrainingJobRepository:
    @staticmethod
    def create_job(db: Session, job_id: str, model_name: str, dataset_path: str, epochs: int, batch_size: int, image_size: int, learning_rate: float, device: str) -> TrainingJobRecord:
        record = TrainingJobRecord(
            job_id=job_id,
            model_name=model_name,
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            device=device,
            status="PENDING"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_job(db: Session, job_id: str) -> Optional[TrainingJobRecord]:
        return db.query(TrainingJobRecord).filter(TrainingJobRecord.job_id == job_id).first()

    @staticmethod
    def update_progress(
        db: Session,
        job_id: str,
        status: str,
        current_epoch: int,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        map50: float = 0.0,
        map50_95: float = 0.0,
        best_model_path: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[TrainingJobRecord]:
        record = TrainingJobRepository.get_job(db, job_id)
        if record:
            record.status = status
            record.current_epoch = current_epoch
            record.train_loss = train_loss
            record.val_loss = val_loss
            record.precision = precision
            record.recall = recall
            record.map50 = map50
            record.map50_95 = map50_95
            if best_model_path:
                record.best_model_path = best_model_path
            if error_message:
                record.error_message = error_message
            if status in ["COMPLETED", "FAILED", "STOPPED"]:
                record.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(record)
        return record

    @staticmethod
    def list_jobs(db: Session, limit: int = 20) -> List[TrainingJobRecord]:
        return db.query(TrainingJobRecord).order_by(desc(TrainingJobRecord.created_at)).limit(limit).all()
