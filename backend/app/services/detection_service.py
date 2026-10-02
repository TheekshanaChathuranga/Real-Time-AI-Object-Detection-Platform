"""Detection service coordinating ML engine, storage, and persistence."""

import base64
from datetime import datetime
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import cv2
from fastapi import UploadFile
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.core.exceptions import InvalidFileError
from backend.app.core.security import ALLOWED_IMAGE_EXTENSIONS, sanitize_filename, validate_file_extension
from backend.app.db.repository import InferenceRepository, ModelRepository
from backend.app.schemas.detection import ImageDetectionResponse
from backend.app.storage.storage_manager import LocalStorageManager
from ml.inference.detector import DetectionEngine

storage = LocalStorageManager(settings.UPLOAD_DIRECTORY)
results_storage = LocalStorageManager(settings.RESULT_DIRECTORY)

# Global Detection Engine singleton
_engine_instance: Optional[DetectionEngine] = None


def get_detection_engine(model_name: Optional[str] = None) -> DetectionEngine:
    global _engine_instance
    target_model = model_name or settings.DEFAULT_MODEL
    if _engine_instance is None:
        _engine_instance = DetectionEngine(
            model_path_or_name=target_model,
            device=settings.DEVICE,
            conf_threshold=settings.DEFAULT_CONF_THRESHOLD,
            iou_threshold=settings.DEFAULT_IOU_THRESHOLD
        )
    elif model_name and _engine_instance.model_name != model_name:
        _engine_instance.load_model(model_name)
    return _engine_instance


class DetectionService:
    @staticmethod
    def detect_image(
        db: Session,
        file: UploadFile,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[str] = None
    ) -> ImageDetectionResponse:
        """Execute full image detection pipeline."""
        filename = sanitize_filename(file.filename or "upload.jpg")
        if not validate_file_extension(filename, ALLOWED_IMAGE_EXTENSIONS):
            raise InvalidFileError(f"Unsupported image format. Allowed: {ALLOWED_IMAGE_EXTENSIONS}")

        file_bytes = file.file.read()
        if not file_bytes:
            raise InvalidFileError("Uploaded file is empty.")

        session_id = str(uuid.uuid4())
        detector = get_detection_engine(model_name)

        allowed_cls = [c.strip().lower() for c in classes.split(",") if c.strip()] if classes else None

        # Predict
        response = detector.predict_image(
            image_input=file_bytes,
            conf=conf_threshold,
            iou=iou_threshold,
            annotate=True,
            allowed_classes=allowed_cls
        )

        # Save source & annotated images
        source_subpath = f"images/sources/{session_id}_{filename}"
        result_subpath = f"images/annotated/{session_id}_annotated.jpg"

        source_abs_path = storage.save_file(file_bytes, source_subpath)

        # Encode annotated image to JPEG bytes
        _, annotated_buf = cv2.imencode(".jpg", response.annotated_image)
        annotated_bytes = annotated_buf.tobytes()
        result_abs_path = results_storage.save_file(annotated_bytes, result_subpath)

        base64_encoded = base64.b64encode(annotated_bytes).decode("utf-8")

        # Persist session & detections in DB
        InferenceRepository.create_session(
            db=db,
            session_id=session_id,
            session_type="image",
            model_name=detector.model_name,
            duration_ms=response.metrics.total_latency_ms,
            object_count=response.object_count,
            fps=response.metrics.fps,
            latency_ms=response.metrics.inference_time_ms,
            source_file=source_abs_path,
            result_file=result_abs_path,
            status="COMPLETED"
        )

        detections_dict = [d.to_dict() for d in response.detections]
        InferenceRepository.add_detections(db, session_id, detections_dict)

        return ImageDetectionResponse(
            session_id=session_id,
            model=detector.model_name,
            image_width=response.image_width,
            image_height=response.image_height,
            object_count=response.object_count,
            detections=detections_dict,
            metrics=response.metrics.to_dict(),
            annotated_image_url=f"/api/v1/detection/results/{session_id}",
            annotated_image_base64=f"data:image/jpeg;base64,{base64_encoded}"
        )
