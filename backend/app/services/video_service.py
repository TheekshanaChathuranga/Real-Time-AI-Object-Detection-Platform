"""Video detection service handling streaming video inference and persistence."""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
from fastapi import UploadFile
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.core.exceptions import InvalidFileError
from backend.app.core.security import ALLOWED_VIDEO_EXTENSIONS, sanitize_filename, validate_file_extension
from backend.app.db.repository import InferenceRepository
from backend.app.schemas.detection import VideoDetectionResponse
from backend.app.services.detection_service import get_detection_engine
from backend.app.storage.storage_manager import LocalStorageManager

storage = LocalStorageManager(settings.UPLOAD_DIRECTORY)
results_storage = LocalStorageManager(settings.RESULT_DIRECTORY)


class VideoService:
    @staticmethod
    def process_video(
        db: Session,
        file: UploadFile,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[str] = None
    ) -> VideoDetectionResponse:
        """Process video frame-by-frame and persist detection results."""
        filename = sanitize_filename(file.filename or "video.mp4")
        if not validate_file_extension(filename, ALLOWED_VIDEO_EXTENSIONS):
            raise InvalidFileError(f"Unsupported video format. Allowed: {ALLOWED_VIDEO_EXTENSIONS}")

        session_id = str(uuid.uuid4())
        detector = get_detection_engine(model_name)

        allowed_cls = [c.strip().lower() for c in classes.split(",") if c.strip()] if classes else None

        # Save source video to disk
        source_subpath = f"videos/sources/{session_id}_{filename}"
        output_filename = f"{session_id}_processed.mp4"
        output_subpath = f"videos/annotated/{output_filename}"

        source_abs_path = storage.save_file(file.file, source_subpath)
        output_abs_path = str((Path(settings.RESULT_DIRECTORY) / output_subpath).resolve())
        Path(output_abs_path).parent.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        total_detections = 0
        fps_sum = 0.0
        latency_sum = 0.0
        final_class_distribution: Dict[str, int] = {}

        # Stream frames through generator with temporal anti-jitter smoothing
        for step in detector.predict_video(
            source_path=source_abs_path,
            output_path=output_abs_path,
            conf=conf_threshold,
            iou=iou_threshold,
            allowed_classes=allowed_cls,
            smooth=True
        ):
            total_frames = step["total_frames"]
            fps_sum += step["fps"]
            latency_sum += step["inference_latency_ms"]
            final_class_distribution = step["class_distribution"]
            total_detections += step["detections_count"]

        avg_fps = fps_sum / total_frames if total_frames > 0 else 0.0
        avg_latency = latency_sum / total_frames if total_frames > 0 else 0.0

        # Persist session
        InferenceRepository.create_session(
            db=db,
            session_id=session_id,
            session_type="video",
            model_name=detector.model_name,
            duration_ms=latency_sum,
            object_count=total_detections,
            fps=avg_fps,
            latency_ms=avg_latency,
            source_file=source_abs_path,
            result_file=output_abs_path,
            status="COMPLETED"
        )

        return VideoDetectionResponse(
            session_id=session_id,
            model=detector.model_name,
            total_frames=total_frames,
            total_detections=total_detections,
            average_fps=round(avg_fps, 1),
            average_latency_ms=round(avg_latency, 2),
            class_distribution=final_class_distribution,
            processed_video_url=f"/api/v1/detection/videos/{session_id}",
            status="COMPLETED"
        )
