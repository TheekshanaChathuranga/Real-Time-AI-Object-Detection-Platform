"""Webcam live streaming service supporting low-latency frame prediction."""

import base64
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from backend.app.schemas.detection import DetectionItemSchema, LiveDetectionResult
from backend.app.services.detection_service import get_detection_engine
from ml.preprocessing.image_transforms import ImagePreprocessor


class WebcamService:
    @staticmethod
    def process_webcam_frame(
        frame_base64: str,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        return_annotated_frame: bool = False,
        allowed_classes: Optional[list] = None
    ) -> Dict[str, Any]:
        """Decode base64 frame, run inference, and return structured detections."""
        # Strip data URL prefix if present
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(frame_base64)
        frame = ImagePreprocessor.load_image(image_bytes)

        detector = get_detection_engine(model_name)
        detections, metrics, annotated = detector.predict_frame(
            frame=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            annotate=return_annotated_frame,
            allowed_classes=allowed_classes,
            smooth=True
        )

        annotated_b64 = None
        if return_annotated_frame and annotated is not None:
            _, buf = cv2.imencode(".jpg", annotated)
            annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"

        return {
            "detections": [d.to_dict() for d in detections],
            "object_count": len(detections),
            "fps": round(metrics.fps, 1),
            "inference_latency_ms": round(metrics.inference_time_ms, 2),
            "annotated_frame_base64": annotated_b64
        }
