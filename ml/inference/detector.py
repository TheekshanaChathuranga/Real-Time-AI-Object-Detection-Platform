"""Core Detection Engine implementing the unified Detector interface.

Designed as an extensible ML layer decoupled from web frameworks.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

from ml.configs.model_configs import detect_device
from ml.inference.postprocessor import BoundingBox, Detection, Postprocessor
from ml.inference.temporal_smoother import TemporalSmoother
from ml.preprocessing.image_transforms import ImagePreprocessor

logger = logging.getLogger("ml.inference.detector")


@dataclass
class PerformanceMetrics:
    """Detailed latency and throughput metrics for inference operations."""
    preprocessing_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0
    number_of_detections: int = 0
    number_of_frames: int = 1

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocessing_time_ms": round(self.preprocessing_time_ms, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "postprocessing_time_ms": round(self.postprocessing_time_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "fps": round(self.fps, 1),
            "number_of_detections": self.number_of_detections,
            "number_of_frames": self.number_of_frames,
        }


@dataclass
class DetectionResponse:
    """Unified detection response returned by the detector."""
    model_name: str
    detections: List[Detection]
    object_count: int
    image_width: int
    image_height: int
    metrics: PerformanceMetrics
    annotated_image: Optional[np.ndarray] = None

    def to_dict(self, include_image_bytes: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "model": self.model_name,
            "object_count": self.object_count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detections": [d.to_dict() for d in self.detections],
            "metrics": self.metrics.to_dict(),
        }
        return result


class DetectionEngine:
    """Modular Object Detection Engine using Ultralytics YOLO with hardware acceleration."""

    def __init__(
        self,
        model_path_or_name: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.device = detect_device(device)
        self.model_name = model_path_or_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model: Optional[YOLO] = None
        self.smoother = TemporalSmoother()
        self._cumulative_frames: int = 0
        self._cumulative_inference_time_ms: float = 0.0
        self._cumulative_detections: int = 0
        
        self.load_model(model_path_or_name, self.device)

    def load_model(self, model_path_or_name: str, device: Optional[str] = None) -> None:
        """Load or switch the active YOLO model weights."""
        if device:
            self.device = detect_device(device)

        logger.info(f"Loading YOLO model: {model_path_or_name} on device: {self.device}")
        try:
            self.model = YOLO(model_path_or_name)
            self.model_name = model_path_or_name
            if hasattr(self, "smoother"):
                self.smoother.reset()
            logger.info(f"Model {model_path_or_name} loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_path_or_name}': {str(e)}")
            raise RuntimeError(f"Error loading model weights '{model_path_or_name}': {e}") from e

    def predict_image(
        self,
        image_input: Union[str, bytes, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        annotate: bool = True,
        allowed_classes: Optional[List[str]] = None
    ) -> DetectionResponse:
        """Run full object detection pipeline on a single image."""
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold

        t_start = time.perf_counter()

        # Step 1: Preprocessing & Decoding
        t_pre_start = time.perf_counter()
        img = ImagePreprocessor.load_image(image_input)
        h, w = img.shape[:2]
        t_pre = (time.perf_counter() - t_pre_start) * 1000.0

        # Step 2: Inference
        t_inf_start = time.perf_counter()
        results = self.model.predict(
            source=img,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        t_inf = (time.perf_counter() - t_inf_start) * 1000.0

        # Step 3: Postprocessing & Parsing
        t_post_start = time.perf_counter()
        first_result = results[0] if results else None
        detections = Postprocessor.parse_ultralytics_result(first_result, allowed_classes=allowed_classes)
        
        annotated_img = None
        if annotate:
            annotated_img = Postprocessor.draw_detections(img, detections)
        t_post = (time.perf_counter() - t_post_start) * 1000.0

        total_latency = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / total_latency if total_latency > 0 else 0.0

        metrics = PerformanceMetrics(
            preprocessing_time_ms=t_pre,
            inference_time_ms=t_inf,
            postprocessing_time_ms=t_post,
            total_latency_ms=total_latency,
            fps=fps,
            number_of_detections=len(detections),
            number_of_frames=1
        )

        self._record_telemetry(1, t_inf, len(detections))

        return DetectionResponse(
            model_name=self.model_name,
            detections=detections,
            object_count=len(detections),
            image_width=w,
            image_height=h,
            metrics=metrics,
            annotated_image=annotated_img
        )

    def predict_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        annotate: bool = False,
        allowed_classes: Optional[List[str]] = None,
        smooth: bool = False
    ) -> Tuple[List[Detection], PerformanceMetrics, Optional[np.ndarray]]:
        """Optimized frame-by-frame inference for video and real-time streaming."""
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold

        t_start = time.perf_counter()
        t_inf_start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        t_inf = (time.perf_counter() - t_inf_start) * 1000.0

        t_post_start = time.perf_counter()
        first_result = results[0] if results else None
        detections = Postprocessor.parse_ultralytics_result(first_result, allowed_classes=allowed_classes)
        
        # Apply temporal smoothing to stop class flickering (e.g. truck <-> bus)
        if smooth:
            detections = self.smoother.update(detections)

        annotated_frame = None
        if annotate:
            annotated_frame = Postprocessor.draw_detections(frame, detections)
        t_post = (time.perf_counter() - t_post_start) * 1000.0

        total_latency = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / total_latency if total_latency > 0 else 0.0

        metrics = PerformanceMetrics(
            preprocessing_time_ms=0.0,
            inference_time_ms=t_inf,
            postprocessing_time_ms=t_post,
            total_latency_ms=total_latency,
            fps=fps,
            number_of_detections=len(detections),
            number_of_frames=1
        )

        self._record_telemetry(1, t_inf, len(detections))

        return detections, metrics, annotated_frame

    def predict_video(
        self,
        source_path: str,
        output_path: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        allowed_classes: Optional[List[str]] = None,
        smooth: bool = True,
        progress_callback: Optional[Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Process video frame-by-frame with anti-jitter temporal smoothing."""
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

        frame_idx = 0
        total_detections = 0
        class_counter: Dict[str, int] = {}
        total_inference_time = 0.0
        video_smoother = TemporalSmoother()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                detections, metrics, _ = self.predict_frame(
                    frame=frame,
                    conf=conf,
                    iou=iou,
                    annotate=False,
                    allowed_classes=allowed_classes,
                    smooth=False
                )

                if smooth:
                    detections = video_smoother.update(detections)

                annotated_frame = None
                if output_path:
                    annotated_frame = Postprocessor.draw_detections(frame, detections)

                total_detections += len(detections)
                total_inference_time += metrics.inference_time_ms

                for det in detections:
                    class_counter[det.class_name] = class_counter.get(det.class_name, 0) + 1

                if writer and annotated_frame is not None:
                    writer.write(annotated_frame)

                progress_pct = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0

                yield {
                    "frame_index": frame_idx,
                    "total_frames": total_frames,
                    "progress_percentage": round(progress_pct, 1),
                    "detections_count": len(detections),
                    "fps": round(metrics.fps, 1),
                    "inference_latency_ms": round(metrics.inference_time_ms, 2),
                    "class_distribution": dict(class_counter)
                }

        finally:
            cap.release()
            if writer:
                writer.release()

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative runtime performance statistics."""
        avg_latency = (
            self._cumulative_inference_time_ms / self._cumulative_frames
            if self._cumulative_frames > 0 else 0.0
        )
        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_frames_processed": self._cumulative_frames,
            "total_detections": self._cumulative_detections,
            "average_inference_latency_ms": round(avg_latency, 2),
            "estimated_fps": round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0
        }

    def _record_telemetry(self, frames: int, inf_time_ms: float, detections: int) -> None:
        """Internal telemetry counter."""
        self._cumulative_frames += frames
        self._cumulative_inference_time_ms += inf_time_ms
        self._cumulative_detections += detections
