"""RTSP stream service managing active camera streams and MJPEG preview generators."""

import logging
import threading
from typing import Dict, Generator, Optional
import cv2
from sqlalchemy.orm import Session

from backend.app.core.exceptions import CameraStreamError
from backend.app.db.repository import CameraRepository
from backend.app.services.detection_service import get_detection_engine
from ml.streaming.rtsp_client import RTSPStreamClient, StreamStatus

logger = logging.getLogger("backend.services.stream")


class StreamService:
    """Manages isolated RTSP stream lifecycles across the application."""

    _active_streams: Dict[str, RTSPStreamClient] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create_client(
        cls,
        camera_id: str,
        rtsp_url: str,
        name: str = "Camera",
        target_fps: int = 30
    ) -> RTSPStreamClient:
        with cls._lock:
            if camera_id not in cls._active_streams:
                client = RTSPStreamClient(
                    camera_id=camera_id,
                    rtsp_url=rtsp_url,
                    name=name,
                    target_fps=target_fps
                )
                cls._active_streams[camera_id] = client
            return cls._active_streams[camera_id]

    @classmethod
    def start_camera_stream(
        cls,
        camera_id: str,
        rtsp_url: str,
        name: str,
        model_name: str,
        db: Session
    ) -> Dict:
        """Start thread-isolated RTSP stream with detection frame processor."""
        client = cls.get_or_create_client(camera_id, rtsp_url, name)
        detector = get_detection_engine(model_name)

        def frame_processor(frame):
            try:
                # Run inference on frame with temporal smoothing
                _, _, annotated = detector.predict_frame(frame, annotate=True, smooth=True)
                if annotated is not None:
                    client.set_latest_annotated_frame(annotated)
            except Exception as e:
                logger.error(f"Error processing RTSP frame for {camera_id}: {e}")

        client.start(frame_processor=frame_processor)
        CameraRepository.update_status(db, camera_id, "RUNNING")
        return client.get_status()

    @classmethod
    def stop_camera_stream(cls, camera_id: str, db: Session) -> Dict:
        with cls._lock:
            client = cls._active_streams.get(camera_id)
        if client:
            client.stop()
            CameraRepository.update_status(db, camera_id, "STOPPED")
            return client.get_status()
        CameraRepository.update_status(db, camera_id, "STOPPED")
        return {"camera_id": camera_id, "status": "STOPPED"}

    @classmethod
    def get_stream_status(cls, camera_id: str) -> Dict:
        with cls._lock:
            client = cls._active_streams.get(camera_id)
        if client:
            return client.get_status()
        return {"camera_id": camera_id, "status": "STOPPED", "fps": 0.0}

    @classmethod
    def generate_mjpeg_stream(cls, camera_id: str) -> Generator[bytes, None, None]:
        """Yield multipart/x-mixed-replace JPEG frames for live browser preview."""
        with cls._lock:
            client = cls._active_streams.get(camera_id)
        if not client:
            raise CameraStreamError(camera_id, "Stream is not currently active.")

        while client.status in [StreamStatus.RUNNING, StreamStatus.CONNECTING]:
            frame = client.get_latest_annotated_frame() or client.get_latest_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode(".jpg", frame)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )
            import time
            time.sleep(1.0 / max(client.target_fps, 10))
