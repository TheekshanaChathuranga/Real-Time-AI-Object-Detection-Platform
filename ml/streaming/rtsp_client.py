"""RTSP Stream Client with thread-isolated lifecycle, auto-reconnection, and telemetry."""

from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional
import cv2
import numpy as np

logger = logging.getLogger("ml.streaming.rtsp")


class StreamStatus(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class RTSPStreamClient:
    """Thread-isolated RTSP stream manager with automatic reconnection and frame buffering."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        name: str = "Camera",
        target_fps: int = 30,
        max_reconnect_attempts: int = 10,
        reconnect_delay_seconds: float = 3.0
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.name = name
        self.target_fps = target_fps
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self.status: StreamStatus = StreamStatus.IDLE
        self.error_message: Optional[str] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Telemetry
        self.fps: float = 0.0
        self.total_frames_received: int = 0
        self.reconnect_count: int = 0
        self.start_time: Optional[float] = None
        self.resolution: str = "Unknown"

    def start(self, frame_processor: Optional[Callable[[np.ndarray], Any]] = None) -> bool:
        """Start background capture thread."""
        with self._lock:
            if self.status in [StreamStatus.RUNNING, StreamStatus.CONNECTING]:
                return True

            self._stop_event.clear()
            self.status = StreamStatus.CONNECTING
            self.error_message = None
            self.start_time = time.time()
            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(frame_processor,),
                name=f"RTSP-{self.camera_id}",
                daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Stop background capture thread gracefully."""
        self._stop_event.set()
        with self._lock:
            self.status = StreamStatus.STOPPED
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Thread-safe retrieval of the most recent frame."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Thread-safe retrieval of the most recent annotated frame."""
        with self._lock:
            return self._latest_annotated_frame.copy() if self._latest_annotated_frame is not None else None

    def set_latest_annotated_frame(self, frame: np.ndarray) -> None:
        """Store annotated frame."""
        with self._lock:
            self._latest_annotated_frame = frame

    def get_status(self) -> Dict[str, Any]:
        """Return real-time stream status and telemetry."""
        uptime = (time.time() - self.start_time) if self.start_time and self.status == StreamStatus.RUNNING else 0.0
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "rtsp_url": self.rtsp_url,
            "status": self.status.value,
            "fps": round(self.fps, 1),
            "resolution": self.resolution,
            "total_frames_received": self.total_frames_received,
            "reconnect_count": self.reconnect_count,
            "uptime_seconds": round(uptime, 1),
            "error_message": self.error_message
        }

    def _capture_loop(self, frame_processor: Optional[Callable[[np.ndarray], Any]]) -> None:
        """Internal capture loop with reconnect resilience."""
        attempts = 0

        while not self._stop_event.is_set():
            logger.info(f"Opening RTSP stream [{self.name}]: {self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            # Set buffer size to minimum to prevent latency accumulation
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                attempts += 1
                self.reconnect_count = attempts
                with self._lock:
                    self.status = StreamStatus.RECONNECTING
                    self.error_message = f"Failed to connect (attempt {attempts}/{self.max_reconnect_attempts})"
                logger.warning(f"RTSP [{self.camera_id}] connect failed: attempt {attempts}")

                if attempts >= self.max_reconnect_attempts:
                    with self._lock:
                        self.status = StreamStatus.ERROR
                        self.error_message = "Exceeded maximum reconnect attempts"
                    break

                time.sleep(self.reconnect_delay_seconds)
                continue

            # Successfully connected
            attempts = 0
            with self._lock:
                self.status = StreamStatus.RUNNING
                self.error_message = None
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.resolution = f"{width}x{height}" if width > 0 else "Active"

            fps_calc_start = time.time()
            frames_in_second = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"RTSP [{self.camera_id}] lost frame signal. Attempting reconnect.")
                    with self._lock:
                        self.status = StreamStatus.RECONNECTING
                    break

                with self._lock:
                    self._latest_frame = frame
                    self.total_frames_received += 1

                frames_in_second += 1
                elapsed = time.time() - fps_calc_start
                if elapsed >= 1.0:
                    self.fps = frames_in_second / elapsed
                    frames_in_second = 0
                    fps_calc_start = time.time()

                if frame_processor:
                    try:
                        frame_processor(frame)
                    except Exception as ex:
                        logger.error(f"Error in RTSP frame processor for [{self.camera_id}]: {ex}")

            cap.release()
            if self._stop_event.is_set():
                break
            time.sleep(self.reconnect_delay_seconds)

        with self._lock:
            if self.status != StreamStatus.ERROR:
                self.status = StreamStatus.STOPPED
