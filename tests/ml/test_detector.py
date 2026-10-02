"""Tests for DetectionEngine, Postprocessor, and Preprocessor."""

import os
import tempfile
import cv2
import numpy as np
import pytest

from ml.inference.detector import DetectionEngine, DetectionResponse, PerformanceMetrics
from ml.inference.postprocessor import BoundingBox, Detection, Postprocessor
from ml.preprocessing.image_transforms import ImagePreprocessor


@pytest.fixture(scope="session")
def detector():
    """Shared DetectionEngine instance using default lightweight weights."""
    return DetectionEngine(model_path_or_name="yolov8n.pt", device="cpu")


@pytest.fixture
def sample_image():
    """Create a synthetic 640x480 RGB image with some shapes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a blue background and a bright rectangle
    img[:] = (50, 50, 50)
    cv2.rectangle(img, (100, 100), (300, 350), (200, 200, 200), -1)
    return img


def test_image_preprocessor_load(sample_image):
    """Test loading from numpy array, bytes, and error handling."""
    # From array
    loaded = ImagePreprocessor.load_image(sample_image)
    assert loaded.shape == (480, 640, 3)

    # From encoded bytes
    _, buf = cv2.imencode(".jpg", sample_image)
    loaded_bytes = ImagePreprocessor.load_image(buf.tobytes())
    assert loaded_bytes.shape == (480, 640, 3)

    # Corrupt bytes test
    with pytest.raises(ValueError):
        ImagePreprocessor.load_image(b"invalid_image_data")


def test_postprocessor_draw_detections(sample_image):
    """Test drawing detections on an image canvas."""
    bbox = BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0)
    det = Detection(class_id=0, class_name="person", confidence=0.89, bbox=bbox)
    
    annotated = Postprocessor.draw_detections(sample_image, [det])
    assert annotated.shape == sample_image.shape
    # Ensure drawing modified pixels
    assert not np.array_equal(annotated, sample_image)


def test_detector_predict_image(detector, sample_image):
    """Test full image prediction pipeline."""
    response = detector.predict_image(sample_image, conf=0.25, iou=0.45, annotate=True)
    
    assert isinstance(response, DetectionResponse)
    assert response.image_width == 640
    assert response.image_height == 480
    assert response.metrics.total_latency_ms > 0
    assert response.metrics.inference_time_ms > 0
    assert response.metrics.fps > 0
    assert response.annotated_image is not None
    assert isinstance(response.detections, list)


def test_detector_predict_video(detector, sample_image):
    """Test video frame generator inference."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_video_path = os.path.join(tmp_dir, "test_input.mp4")
        output_video_path = os.path.join(tmp_dir, "test_output.mp4")

        # Create a 5-frame synthetic video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(input_video_path, fourcc, 10.0, (640, 480))
        for _ in range(5):
            writer.write(sample_image)
        writer.release()

        # Run video generator
        progress_steps = list(detector.predict_video(
            source_path=input_video_path,
            output_path=output_video_path,
            conf=0.25
        ))

        assert len(progress_steps) == 5
        assert progress_steps[-1]["progress_percentage"] == 100.0
        assert os.path.exists(output_video_path)
        assert os.path.getsize(output_video_path) > 0


def test_detector_telemetry(detector):
    """Test cumulative telemetry reporting."""
    metrics = detector.get_metrics()
    assert "total_frames_processed" in metrics
    assert "average_inference_latency_ms" in metrics
    assert metrics["total_frames_processed"] > 0
