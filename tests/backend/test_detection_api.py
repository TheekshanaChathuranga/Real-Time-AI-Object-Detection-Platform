"""Detection API endpoints integration tests."""

import io
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
from backend.app.main import app

client = TestClient(app)


@pytest.fixture
def encoded_test_image():
    """Create a 320x240 image with a white shape on dark background, encoded as JPEG bytes."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    ret, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_image_detection_endpoint(encoded_test_image):
    files = {"file": ("test.jpg", io.BytesIO(encoded_test_image), "image/jpeg")}
    data = {"conf_threshold": "0.25", "iou_threshold": "0.45"}

    response = client.post("/api/v1/detection/image", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert "session_id" in res_data
    assert res_data["image_width"] == 320
    assert res_data["image_height"] == 240
    assert "metrics" in res_data
    assert res_data["metrics"]["fps"] > 0
    assert "detections" in res_data
    assert "annotated_image_base64" in res_data

    # Test retrieving session info
    session_id = res_data["session_id"]
    sess_res = client.get(f"/api/v1/detection/{session_id}")
    assert sess_res.status_code == 200
    sess_data = sess_res.json()
    assert sess_data["session_id"] == session_id


def test_invalid_image_upload():
    files = {"file": ("malicious.exe", io.BytesIO(b"fake_binary_payload"), "application/octet-stream")}
    response = client.post("/api/v1/detection/image", files=files)
    assert response.status_code in [400, 422]
