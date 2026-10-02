"""Models and Registry API integration tests."""

from fastapi.testclient import TestClient
import pytest
from backend.app.main import app

client = TestClient(app)


def test_list_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0
    # yolov8n.pt should be registered
    names = [m["name"] for m in models]
    assert "yolov8n.pt" in names


def test_activate_model():
    response = client.post("/api/v1/models/activate", json={"model_name": "yolov8n.pt"})
    assert response.status_code == 200
    model = response.json()
    assert model["name"] == "yolov8n.pt"
    assert model["is_active"] is True
