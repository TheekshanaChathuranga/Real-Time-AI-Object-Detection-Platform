"""Health and Root API integration tests."""

from fastapi.testclient import TestClient
import pytest
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert data["docs"] == "/docs"


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "compute_device" in data
    assert "default_model" in data
