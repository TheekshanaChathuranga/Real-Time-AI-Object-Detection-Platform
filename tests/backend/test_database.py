"""Database and Repository unit tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.db.models import CameraRecord, ModelRecord
from backend.app.db.repository import CameraRepository, InferenceRepository, ModelRepository

# In-memory SQLite for fast testing
TEST_ENGINE = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=TEST_ENGINE)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=TEST_ENGINE)


def test_model_repository_operations(db_session):
    # Create model
    model = ModelRepository.create_or_update(
        db=db_session,
        name="test_model.pt",
        description="Unit test model",
        is_active=False
    )
    assert model.name == "test_model.pt"

    # Add version
    version = ModelRepository.add_version(
        db=db_session,
        model_name="test_model.pt",
        version="v1.0",
        file_path="test_model.pt",
        map50=0.72
    )
    assert version.map50 == 0.72

    # Set active
    active = ModelRepository.set_active(db_session, "test_model.pt")
    assert active.is_active is True


def test_camera_repository_operations(db_session):
    cam = CameraRepository.create(
        db=db_session,
        camera_id="cam-123",
        name="Entrance Camera",
        rtsp_url="rtsp://localhost:554/live"
    )
    assert cam.camera_id == "cam-123"
    assert cam.status == "STOPPED"

    updated = CameraRepository.update_status(db_session, "cam-123", "RUNNING")
    assert updated.status == "RUNNING"
