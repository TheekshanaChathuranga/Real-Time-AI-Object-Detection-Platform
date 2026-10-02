"""RTSP camera management and streaming endpoints."""

from typing import List
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.repository import CameraRepository
from backend.app.schemas.camera import CameraCreate, CameraResponse
from backend.app.services.stream_service import StreamService

router = APIRouter()


@router.get("", response_model=List[CameraResponse], summary="List Configured Cameras")
def list_cameras(db: Session = Depends(get_db)):
    """Retrieve all configured RTSP / IP cameras and their current statuses."""
    return CameraRepository.list_all(db)


@router.post("", response_model=CameraResponse, summary="Register New RTSP Camera")
def create_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    """Add a new RTSP camera stream configuration to the system."""
    camera_id = f"cam-{uuid.uuid4().hex[:6]}"
    record = CameraRepository.create(
        db=db,
        camera_id=camera_id,
        name=camera_in.name,
        rtsp_url=camera_in.rtsp_url,
        resolution=camera_in.resolution,
        target_fps=camera_in.target_fps,
        model_name=camera_in.model_name
    )
    return record


@router.post("/{camera_id}/start", summary="Start Camera Stream")
def start_camera(camera_id: str, db: Session = Depends(get_db)):
    """Start isolated background capture and detection pipeline for camera."""
    cam = CameraRepository.get_by_id(db, camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")

    status_info = StreamService.start_camera_stream(
        camera_id=cam.camera_id,
        rtsp_url=cam.rtsp_url,
        name=cam.name,
        model_name=cam.model_name,
        db=db
    )
    return status_info


@router.post("/{camera_id}/stop", summary="Stop Camera Stream")
def stop_camera(camera_id: str, db: Session = Depends(get_db)):
    """Gracefully stop background stream."""
    return StreamService.stop_camera_stream(camera_id, db)


@router.get("/{camera_id}/status", summary="Get Camera Stream Telemetry")
def get_camera_status(camera_id: str):
    """Retrieve real-time telemetry (FPS, reconnect count, uptime)."""
    return StreamService.get_stream_status(camera_id)


@router.get("/{camera_id}/preview", summary="Live MJPEG Stream Preview")
def preview_camera(camera_id: str):
    """Multipart MJPEG video stream for direct browser rendering."""
    return StreamingResponse(
        StreamService.generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.delete("/{camera_id}", summary="Delete Camera Configuration")
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    """Stop stream if running and remove camera configuration from database."""
    StreamService.stop_camera_stream(camera_id, db)
    success = CameraRepository.delete(db, camera_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    return {"message": f"Camera {camera_id} successfully deleted."}
