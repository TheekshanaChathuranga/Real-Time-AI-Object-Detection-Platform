"""Image and video detection endpoints."""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.repository import InferenceRepository
from backend.app.schemas.detection import ImageDetectionResponse, VideoDetectionResponse
from backend.app.services.detection_service import DetectionService
from backend.app.services.video_service import VideoService

router = APIRouter()


@router.post(
    "/image",
    response_model=ImageDetectionResponse,
    summary="Run Object Detection on Image"
)
def detect_image(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP)"),
    model_name: Optional[str] = Form(None, description="Model to use (defaults to active model)"),
    conf_threshold: Optional[float] = Form(None, description="Confidence threshold [0.0 - 1.0]"),
    iou_threshold: Optional[float] = Form(None, description="IoU threshold [0.0 - 1.0]"),
    classes: Optional[str] = Form(None, description="Comma-separated class filter, e.g. car,truck,bus"),
    db: Session = Depends(get_db)
):
    """Run full object detection pipeline on uploaded image with bounding boxes and performance telemetry."""
    return DetectionService.detect_image(
        db=db,
        file=file,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes
    )


@router.post(
    "/video",
    response_model=VideoDetectionResponse,
    summary="Run Object Detection on Video"
)
def detect_video(
    file: UploadFile = File(..., description="Target video file (MP4, AVI, MOV)"),
    model_name: Optional[str] = Form(None, description="Model to use"),
    conf_threshold: Optional[float] = Form(None, description="Confidence threshold"),
    iou_threshold: Optional[float] = Form(None, description="IoU threshold"),
    classes: Optional[str] = Form(None, description="Comma-separated class filter, e.g. car,truck,bus"),
    db: Session = Depends(get_db)
):
    """Run streaming frame-by-frame detection on uploaded video and generate annotated video file."""
    return VideoService.process_video(
        db=db,
        file=file,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes
    )


@router.get("/{session_id}", summary="Get Detection Session Metadata")
def get_session_info(session_id: str, db: Session = Depends(get_db)):
    """Retrieve details and detected items for a specific inference session."""
    session = InferenceRepository.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference session '{session_id}' not found."
        )

    detections = [
        {
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}
        }
        for d in session.detections
    ]

    return {
        "session_id": session.session_id,
        "session_type": session.session_type,
        "model_name": session.model_name,
        "duration_ms": session.duration_ms,
        "object_count": session.object_count,
        "fps": session.fps,
        "latency_ms": session.latency_ms,
        "status": session.status,
        "detections": detections
    }


@router.get("/results/{session_id}", summary="Download Annotated Image Result")
def get_annotated_image(session_id: str, db: Session = Depends(get_db)):
    """Serve the annotated output image for a given session."""
    session = InferenceRepository.get_session(db, session_id)
    if not session or not session.result_file or not os.path.exists(session.result_file):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated image not found.")
    return FileResponse(session.result_file, media_type="image/jpeg")


@router.get("/videos/{session_id}", summary="Download Processed Video Result")
def get_processed_video(session_id: str, db: Session = Depends(get_db)):
    """Serve the processed annotated video for a given session."""
    session = InferenceRepository.get_session(db, session_id)
    if not session or not session.result_file or not os.path.exists(session.result_file):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video not found.")
    return FileResponse(session.result_file, media_type="video/mp4")
