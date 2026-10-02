"""WebSocket and REST endpoints for real-time live webcam detection."""

import json
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.app.schemas.detection import LiveDetectionResult, LiveFramePayload
from backend.app.services.webcam_service import WebcamService

logger = logging.getLogger("backend.api.websocket")
router = APIRouter()


@router.websocket("/ws/live")
async def websocket_live_detection(websocket: WebSocket):
    """Real-time bi-directional WebSocket connection for webcam frames.
    
    Accepts JSON messages:
    {
        "frame": "data:image/jpeg;base64,...",
        "model": "yolov8n.pt",
        "conf": 0.25,
        "iou": 0.45,
        "annotate": false
    }
    """
    await websocket.accept()
    logger.info("Webcam WebSocket client connected.")

    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                data = json.loads(text_data)
                frame_b64 = data.get("frame", "")
                if not frame_b64:
                    await websocket.send_json({"error": "Empty frame payload"})
                    continue

                model_name = data.get("model")
                conf = float(data.get("conf", 0.50))
                iou = float(data.get("iou", 0.45))
                annotate = bool(data.get("annotate", False))
                classes_data = data.get("classes")
                allowed_cls = [c.strip().lower() for c in classes_data.split(",") if c.strip()] if isinstance(classes_data, str) and classes_data else None

                result = WebcamService.process_webcam_frame(
                    frame_base64=frame_b64,
                    model_name=model_name,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    return_annotated_frame=annotate,
                    allowed_classes=allowed_cls
                )

                await websocket.send_json(result)

            except Exception as e:
                logger.error(f"Error processing live WebSocket frame: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("Webcam WebSocket client disconnected.")


@router.post("/live-frame", response_model=LiveDetectionResult, summary="Process Single Live Camera Frame")
def process_live_frame(payload: LiveFramePayload):
    """HTTP REST fallback endpoint for single live webcam frame processing."""
    result = WebcamService.process_webcam_frame(
        frame_base64=payload.frame_base64,
        model_name=payload.model_name,
        conf_threshold=payload.conf_threshold or 0.25,
        iou_threshold=payload.iou_threshold or 0.45,
        return_annotated_frame=True
    )
    return LiveDetectionResult(
        detections=result["detections"],
        object_count=result["object_count"],
        fps=result["fps"],
        inference_latency_ms=result["inference_latency_ms"],
        annotated_frame_base64=result["annotated_frame_base64"]
    )
