"""Model configurations, hardware detection, and default parameters for ML engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import torch


def detect_device(preferred_device: str = "auto") -> str:
    """Detect available compute hardware.
    
    Supports 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.
    Falls back gracefully to 'cpu' if requested device is unavailable.
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    if preferred_device.startswith("cuda"):
        if torch.cuda.is_available():
            return preferred_device
        return "cpu"
        
    if preferred_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
        
    return "cpu"


@dataclass
class ModelInfo:
    """Metadata describing a supported or registered model."""
    name: str
    filename: str
    task: str = "detect"
    description: str = ""
    is_custom: bool = False
    file_path: Optional[str] = None
    default_conf: float = 0.25
    default_iou: float = 0.45
    input_size: int = 640
    supported_classes: List[str] = field(default_factory=list)


# Standard COCO 80 classes for standard YOLO models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

PRETRAINED_MODELS: Dict[str, ModelInfo] = {
    "yolov8n.pt": ModelInfo(
        name="YOLOv8 Nano",
        filename="yolov8n.pt",
        description="Fastest, lightweight model ideal for CPU and real-time edge streaming.",
        is_custom=False,
        default_conf=0.25,
        default_iou=0.45,
        input_size=640,
        supported_classes=COCO_CLASSES
    ),
    "yolov8s.pt": ModelInfo(
        name="YOLOv8 Small",
        filename="yolov8s.pt",
        description="Balanced speed and accuracy for real-time video surveillance.",
        is_custom=False,
        default_conf=0.25,
        default_iou=0.45,
        input_size=640,
        supported_classes=COCO_CLASSES
    ),
    "yolov8m.pt": ModelInfo(
        name="YOLOv8 Medium",
        filename="yolov8m.pt",
        description="Higher precision detector suitable for higher powered server GPUs.",
        is_custom=False,
        default_conf=0.25,
        default_iou=0.45,
        input_size=640,
        supported_classes=COCO_CLASSES
    )
}
