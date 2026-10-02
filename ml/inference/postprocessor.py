"""Detection post-processing, bounding box parsing, and image annotation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2)
        }


@dataclass
class Detection:
    """Standardized detection result entity for YOLO and future CV architectures."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict()
        }


class Postprocessor:
    """Parses raw detector predictions into standardized detection outputs and draws annotations."""

    # Curated modern HSL-based distinct color palette for up to 80 classes
    @staticmethod
    def get_class_color(class_id: int) -> Tuple[int, int, int]:
        """Generate a consistent, aesthetically pleasing BGR color for any class index."""
        # Golden ratio hue spacing
        golden_ratio = 0.618033988749895
        h = int(((class_id * golden_ratio) % 1.0) * 180)
        s = 200
        v = 240
        hsv_pixel = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # Minimum confidence floors for classes that commonly trigger false alarms on road infrastructure
    CLASS_CONF_MINIMUMS: Dict[str, float] = {
        "train": 0.60,
        "boat": 0.60,
        "airplane": 0.60,
    }

    @classmethod
    def parse_ultralytics_result(
        cls,
        result: Any,
        names_dict: Optional[Dict[int, str]] = None,
        allowed_classes: Optional[List[str]] = None
    ) -> List[Detection]:
        """Convert an Ultralytics YOLO Results object into standardized Detection list."""
        detections: List[Detection] = []
        if result is None or result.boxes is None:
            return detections

        names = names_dict or getattr(result, "names", {})

        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
        confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else result.boxes.conf
        class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, "cpu") else result.boxes.cls

        orig_shape = getattr(result, "orig_shape", None)
        img_h = orig_shape[0] if orig_shape else 1080.0

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            c_id = int(cls_id)
            c_name = names.get(c_id, f"class_{c_id}")
            c_conf = float(conf)
            c_name_lower = c_name.lower()

            # 1. Enforce class-specific confidence floor (filters out weak false alarms like 'train 0.46')
            min_conf = cls.CLASS_CONF_MINIMUMS.get(c_name_lower, 0.0)
            if c_conf < min_conf:
                continue

            # 2. Filter out overhead roadside structures / gantries falsely detected as 'train'
            if c_name_lower == "train":
                box_w = float(box[2] - box[0])
                box_h = float(box[3] - box[1])
                aspect_ratio = box_w / max(1.0, box_h)
                # If high in the frame (upper 45%), wide aspect ratio (> 2.5), and not ultra-high confidence, reject
                if float(box[1]) < 0.45 * img_h and aspect_ratio > 2.2 and c_conf < 0.75:
                    continue

            # 3. Allowed classes filter if specified
            if allowed_classes is not None:
                allowed_lower = [str(ac).lower() for ac in allowed_classes]
                if c_name_lower not in allowed_lower and str(c_id) not in allowed_lower:
                    continue

            bbox = BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3])
            )
            detections.append(Detection(
                class_id=c_id,
                class_name=c_name,
                confidence=c_conf,
                bbox=bbox
            ))

        return detections

    @classmethod
    def draw_detections(
        cls,
        image: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2,
        font_scale: float = 0.5,
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and class labels onto an image copy with high aesthetic finish."""
        annotated = image.copy()

        for det in detections:
            color = cls.get_class_color(det.class_id)
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

            if draw_labels:
                label = f"{det.class_name} {det.confidence:.2f}"
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                
                # Ensure label box stays within image bounds
                label_y1 = max(0, y1 - h - baseline - 4)
                label_y2 = y1
                label_x2 = min(annotated.shape[1], x1 + w + 8)

                # Draw solid label background badge
                cv2.rectangle(
                    annotated,
                    (x1, label_y1),
                    (label_x2, label_y2),
                    color,
                    -1
                )
                # Draw contrasting text (white or black depending on color luminance)
                luminance = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
                text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA
                )

        return annotated
