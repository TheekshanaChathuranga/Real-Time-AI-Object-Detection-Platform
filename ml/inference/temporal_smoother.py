"""Temporal smoothing and anti-jitter tracking for video and streaming detection.

Solves inter-frame class flickering/toggling (e.g. truck <-> bus) and stabilizes
bounding boxes across consecutive frames.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ml.inference.postprocessor import BoundingBox, Detection


def compute_box_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1.x2 - box1.x1) * (box1.y2 - box1.y1))
    area2 = max(0.0, (box2.x2 - box2.x1) * (box2.y2 - box2.y1))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


# Confusing vehicle class pairs prone to label jitter
CONFUSING_PAIRS = {
    ("truck", "bus"),
    ("bus", "truck"),
    ("car", "truck"),
    ("truck", "car"),
}


@dataclass
class TrackedEntity:
    """Represents a tracked object across time frames with class voting history."""
    track_id: int
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    missing_frames: int = 0
    total_frames_seen: int = 1

    def update(self, det: Detection) -> None:
        """Update track with new frame detection and smooth the classification."""
        self.history.append((det.class_id, det.class_name, det.confidence))
        self.missing_frames = 0
        self.total_frames_seen += 1

        # Smooth bounding box (exponential moving average for stable edges)
        alpha = 0.80  # Weight for new frame
        smoothed_bbox = BoundingBox(
            x1=alpha * det.bbox.x1 + (1.0 - alpha) * self.bbox.x1,
            y1=alpha * det.bbox.y1 + (1.0 - alpha) * self.bbox.y1,
            x2=alpha * det.bbox.x2 + (1.0 - alpha) * self.bbox.x2,
            y2=alpha * det.bbox.y2 + (1.0 - alpha) * self.bbox.y2,
        )
        self.bbox = smoothed_bbox

        # Class stabilization via confidence-weighted voting over history window
        class_weights: Dict[str, float] = {}
        class_ids: Dict[str, int] = {}
        for cid, cname, conf in self.history:
            class_weights[cname] = class_weights.get(cname, 0.0) + conf
            class_ids[cname] = cid

        dominant_class = max(class_weights.items(), key=lambda item: item[1])[0]

        # Prevent toggling between bus and truck if track was already established as one
        pair = (self.class_name, det.class_name)
        if pair in CONFUSING_PAIRS and self.total_frames_seen >= 3:
            # Require strong confidence (> 0.65) and sustained history to flip an established vehicle
            if det.confidence < 0.65 and dominant_class != det.class_name:
                det.class_name = self.class_name
                det.class_id = self.class_id
            else:
                self.class_name = dominant_class
                self.class_id = class_ids[dominant_class]
        else:
            self.class_name = dominant_class
            self.class_id = class_ids[dominant_class]

        self.confidence = det.confidence


class TemporalSmoother:
    """Anti-jitter and temporal consistency filter across consecutive video frames."""

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_missing_frames: int = 5,
        history_window: int = 8
    ):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.history_window = history_window
        self.tracks: Dict[int, TrackedEntity] = {}
        self.next_track_id: int = 1

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracks with current frame detections and return stabilized detections."""
        if not detections and not self.tracks:
            return []

        matched_tracks = set()
        matched_detections = set()
        matches: List[Tuple[float, int, int]] = []

        # Compute IoU between all active tracks and new detections
        for t_id, track in self.tracks.items():
            for d_idx, det in enumerate(detections):
                iou = compute_box_iou(track.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    matches.append((iou, t_id, d_idx))

        # Greedy match from highest IoU
        matches.sort(key=lambda x: x[0], reverse=True)
        for _, t_id, d_idx in matches:
            if t_id not in matched_tracks and d_idx not in matched_detections:
                matched_tracks.add(t_id)
                matched_detections.add(d_idx)
                self.tracks[t_id].update(detections[d_idx])

                # Update the detection with stabilized values
                detections[d_idx].class_id = self.tracks[t_id].class_id
                detections[d_idx].class_name = self.tracks[t_id].class_name
                detections[d_idx].bbox = self.tracks[t_id].bbox

        # Age unmatched existing tracks
        for t_id in list(self.tracks.keys()):
            if t_id not in matched_tracks:
                self.tracks[t_id].missing_frames += 1
                if self.tracks[t_id].missing_frames > self.max_missing_frames:
                    del self.tracks[t_id]

        # Register new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = TrackedEntity(
                    track_id=self.next_track_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence
                )
                new_track.history.append((det.class_id, det.class_name, det.confidence))
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        return detections

    def reset(self) -> None:
        """Reset all active tracks."""
        self.tracks.clear()
        self.next_track_id = 1
