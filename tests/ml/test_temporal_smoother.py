"""Unit tests for TemporalSmoother and postprocessing false-alarm filters."""

import pytest
from ml.inference.postprocessor import BoundingBox, Detection, Postprocessor
from ml.inference.temporal_smoother import TemporalSmoother


def test_train_false_positive_suppression():
    """Verify that train false positives on overhead structures are suppressed."""
    # Mock result with a train detection at 0.46 confidence (like the user's gantry)
    class MockBoxes:
        def __init__(self):
            import numpy as np
            self.xyxy = np.array([[800.0, 50.0, 1100.0, 120.0]])
            self.conf = np.array([0.46])
            self.cls = np.array([6])  # train

    class MockResult:
        def __init__(self):
            self.boxes = MockBoxes()
            self.names = {6: "train"}
            self.orig_shape = (720, 1280)

    res = MockResult()
    detections = Postprocessor.parse_ultralytics_result(res)
    # Must be filtered out because confidence 0.46 < 0.60 and overhead aspect ratio
    assert len(detections) == 0


def test_temporal_smoother_prevents_truck_bus_toggling():
    """Verify that TemporalSmoother stabilizes an established truck from toggling into a bus."""
    smoother = TemporalSmoother()

    box = BoundingBox(x1=100.0, y1=200.0, x2=250.0, y2=400.0)

    # Frame 1: Truck 0.62
    det1 = [Detection(class_id=7, class_name="truck", confidence=0.62, bbox=box)]
    out1 = smoother.update(det1)
    assert out1[0].class_name == "truck"

    # Frame 2: Truck 0.60
    det2 = [Detection(class_id=7, class_name="truck", confidence=0.60, bbox=box)]
    out2 = smoother.update(det2)
    assert out2[0].class_name == "truck"

    # Frame 3: Model fluctuates to bus with weak 0.41 confidence
    det3 = [Detection(class_id=5, class_name="bus", confidence=0.41, bbox=box)]
    out3 = smoother.update(det3)
    # The smoother must suppress the jitter and preserve 'truck'!
    assert out3[0].class_name == "truck"
