"""Tests for DatasetValidator."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import yaml

from ml.training.dataset_validator import DatasetValidator


@pytest.fixture
def dummy_yolo_dataset():
    """Create a temporary valid YOLO dataset with images and labels."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_img_dir = os.path.join(tmp_dir, "images", "train")
        val_img_dir = os.path.join(tmp_dir, "images", "val")
        train_lbl_dir = os.path.join(tmp_dir, "labels", "train")
        val_lbl_dir = os.path.join(tmp_dir, "labels", "val")

        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Write train image + label
        cv2.imwrite(os.path.join(train_img_dir, "img1.jpg"), dummy_img)
        with open(os.path.join(train_lbl_dir, "img1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Write val image + label
        cv2.imwrite(os.path.join(val_img_dir, "val1.jpg"), dummy_img)
        with open(os.path.join(val_lbl_dir, "val1.txt"), "w") as f:
            f.write("0 0.4 0.4 0.3 0.3\n")

        # Write data.yaml
        yaml_content = {
            "path": tmp_dir,
            "train": "images/train",
            "val": "images/val",
            "names": {0: "object"}
        }
        yaml_path = os.path.join(tmp_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        yield yaml_path, tmp_dir


def test_valid_dataset_validation(dummy_yolo_dataset):
    """Test validation of well-formed dataset."""
    yaml_path, _ = dummy_yolo_dataset
    report = DatasetValidator.validate_yaml_dataset(yaml_path)

    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.train_images_count == 1
    assert report.val_images_count == 1
    assert report.num_classes == 1
    assert report.total_annotations == 2


def test_invalid_coordinates_detected(dummy_yolo_dataset):
    """Test that out-of-bounds coordinates (> 1.0) trigger validation error."""
    yaml_path, tmp_dir = dummy_yolo_dataset
    train_lbl_dir = os.path.join(tmp_dir, "labels", "train")

    # Overwrite label with out-of-bounds coordinate (1.5)
    with open(os.path.join(train_lbl_dir, "img1.txt"), "w") as f:
        f.write("0 1.5 0.5 0.2 0.2\n")

    report = DatasetValidator.validate_yaml_dataset(yaml_path)
    assert report.is_valid is False
    assert any("out of bounds" in err for err in report.errors)
