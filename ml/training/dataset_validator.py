"""Dataset validation pipeline for YOLO format datasets."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml
from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetValidationReport:
    """Detailed dataset validation report with error list and metrics."""
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    train_images_count: int = 0
    val_images_count: int = 0
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    total_annotations: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "train_images_count": self.train_images_count,
            "val_images_count": self.val_images_count,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "total_annotations": self.total_annotations,
            "class_distribution": self.class_distribution
        }


class DatasetValidator:
    """Validates YOLO-format datasets for structure, images, labels, and coordinate bounds."""

    @classmethod
    def validate_yaml_dataset(cls, data_yaml_path: str) -> DatasetValidationReport:
        """Validate dataset based on its data.yaml configuration."""
        report = DatasetValidationReport()
        yaml_file = Path(data_yaml_path)

        if not yaml_file.exists():
            report.errors.append(f"data.yaml does not exist at: {data_yaml_path}")
            return report

        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data_config = yaml.safe_load(f)
        except Exception as e:
            report.errors.append(f"Failed to parse YAML file: {e}")
            return report

        if not isinstance(data_config, dict):
            report.errors.append("Invalid data.yaml format; expected a YAML dictionary.")
            return report

        # Extract paths
        base_dir = yaml_file.parent
        train_path_raw = data_config.get("train")
        val_path_raw = data_config.get("val")

        if not train_path_raw:
            report.errors.append("Missing 'train' path in data.yaml.")
        if not val_path_raw:
            report.errors.append("Missing 'val' path in data.yaml.")

        # Extract classes
        names = data_config.get("names")
        if not names:
            report.errors.append("Missing 'names' field in data.yaml.")
            return report

        if isinstance(names, dict):
            class_names = [names[k] for k in sorted(names.keys())]
        elif isinstance(names, list):
            class_names = names
        else:
            report.errors.append("'names' field must be a list or dict of class names.")
            return report

        report.num_classes = len(class_names)
        report.class_names = class_names
        report.class_distribution = {name: 0 for name in class_names}

        # Resolve paths
        train_dir = cls._resolve_path(base_dir, train_path_raw)
        val_dir = cls._resolve_path(base_dir, val_path_raw)

        if not train_dir.exists():
            report.errors.append(f"Train directory does not exist: {train_dir}")
        else:
            cls._validate_split(train_dir, "train", report)

        if not val_dir.exists():
            report.errors.append(f"Validation directory does not exist: {val_dir}")
        else:
            cls._validate_split(val_dir, "val", report)

        # Final checks
        if report.train_images_count == 0:
            report.errors.append("Training set contains 0 valid images.")
        if report.val_images_count == 0:
            report.warnings.append("Validation set contains 0 images.")

        report.is_valid = len(report.errors) == 0
        return report

    @classmethod
    def _resolve_path(cls, base_dir: Path, target_path: str) -> Path:
        p = Path(target_path)
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()

    @classmethod
    def _validate_split(cls, split_dir: Path, split_name: str, report: DatasetValidationReport) -> None:
        """Validate images and corresponding YOLO label txt files in a split."""
        image_files: List[Path] = []
        for root, _, files in os.walk(split_dir):
            for file in files:
                if Path(file).suffix.lower() in VALID_IMAGE_EXTS:
                    image_files.append(Path(root) / file)

        if split_name == "train":
            report.train_images_count = len(image_files)
        else:
            report.val_images_count = len(image_files)

        for img_path in image_files:
            # Check image integrity
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                report.errors.append(f"Corrupt image file: {img_path}")
                continue

            # Look for matching label file
            label_path = cls._find_label_file(img_path)
            if not label_path or not label_path.exists():
                report.warnings.append(f"Missing label file for: {img_path.name}")
                continue

            # Check label coordinates & classes
            try:
                with open(label_path, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                    for line_idx, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            report.errors.append(
                                f"Malformed label in {label_path.name}:{line_idx+1} (expected 5 items, got {len(parts)})"
                            )
                            continue

                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        if cls_id < 0 or cls_id >= report.num_classes:
                            report.errors.append(
                                f"Class ID {cls_id} out of bounds in {label_path.name}:{line_idx+1}"
                            )

                        for val, name in [(x_center, "x_center"), (y_center, "y_center"), (width, "width"), (height, "height")]:
                            if not (0.0 <= val <= 1.0):
                                report.errors.append(
                                    f"Coordinate {name}={val} out of bounds [0, 1] in {label_path.name}:{line_idx+1}"
                                )

                        report.total_annotations += 1
                        if 0 <= cls_id < report.num_classes:
                            cls_name = report.class_names[cls_id]
                            report.class_distribution[cls_name] = report.class_distribution.get(cls_name, 0) + 1

            except Exception as e:
                report.errors.append(f"Error reading label {label_path.name}: {e}")

    @classmethod
    def _find_label_file(cls, img_path: Path) -> Optional[Path]:
        """Find matching .txt label file according to standard YOLO conventions."""
        # Check in same directory
        same_dir = img_path.with_suffix(".txt")
        if same_dir.exists():
            return same_dir

        # Check 'images' -> 'labels' directory sibling replacement
        parts = list(img_path.parts)
        for i in reversed(range(len(parts))):
            if parts[i] == "images":
                parts[i] = "labels"
                candidate = Path(*parts).with_suffix(".txt")
                if candidate.exists():
                    return candidate

        return same_dir
