"""Model Evaluation engine for computing Precision, Recall, F1, mAP, and per-class metrics."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from ultralytics import YOLO

from ml.configs.model_configs import detect_device

logger = logging.getLogger("ml.evaluation.evaluator")


@dataclass
class ClassMetric:
    """Evaluation metrics for an individual class."""
    class_name: str
    class_id: int
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4)
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a trained YOLO model."""
    model_name: str
    dataset_yaml: str
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    inference_speed_ms: float = 0.0
    per_class_metrics: List[ClassMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_yaml": self.dataset_yaml,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "inference_speed_ms": round(self.inference_speed_ms, 2),
            "per_class_metrics": [c.to_dict() for c in self.per_class_metrics]
        }


class ModelEvaluator:
    """Evaluates YOLO models on validation datasets and computes standard COCO benchmark metrics."""

    @classmethod
    def evaluate(
        cls,
        model_path: str,
        data_yaml: str,
        device: str = "auto",
        batch_size: int = 16,
        image_size: int = 640
    ) -> EvaluationReport:
        """Run validation on a model and return an EvaluationReport."""
        dev = detect_device(device)
        logger.info(f"Evaluating {model_path} on {data_yaml} with device {dev}")

        model = YOLO(model_path)
        metrics = model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=image_size,
            device=dev,
            verbose=False
        )

        p = float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0
        r = float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0
        map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
        map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        # Speeds: preprocess, inference, loss, postprocess
        speed_ms = 0.0
        if hasattr(metrics, "speed") and isinstance(metrics.speed, dict):
            speed_ms = float(metrics.speed.get("inference", 0.0))

        # Per-class breakdown
        per_class_list: List[ClassMetric] = []
        names = metrics.names if hasattr(metrics, "names") else {}

        if hasattr(metrics.box, "p") and hasattr(metrics.box, "r"):
            box_p = metrics.box.p
            box_r = metrics.box.r
            box_map50 = getattr(metrics.box, "all_ap", None)
            
            for idx, name in names.items():
                cp = float(box_p[idx]) if idx < len(box_p) else p
                cr = float(box_r[idx]) if idx < len(box_r) else r
                cf1 = 2 * (cp * cr) / (cp + cr) if (cp + cr) > 0 else 0.0
                cmap50 = map50
                if box_map50 is not None and len(box_map50) > idx and len(box_map50[idx]) > 0:
                    cmap50 = float(box_map50[idx][0])

                per_class_list.append(ClassMetric(
                    class_name=name,
                    class_id=idx,
                    precision=cp,
                    recall=cr,
                    f1_score=cf1,
                    map50=cmap50,
                    map50_95=map50_95
                ))

        return EvaluationReport(
            model_name=Path(model_path).name,
            dataset_yaml=data_yaml,
            precision=p,
            recall=r,
            f1_score=f1,
            map50=map50,
            map50_95=map50_95,
            inference_speed_ms=speed_ms,
            per_class_metrics=per_class_list
        )
