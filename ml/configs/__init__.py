"""ML configuration exports."""

from ml.configs.model_configs import (
    ModelInfo,
    PRETRAINED_MODELS,
    COCO_CLASSES,
    detect_device
)

__all__ = ["ModelInfo", "PRETRAINED_MODELS", "COCO_CLASSES", "detect_device"]
