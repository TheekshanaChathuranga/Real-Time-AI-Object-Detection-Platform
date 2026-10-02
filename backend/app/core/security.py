"""Security helpers for safe file uploads, safe paths, and input validation."""

import os
from pathlib import Path
import re
from typing import Set

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_MODEL_EXTENSIONS: Set[str] = {".pt", ".onnx", ".torchscript"}


def sanitize_filename(filename: str) -> str:
    """Sanitize user-submitted filename to prevent path injection."""
    base = os.path.basename(filename)
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return clean or "unnamed_file"


def validate_file_extension(filename: str, allowed_extensions: Set[str]) -> bool:
    """Check if file has an approved extension."""
    suffix = Path(filename).suffix.lower()
    return suffix in allowed_extensions


def validate_file_size(size_bytes: int, max_mb: int = 100) -> bool:
    """Ensure file size does not exceed configured ceiling."""
    return size_bytes <= (max_mb * 1024 * 1024)
