"""Storage abstraction for saving, reading, and managing images, videos, datasets, and weights."""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import shutil
from typing import BinaryIO, Optional, Union
import uuid


class StorageManager(ABC):
    """Abstract interface for local and cloud object storage (S3 / MinIO)."""

    @abstractmethod
    def save_file(self, file_data: Union[bytes, BinaryIO], destination_subpath: str) -> str:
        """Save raw bytes or file-like object and return relative or absolute stored path."""
        pass

    @abstractmethod
    def get_file_path(self, relative_path: str) -> Optional[str]:
        """Return accessible file path if existing."""
        pass

    @abstractmethod
    def delete_file(self, relative_path: str) -> bool:
        """Delete stored file."""
        pass


class LocalStorageManager(StorageManager):
    """Local filesystem storage implementation with traversal security and directory isolation."""

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, destination_subpath: str) -> Path:
        """Prevent directory traversal attacks."""
        clean_subpath = Path(destination_subpath).as_posix().lstrip("/\\")
        dest = (self.base_dir / clean_subpath).resolve()
        if not str(dest).startswith(str(self.base_dir)):
            raise ValueError("Path traversal attempt detected.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def save_file(self, file_data: Union[bytes, BinaryIO], destination_subpath: str) -> str:
        target_path = self._resolve_safe_path(destination_subpath)
        if isinstance(file_data, bytes):
            with open(target_path, "wb") as f:
                f.write(file_data)
        else:
            with open(target_path, "wb") as f:
                shutil.copyfileobj(file_data, f)
        return str(target_path)

    def get_file_path(self, relative_path: str) -> Optional[str]:
        try:
            target_path = self._resolve_safe_path(relative_path)
            return str(target_path) if target_path.exists() else None
        except ValueError:
            return None

    def delete_file(self, relative_path: str) -> bool:
        try:
            target_path = self._resolve_safe_path(relative_path)
            if target_path.exists():
                os.remove(target_path)
                return True
            return False
        except (ValueError, OSError):
            return False
