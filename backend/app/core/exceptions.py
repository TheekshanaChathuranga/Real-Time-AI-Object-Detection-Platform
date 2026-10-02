"""Custom exception classes and centralized error definitions."""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class PlatformException(Exception):
    """Base exception for platform domain errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class ModelNotFoundError(PlatformException):
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' was not found in registry or disk.",
            status_code=status.HTTP_404_NOT_FOUND
        )


class ModelLoadingError(PlatformException):
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class InvalidFileError(PlatformException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid file payload: {reason}",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class CameraStreamError(PlatformException):
    def __init__(self, camera_id: str, reason: str):
        super().__init__(
            message=f"RTSP camera '{camera_id}' error: {reason}",
            status_code=status.HTTP_502_BAD_GATEWAY
        )


class DatasetValidationError(PlatformException):
    def __init__(self, errors: list):
        super().__init__(
            message="Dataset validation failed.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=errors
        )
