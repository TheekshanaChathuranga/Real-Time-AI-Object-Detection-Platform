"""Health check endpoint."""

from datetime import datetime
from fastapi import APIRouter
from backend.app.config import settings
from ml.configs.model_configs import detect_device

router = APIRouter()


@router.get("/health", summary="Platform Health Check")
def health_check():
    """Returns platform operational status, compute device, and default model."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT,
        "compute_device": detect_device(settings.DEVICE),
        "default_model": settings.DEFAULT_MODEL,
        "version": "1.0.0"
    }
