"""Application configuration and environment settings."""

from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic BaseSettings class for unified environment configuration."""

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000"
    ]

    # Database
    DATABASE_URL: str = "sqlite:///./yolo_platform.db"

    # Storage paths
    STORAGE_BACKEND: str = "local"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIRECTORY: str = "./data/uploads"
    RESULT_DIRECTORY: str = "./data/results"
    DATASET_DIRECTORY: str = "./data/datasets"
    MODEL_DIRECTORY: str = "./models/weights"
    MAX_UPLOAD_SIZE_MB: int = 100

    # ML Inference
    DEFAULT_MODEL: str = "yolov8n.pt"
    DEVICE: str = "auto"
    DEFAULT_CONF_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.45

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


settings = Settings()
