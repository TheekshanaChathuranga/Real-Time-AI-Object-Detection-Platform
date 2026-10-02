"""Database engine, session management, and connection initialization."""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.config import settings
from backend.app.db.base import Base

logger = logging.getLogger("backend.db.session")

connect_args = {}
database_url = settings.DATABASE_URL

if database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    # Test connection
    with engine.connect() as conn:
        logger.info(f"Database connection established successfully: {engine.url.drivername}")
except Exception as e:
    logger.warning(
        f"Failed to connect to configured DATABASE_URL ({database_url}): {e}. "
        "Falling back to local SQLite database: sqlite:///./yolo_platform.db"
    )
    fallback_url = "sqlite:///./yolo_platform.db"
    engine = create_engine(fallback_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables if they do not already exist."""
    import backend.app.db.models  # noqa: F401 - ensure models are registered on Base.metadata
    Base.metadata.create_all(bind=engine)
    logger.info("Database schemas initialized.")


# Auto-initialize tables upon module load
init_db()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
