"""FastAPI application entrypoint, middleware, lifespan, and error handling."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.api import api_router
from backend.app.config import settings
from backend.app.core.exceptions import PlatformException
from backend.app.core.logging import setup_logging
from backend.app.db.session import SessionLocal, init_db
from backend.app.services.model_service import ModelService

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for database initialization and model synchronization."""
    logger.info("Starting up Real-Time AI Object Detection Platform...")
    init_db()

    # Sync default models into DB
    try:
        with SessionLocal() as db:
            ModelService.sync_models_with_db(db)
        logger.info("Model registry synchronized with database.")
    except Exception as e:
        logger.error(f"Error synchronizing model registry during startup: {e}")

    yield

    logger.info("Shutting down Real-Time AI Object Detection Platform...")


app = FastAPI(
    title="Real-Time AI Object Detection Platform API",
    description="Production-grade, modular, scalable AI Object Detection Platform using YOLO.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(PlatformException)
async def platform_exception_handler(request: Request, exc: PlatformException):
    logger.warning(f"PlatformException on {request.method} {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "details": exc.details}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An internal server error occurred. Please check server logs."}
    )


# Register API v1 Router
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "title": "Real-Time AI Object Detection Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
