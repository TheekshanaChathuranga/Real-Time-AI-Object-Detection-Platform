"""Analytics service for aggregating system and detection telemetry."""

from typing import Any, Dict, List
import psutil
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db.models import CameraRecord, InferenceSessionRecord, ModelRecord
from backend.app.db.repository import InferenceRepository, ModelRepository
from backend.app.schemas.analytics import DashboardStatsResponse, RecentSessionSummary
from ml.configs.model_configs import detect_device


class AnalyticsService:
    @staticmethod
    def get_dashboard_stats(db: Session) -> DashboardStatsResponse:
        summary = InferenceRepository.get_summary_metrics(db)
        active_cams = db.query(CameraRecord).filter(CameraRecord.status == "RUNNING").count()
        avail_models = db.query(ModelRecord).count()
        recent = InferenceRepository.list_recent_sessions(db, limit=10)

        recent_summaries = [
            RecentSessionSummary(
                session_id=s.session_id,
                session_type=s.session_type,
                model_name=s.model_name,
                start_time=s.start_time,
                duration_ms=s.duration_ms,
                object_count=s.object_count,
                fps=s.fps,
                latency_ms=s.latency_ms,
                status=s.status
            )
            for s in recent
        ]

        return DashboardStatsResponse(
            total_sessions=summary["total_sessions"],
            total_detected_objects=summary["total_detected_objects"],
            active_cameras=active_cams,
            available_models=avail_models,
            average_fps=summary["average_fps"],
            average_latency_ms=summary["average_latency_ms"],
            hardware_device=detect_device(settings.DEVICE),
            recent_sessions=recent_summaries
        )

    @staticmethod
    def get_analytics_breakdown(db: Session) -> Dict[str, Any]:
        summary = InferenceRepository.get_summary_metrics(db)
        
        # Session type distribution
        types = ["image", "video", "webcam", "rtsp"]
        type_dist = {
            t: db.query(InferenceSessionRecord).filter(InferenceSessionRecord.session_type == t).count()
            for t in types
        }

        # Hardware metrics
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        return {
            "total_sessions": summary["total_sessions"],
            "total_detected_objects": summary["total_detected_objects"],
            "average_fps": summary["average_fps"],
            "average_latency_ms": summary["average_latency_ms"],
            "class_distribution": summary["class_distribution"],
            "session_type_distribution": type_dist,
            "system_hardware": {
                "cpu_percent": cpu,
                "memory_percent": mem,
                "compute_device": detect_device(settings.DEVICE)
            }
        }
