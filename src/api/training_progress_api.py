"""
Training Progress Monitoring API Endpoints
Provides REST API endpoints for training progress monitoring, notifications, and version management
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio
import json

# Import training progress components
from ..models.training_progress_monitor import (
    TrainingProgressMonitor, ProgressUpdate, TrainingNotification, 
    DataQualityValidation, ProgressStage, NotificationType
)
from ..models.model_version_manager import (
    ModelVersionManager, RollbackReason, VersionComparison
)
from ..models.automated_training_pipeline import AutomatedTrainingPipeline
from ..models.ensemble_forecasting_engine import EnsembleForecastingEngine

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/training-progress", tags=["Training Progress"])

# Global instances (will be initialized by main app)
progress_monitor: Optional[TrainingProgressMonitor] = None
version_manager: Optional[ModelVersionManager] = None
training_pipeline: Optional[AutomatedTrainingPipeline] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models for API
class ProgressUpdateResponse(BaseModel):
    job_id: str
    stage: str
    progress_percentage: float
    message: str
    timestamp: str
    model_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class NotificationResponse(BaseModel):
    notification_id: str
    job_id: str
    type: str
    title: str
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None
    read: bool = False

class DataQualityResponse(BaseModel):
    is_valid: bool
    quality_score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    validation_details: Dict[str, Any]

class RollbackRequest(BaseModel):
    model_name: str
    target_version_id: Optional[str] = None
    reason: str = "manual_request"
    notes: Optional[str] = None

class VersionComparisonRequest(BaseModel):
    model_name: str
    version_1_id: str
    version_2_id: str

def get_progress_monitor() -> TrainingProgressMonitor:
    """Get progress monitor instance"""
    if progress_monitor is None:
        raise HTTPException(status_code=503, detail="Training progress monitor not initialized")
    return progress_monitor

def get_version_manager() -> ModelVersionManager:
    """Get version manager instance"""
    if version_manager is None:
        raise HTTPException(status_code=503, detail="Model version manager not initialized")
    return version_manager

def init_training_progress_api(
    ensemble_engine: EnsembleForecastingEngine,
    training_pipeline_instance: AutomatedTrainingPipeline
):
    """Initialize the training progress API with required components"""
    global progress_monitor, version_manager, training_pipeline
    
    training_pipeline = training_pipeline_instance
    progress_monitor = TrainingProgressMonitor(training_pipeline)
    version_manager = ModelVersionManager(ensemble_engine)
    
    # Register callbacks for real-time updates
    progress_monitor.register_progress_callback(_on_progress_update)
    progress_monitor.register_notification_callback(_on_notification)
    
    logger.info("Training progress API initialized")

async def _on_progress_update(progress_update: ProgressUpdate):
    """Handle progress updates and broadcast to WebSocket clients"""
    try:
        message = {
            "type": "progress_update",
            "data": {
                "job_id": progress_update.job_id,
                "stage": progress_update.stage.value,
                "progress_percentage": progress_update.progress_percentage,
                "message": progress_update.message,
                "timestamp": progress_update.timestamp.isoformat(),
                "model_name": progress_update.model_name,
                "details": progress_update.details
            }
        }
        await manager.broadcast(message)
    except Exception as e:
        logger.error(f"Failed to broadcast progress update: {e}")

async def _on_notification(notification: TrainingNotification):
    """Handle notifications and broadcast to WebSocket clients"""
    try:
        message = {
            "type": "notification",
            "data": {
                "notification_id": notification.notification_id,
                "job_id": notification.job_id,
                "type": notification.type.value,
                "title": notification.title,
                "message": notification.message,
                "timestamp": notification.timestamp.isoformat(),
                "data": notification.data,
                "read": notification.read
            }
        }
        await manager.broadcast(message)
    except Exception as e:
        logger.error(f"Failed to broadcast notification: {e}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.post("/start-monitoring/{job_id}")
async def start_monitoring_job(job_id: str):
    """Start monitoring a training job"""
    try:
        monitor = get_progress_monitor()
        success = await monitor.start_monitoring_job(job_id)
        
        if success:
            return {
                "message": f"Started monitoring training job {job_id}",
                "job_id": job_id,
                "monitoring_started": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start monitoring job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{job_id}", response_model=List[ProgressUpdateResponse])
async def get_job_progress(job_id: str):
    """Get progress history for a training job"""
    try:
        monitor = get_progress_monitor()
        progress_data = monitor.get_job_progress(job_id)
        
        return [ProgressUpdateResponse(**update) for update in progress_data]
        
    except Exception as e:
        logger.error(f"Failed to get job progress for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active-jobs")
async def get_active_jobs():
    """Get currently active training jobs"""
    try:
        monitor = get_progress_monitor()
        active_jobs = monitor.get_active_jobs()
        
        return {
            "active_jobs": active_jobs,
            "total_active": len(active_jobs),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    job_id: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 50
):
    """Get training notifications"""
    try:
        monitor = get_progress_monitor()
        notifications = monitor.get_notifications(job_id=job_id, unread_only=unread_only)
        
        # Limit results
        notifications = notifications[:limit]
        
        return [NotificationResponse(**notification) for notification in notifications]
        
    except Exception as e:
        logger.error(f"Failed to get notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/{notification_id}/mark-read")
async def mark_notification_read(notification_id: str):
    """Mark a notification as read"""
    try:
        monitor = get_progress_monitor()
        success = monitor.mark_notification_read(notification_id)
        
        if success:
            return {
                "message": f"Notification {notification_id} marked as read",
                "notification_id": notification_id,
                "success": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"Notification {notification_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark notification as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-data-quality", response_model=DataQualityResponse)
async def validate_data_quality(company_id: str, data: Dict[str, Any]):
    """Validate data quality before training initiation"""
    try:
        monitor = get_progress_monitor()
        
        # Convert data to DataFrame (simplified for API)
        import pandas as pd
        df = pd.DataFrame(data.get('records', []))
        
        validation_result = await monitor.validate_data_quality(df, company_id)
        
        return DataQualityResponse(
            is_valid=validation_result.is_valid,
            quality_score=validation_result.quality_score,
            issues=validation_result.issues,
            warnings=validation_result.warnings,
            recommendations=validation_result.recommendations,
            validation_details=validation_result.validation_details
        )
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{model_name}")
async def get_model_version_history(model_name: str):
    """Get version history for a model"""
    try:
        version_mgr = get_version_manager()
        history = version_mgr.get_version_history(model_name)
        
        return {
            "model_name": model_name,
            "version_history": history,
            "total_versions": len(history),
            "active_version": version_mgr.active_versions.get(model_name)
        }
        
    except Exception as e:
        logger.error(f"Failed to get version history for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions")
async def get_all_active_versions():
    """Get currently active versions for all models"""
    try:
        version_mgr = get_version_manager()
        active_versions = version_mgr.get_active_versions()
        
        return {
            "active_versions": active_versions,
            "total_models": len(active_versions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback")
async def rollback_model_version(request: RollbackRequest):
    """Rollback a model to a previous version"""
    try:
        version_mgr = get_version_manager()
        
        # Convert reason string to enum
        try:
            reason = RollbackReason(request.reason)
        except ValueError:
            reason = RollbackReason.MANUAL_REQUEST
        
        success = await version_mgr.rollback_model(
            model_name=request.model_name,
            target_version_id=request.target_version_id,
            reason=reason,
            triggered_by="api_user"
        )
        
        if success:
            return {
                "message": f"Successfully rolled back {request.model_name}",
                "model_name": request.model_name,
                "target_version_id": request.target_version_id,
                "reason": request.reason,
                "success": True
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to rollback {request.model_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-versions")
async def compare_model_versions(request: VersionComparisonRequest):
    """Compare two model versions"""
    try:
        version_mgr = get_version_manager()
        
        comparison = version_mgr.compare_versions(
            model_name=request.model_name,
            version_1_id=request.version_1_id,
            version_2_id=request.version_2_id
        )
        
        if comparison:
            return {
                "model_name": comparison.model_name,
                "version_1": comparison.version_1,
                "version_2": comparison.version_2,
                "performance_metrics_comparison": comparison.performance_metrics_comparison,
                "parameter_differences": comparison.parameter_differences,
                "recommendation": comparison.recommendation,
                "confidence_score": comparison.confidence_score,
                "comparison_timestamp": comparison.comparison_timestamp.isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Could not compare versions for {request.model_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Version comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rollback-history")
async def get_rollback_history(model_name: Optional[str] = None):
    """Get rollback history"""
    try:
        version_mgr = get_version_manager()
        history = version_mgr.get_rollback_history(model_name)
        
        return {
            "rollback_history": history,
            "total_rollbacks": len(history),
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get rollback history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check-rollback-needed/{model_name}")
async def check_rollback_needed(model_name: str):
    """Check if a model needs to be rolled back"""
    try:
        version_mgr = get_version_manager()
        needs_rollback, reason = version_mgr.check_rollback_needed(model_name)
        
        return {
            "model_name": model_name,
            "needs_rollback": needs_rollback,
            "reason": reason.value if reason else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to check rollback need for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-versions/{model_name}")
async def cleanup_old_versions(model_name: str, keep_count: int = 10):
    """Clean up old model versions"""
    try:
        version_mgr = get_version_manager()
        cleaned_count = version_mgr.cleanup_old_versions(model_name, keep_count)
        
        return {
            "message": f"Cleaned up {cleaned_count} old versions for {model_name}",
            "model_name": model_name,
            "versions_cleaned": cleaned_count,
            "versions_kept": keep_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup versions for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring-status")
async def get_monitoring_status():
    """Get overall monitoring system status"""
    try:
        monitor = get_progress_monitor()
        version_mgr = get_version_manager()
        
        # Get active jobs
        active_jobs = monitor.get_active_jobs()
        
        # Get unread notifications
        unread_notifications = monitor.get_notifications(unread_only=True)
        
        # Get active versions
        active_versions = version_mgr.get_active_versions()
        
        # Get recent rollbacks
        recent_rollbacks = version_mgr.get_rollback_history()[:5]
        
        return {
            "monitoring_status": {
                "active_jobs": len(active_jobs),
                "unread_notifications": len(unread_notifications),
                "active_models": len(active_versions),
                "recent_rollbacks": len(recent_rollbacks),
                "websocket_connections": len(manager.active_connections)
            },
            "active_jobs": active_jobs,
            "recent_notifications": unread_notifications[:10],
            "active_versions": active_versions,
            "recent_rollbacks": recent_rollbacks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_monitoring_health():
    """Get health status of the monitoring system"""
    try:
        monitor = get_progress_monitor()
        version_mgr = get_version_manager()
        
        # Check if components are initialized
        components_status = {
            "progress_monitor": monitor is not None,
            "version_manager": version_mgr is not None,
            "training_pipeline": training_pipeline is not None
        }
        
        # Calculate health score
        health_score = sum(components_status.values()) / len(components_status)
        
        # Determine health status
        if health_score == 1.0:
            health_status = "healthy"
        elif health_score >= 0.5:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        return {
            "health_status": health_status,
            "health_score": health_score,
            "components": components_status,
            "websocket_connections": len(manager.active_connections),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring health: {e}")
        raise HTTPException(status_code=500, detail=str(e))