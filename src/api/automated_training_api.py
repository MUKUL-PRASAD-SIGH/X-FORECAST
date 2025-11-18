"""
Automated Training Pipeline API Endpoints
Provides REST API endpoints for managing automated model training and retraining
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Import training pipeline components
from ..models.training_pipeline_integration import TrainingPipelineIntegration
from ..models.ensemble_forecasting_engine import EnsembleForecastingEngine
from ..models.model_performance_tracker import ModelPerformanceTracker

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/training", tags=["Automated Training"])

# Global instances (will be initialized by main app)
training_integration: Optional[TrainingPipelineIntegration] = None
ensemble_engine: Optional[EnsembleForecastingEngine] = None

# Pydantic models for API
class TrainingJobResponse(BaseModel):
    job_id: str
    company_id: str
    status: str
    trigger_reason: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_names: List[str]
    error_message: Optional[str] = None
    new_versions: Optional[Dict[str, str]] = None
    performance_comparison: Optional[Dict[str, Any]] = None

class ModelVersionResponse(BaseModel):
    version_id: str
    training_timestamp: str
    performance_metrics: Dict[str, float]
    model_parameters: Dict[str, Any]
    training_data_size: int
    seasonal_parameters: Dict[str, Any]
    is_active: bool
    pattern_characteristics: Optional[Dict[str, Any]] = None

class TrainingStatusResponse(BaseModel):
    total_jobs: int
    status_counts: Dict[str, int]
    queue_length: int
    active_versions: Dict[str, str]
    total_model_versions: int
    recent_jobs: List[Dict[str, Any]]
    storage_dir: str
    max_versions_per_model: int

class RetrainingRecommendation(BaseModel):
    model_name: str
    urgency: str
    reasons: List[str]
    recommended_action: str

class ManualRetrainRequest(BaseModel):
    company_id: str
    model_names: Optional[List[str]] = None
    reason: str = "manual_trigger"

class RollbackRequest(BaseModel):
    model_name: str
    target_version_id: str

def get_training_integration() -> TrainingPipelineIntegration:
    """Get training integration instance"""
    if training_integration is None:
        raise HTTPException(status_code=503, detail="Training pipeline not initialized")
    return training_integration

def init_training_api(ensemble_eng: EnsembleForecastingEngine, 
                     performance_tracker: Optional[ModelPerformanceTracker] = None):
    """Initialize the training API with required components"""
    global training_integration, ensemble_engine
    
    ensemble_engine = ensemble_eng
    training_integration = TrainingPipelineIntegration(
        ensemble_engine=ensemble_eng,
        performance_tracker=performance_tracker
    )
    
    logger.info("Automated training API initialized")

@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_pipeline_status():
    """Get overall training pipeline status"""
    try:
        integration = get_training_integration()
        status = integration.get_training_status()
        
        return TrainingStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get training pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{company_id}")
async def get_company_training_status(company_id: str):
    """Get training status for a specific company"""
    try:
        integration = get_training_integration()
        status = integration.get_training_status(company_id)
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get training status for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job/{job_id}", response_model=TrainingJobResponse)
async def get_training_job_status(job_id: str):
    """Get status of a specific training job"""
    try:
        integration = get_training_integration()
        job_status = integration.training_pipeline.get_training_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        return TrainingJobResponse(**job_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/versions", response_model=List[ModelVersionResponse])
async def get_model_versions(model_name: str):
    """Get version history for a specific model"""
    try:
        integration = get_training_integration()
        versions = integration.get_model_versions(model_name)
        
        return [ModelVersionResponse(**version) for version in versions]
        
    except Exception as e:
        logger.error(f"Failed to get versions for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/manual-retrain")
async def trigger_manual_retraining(
    request: ManualRetrainRequest,
    background_tasks: BackgroundTasks
):
    """Manually trigger retraining for specific models"""
    try:
        integration = get_training_integration()
        
        # Trigger retraining in background
        job_id = await integration.manual_retrain(
            company_id=request.company_id,
            model_names=request.model_names,
            reason=request.reason
        )
        
        return {
            "message": "Manual retraining triggered successfully",
            "job_id": job_id,
            "company_id": request.company_id,
            "model_names": request.model_names or "all",
            "reason": request.reason
        }
        
    except Exception as e:
        logger.error(f"Manual retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback")
async def rollback_model_version(request: RollbackRequest):
    """Rollback a model to a specific version"""
    try:
        integration = get_training_integration()
        
        success = await integration.rollback_model(
            model_name=request.model_name,
            target_version_id=request.target_version_id
        )
        
        if success:
            return {
                "message": f"Successfully rolled back {request.model_name} to version {request.target_version_id}",
                "model_name": request.model_name,
                "target_version_id": request.target_version_id,
                "success": True
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to rollback {request.model_name} to version {request.target_version_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{company_id}", response_model=List[RetrainingRecommendation])
async def get_retraining_recommendations(company_id: str):
    """Get retraining recommendations for a company"""
    try:
        integration = get_training_integration()
        recommendations = integration.get_retraining_recommendations(company_id)
        
        return [RetrainingRecommendation(**rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Failed to get retraining recommendations for {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_models():
    """Get list of available models in the ensemble"""
    try:
        if ensemble_engine is None:
            raise HTTPException(status_code=503, detail="Ensemble engine not initialized")
        
        models = []
        for model_name, status in ensemble_engine.model_status.items():
            model_info = {
                "model_name": model_name,
                "initialized": status.initialized,
                "last_trained": status.last_trained.isoformat() if status.last_trained else None,
                "training_data_points": status.training_data_points,
                "current_weight": status.current_weight,
                "error_count": status.error_count,
                "status_message": status.status_message
            }
            
            if status.performance_metrics:
                model_info["performance_metrics"] = {
                    "mae": status.performance_metrics.mae,
                    "mape": status.performance_metrics.mape,
                    "rmse": status.performance_metrics.rmse,
                    "r_squared": status.performance_metrics.r_squared,
                    "evaluation_date": status.performance_metrics.evaluation_date.isoformat()
                }
            
            models.append(model_info)
        
        return {
            "total_models": len(models),
            "models": models,
            "ensemble_weights": ensemble_engine.model_weights
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_training_pipeline_health():
    """Get health status of the training pipeline"""
    try:
        integration = get_training_integration()
        
        # Get pipeline status
        pipeline_status = integration.training_pipeline.get_pipeline_status()
        
        # Calculate health metrics
        total_jobs = pipeline_status['total_jobs']
        failed_jobs = pipeline_status['status_counts'].get('failed', 0)
        completed_jobs = pipeline_status['status_counts'].get('completed', 0)
        
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 100
        failure_rate = (failed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        # Determine health status
        if failure_rate > 20:
            health_status = "unhealthy"
        elif failure_rate > 10:
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "health_status": health_status,
            "success_rate": round(success_rate, 2),
            "failure_rate": round(failure_rate, 2),
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "queue_length": pipeline_status['queue_length'],
            "active_models": len(pipeline_status['active_versions']),
            "total_model_versions": pipeline_status['total_model_versions'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get training pipeline health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/job/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a pending or in-progress training job"""
    try:
        integration = get_training_integration()
        
        # Get job status
        job_status = integration.training_pipeline.get_training_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        current_status = job_status['status']
        
        if current_status in ['completed', 'failed', 'rolled_back']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job in status: {current_status}"
            )
        
        # For now, we'll just mark it as failed
        # In a full implementation, you'd want to actually stop the running process
        job = integration.training_pipeline.training_jobs[job_id]
        job.status = integration.training_pipeline.TrainingStatus.FAILED
        job.error_message = "Cancelled by user"
        job.completed_at = datetime.now()
        
        integration.training_pipeline._save_training_jobs()
        
        return {
            "message": f"Training job {job_id} cancelled successfully",
            "job_id": job_id,
            "previous_status": current_status,
            "new_status": "failed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_training_metrics():
    """Get training pipeline metrics and statistics"""
    try:
        integration = get_training_integration()
        pipeline_status = integration.training_pipeline.get_pipeline_status()
        
        # Calculate metrics
        total_jobs = pipeline_status['total_jobs']
        status_counts = pipeline_status['status_counts']
        
        # Calculate average training time (mock data for now)
        avg_training_time_minutes = 15  # This would be calculated from actual job data
        
        # Get model-specific metrics
        model_metrics = {}
        if ensemble_engine:
            for model_name, status in ensemble_engine.model_status.items():
                model_metrics[model_name] = {
                    "total_trainings": 1 if status.initialized else 0,
                    "last_training": status.last_trained.isoformat() if status.last_trained else None,
                    "current_weight": status.current_weight,
                    "error_count": status.error_count,
                    "performance": {
                        "mae": status.performance_metrics.mae if status.performance_metrics else None,
                        "mape": status.performance_metrics.mape if status.performance_metrics else None,
                        "rmse": status.performance_metrics.rmse if status.performance_metrics else None,
                        "r_squared": status.performance_metrics.r_squared if status.performance_metrics else None
                    }
                }
        
        return {
            "pipeline_metrics": {
                "total_jobs": total_jobs,
                "success_rate": (status_counts.get('completed', 0) / total_jobs * 100) if total_jobs > 0 else 100,
                "failure_rate": (status_counts.get('failed', 0) / total_jobs * 100) if total_jobs > 0 else 0,
                "avg_training_time_minutes": avg_training_time_minutes,
                "queue_length": pipeline_status['queue_length']
            },
            "status_distribution": status_counts,
            "model_metrics": model_metrics,
            "active_versions": pipeline_status['active_versions'],
            "total_model_versions": pipeline_status['total_model_versions'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))