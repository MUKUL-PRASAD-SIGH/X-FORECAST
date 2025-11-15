"""
API endpoints for managing adaptive ensemble configuration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

from ..models.adaptive_config_manager import get_config_manager, AdaptiveEnsembleConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/adaptive-config", tags=["adaptive-config"])

class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates"""
    adaptive_learning_enabled: Optional[bool] = None
    confidence_intervals_enabled: Optional[bool] = None
    pattern_detection_enabled: Optional[bool] = None
    backtesting_enabled: Optional[bool] = None
    learning_window_days: Optional[int] = None
    weight_update_frequency: Optional[str] = None
    min_model_weight: Optional[float] = None
    max_weight_change_per_update: Optional[float] = None
    performance_tracking_enabled: Optional[bool] = None
    performance_alert_threshold: Optional[float] = None
    weight_change_alert_threshold: Optional[float] = None
    dashboard_monitoring_enabled: Optional[bool] = None
    system_monitoring_enabled: Optional[bool] = None
    auto_update_enabled: Optional[bool] = None

class RolloutConfigRequest(BaseModel):
    """Request model for rollout configuration"""
    enabled: bool
    percentage: Optional[float] = 100.0
    user_groups: Optional[list] = None

@router.get("/status")
async def get_config_status():
    """Get current adaptive ensemble configuration status"""
    try:
        config_manager = get_config_manager()
        status = config_manager.get_status_summary()
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get config status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_current_config():
    """Get current adaptive ensemble configuration"""
    try:
        config_manager = get_config_manager()
        config = config_manager.export_config()
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """Update adaptive ensemble configuration"""
    try:
        config_manager = get_config_manager()
        
        # Convert request to dict, excluding None values
        update_params = {
            key: value for key, value in request.dict().items() 
            if value is not None
        }
        
        if not update_params:
            raise HTTPException(status_code=400, detail="No configuration parameters provided")
        
        success = config_manager.update_config(**update_params)
        
        if success:
            return {
                "success": True,
                "message": "Configuration updated successfully",
                "updated_params": update_params
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration validation failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/enable")
async def enable_adaptive_features():
    """Enable all adaptive features"""
    try:
        config_manager = get_config_manager()
        success = config_manager.enable_adaptive_features(True)
        
        if success:
            return {
                "success": True,
                "message": "Adaptive features enabled successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to enable adaptive features")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable adaptive features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/disable")
async def disable_adaptive_features():
    """Disable all adaptive features"""
    try:
        config_manager = get_config_manager()
        success = config_manager.enable_adaptive_features(False)
        
        if success:
            return {
                "success": True,
                "message": "Adaptive features disabled successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to disable adaptive features")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable adaptive features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollout/configure")
async def configure_rollout(request: RolloutConfigRequest):
    """Configure gradual rollout of adaptive features"""
    try:
        config_manager = get_config_manager()
        
        if request.enabled:
            success = config_manager.enable_gradual_rollout(
                percentage=request.percentage,
                user_groups=request.user_groups
            )
            message = f"Gradual rollout enabled at {request.percentage}%"
        else:
            success = config_manager.disable_gradual_rollout()
            message = "Gradual rollout disabled (full deployment)"
        
        if success:
            return {
                "success": True,
                "message": message,
                "rollout_config": {
                    "enabled": request.enabled,
                    "percentage": request.percentage,
                    "user_groups": request.user_groups
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to configure rollout")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure rollout: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rollout/status")
async def get_rollout_status():
    """Get current rollout status"""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        return {
            "success": True,
            "rollout_status": {
                "enabled": config.gradual_rollout_enabled,
                "percentage": config.rollout_percentage,
                "user_groups": config.rollout_user_groups,
                "fallback_enabled": config.fallback_to_basic_on_error
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get rollout status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/should-use-adaptive")
async def check_user_adaptive_eligibility(user_id: str):
    """Check if a specific user should use adaptive features"""
    try:
        config_manager = get_config_manager()
        should_use = config_manager.should_use_adaptive_features(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "should_use_adaptive": should_use,
            "reason": "gradual_rollout" if not should_use else "enabled"
        }
        
    except Exception as e:
        logger.error(f"Failed to check user eligibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/import")
async def import_config(config_data: Dict[str, Any]):
    """Import configuration from external source"""
    try:
        config_manager = get_config_manager()
        success = config_manager.import_config(config_data)
        
        if success:
            return {
                "success": True,
                "message": "Configuration imported successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration import failed validation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/metrics")
async def get_monitoring_metrics():
    """Get adaptive ensemble monitoring metrics"""
    try:
        # Import here to avoid circular imports
        from ..monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        metrics = monitor.get_adaptive_ensemble_metrics()
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for adaptive configuration API"""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        return {
            "success": True,
            "status": "healthy",
            "adaptive_features_enabled": config.adaptive_learning_enabled,
            "config_file_exists": True
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }