"""
Weight Adaptation API Endpoints
Provides REST API endpoints for real-time model weight adaptation system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from src.models.ensemble_forecasting_engine import EnsembleForecastingEngine
from src.models.weight_adaptation_system import WeightUpdateTrigger
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/weight-adaptation", tags=["Weight Adaptation"])

# Global ensemble engine instance (in production, this would be managed differently)
ensemble_engine: Optional[EnsembleForecastingEngine] = None

def get_ensemble_engine() -> EnsembleForecastingEngine:
    """Get the global ensemble engine instance"""
    global ensemble_engine
    if ensemble_engine is None:
        ensemble_engine = EnsembleForecastingEngine()
    return ensemble_engine

@router.get("/status")
async def get_weight_adaptation_status(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current weight adaptation system status"""
    try:
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        status = engine.weight_adaptation_system.get_system_status()
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get weight adaptation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weights/current")
async def get_current_weights(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current model weights"""
    try:
        engine = get_ensemble_engine()
        
        current_weights = engine.model_weights.copy()
        
        # Add additional weight information
        weight_info = {}
        for model_name, weight in current_weights.items():
            status = engine.model_status.get(model_name)
            weight_info[model_name] = {
                "weight": weight,
                "status": "active" if status and status.initialized else "inactive",
                "last_updated": status.last_trained.isoformat() if status and status.last_trained else None,
                "error_count": status.error_count if status else 0
            }
        
        return {
            "success": True,
            "data": {
                "weights": current_weights,
                "weight_info": weight_info,
                "last_update": engine.weight_adaptation_system.last_update.isoformat() 
                              if engine.weight_adaptation_system and engine.weight_adaptation_system.last_update 
                              else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get current weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weights/evolution")
async def get_weight_evolution(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get weight evolution data for visualization"""
    try:
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=400, 
                detail="Days parameter must be between 1 and 365"
            )
        
        engine = get_ensemble_engine()
        evolution_data = engine.get_weight_evolution_data(days)
        
        return {
            "success": True,
            "data": evolution_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get weight evolution data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/metrics")
async def get_performance_metrics(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get real-time performance metrics for all models"""
    try:
        engine = get_ensemble_engine()
        metrics = engine.get_performance_metrics()
        
        return {
            "success": True,
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/models")
async def get_model_health(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get model health summary"""
    try:
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        health_summary = engine.weight_adaptation_system.health_monitor.get_model_health_summary()
        
        return {
            "success": True,
            "data": health_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retraining/recommendations")
async def get_retraining_recommendations(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get model retraining recommendations"""
    try:
        engine = get_ensemble_engine()
        recommendations = engine.get_retraining_recommendations()
        
        return {
            "success": True,
            "data": {
                "recommendations": recommendations,
                "total_count": len(recommendations),
                "urgent_count": len([r for r in recommendations if r.get('urgency') in ['high', 'critical']])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get retraining recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/weights/update")
async def manual_weight_update(
    weights: Dict[str, float],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Manually update model weights"""
    try:
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        # Validate weights
        if not weights:
            raise HTTPException(status_code=400, detail="Weights cannot be empty")
        
        if abs(sum(weights.values()) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400, 
                detail="Weights must sum to 1.0"
            )
        
        # Check if all required models are present
        current_models = set(engine.model_weights.keys())
        provided_models = set(weights.keys())
        
        if current_models != provided_models:
            raise HTTPException(
                status_code=400,
                detail=f"Weight keys must match current models. Expected: {list(current_models)}, Got: {list(provided_models)}"
            )
        
        # Update weights
        success = engine.weight_adaptation_system.manual_weight_update(weights)
        
        if success:
            return {
                "success": True,
                "message": "Weights updated successfully",
                "data": {
                    "new_weights": engine.weight_adaptation_system.get_current_weights(),
                    "update_time": datetime.now().isoformat()
                }
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to update weights"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_name}/retrain")
async def trigger_model_retraining(
    model_name: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Trigger retraining for a specific model"""
    try:
        engine = get_ensemble_engine()
        
        if model_name not in engine.models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found"
            )
        
        # Trigger retraining in background
        success = engine.trigger_model_retraining(model_name)
        
        if success:
            return {
                "success": True,
                "message": f"Retraining triggered for model '{model_name}'",
                "data": {
                    "model_name": model_name,
                    "trigger_time": datetime.now().isoformat(),
                    "status": "retraining_started"
                }
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to trigger retraining for model '{model_name}'"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger retraining for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_name}/recover")
async def recover_failed_model(
    model_name: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Recover a previously failed model"""
    try:
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        if model_name not in engine.models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found"
            )
        
        success = engine.weight_adaptation_system.recover_model(model_name)
        
        if success:
            return {
                "success": True,
                "message": f"Model '{model_name}' recovered successfully",
                "data": {
                    "model_name": model_name,
                    "recovery_time": datetime.now().isoformat(),
                    "new_weights": engine.weight_adaptation_system.get_current_weights()
                }
            }
        else:
            return {
                "success": False,
                "message": f"Model '{model_name}' was not in failed state or recovery failed"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recover model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/weights/trigger-update")
async def trigger_weight_update(
    trigger_type: str = "manual",
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Manually trigger a weight update"""
    try:
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        # Validate trigger type
        valid_triggers = [t.value for t in WeightUpdateTrigger]
        if trigger_type not in valid_triggers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trigger type. Must be one of: {valid_triggers}"
            )
        
        trigger = WeightUpdateTrigger(trigger_type)
        success = engine.weight_adaptation_system._trigger_weight_update(trigger)
        
        if success:
            return {
                "success": True,
                "message": "Weight update triggered successfully",
                "data": {
                    "trigger_type": trigger_type,
                    "trigger_time": datetime.now().isoformat(),
                    "new_weights": engine.weight_adaptation_system.get_current_weights()
                }
            }
        else:
            return {
                "success": False,
                "message": "Weight update was not needed or failed"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger weight update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_weight_adaptation_config(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get weight adaptation system configuration"""
    try:
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        config = engine.weight_adaptation_system.config
        
        return {
            "success": True,
            "data": {
                "weight_update_frequency": config.weight_update_frequency,
                "min_model_weight": config.min_model_weight,
                "max_model_weight": config.max_model_weight,
                "weight_smoothing_factor": config.weight_smoothing_factor,
                "performance_window_days": config.performance_window_days,
                "degradation_alert_threshold": config.degradation_alert_threshold,
                "max_weight_change_per_update": config.max_weight_change_per_update,
                "adaptive_learning_enabled": config.adaptive_learning_enabled
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get weight adaptation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_weight_adaptation_alerts(
    days: int = 7,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get weight adaptation system alerts"""
    try:
        if days < 1 or days > 30:
            raise HTTPException(
                status_code=400, 
                detail="Days parameter must be between 1 and 30"
            )
        
        engine = get_ensemble_engine()
        
        if not engine.weight_adaptation_system:
            raise HTTPException(
                status_code=503, 
                detail="Weight adaptation system not available"
            )
        
        # Get alerts from health monitor
        health_summary = engine.weight_adaptation_system.health_monitor.get_model_health_summary()
        retraining_recommendations = engine.weight_adaptation_system.health_monitor.get_retraining_recommendations()
        
        # Create alert summary
        alerts = []
        
        # Add model health alerts
        for model_name, model_info in health_summary.get('models', {}).items():
            if model_info.get('health_status') in ['degraded', 'failed']:
                alerts.append({
                    "type": "model_health",
                    "severity": "high" if model_info.get('health_status') == 'failed' else "medium",
                    "model_name": model_name,
                    "message": f"Model {model_name} health status: {model_info.get('health_status')}",
                    "details": model_info
                })
        
        # Add retraining alerts
        for rec in retraining_recommendations:
            alerts.append({
                "type": "retraining_needed",
                "severity": rec.get('urgency', 'medium'),
                "model_name": rec.get('model_name'),
                "message": f"Model {rec.get('model_name')} needs retraining: {rec.get('trigger_reason')}",
                "details": rec
            })
        
        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "total_count": len(alerts),
                "by_severity": {
                    "critical": len([a for a in alerts if a['severity'] == 'critical']),
                    "high": len([a for a in alerts if a['severity'] == 'high']),
                    "medium": len([a for a in alerts if a['severity'] == 'medium']),
                    "low": len([a for a in alerts if a['severity'] == 'low'])
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get weight adaptation alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))