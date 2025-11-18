"""
Model Performance Tracking API Endpoints
Provides REST API endpoints for comprehensive model performance monitoring
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import os
import asyncio

from ..models.model_performance_tracker import (
    ModelPerformanceTracker, DriftType, HealthStatus,
    RealTimeMetrics, SystemHealthMetrics, DriftDetectionResult,
    RetrainingRecommendation
)
from ..models.ensemble import AdaptiveConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/model-performance", tags=["Model Performance"])

# Initialize performance tracker (singleton pattern)
_performance_tracker: Optional[ModelPerformanceTracker] = None

def get_performance_tracker() -> ModelPerformanceTracker:
    """Get or create the performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        config = AdaptiveConfig()
        _performance_tracker = ModelPerformanceTracker(config)
        logger.info("ModelPerformanceTracker initialized")
    return _performance_tracker

@router.post("/track-prediction")
async def track_prediction(
    model_name: str,
    predicted_value: float,
    actual_value: Optional[float] = None,
    response_time_ms: float = 0.0,
    additional_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Track a single prediction and update real-time metrics"""
    try:
        tracker = get_performance_tracker()
        
        metrics = await tracker.track_prediction(
            model_name=model_name,
            predicted_value=predicted_value,
            actual_value=actual_value,
            response_time_ms=response_time_ms,
            additional_metrics=additional_metrics
        )
        
        return {
            'success': True,
            'timestamp': metrics.timestamp.isoformat(),
            'model_name': model_name,
            'metrics': {
                'mae': metrics.mae,
                'mape': metrics.mape,
                'rmse': metrics.rmse,
                'r_squared': metrics.r_squared,
                'response_time_ms': metrics.response_time_ms,
                'accuracy_trend': metrics.accuracy_trend,
                'drift_score': metrics.drift_score,
                'health_score': metrics.health_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to track prediction for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/real-time-metrics")
async def get_real_time_metrics(
    model_name: str,
    window_minutes: int = Query(60, description="Time window in minutes")
) -> Dict[str, Any]:
    """Get real-time accuracy metrics for a specific model"""
    try:
        tracker = get_performance_tracker()
        
        metrics = await tracker.calculate_real_time_accuracy_metrics(
            model_name=model_name,
            window_minutes=window_minutes
        )
        
        return {
            'model_name': model_name,
            'time_window_minutes': window_minutes,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get real-time metrics for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/drift-detection")
async def detect_model_drift(
    model_name: str,
    detection_window: Optional[int] = Query(None, description="Number of predictions to analyze")
) -> Dict[str, Any]:
    """Detect model drift for a specific model"""
    try:
        tracker = get_performance_tracker()
        
        drift_result = await tracker.detect_model_drift(
            model_name=model_name,
            detection_window=detection_window
        )
        
        return {
            'model_name': model_name,
            'drift_detection': {
                'drift_type': drift_result.drift_type.value,
                'drift_score': drift_result.drift_score,
                'confidence': drift_result.confidence,
                'detected_at': drift_result.detected_at.isoformat(),
                'details': drift_result.details,
                'recommended_action': drift_result.recommended_action
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to detect drift for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-health")
async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health metrics"""
    try:
        tracker = get_performance_tracker()
        
        health_metrics = await tracker.calculate_system_health_score()
        
        return {
            'timestamp': health_metrics.last_updated.isoformat(),
            'system_health': {
                'overall_health_score': health_metrics.overall_health_score,
                'health_status': health_metrics.health_status.value,
                'active_models': health_metrics.active_models,
                'failed_models': health_metrics.failed_models,
                'avg_accuracy': health_metrics.avg_accuracy,
                'avg_response_time': health_metrics.avg_response_time,
                'drift_alerts': health_metrics.drift_alerts,
                'performance_alerts': health_metrics.performance_alerts,
                'model_health_scores': health_metrics.model_health_scores
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retraining-recommendations")
async def get_retraining_recommendations() -> Dict[str, Any]:
    """Get current retraining recommendations"""
    try:
        tracker = get_performance_tracker()
        
        recommendations = tracker.get_retraining_recommendations()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'recommendations': [
                {
                    'model_name': rec.model_name,
                    'urgency': rec.urgency,
                    'trigger_reason': rec.trigger_reason,
                    'performance_degradation': rec.performance_degradation,
                    'drift_score': rec.drift_score,
                    'recommended_action': rec.recommended_action,
                    'estimated_improvement': rec.estimated_improvement,
                    'last_training_date': rec.last_training_date.isoformat() if rec.last_training_date else None
                }
                for rec in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get retraining recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_name}/trigger-retraining")
async def trigger_model_retraining(
    model_name: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Trigger automatic retraining for a specific model"""
    try:
        tracker = get_performance_tracker()
        
        success = tracker.trigger_automatic_retraining(model_name)
        
        if success:
            return {
                'success': True,
                'message': f'Retraining triggered for model {model_name}',
                'model_name': model_name,
                'triggered_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to trigger retraining for model {model_name}"
            )
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard-data")
async def get_performance_dashboard_data(
    hours: int = Query(24, description="Number of hours to include in dashboard data")
) -> Dict[str, Any]:
    """Get comprehensive performance data for dashboard display"""
    try:
        tracker = get_performance_tracker()
        
        dashboard_data = tracker.get_performance_dashboard_data(hours=hours)
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comprehensive-dashboard")
async def get_comprehensive_dashboard_data(
    hours: int = Query(24, description="Number of hours to include in dashboard data")
) -> Dict[str, Any]:
    """Get enhanced comprehensive performance data with health monitoring and alerts"""
    try:
        tracker = get_performance_tracker()
        
        # Get comprehensive dashboard data
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'time_window_hours': hours,
            'system_health': {},
            'model_health_summaries': {},
            'alerts': {},
            'performance_trends': {},
            'monitoring_status': {}
        }
        
        # Get system health
        system_health = await tracker.calculate_system_health_score()
        dashboard_data['system_health'] = {
            'overall_health_score': system_health.overall_health_score,
            'health_status': system_health.health_status.value,
            'active_models': system_health.active_models,
            'failed_models': system_health.failed_models,
            'avg_accuracy': system_health.avg_accuracy,
            'avg_response_time': system_health.avg_response_time,
            'drift_alerts': system_health.drift_alerts,
            'performance_alerts': system_health.performance_alerts,
            'last_updated': system_health.last_updated.isoformat()
        }
        
        # Get individual model health summaries
        for model_name in tracker.real_time_metrics.keys():
            model_health = await tracker.get_model_health_summary(model_name)
            dashboard_data['model_health_summaries'][model_name] = model_health
        
        # Get system alerts
        dashboard_data['alerts'] = {
            'critical_alerts': tracker.get_system_alerts(hours, 'critical'),
            'warning_alerts': tracker.get_system_alerts(hours, 'warning'),
            'all_alerts': tracker.get_system_alerts(hours),
            'total_count': len(tracker.get_system_alerts(hours))
        }
        
        # Get monitoring status
        dashboard_data['monitoring_status'] = {
            'continuous_monitoring_active': tracker.monitoring_active,
            'last_health_check': tracker.last_health_check.isoformat(),
            'monitoring_interval_seconds': tracker.monitoring_interval_seconds,
            'total_models_monitored': len(tracker.real_time_metrics)
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_monitored_models() -> Dict[str, Any]:
    """Get list of all monitored models with basic status"""
    try:
        tracker = get_performance_tracker()
        
        models = {}
        for model_name in tracker.real_time_metrics.keys():
            recent_metrics = await tracker.calculate_real_time_accuracy_metrics(model_name, 60)
            models[model_name] = {
                'status': 'active' if recent_metrics['prediction_count'] > 0 else 'inactive',
                'health_score': recent_metrics.get('health_score', 0.0),
                'prediction_count_last_hour': recent_metrics['prediction_count'],
                'avg_response_time': recent_metrics['avg_response_time'],
                'accuracy_trend': recent_metrics.get('accuracy_trend', 'unknown')
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(models),
            'models': models
        }
        
    except Exception as e:
        logger.error(f"Failed to list monitored models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/performance-history")
async def get_model_performance_history(
    model_name: str,
    hours: int = Query(24, description="Number of hours of history to retrieve")
) -> Dict[str, Any]:
    """Get performance history for a specific model"""
    try:
        tracker = get_performance_tracker()
        
        if model_name not in tracker.real_time_metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found in performance tracking"
            )
        
        # Get recent metrics
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            {
                'timestamp': m.timestamp.isoformat(),
                'mae': m.mae,
                'mape': m.mape,
                'rmse': m.rmse,
                'r_squared': m.r_squared,
                'response_time_ms': m.response_time_ms,
                'accuracy_trend': m.accuracy_trend,
                'drift_score': m.drift_score,
                'health_score': m.health_score
            }
            for m in tracker.real_time_metrics[model_name]
            if m.timestamp >= cutoff_time
        ]
        
        return {
            'model_name': model_name,
            'time_window_hours': hours,
            'total_records': len(recent_metrics),
            'performance_history': recent_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance history for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_performance_alerts(
    hours: int = Query(24, description="Number of hours to look back for alerts"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type: drift, performance, retraining")
) -> Dict[str, Any]:
    """Get performance alerts and notifications"""
    try:
        tracker = get_performance_tracker()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = {
            'drift_alerts': [],
            'performance_alerts': [],
            'retraining_recommendations': []
        }
        
        # Get drift alerts
        if alert_type is None or alert_type == 'drift':
            drift_alerts = [
                {
                    'model_name': r.model_name,
                    'drift_type': r.drift_type.value,
                    'drift_score': r.drift_score,
                    'confidence': r.confidence,
                    'detected_at': r.detected_at.isoformat(),
                    'recommended_action': r.recommended_action,
                    'details': r.details
                }
                for r in tracker.drift_detection_results
                if r.detected_at >= cutoff_time and r.drift_type != DriftType.NO_DRIFT
            ]
            alerts['drift_alerts'] = drift_alerts
        
        # Get retraining recommendations
        if alert_type is None or alert_type == 'retraining':
            recommendations = tracker.get_retraining_recommendations()
            alerts['retraining_recommendations'] = [
                {
                    'model_name': rec.model_name,
                    'urgency': rec.urgency,
                    'trigger_reason': rec.trigger_reason,
                    'performance_degradation': rec.performance_degradation,
                    'recommended_action': rec.recommended_action
                }
                for rec in recommendations
            ]
        
        # Performance alerts (simplified - based on current metrics)
        if alert_type is None or alert_type == 'performance':
            performance_alerts = []
            for model_name in tracker.real_time_metrics.keys():
                recent_metrics = await tracker.calculate_real_time_accuracy_metrics(model_name, 60)
                
                if recent_metrics['mape'] > 25:  # MAPE > 25% is concerning
                    performance_alerts.append({
                        'model_name': model_name,
                        'alert_type': 'high_error_rate',
                        'mape': recent_metrics['mape'],
                        'threshold': 25,
                        'detected_at': datetime.now().isoformat()
                    })
                
                if recent_metrics['avg_response_time'] > 1000:  # Response time > 1s
                    performance_alerts.append({
                        'model_name': model_name,
                        'alert_type': 'slow_response',
                        'response_time_ms': recent_metrics['avg_response_time'],
                        'threshold': 1000,
                        'detected_at': datetime.now().isoformat()
                    })
            
            alerts['performance_alerts'] = performance_alerts
        
        return {
            'timestamp': datetime.now().isoformat(),
            'time_window_hours': hours,
            'alert_type_filter': alert_type,
            'total_alerts': sum(len(alert_list) for alert_list in alerts.values()),
            'alerts': alerts
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_performance_data(
    format: str = Query("json", description="Export format: json or csv"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, str]:
    """Export comprehensive performance data"""
    try:
        tracker = get_performance_tracker()
        
        if format.lower() not in ['json', 'csv']:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        # Create export directory if it doesn't exist
        export_dir = "exports/performance"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_data_{timestamp}.{format.lower()}"
        filepath = os.path.join(export_dir, filename)
        
        # Export data
        success = tracker.export_performance_data(filepath, format.lower())
        
        if success:
            return {
                'success': True,
                'message': 'Performance data exported successfully',
                'filepath': filepath,
                'format': format.lower(),
                'exported_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to export performance data"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{filename}")
async def download_export_file(filename: str):
    """Download exported performance data file"""
    try:
        filepath = f"exports/performance/{filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Export file not found")
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download export file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/health-summary")
async def get_model_health_summary(model_name: str) -> Dict[str, Any]:
    """Get comprehensive health summary for a specific model"""
    try:
        tracker = get_performance_tracker()
        
        health_summary = await tracker.get_model_health_summary(model_name)
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Failed to get health summary for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-alerts")
async def get_system_alerts(
    hours: int = Query(24, description="Number of hours to look back for alerts"),
    severity: Optional[str] = Query(None, description="Filter by severity: critical, warning")
) -> Dict[str, Any]:
    """Get system alerts with optional filtering"""
    try:
        tracker = get_performance_tracker()
        
        alerts = tracker.get_system_alerts(hours, severity)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'time_window_hours': hours,
            'severity_filter': severity,
            'total_alerts': len(alerts),
            'alerts': alerts
        }
        
    except Exception as e:
        logger.error(f"Failed to get system alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
    """Acknowledge a system alert"""
    try:
        tracker = get_performance_tracker()
        
        success = tracker.acknowledge_alert(alert_id)
        
        if success:
            return {
                'success': True,
                'message': f'Alert {alert_id} acknowledged successfully',
                'alert_id': alert_id,
                'acknowledged_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/models/{model_name}/thresholds")
async def update_performance_thresholds(
    model_name: str,
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Update performance thresholds for a specific model"""
    try:
        tracker = get_performance_tracker()
        
        success = tracker.update_performance_thresholds(model_name, thresholds)
        
        if success:
            return {
                'success': True,
                'message': f'Thresholds updated for model {model_name}',
                'model_name': model_name,
                'updated_thresholds': thresholds,
                'updated_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update thresholds for model {model_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update thresholds for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/start")
async def start_continuous_monitoring() -> Dict[str, Any]:
    """Start continuous monitoring for all models"""
    try:
        tracker = get_performance_tracker()
        
        await tracker.start_continuous_monitoring()
        
        return {
            'success': True,
            'message': 'Continuous monitoring started',
            'monitoring_active': tracker.monitoring_active,
            'started_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start continuous monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/stop")
async def stop_continuous_monitoring() -> Dict[str, Any]:
    """Stop continuous monitoring for all models"""
    try:
        tracker = get_performance_tracker()
        
        await tracker.stop_continuous_monitoring()
        
        return {
            'success': True,
            'message': 'Continuous monitoring stopped',
            'monitoring_active': tracker.monitoring_active,
            'stopped_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop continuous monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/monitoring/{token}")
async def websocket_monitoring_endpoint(websocket, token: str):
    """WebSocket endpoint for real-time performance monitoring updates"""
    await websocket.accept()
    
    try:
        tracker = get_performance_tracker()
        
        # Send initial data
        system_health = await tracker.calculate_system_health_score()
        await websocket.send_json({
            'type': 'system_health_update',
            'data': {
                'overall_health_score': system_health.overall_health_score,
                'health_status': system_health.health_status.value,
                'active_models': system_health.active_models,
                'failed_models': system_health.failed_models,
                'avg_accuracy': system_health.avg_accuracy,
                'avg_response_time': system_health.avg_response_time,
                'drift_alerts': system_health.drift_alerts,
                'performance_alerts': system_health.performance_alerts,
                'last_updated': system_health.last_updated.isoformat(),
                'model_health_scores': system_health.model_health_scores
            }
        })
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for 30 seconds
                await asyncio.sleep(30)
                
                # Send updated system health
                system_health = await tracker.calculate_system_health_score()
                await websocket.send_json({
                    'type': 'system_health_update',
                    'data': {
                        'overall_health_score': system_health.overall_health_score,
                        'health_status': system_health.health_status.value,
                        'active_models': system_health.active_models,
                        'failed_models': system_health.failed_models,
                        'avg_accuracy': system_health.avg_accuracy,
                        'avg_response_time': system_health.avg_response_time,
                        'drift_alerts': system_health.drift_alerts,
                        'performance_alerts': system_health.performance_alerts,
                        'last_updated': system_health.last_updated.isoformat(),
                        'model_health_scores': system_health.model_health_scores
                    }
                })
                
                # Send model rankings update
                models_data = {}
                for model_name in tracker.real_time_metrics.keys():
                    model_metrics = await tracker.calculate_real_time_accuracy_metrics(model_name, 60)
                    models_data[model_name] = {
                        'status': 'active' if model_metrics['prediction_count'] > 0 else 'inactive',
                        'health_score': model_metrics.get('health_score', 0.0),
                        'prediction_count_last_hour': model_metrics['prediction_count'],
                        'avg_response_time': model_metrics['avg_response_time'],
                        'accuracy_trend': model_metrics.get('accuracy_trend', 'unknown')
                    }
                
                await websocket.send_json({
                    'type': 'model_ranking_update',
                    'data': models_data
                })
                
            except Exception as e:
                logger.error(f"Error in WebSocket monitoring loop: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket monitoring error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for model performance API"""
    try:
        tracker = get_performance_tracker()
        
        # Enhanced health check
        model_count = len(tracker.real_time_metrics)
        system_health = await tracker.calculate_system_health_score()
        
        return {
            'status': 'healthy' if system_health.health_status.value in ['excellent', 'good'] else 'degraded',
            'service': 'model-performance-api',
            'monitored_models': model_count,
            'system_health_score': system_health.overall_health_score,
            'system_health_status': system_health.health_status.value,
            'continuous_monitoring_active': tracker.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'service': 'model-performance-api',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Note: Exception handlers are typically added to the main FastAPI app, not individual routers