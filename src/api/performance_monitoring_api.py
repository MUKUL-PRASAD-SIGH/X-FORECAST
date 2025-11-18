"""
Performance Monitoring and Optimization API Endpoints
Provides REST API endpoints for performance optimization and system monitoring
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio

from ..models.performance_optimization import PerformanceOptimizer
from ..models.system_monitoring import SystemMonitor, SystemComponent, AlertSeverity
from ..models.ensemble import AdaptiveConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/performance-monitoring", tags=["Performance Monitoring"])

# Global instances (singleton pattern)
_performance_optimizer: Optional[PerformanceOptimizer] = None
_system_monitor: Optional[SystemMonitor] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create the performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        config = {
            'max_memory_gb': 4.0,
            'cache_size_mb': 512.0,
            'cache_ttl_seconds': 3600,
            'chunk_size': 10000,
            'max_workers': 4,
            'db_path': 'performance_optimization.db'
        }
        _performance_optimizer = PerformanceOptimizer(config)
        logger.info("PerformanceOptimizer initialized")
    return _performance_optimizer

def get_system_monitor() -> SystemMonitor:
    """Get or create the system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        config = {
            'alerts': {
                'email_enabled': False,  # Configure as needed
                'webhook_enabled': False,  # Configure as needed
            }
        }
        _system_monitor = SystemMonitor(config)
        logger.info("SystemMonitor initialized")
    return _system_monitor

@router.post("/start-monitoring")
async def start_monitoring() -> Dict[str, Any]:
    """Start comprehensive performance and system monitoring"""
    try:
        system_monitor = get_system_monitor()
        await system_monitor.start_monitoring()
        
        return {
            'success': True,
            'message': 'Performance and system monitoring started',
            'started_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-monitoring")
async def stop_monitoring() -> Dict[str, Any]:
    """Stop performance and system monitoring"""
    try:
        system_monitor = get_system_monitor()
        await system_monitor.stop_monitoring()
        
        return {
            'success': True,
            'message': 'Performance and system monitoring stopped',
            'stopped_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    try:
        system_monitor = get_system_monitor()
        status = system_monitor.get_system_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization-stats")
async def get_optimization_stats() -> Dict[str, Any]:
    """Get performance optimization statistics"""
    try:
        optimizer = get_performance_optimizer()
        stats = optimizer.get_optimization_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_statistics': stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache-stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache performance statistics"""
    try:
        optimizer = get_performance_optimizer()
        cache_stats = optimizer.cache.get_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_statistics': cache_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_cache() -> Dict[str, Any]:
    """Clear performance cache"""
    try:
        optimizer = get_performance_optimizer()
        optimizer.cache.clear()
        
        return {
            'success': True,
            'message': 'Cache cleared successfully',
            'cleared_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory-usage")
async def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics"""
    try:
        optimizer = get_performance_optimizer()
        memory_usage = optimizer.memory_manager.get_memory_usage()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_usage': memory_usage,
            'memory_pressure': optimizer.memory_manager.check_memory_pressure()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-memory")
async def cleanup_memory() -> Dict[str, Any]:
    """Trigger memory cleanup"""
    try:
        optimizer = get_performance_optimizer()
        
        memory_before = optimizer.memory_manager.get_memory_usage()
        await optimizer.memory_manager.cleanup_if_needed()
        memory_after = optimizer.memory_manager.get_memory_usage()
        
        return {
            'success': True,
            'message': 'Memory cleanup completed',
            'memory_before_mb': memory_before['rss_mb'],
            'memory_after_mb': memory_after['rss_mb'],
            'memory_freed_mb': memory_before['rss_mb'] - memory_after['rss_mb'],
            'cleaned_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts(
    component: Optional[str] = Query(None, description="Filter by component"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, description="Maximum number of alerts to return")
) -> Dict[str, Any]:
    """Get system alerts"""
    try:
        system_monitor = get_system_monitor()
        
        # Convert string parameters to enums if provided
        component_filter = None
        if component:
            try:
                component_filter = SystemComponent(component.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        alerts = system_monitor.alert_manager.get_active_alerts(
            component=component_filter,
            severity=severity_filter
        )
        
        # Limit results
        alerts = alerts[:limit]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(alerts),
            'filters': {
                'component': component,
                'severity': severity
            },
            'alerts': [
                {
                    'id': alert.id,
                    'component': alert.component.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'details': alert.details,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'acknowledged': alert.acknowledged,
                    'acknowledged_by': alert.acknowledged_by,
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                }
                for alert in alerts
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query("api_user", description="Who is acknowledging the alert")
) -> Dict[str, Any]:
    """Acknowledge a system alert"""
    try:
        system_monitor = get_system_monitor()
        
        success = system_monitor.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return {
                'success': True,
                'message': f'Alert {alert_id} acknowledged successfully',
                'alert_id': alert_id,
                'acknowledged_by': acknowledged_by,
                'acknowledged_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """Resolve a system alert"""
    try:
        system_monitor = get_system_monitor()
        
        success = system_monitor.alert_manager.resolve_alert(alert_id)
        
        if success:
            return {
                'success': True,
                'message': f'Alert {alert_id} resolved successfully',
                'alert_id': alert_id,
                'resolved_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capacity-recommendations")
async def get_capacity_recommendations() -> Dict[str, Any]:
    """Get capacity planning recommendations"""
    try:
        system_monitor = get_system_monitor()
        recommendations = system_monitor.capacity_planner.get_capacity_recommendations()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'recommendations': [
                {
                    'component': rec.component.value,
                    'current_utilization': rec.current_utilization,
                    'projected_utilization_24h': rec.projected_utilization_24h,
                    'projected_utilization_7d': rec.projected_utilization_7d,
                    'capacity_exhaustion_eta': rec.capacity_exhaustion_eta.isoformat() if rec.capacity_exhaustion_eta else None,
                    'recommended_action': rec.recommended_action,
                    'confidence': rec.confidence,
                    'timestamp': rec.timestamp.isoformat()
                }
                for rec in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get capacity recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-checks")
async def get_health_checks() -> Dict[str, Any]:
    """Get latest health check results"""
    try:
        system_monitor = get_system_monitor()
        
        # Run fresh health checks
        health_results = await system_monitor.health_monitor.run_all_health_checks()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': all(result.healthy for result in health_results.values()),
            'health_checks': {
                component.value: {
                    'healthy': result.healthy,
                    'score': result.score,
                    'message': result.message,
                    'metrics': result.metrics,
                    'response_time_ms': result.response_time_ms,
                    'timestamp': result.timestamp.isoformat()
                }
                for component, result in health_results.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get health checks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-checks/{component}")
async def get_component_health_check(component: str) -> Dict[str, Any]:
    """Get health check for a specific component"""
    try:
        # Convert string to enum
        try:
            component_enum = SystemComponent(component.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        system_monitor = get_system_monitor()
        result = await system_monitor.health_monitor.run_health_check(component_enum)
        
        return {
            'component': component_enum.value,
            'healthy': result.healthy,
            'score': result.score,
            'message': result.message,
            'metrics': result.metrics,
            'response_time_ms': result.response_time_ms,
            'timestamp': result.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health check for {component}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-ensemble")
async def optimize_ensemble_processing(
    background_tasks: BackgroundTasks,
    data_size_mb: float = Query(..., description="Size of data to process in MB"),
    model_count: int = Query(5, description="Number of models in ensemble")
) -> Dict[str, Any]:
    """Optimize ensemble processing (simulation endpoint)"""
    try:
        optimizer = get_performance_optimizer()
        
        # This is a simulation - in real implementation, this would process actual data
        import pandas as pd
        import numpy as np
        
        # Create sample data for optimization testing
        rows = int(data_size_mb * 1000)  # Approximate rows for given MB
        sample_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=rows, freq='D'),
            'sales_amount': np.random.normal(1000, 200, rows),
            'product_category': np.random.choice(['A', 'B', 'C'], rows),
            'region': np.random.choice(['North', 'South', 'East', 'West'], rows)
        })
        
        # Simulate model dictionary
        models = {f'model_{i}': f'mock_model_{i}' for i in range(model_count)}
        
        # Optimize processing
        result = await optimizer.optimize_ensemble_processing(sample_data, models)
        
        return {
            'success': True,
            'message': 'Ensemble processing optimized',
            'optimization_result': result,
            'optimized_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize ensemble processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-store-performance")
async def batch_store_performance_data(
    performance_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Store performance data in batch for efficiency"""
    try:
        optimizer = get_performance_optimizer()
        
        # Validate performance data
        required_fields = ['model_name', 'timestamp', 'mae', 'mape']
        for i, data in enumerate(performance_data):
            for field in required_fields:
                if field not in data:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Missing required field '{field}' in record {i}"
                    )
        
        # Store data
        await optimizer.batch_store_performance_data(performance_data)
        
        return {
            'success': True,
            'message': f'Stored {len(performance_data)} performance records',
            'records_stored': len(performance_data),
            'stored_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch store performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database-stats")
async def get_database_stats() -> Dict[str, Any]:
    """Get database optimization statistics"""
    try:
        optimizer = get_performance_optimizer()
        
        # Get database connection pool stats
        db_optimizer = optimizer.db_optimizer
        
        return {
            'timestamp': datetime.now().isoformat(),
            'database_path': db_optimizer.db_path,
            'connection_pool_size': len(db_optimizer.connection_pool),
            'max_pool_size': db_optimizer.pool_size,
            'database_optimization_active': True
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-old-data")
async def cleanup_old_data(
    days_to_keep: int = Query(30, description="Number of days of data to keep")
) -> Dict[str, Any]:
    """Clean up old performance data"""
    try:
        optimizer = get_performance_optimizer()
        
        # Run cleanup in background
        loop = asyncio.get_event_loop()
        deleted_count = await loop.run_in_executor(
            None, 
            optimizer.db_optimizer.cleanup_old_data, 
            days_to_keep
        )
        
        return {
            'success': True,
            'message': f'Cleaned up old data (keeping {days_to_keep} days)',
            'records_deleted': deleted_count,
            'cleaned_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-trends")
async def get_performance_trends(
    hours: int = Query(24, description="Number of hours to analyze"),
    component: Optional[str] = Query(None, description="Filter by component")
) -> Dict[str, Any]:
    """Get performance trends analysis"""
    try:
        system_monitor = get_system_monitor()
        
        # This would analyze performance trends from collected data
        # For now, return a placeholder response
        
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'component_filter': component,
            'trends': {
                'cpu_utilization': {
                    'trend_direction': 'stable',
                    'trend_strength': 0.2,
                    'current_value': 45.2,
                    'predicted_value_24h': 46.1,
                    'confidence': 0.8
                },
                'memory_utilization': {
                    'trend_direction': 'improving',
                    'trend_strength': 0.3,
                    'current_value': 62.1,
                    'predicted_value_24h': 58.5,
                    'confidence': 0.7
                },
                'response_time': {
                    'trend_direction': 'degrading',
                    'trend_strength': 0.4,
                    'current_value': 245.3,
                    'predicted_value_24h': 267.8,
                    'confidence': 0.6
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring-dashboard")
async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data"""
    try:
        system_monitor = get_system_monitor()
        optimizer = get_performance_optimizer()
        
        # Get system status
        system_status = system_monitor.get_system_status()
        
        # Get optimization stats
        optimization_stats = optimizer.get_optimization_stats()
        
        # Get active alerts
        active_alerts = system_monitor.alert_manager.get_active_alerts()
        
        # Get capacity recommendations
        capacity_recommendations = system_monitor.capacity_planner.get_capacity_recommendations()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'dashboard_data': {
                'system_status': system_status,
                'optimization_statistics': optimization_stats,
                'active_alerts_count': len(active_alerts),
                'critical_alerts_count': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'capacity_warnings_count': len([r for r in capacity_recommendations if r.projected_utilization_7d > 85]),
                'monitoring_active': system_status.get('monitoring_active', False)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shutdown")
async def shutdown_monitoring() -> Dict[str, Any]:
    """Shutdown monitoring and optimization systems"""
    try:
        global _system_monitor, _performance_optimizer
        
        if _system_monitor:
            await _system_monitor.stop_monitoring()
        
        if _performance_optimizer:
            await _performance_optimizer.shutdown()
        
        return {
            'success': True,
            'message': 'Monitoring and optimization systems shutdown completed',
            'shutdown_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to shutdown monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for performance monitoring API"""
    try:
        # Basic health check
        return {
            'status': 'healthy',
            'service': 'performance-monitoring-api',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))