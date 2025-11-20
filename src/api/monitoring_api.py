"""
Monitoring and Analytics API
Provides endpoints for system monitoring, logging analytics, and performance tracking
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

# Import logging system
from src.utils.logging_config import comprehensive_logger, EventType, LogLevel

# Import existing monitoring systems
try:
    from src.rag.health_monitor import health_monitor, RAGHealthMonitor
    from src.rag.diagnostic_engine import diagnostic_engine, RAGDiagnosticEngine
    RAG_MONITORING_AVAILABLE = True
except ImportError:
    RAG_MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["System Monitoring"])

# Pydantic models for API responses
class AuthStatistics(BaseModel):
    period_days: int
    total_login_attempts: int
    successful_logins: int
    failed_logins: int
    unique_users: int
    new_registrations: int
    success_rate: float

class FileProcessingStats(BaseModel):
    period_days: int
    by_file_type: Dict[str, Dict[str, Any]]

class SecurityEventsSummary(BaseModel):
    period_days: int
    events_by_type: Dict[str, Dict[str, int]]
    top_offending_ips: List[Dict[str, Any]]

class SystemHealthResponse(BaseModel):
    timestamp: datetime
    overall_status: str
    overall_score: int
    components: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]

class MonitoringDashboardResponse(BaseModel):
    timestamp: datetime
    auth_stats: AuthStatistics
    file_processing_stats: FileProcessingStats
    security_events: SecurityEventsSummary
    system_health: Optional[SystemHealthResponse] = None
    rag_performance: Optional[Dict[str, Any]] = None

# Authentication dependency (simplified for monitoring)
def get_current_user_for_monitoring(request):
    """Simplified auth check for monitoring endpoints"""
    # In production, this should verify admin/monitoring permissions
    return {"user_id": "admin", "role": "admin"}

@router.get("/dashboard", response_model=MonitoringDashboardResponse)
async def get_monitoring_dashboard(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze"),
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """
    Get comprehensive monitoring dashboard data
    """
    try:
        # Get authentication statistics
        auth_stats = comprehensive_logger.get_auth_statistics(days)
        
        # Get file processing statistics
        file_stats = comprehensive_logger.get_file_processing_statistics(days)
        
        # Get security events
        security_events = comprehensive_logger.get_security_events(days)
        
        # Get system health if available
        system_health = None
        if RAG_MONITORING_AVAILABLE:
            try:
                health_result = health_monitor.get_current_health_snapshot()
                if health_result:
                    system_health = SystemHealthResponse(
                        timestamp=health_result.timestamp,
                        overall_status=health_result.overall_score > 70 and "healthy" or "degraded",
                        overall_score=health_result.overall_score,
                        components=health_result.component_scores,
                        performance_metrics=health_result.performance_metrics,
                        active_alerts=[]  # Would be populated from health monitor
                    )
            except Exception as e:
                logger.warning(f"Could not get system health: {e}")
        
        # Get RAG performance metrics
        rag_performance = None
        if RAG_MONITORING_AVAILABLE:
            try:
                rag_performance = get_rag_performance_metrics(days)
            except Exception as e:
                logger.warning(f"Could not get RAG performance: {e}")
        
        return MonitoringDashboardResponse(
            timestamp=datetime.now(),
            auth_stats=AuthStatistics(**auth_stats),
            file_processing_stats=FileProcessingStats(**file_stats),
            security_events=SecurityEventsSummary(**security_events),
            system_health=system_health,
            rag_performance=rag_performance
        )
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")

@router.get("/auth/statistics")
async def get_auth_statistics(
    days: int = Query(default=7, ge=1, le=30),
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get detailed authentication statistics"""
    try:
        stats = comprehensive_logger.get_auth_statistics(days)
        return AuthStatistics(**stats)
    except Exception as e:
        logger.error(f"Error getting auth statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/statistics")
async def get_file_processing_statistics(
    days: int = Query(default=7, ge=1, le=30),
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get detailed file processing statistics"""
    try:
        stats = comprehensive_logger.get_file_processing_statistics(days)
        return FileProcessingStats(**stats)
    except Exception as e:
        logger.error(f"Error getting file processing statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security/events")
async def get_security_events(
    days: int = Query(default=7, ge=1, le=30),
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get security events summary"""
    try:
        events = comprehensive_logger.get_security_events(days)
        return SecurityEventsSummary(**events)
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health")
async def get_system_health(
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get current system health status"""
    try:
        if not RAG_MONITORING_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={"message": "System health monitoring not available"}
            )
        
        # Run comprehensive diagnostics
        diagnostic_result = diagnostic_engine.run_comprehensive_diagnostics()
        
        # Get current health snapshot
        health_snapshot = health_monitor.get_current_health_snapshot()
        
        return {
            "timestamp": datetime.now(),
            "diagnostic_result": {
                "overall_status": diagnostic_result.overall_status.value,
                "overall_score": diagnostic_result.overall_score,
                "components": {
                    name: {
                        "status": component.status.value,
                        "score": component.score,
                        "issues_count": len(component.issues)
                    }
                    for name, component in diagnostic_result.components.items()
                },
                "performance_metrics": {
                    "database_query_time": diagnostic_result.performance_metrics.database_query_time,
                    "memory_usage_mb": diagnostic_result.performance_metrics.memory_usage_mb,
                    "cpu_usage_percent": diagnostic_result.performance_metrics.cpu_usage_percent,
                    "disk_usage_mb": diagnostic_result.performance_metrics.disk_usage_mb
                }
            },
            "health_snapshot": {
                "overall_score": health_snapshot.overall_score if health_snapshot else 0,
                "component_scores": health_snapshot.component_scores if health_snapshot else {},
                "issue_count": health_snapshot.issue_count if health_snapshot else 0
            } if health_snapshot else None
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/performance")
async def get_rag_performance_metrics(
    days: int = Query(default=7, ge=1, le=30),
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get RAG system performance metrics"""
    try:
        # Get RAG performance from logs
        import sqlite3
        from src.utils.logging_config import comprehensive_logger
        
        conn = sqlite3.connect(comprehensive_logger.log_db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Get RAG query statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_queries,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                AVG(response_time_ms) as avg_response_time,
                MAX(response_time_ms) as max_response_time,
                MIN(response_time_ms) as min_response_time,
                COUNT(DISTINCT user_id) as unique_users
            FROM rag_events 
            WHERE event_type LIKE 'rag_query_%'
            AND timestamp >= ?
        ''', (since_date,))
        
        query_stats = cursor.fetchone()
        
        # Get RAG initialization statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_initializations,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_initializations
            FROM rag_events 
            WHERE event_type LIKE 'rag_initialization_%'
            AND timestamp >= ?
        ''', (since_date,))
        
        init_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            "period_days": days,
            "query_performance": {
                "total_queries": query_stats[0] or 0,
                "successful_queries": query_stats[1] or 0,
                "success_rate": (query_stats[1] / query_stats[0] * 100) if query_stats[0] > 0 else 0,
                "avg_response_time_ms": query_stats[2] or 0,
                "max_response_time_ms": query_stats[3] or 0,
                "min_response_time_ms": query_stats[4] or 0,
                "unique_users": query_stats[5] or 0
            },
            "initialization_performance": {
                "total_initializations": init_stats[0] or 0,
                "successful_initializations": init_stats[1] or 0,
                "success_rate": (init_stats[1] / init_stats[0] * 100) if init_stats[0] > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/active")
async def get_active_alerts(
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get active system alerts"""
    try:
        alerts = []
        
        # Check for recent security events
        security_events = comprehensive_logger.get_security_events(1)  # Last 24 hours
        
        for event_type, data in security_events["events_by_type"].items():
            if data["count"] > 10:  # Threshold for alert
                alerts.append({
                    "id": f"security_{event_type}",
                    "level": "warning",
                    "title": f"High {event_type.replace('_', ' ').title()} Activity",
                    "description": f"{data['count']} {event_type} events in the last 24 hours",
                    "timestamp": datetime.now(),
                    "component": "security"
                })
        
        # Check authentication failure rate
        auth_stats = comprehensive_logger.get_auth_statistics(1)
        if auth_stats["total_login_attempts"] > 0:
            failure_rate = (auth_stats["failed_logins"] / auth_stats["total_login_attempts"]) * 100
            if failure_rate > 20:  # More than 20% failure rate
                alerts.append({
                    "id": "auth_high_failure_rate",
                    "level": "warning" if failure_rate < 50 else "critical",
                    "title": "High Authentication Failure Rate",
                    "description": f"Authentication failure rate is {failure_rate:.1f}% in the last 24 hours",
                    "timestamp": datetime.now(),
                    "component": "authentication"
                })
        
        # Check file processing errors
        file_stats = comprehensive_logger.get_file_processing_statistics(1)
        for file_type, stats in file_stats["by_file_type"].items():
            if stats["success_rate"] < 80:  # Less than 80% success rate
                alerts.append({
                    "id": f"file_processing_{file_type}",
                    "level": "warning",
                    "title": f"Low {file_type.upper()} Processing Success Rate",
                    "description": f"{file_type.upper()} processing success rate is {stats['success_rate']:.1f}%",
                    "timestamp": datetime.now(),
                    "component": "file_processing"
                })
        
        return {
            "timestamp": datetime.now(),
            "alert_count": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/current")
async def get_current_performance_metrics(
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get current performance metrics and status"""
    try:
        from src.utils.performance_monitor import performance_monitor
        from src.utils.performance_optimizer import pdf_optimizer, rag_cache
        
        # Get current performance report
        performance_report = performance_monitor.get_performance_report()
        
        # Add PDF processing statistics
        recent_pdf_metrics = [m for m in pdf_optimizer.performance_metrics 
                            if (datetime.now() - m.timestamp).total_seconds() < 300]
        
        pdf_stats = {
            "recent_operations": len(recent_pdf_metrics),
            "avg_processing_time_ms": sum(m.duration_ms for m in recent_pdf_metrics) / len(recent_pdf_metrics) if recent_pdf_metrics else 0,
            "success_rate": (sum(1 for m in recent_pdf_metrics if m.success) / len(recent_pdf_metrics) * 100) if recent_pdf_metrics else 100,
            "cache_size": len(getattr(pdf_optimizer, 'processing_cache', {}))
        }
        
        # Add RAG cache statistics
        cache_stats = rag_cache.get_cache_statistics()
        
        performance_report.update({
            "pdf_processing": pdf_stats,
            "rag_cache": cache_stats
        })
        
        return performance_report
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/start-monitoring")
async def start_performance_monitoring(
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Start performance monitoring"""
    try:
        from src.utils.performance_monitor import performance_monitor
        
        performance_monitor.start_monitoring()
        
        return {
            "success": True,
            "message": "Performance monitoring started",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error starting performance monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/stop-monitoring")
async def stop_performance_monitoring(
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Stop performance monitoring"""
    try:
        from src.utils.performance_monitor import performance_monitor
        
        performance_monitor.stop_monitoring()
        
        return {
            "success": True,
            "message": "Performance monitoring stopped",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error stopping performance monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/alerts")
async def get_performance_alerts(
    active_only: bool = Query(default=True, description="Return only active alerts"),
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Get performance alerts"""
    try:
        from src.utils.performance_monitor import performance_monitor
        
        alerts = performance_monitor.active_alerts
        
        if active_only:
            alerts = [alert for alert in alerts if not alert.resolved]
        
        return {
            "timestamp": datetime.now(),
            "alert_count": len(alerts),
            "alerts": [
                {
                    "id": alert.id,
                    "type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "auto_resolved": alert.auto_resolved,
                    "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
                }
                for alert in sorted(alerts, key=lambda x: x.timestamp, reverse=True)
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/alerts/{alert_id}/resolve")
async def resolve_performance_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Manually resolve a performance alert"""
    try:
        from src.utils.performance_monitor import performance_monitor
        
        # Find and resolve alert
        alert_found = False
        for alert in performance_monitor.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                alert.auto_resolved = False
                
                # Update in database
                performance_monitor._update_alert_resolution(alert)
                alert_found = True
                break
        
        if not alert_found:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "success": True,
            "message": f"Alert {alert_id} resolved",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test/log-event")
async def test_log_event(
    event_type: str,
    level: str = "info",
    user_id: Optional[str] = None,
    message: Optional[str] = None,
    current_user: dict = Depends(get_current_user_for_monitoring)
):
    """Test endpoint for logging events (development/testing only)"""
    try:
        from src.utils.logging_config import LogEvent, EventType, LogLevel
        
        # Create test event
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType(event_type),
            level=LogLevel(level.upper()),
            user_id=user_id,
            error_message=message,
            details={"test_event": True}
        )
        
        comprehensive_logger.log_event(event)
        
        return {
            "success": True,
            "message": f"Test event logged: {event_type}",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error logging test event: {e}")
        raise HTTPException(status_code=500, detail=str(e))