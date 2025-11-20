"""
Health Monitoring API Endpoints
Provides real-time health monitoring and circuit breaker status endpoints
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime

from src.utils.health_monitor import live_health_monitor, ServiceStatus
from src.utils.error_handling import service_health_monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["Health Monitoring"])

# WebSocket connections for real-time health updates
health_websocket_connections: List[WebSocket] = []

@router.get("/status")
async def get_system_health_status():
    """Get overall system health status"""
    try:
        health_summary = live_health_monitor.get_system_health_summary()
        
        return JSONResponse(content={
            "success": True,
            "data": health_summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/services")
async def get_all_services_health():
    """Get health status for all monitored services"""
    try:
        services_health = live_health_monitor.get_all_services_health()
        
        # Convert to JSON-serializable format
        services_data = {}
        for service_name, health_info in services_health.items():
            services_data[service_name] = {
                "service_name": health_info.service_name,
                "status": health_info.status.value,
                "health_score": health_info.health_score,
                "last_check": health_info.last_check.isoformat(),
                "consecutive_failures": health_info.consecutive_failures,
                "consecutive_successes": health_info.consecutive_successes,
                "uptime_percentage": health_info.uptime_percentage,
                "average_response_time": health_info.average_response_time,
                "circuit_breaker_state": health_info.circuit_breaker_state.value,
                "recent_checks": [
                    {
                        "status": check.status.value,
                        "response_time_ms": check.response_time_ms,
                        "timestamp": check.timestamp.isoformat(),
                        "message": check.message,
                        "error": check.error
                    }
                    for check in health_info.recent_checks[-5:]  # Last 5 checks
                ],
                "alerts": health_info.alerts
            }
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "services": services_data,
                "total_services": len(services_data),
                "healthy_services": len([s for s in services_health.values() if s.status == ServiceStatus.HEALTHY]),
                "degraded_services": len([s for s in services_health.values() if s.status == ServiceStatus.DEGRADED]),
                "unhealthy_services": len([s for s in services_health.values() if s.status == ServiceStatus.UNHEALTHY])
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting services health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/services/{service_name}")
async def get_service_health(service_name: str):
    """Get detailed health information for a specific service"""
    try:
        health_info = live_health_monitor.get_service_health(service_name)
        
        if not health_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        # Get circuit breaker stats
        circuit_breaker_stats = {}
        if service_name in live_health_monitor.circuit_breakers:
            circuit_breaker_stats = live_health_monitor.circuit_breakers[service_name].get_stats()
        
        service_data = {
            "service_name": health_info.service_name,
            "status": health_info.status.value,
            "health_score": health_info.health_score,
            "last_check": health_info.last_check.isoformat(),
            "consecutive_failures": health_info.consecutive_failures,
            "consecutive_successes": health_info.consecutive_successes,
            "uptime_percentage": health_info.uptime_percentage,
            "average_response_time": health_info.average_response_time,
            "circuit_breaker": {
                "state": health_info.circuit_breaker_state.value,
                "stats": circuit_breaker_stats
            },
            "recent_checks": [
                {
                    "status": check.status.value,
                    "response_time_ms": check.response_time_ms,
                    "timestamp": check.timestamp.isoformat(),
                    "message": check.message,
                    "details": check.details,
                    "error": check.error
                }
                for check in health_info.recent_checks
            ],
            "alerts": health_info.alerts,
            "configuration": {
                "check_interval": live_health_monitor.health_checks[service_name].interval_seconds,
                "failure_threshold": live_health_monitor.health_checks[service_name].failure_threshold,
                "recovery_threshold": live_health_monitor.health_checks[service_name].recovery_threshold,
                "timeout_seconds": live_health_monitor.health_checks[service_name].timeout_seconds
            }
        }
        
        return JSONResponse(content={
            "success": True,
            "data": service_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service health for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/services/{service_name}/check")
async def force_health_check(service_name: str):
    """Force an immediate health check for a specific service"""
    try:
        result = await live_health_monitor.force_health_check(service_name)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        # Broadcast update to WebSocket connections
        await broadcast_health_update(service_name, result)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "service_name": result.service_name,
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "timestamp": result.timestamp.isoformat(),
                "message": result.message,
                "details": result.details,
                "error": result.error
            },
            "message": f"Health check completed for {service_name}",
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forcing health check for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    try:
        circuit_breaker_status = live_health_monitor.get_circuit_breaker_status()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "circuit_breakers": circuit_breaker_status,
                "total_breakers": len(circuit_breaker_status),
                "open_breakers": len([cb for cb in circuit_breaker_status.values() if cb.get('state') == 'open']),
                "half_open_breakers": len([cb for cb in circuit_breaker_status.values() if cb.get('state') == 'half_open']),
                "closed_breakers": len([cb for cb in circuit_breaker_status.values() if cb.get('state') == 'closed'])
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuit-breakers/{service_name}")
async def get_service_circuit_breaker(service_name: str):
    """Get circuit breaker status for a specific service"""
    try:
        if service_name not in live_health_monitor.circuit_breakers:
            raise HTTPException(status_code=404, detail=f"Circuit breaker for '{service_name}' not found")
        
        circuit_breaker = live_health_monitor.circuit_breakers[service_name]
        stats = circuit_breaker.get_stats()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "service_name": service_name,
                "circuit_breaker": stats,
                "recommendations": get_circuit_breaker_recommendations(stats)
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting circuit breaker for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_health_dashboard_data():
    """Get comprehensive health dashboard data"""
    try:
        # Ensure monitoring is started
        await live_health_monitor.start_background_monitoring()
        
        # Get system health summary
        system_health = live_health_monitor.get_system_health_summary()
        
        # Get circuit breaker status
        circuit_breakers = live_health_monitor.get_circuit_breaker_status()
        
        # Calculate additional metrics
        services_health = live_health_monitor.get_all_services_health()
        
        # Response time statistics
        response_times = []
        for health_info in services_health.values():
            if health_info.recent_checks:
                response_times.extend([check.response_time_ms for check in health_info.recent_checks])
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Service availability over time (simplified)
        availability_data = {}
        for service_name, health_info in services_health.items():
            availability_data[service_name] = {
                "uptime_percentage": health_info.uptime_percentage,
                "status": health_info.status.value,
                "health_score": health_info.health_score
            }
        
        # Convert services health to the format expected by frontend
        services_data = {}
        for service_name, health_info in services_health.items():
            services_data[service_name] = {
                "service_name": health_info.service_name,
                "status": health_info.status.value,
                "health_score": health_info.health_score,
                "last_check": health_info.last_check.isoformat(),
                "consecutive_failures": health_info.consecutive_failures,
                "consecutive_successes": health_info.consecutive_successes,
                "uptime_percentage": health_info.uptime_percentage,
                "average_response_time": health_info.average_response_time,
                "circuit_breaker_state": health_info.circuit_breaker_state.value,
                "recent_checks": [
                    {
                        "status": check.status.value,
                        "response_time_ms": check.response_time_ms,
                        "timestamp": check.timestamp.isoformat(),
                        "message": check.message,
                        "error": check.error
                    }
                    for check in health_info.recent_checks[-5:]  # Last 5 checks
                ],
                "alerts": health_info.alerts
            }
        
        dashboard_data = {
            "overall_status": system_health.get('overall_status', 'unknown'),
            "overall_health_score": system_health.get('overall_health_score', 0.0),
            "service_counts": system_health.get('service_counts', {
                'healthy': 0, 'degraded': 0, 'unhealthy': 0, 'unknown': 0
            }),
            "total_services": system_health.get('total_services', 0),
            "services": services_data,
            "circuit_breakers": circuit_breakers,
            "timestamp": system_health.get('timestamp', datetime.now().isoformat()),
            "performance_metrics": {
                "average_response_time_ms": round(avg_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2),
                "total_health_checks_today": len(response_times),  # Simplified
                "system_uptime_percentage": round(
                    sum(info.uptime_percentage for info in services_health.values()) / len(services_health) 
                    if services_health else 0, 2
                )
            },
            "circuit_breaker_summary": {
                "total_breakers": len(circuit_breakers),
                "open_breakers": len([cb for cb in circuit_breakers.values() if cb.get('state') == 'open']),
                "half_open_breakers": len([cb for cb in circuit_breakers.values() if cb.get('state') == 'half_open']),
                "recent_failures": sum(cb.get('failure_count', 0) for cb in circuit_breakers.values())
            },
            "service_availability": availability_data,
            "alerts": get_active_health_alerts(services_health),
            "recommendations": get_system_health_recommendations(system_health, services_health)
        }
        
        return JSONResponse(content={
            "success": True,
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting health dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/live")
async def health_monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time health monitoring updates"""
    await websocket.accept()
    health_websocket_connections.append(websocket)
    
    try:
        # Send initial health status
        initial_data = live_health_monitor.get_system_health_summary()
        await websocket.send_text(json.dumps({
            "type": "initial_health_status",
            "data": initial_data,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get("type") == "subscribe_service":
                    service_name = message.get("service_name")
                    if service_name:
                        # Send current service status
                        health_info = live_health_monitor.get_service_health(service_name)
                        if health_info:
                            await websocket.send_text(json.dumps({
                                "type": "service_status",
                                "service_name": service_name,
                                "data": {
                                    "status": health_info.status.value,
                                    "health_score": health_info.health_score,
                                    "uptime_percentage": health_info.uptime_percentage,
                                    "circuit_breaker_state": health_info.circuit_breaker_state.value
                                },
                                "timestamp": datetime.now().isoformat()
                            }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in health_websocket_connections:
            health_websocket_connections.remove(websocket)

# Helper functions
def get_circuit_breaker_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Get recommendations based on circuit breaker status"""
    recommendations = []
    
    state = stats.get('state', 'closed')
    failure_count = stats.get('failure_count', 0)
    
    if state == 'open':
        recommendations.append("Circuit breaker is open - service calls are being blocked")
        recommendations.append("Check service health and resolve underlying issues")
        recommendations.append("Monitor for automatic recovery")
    elif state == 'half_open':
        recommendations.append("Circuit breaker is testing service recovery")
        recommendations.append("Avoid manual service calls during recovery testing")
    elif failure_count > 0:
        recommendations.append(f"Service has {failure_count} recent failures")
        recommendations.append("Monitor service closely to prevent circuit breaker activation")
    else:
        recommendations.append("Circuit breaker is healthy")
    
    return recommendations

def get_active_health_alerts(services_health: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get active health alerts"""
    alerts = []
    
    for service_name, health_info in services_health.items():
        if health_info.status == ServiceStatus.UNHEALTHY:
            alerts.append({
                "type": "service_unhealthy",
                "service_name": service_name,
                "severity": "high",
                "message": f"Service {service_name} is unhealthy",
                "consecutive_failures": health_info.consecutive_failures,
                "timestamp": health_info.last_check.isoformat()
            })
        elif health_info.status == ServiceStatus.DEGRADED:
            alerts.append({
                "type": "service_degraded",
                "service_name": service_name,
                "severity": "medium",
                "message": f"Service {service_name} is degraded",
                "consecutive_failures": health_info.consecutive_failures,
                "timestamp": health_info.last_check.isoformat()
            })
        
        if health_info.uptime_percentage < 95:
            alerts.append({
                "type": "low_uptime",
                "service_name": service_name,
                "severity": "medium",
                "message": f"Service {service_name} has low uptime: {health_info.uptime_percentage:.1f}%",
                "uptime_percentage": health_info.uptime_percentage,
                "timestamp": health_info.last_check.isoformat()
            })
    
    return alerts

def get_system_health_recommendations(system_health: Dict[str, Any], services_health: Dict[str, Any]) -> List[str]:
    """Get system health recommendations"""
    recommendations = []
    
    overall_score = system_health.get('overall_health_score', 0)
    overall_status = system_health.get('overall_status', 'unknown')
    
    if overall_status == 'unhealthy':
        recommendations.append("System health is critical - immediate attention required")
        recommendations.append("Check unhealthy services and resolve issues")
    elif overall_status == 'degraded':
        recommendations.append("System health is degraded - monitor closely")
        recommendations.append("Address degraded services to prevent further issues")
    elif overall_score < 0.95:
        recommendations.append("System health is good but could be improved")
    
    # Check for specific issues
    unhealthy_services = [name for name, info in services_health.items() if info.status == ServiceStatus.UNHEALTHY]
    if unhealthy_services:
        recommendations.append(f"Unhealthy services need attention: {', '.join(unhealthy_services)}")
    
    # Check circuit breakers
    circuit_breakers = system_health.get('circuit_breakers', {})
    open_breakers = [name for name, stats in circuit_breakers.items() if stats.get('state') == 'open']
    if open_breakers:
        recommendations.append(f"Circuit breakers are open for: {', '.join(open_breakers)}")
    
    if not recommendations:
        recommendations.append("System health is excellent - no action required")
    
    return recommendations

async def broadcast_health_update(service_name: str, result: Any):
    """Broadcast health update to all WebSocket connections"""
    if not health_websocket_connections:
        return
    
    message = {
        "type": "health_update",
        "service_name": service_name,
        "data": {
            "status": result.status.value,
            "response_time_ms": result.response_time_ms,
            "message": result.message,
            "error": result.error
        },
        "timestamp": result.timestamp.isoformat()
    }
    
    # Send to all connected clients
    disconnected = []
    for websocket in health_websocket_connections:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error broadcasting health update: {e}")
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        if websocket in health_websocket_connections:
            health_websocket_connections.remove(websocket)

# Background task to periodically broadcast system health
async def periodic_health_broadcast():
    """Periodically broadcast system health to WebSocket connections"""
    while True:
        try:
            if health_websocket_connections:
                system_health = live_health_monitor.get_system_health_summary()
                
                message = {
                    "type": "system_health_update",
                    "data": system_health,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send to all connected clients
                disconnected = []
                for websocket in health_websocket_connections:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error broadcasting system health: {e}")
                        disconnected.append(websocket)
                
                # Remove disconnected clients
                for websocket in disconnected:
                    if websocket in health_websocket_connections:
                        health_websocket_connections.remove(websocket)
            
            # Wait 30 seconds before next broadcast
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in periodic health broadcast: {e}")
            await asyncio.sleep(30)