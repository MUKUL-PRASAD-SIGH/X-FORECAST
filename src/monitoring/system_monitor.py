"""
Comprehensive System Monitoring and Alerting for Cyberpunk AI Dashboard
Real-time performance metrics, health checks, and automated alerting
"""

import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"

@dataclass
class SystemMetric:
    """System metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: MetricType

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Health check result"""
    service_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time: float
    timestamp: datetime
    details: Dict[str, Any]

class MetricsCollector:
    """Collect system and application metrics"""
    
    def __init__(self):
        # Prometheus metrics
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('disk_usage_percent', 'Disk usage percentage')
        self.api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
        self.api_response_time = Histogram('api_response_time_seconds', 'API response time')
        self.active_connections = Gauge('active_connections', 'Active WebSocket connections')
        
        # Internal storage
        self.metrics_history: List[SystemMetric] = []
        self.max_history_size = 10000
    
    def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect comprehensive system metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = SystemMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                tags={"type": "system"},
                metric_type=MetricType.SYSTEM
            )
            metrics.append(cpu_metric)
            self.cpu_usage.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_metric = SystemMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                timestamp=timestamp,
                tags={"type": "system"},
                metric_type=MetricType.SYSTEM
            )
            metrics.append(memory_metric)
            self.memory_usage.set(memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = SystemMetric(
                name="disk_usage",
                value=disk_percent,
                unit="percent",
                timestamp=timestamp,
                tags={"type": "system"},
                metric_type=MetricType.SYSTEM
            )
            metrics.append(disk_metric)
            self.disk_usage.set(disk_percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_metric = SystemMetric(
                name="network_bytes_sent",
                value=network.bytes_sent,
                unit="bytes",
                timestamp=timestamp,
                tags={"type": "network"},
                metric_type=MetricType.SYSTEM
            )
            metrics.append(network_metric)
            
            # Store metrics
            self.metrics_history.extend(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return []
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[SystemMetric]:
        """Get historical data for a specific metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics_history
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        
        # Default alert rules
        self.setup_default_rules()
    
    def setup_default_rules(self):
        """Setup default alerting rules"""
        self.alert_rules = {
            "cpu_usage": {
                "threshold": 85.0,
                "operator": ">",
                "severity": AlertSeverity.WARNING,
                "description": "High CPU usage detected"
            },
            "memory_usage": {
                "threshold": 90.0,
                "operator": ">",
                "severity": AlertSeverity.ERROR,
                "description": "High memory usage detected"
            },
            "disk_usage": {
                "threshold": 85.0,
                "operator": ">",
                "severity": AlertSeverity.WARNING,
                "description": "High disk usage detected"
            },
            "api_response_time": {
                "threshold": 5.0,
                "operator": ">",
                "severity": AlertSeverity.WARNING,
                "description": "Slow API response time detected"
            }
        }
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel"""
        self.notification_channels.append(channel)
    
    def evaluate_metrics(self, metrics: List[SystemMetric]) -> List[Alert]:
        """Evaluate metrics against alert rules"""
        new_alerts = []
        
        for metric in metrics:
            if metric.name in self.alert_rules:
                rule = self.alert_rules[metric.name]
                
                # Check if alert condition is met
                if self._evaluate_condition(metric.value, rule["threshold"], rule["operator"]):
                    alert_id = f"{metric.name}_{int(time.time())}"
                    
                    # Check if similar alert is already active
                    existing_alert_key = f"active_{metric.name}"
                    if existing_alert_key not in self.active_alerts:
                        alert = Alert(
                            alert_id=alert_id,
                            severity=rule["severity"],
                            title=f"{metric.name.replace('_', ' ').title()} Alert",
                            description=rule["description"],
                            metric_name=metric.name,
                            current_value=metric.value,
                            threshold_value=rule["threshold"],
                            timestamp=datetime.now()
                        )
                        
                        self.active_alerts[existing_alert_key] = alert
                        new_alerts.append(alert)
                        
                        # Send notifications
                        asyncio.create_task(self._send_notifications(alert))
                
                else:
                    # Check if we should resolve an existing alert
                    existing_alert_key = f"active_{metric.name}"
                    if existing_alert_key in self.active_alerts:
                        alert = self.active_alerts[existing_alert_key]
                        alert.resolved = True
                        alert.resolved_at = datetime.now()
                        del self.active_alerts[existing_alert_key]
        
        return new_alerts
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        return False
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through all channels"""
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

class HealthChecker:
    """Perform health checks on system components"""
    
    def __init__(self):
        self.health_endpoints = {
            "api": "http://localhost:8000/api/v1/status",
            "database": "postgresql://localhost:5432",
            "redis": "redis://localhost:6379",
            "frontend": "http://localhost:3000"
        }
        self.health_history: List[HealthCheck] = []
    
    async def check_all_services(self) -> List[HealthCheck]:
        """Check health of all services"""
        health_checks = []
        
        for service_name, endpoint in self.health_endpoints.items():
            health_check = await self._check_service_health(service_name, endpoint)
            health_checks.append(health_check)
            self.health_history.append(health_check)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.health_history = [
            hc for hc in self.health_history if hc.timestamp >= cutoff_time
        ]
        
        return health_checks
    
    async def _check_service_health(self, service_name: str, endpoint: str) -> HealthCheck:
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            if endpoint.startswith("http"):
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            status = "healthy"
                            details = {"status_code": response.status}
                        else:
                            status = "degraded"
                            details = {"status_code": response.status, "error": "Non-200 response"}
            
            elif endpoint.startswith("postgresql"):
                # Mock database health check
                response_time = time.time() - start_time
                status = "healthy"
                details = {"connection": "ok"}
            
            elif endpoint.startswith("redis"):
                # Mock Redis health check
                response_time = time.time() - start_time
                status = "healthy"
                details = {"connection": "ok"}
            
            else:
                response_time = time.time() - start_time
                status = "unknown"
                details = {"error": "Unknown endpoint type"}
        
        except Exception as e:
            response_time = time.time() - start_time
            status = "unhealthy"
            details = {"error": str(e)}
        
        return HealthCheck(
            service_name=service_name,
            status=status,
            response_time=response_time,
            timestamp=datetime.now(),
            details=details
        )

class SystemMonitor:
    """Main system monitoring orchestrator"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        
        # Redis for storing monitoring data
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.alert_manager.redis_client = self.redis_client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        
        # Setup notification channels
        self.setup_notification_channels()
        
        # Start Prometheus metrics server
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    def setup_notification_channels(self):
        """Setup notification channels"""
        # Add console notification
        self.alert_manager.add_notification_channel(self._console_notification)
        
        # Add Redis notification (if available)
        if self.redis_client:
            self.alert_manager.add_notification_channel(self._redis_notification)
    
    async def start_monitoring(self):
        """Start continuous system monitoring"""
        self.monitoring_active = True
        logger.info("Starting system monitoring...")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_system_metrics()
                
                # Evaluate alerts
                new_alerts = self.alert_manager.evaluate_metrics(metrics)
                
                # Perform health checks
                health_checks = await self.health_checker.check_all_services()
                
                # Store monitoring data
                await self._store_monitoring_data(metrics, new_alerts, health_checks)
                
                # Log summary
                if new_alerts:
                    logger.info(f"Generated {len(new_alerts)} new alerts")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        logger.info("System monitoring stopped")
    
    async def _store_monitoring_data(self, metrics: List[SystemMetric], 
                                   alerts: List[Alert], health_checks: List[HealthCheck]):
        """Store monitoring data in Redis"""
        if not self.redis_client:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            
            # Store metrics
            for metric in metrics:
                key = f"metrics:{metric.name}:{timestamp}"
                self.redis_client.setex(key, 3600, json.dumps(asdict(metric), default=str))
            
            # Store alerts
            for alert in alerts:
                key = f"alerts:{alert.alert_id}"
                self.redis_client.setex(key, 86400, json.dumps(asdict(alert), default=str))
            
            # Store health checks
            for health_check in health_checks:
                key = f"health:{health_check.service_name}:{timestamp}"
                self.redis_client.setex(key, 3600, json.dumps(asdict(health_check), default=str))
            
            # Update monitoring summary
            summary = {
                "last_update": timestamp,
                "active_alerts": len(self.alert_manager.active_alerts),
                "healthy_services": len([hc for hc in health_checks if hc.status == "healthy"]),
                "total_services": len(health_checks)
            }
            self.redis_client.setex("monitoring:summary", 300, json.dumps(summary))
            
        except Exception as e:
            logger.error(f"Error storing monitoring data: {e}")
    
    async def _console_notification(self, alert: Alert):
        """Send alert notification to console"""
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.description}")
    
    async def _redis_notification(self, alert: Alert):
        """Send alert notification to Redis pub/sub"""
        if self.redis_client:
            try:
                alert_data = asdict(alert)
                self.redis_client.publish("alerts", json.dumps(alert_data, default=str))
            except Exception as e:
                logger.error(f"Error publishing alert to Redis: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get current monitoring summary"""
        try:
            # Get recent metrics
            recent_metrics = self.metrics_collector.metrics_history[-10:] if self.metrics_collector.metrics_history else []
            
            # Get active alerts
            active_alerts = list(self.alert_manager.active_alerts.values())
            
            # Get recent health checks
            recent_health = self.health_checker.health_history[-len(self.health_checker.health_endpoints):] if self.health_checker.health_history else []
            
            return {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "recent_metrics_count": len(recent_metrics),
                "active_alerts_count": len(active_alerts),
                "active_alerts": [asdict(alert) for alert in active_alerts],
                "service_health": {
                    hc.service_name: hc.status for hc in recent_health
                },
                "system_status": self._calculate_system_status(recent_metrics, active_alerts, recent_health)
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring summary: {e}")
            return {"error": str(e)}
    
    def _calculate_system_status(self, metrics: List[SystemMetric], 
                               alerts: List[Alert], health_checks: List[HealthCheck]) -> str:
        """Calculate overall system status"""
        if any(alert.severity == AlertSeverity.CRITICAL for alert in alerts):
            return "critical"
        elif any(alert.severity == AlertSeverity.ERROR for alert in alerts):
            return "error"
        elif any(hc.status == "unhealthy" for hc in health_checks):
            return "degraded"
        elif any(alert.severity == AlertSeverity.WARNING for alert in alerts):
            return "warning"
        else:
            return "healthy"

# Example usage and testing
if __name__ == "__main__":
    async def test_system_monitor():
        monitor = SystemMonitor()
        
        try:
            # Start monitoring for 30 seconds
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            
            # Let it run for a bit
            await asyncio.sleep(30)
            
            # Get summary
            summary = monitor.get_monitoring_summary()
            print("Monitoring Summary:")
            print(json.dumps(summary, indent=2, default=str))
            
        except KeyboardInterrupt:
            logger.info("Stopping monitoring...")
        finally:
            monitor.stop_monitoring()
    
    # Run test
    asyncio.run(test_system_monitor())