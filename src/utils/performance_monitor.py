"""
Performance Monitoring and Alerting System
Provides real-time performance monitoring, alerting, and automated responses
"""

import os
import time
import threading
import sqlite3
import logging
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

# Import existing monitoring systems
from src.utils.logging_config import comprehensive_logger
from src.utils.performance_optimizer import pdf_optimizer, rag_cache

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of performance alerts"""
    PDF_PROCESSING_SLOW = "pdf_processing_slow"
    RAG_QUERY_SLOW = "rag_query_slow"
    HIGH_MEMORY_USAGE = "high_memory_usage"
    HIGH_CPU_USAGE = "high_cpu_usage"
    LOW_DISK_SPACE = "low_disk_space"
    DATABASE_SLOW = "database_slow"
    CACHE_MISS_RATE_HIGH = "cache_miss_rate_high"
    ERROR_RATE_HIGH = "error_rate_high"
    SYSTEM_OVERLOAD = "system_overload"

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    check_interval_seconds: int = 60
    consecutive_violations_for_alert: int = 3

@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    auto_resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """Current system performance metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_connections: int
    pdf_processing_queue_size: int
    rag_query_response_time_avg: float
    cache_hit_rate_percent: float
    error_rate_percent: float
    database_query_time_avg: float

class PerformanceMonitor:
    """
    Real-time performance monitoring and alerting system
    """
    
    def __init__(self, monitoring_db_path: str = "performance_monitoring.db"):
        self.monitoring_db_path = monitoring_db_path
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.active_alerts: List[PerformanceAlert] = []
        self.metrics_history: List[SystemMetrics] = []
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": PerformanceThreshold("cpu_usage", 80.0, 95.0, "%"),
            "memory_usage": PerformanceThreshold("memory_usage", 85.0, 95.0, "%"),
            "disk_usage": PerformanceThreshold("disk_usage", 90.0, 98.0, "%"),
            "pdf_processing_time": PerformanceThreshold("pdf_processing_time", 30000.0, 60000.0, "ms"),
            "rag_query_time": PerformanceThreshold("rag_query_time", 5000.0, 15000.0, "ms"),
            "database_query_time": PerformanceThreshold("database_query_time", 1000.0, 5000.0, "ms"),
            "cache_hit_rate": PerformanceThreshold("cache_hit_rate", 70.0, 50.0, "%", consecutive_violations_for_alert=5),
            "error_rate": PerformanceThreshold("error_rate", 5.0, 15.0, "%")
        }
        
        # Violation counters for consecutive threshold violations
        self.violation_counters = {name: 0 for name in self.thresholds.keys()}
        
        self._init_monitoring_database()
    
    def _init_monitoring_database(self):
        """Initialize performance monitoring database"""
        try:
            conn = sqlite3.connect(self.monitoring_db_path)
            cursor = conn.cursor()
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage_percent REAL,
                    memory_usage_percent REAL,
                    disk_usage_percent REAL,
                    active_connections INTEGER,
                    pdf_processing_queue_size INTEGER,
                    rag_query_response_time_avg REAL,
                    cache_hit_rate_percent REAL,
                    error_rate_percent REAL,
                    database_query_time_avg REAL
                )
            ''')
            
            # Performance alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    current_value REAL,
                    threshold_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    resolution_time TIMESTAMP,
                    auto_resolved BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            ''')
            
            # Performance trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    trend_direction TEXT NOT NULL,
                    trend_strength REAL,
                    analysis_period_hours INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Performance monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing performance monitoring database: {str(e)}")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                self.metrics_history.append(metrics)
                
                # Check thresholds and generate alerts
                self._check_performance_thresholds(metrics)
                
                # Analyze trends
                if len(self.metrics_history) > 10:
                    self._analyze_performance_trends()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # System resource metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('.')
            
            # Application-specific metrics
            pdf_queue_size = len(getattr(pdf_optimizer, 'processing_queue', []))
            
            # RAG query performance
            rag_stats = rag_cache.get_cache_statistics()
            cache_hit_rate = rag_stats.get("hit_rate_percent", 0)
            
            # Calculate average response times from recent metrics
            recent_pdf_metrics = [m for m in pdf_optimizer.performance_metrics 
                                if (datetime.now() - m.timestamp).total_seconds() < 300]  # Last 5 minutes
            
            avg_pdf_time = 0
            if recent_pdf_metrics:
                avg_pdf_time = sum(m.duration_ms for m in recent_pdf_metrics) / len(recent_pdf_metrics)
            
            # Get database performance from logs
            db_query_time = self._get_average_database_query_time()
            
            # Calculate error rate
            error_rate = self._calculate_error_rate()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_info.percent,
                disk_usage_percent=(disk_info.used / disk_info.total) * 100,
                active_connections=0,  # Would need to implement connection tracking
                pdf_processing_queue_size=pdf_queue_size,
                rag_query_response_time_avg=rag_stats.get("avg_response_time_ms", 0),
                cache_hit_rate_percent=cache_hit_rate,
                error_rate_percent=error_rate,
                database_query_time_avg=db_query_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=0,
                memory_usage_percent=0,
                disk_usage_percent=0,
                active_connections=0,
                pdf_processing_queue_size=0,
                rag_query_response_time_avg=0,
                cache_hit_rate_percent=0,
                error_rate_percent=0,
                database_query_time_avg=0
            )
    
    def _get_average_database_query_time(self) -> float:
        """Get average database query time from recent operations"""
        try:
            # This would integrate with the logging system to get recent DB performance
            # For now, return a placeholder
            return 100.0  # ms
        except Exception:
            return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent operations"""
        try:
            # Get recent metrics from PDF optimizer
            recent_metrics = [m for m in pdf_optimizer.performance_metrics 
                            if (datetime.now() - m.timestamp).total_seconds() < 300]
            
            if not recent_metrics:
                return 0.0
            
            error_count = sum(1 for m in recent_metrics if not m.success)
            return (error_count / len(recent_metrics)) * 100
            
        except Exception:
            return 0.0
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in database"""
        try:
            conn = sqlite3.connect(self.monitoring_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, cpu_usage_percent, memory_usage_percent, disk_usage_percent,
                 active_connections, pdf_processing_queue_size, rag_query_response_time_avg,
                 cache_hit_rate_percent, error_rate_percent, database_query_time_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.cpu_usage_percent, metrics.memory_usage_percent,
                metrics.disk_usage_percent, metrics.active_connections, 
                metrics.pdf_processing_queue_size, metrics.rag_query_response_time_avg,
                metrics.cache_hit_rate_percent, metrics.error_rate_percent,
                metrics.database_query_time_avg
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metrics: {str(e)}")
    
    def _check_performance_thresholds(self, metrics: SystemMetrics):
        """Check performance thresholds and generate alerts"""
        try:
            # Check CPU usage
            self._check_threshold("cpu_usage", metrics.cpu_usage_percent, 
                                AlertType.HIGH_CPU_USAGE, "High CPU Usage")
            
            # Check memory usage
            self._check_threshold("memory_usage", metrics.memory_usage_percent,
                                AlertType.HIGH_MEMORY_USAGE, "High Memory Usage")
            
            # Check disk usage
            self._check_threshold("disk_usage", metrics.disk_usage_percent,
                                AlertType.LOW_DISK_SPACE, "Low Disk Space")
            
            # Check RAG query performance
            self._check_threshold("rag_query_time", metrics.rag_query_response_time_avg,
                                AlertType.RAG_QUERY_SLOW, "Slow RAG Query Performance")
            
            # Check database performance
            self._check_threshold("database_query_time", metrics.database_query_time_avg,
                                AlertType.DATABASE_SLOW, "Slow Database Performance")
            
            # Check cache performance (inverted - lower is worse)
            self._check_threshold_inverted("cache_hit_rate", metrics.cache_hit_rate_percent,
                                         AlertType.CACHE_MISS_RATE_HIGH, "Low Cache Hit Rate")
            
            # Check error rate
            self._check_threshold("error_rate", metrics.error_rate_percent,
                                AlertType.ERROR_RATE_HIGH, "High Error Rate")
            
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {str(e)}")
    
    def _check_threshold(self, threshold_name: str, current_value: float, 
                        alert_type: AlertType, alert_title: str):
        """Check a performance threshold"""
        threshold = self.thresholds[threshold_name]
        
        # Skip if current value is None or invalid
        if current_value is None or not isinstance(current_value, (int, float)):
            return
        
        # Check if threshold is violated
        if current_value >= threshold.critical_threshold:
            self.violation_counters[threshold_name] += 1
            
            if self.violation_counters[threshold_name] >= threshold.consecutive_violations_for_alert:
                self._generate_alert(
                    alert_type, AlertSeverity.CRITICAL, alert_title,
                    f"{alert_title}: {current_value:.1f}{threshold.unit} (Critical threshold: {threshold.critical_threshold}{threshold.unit})",
                    current_value, threshold.critical_threshold
                )
        
        elif current_value >= threshold.warning_threshold:
            self.violation_counters[threshold_name] += 1
            
            if self.violation_counters[threshold_name] >= threshold.consecutive_violations_for_alert:
                self._generate_alert(
                    alert_type, AlertSeverity.HIGH, alert_title,
                    f"{alert_title}: {current_value:.1f}{threshold.unit} (Warning threshold: {threshold.warning_threshold}{threshold.unit})",
                    current_value, threshold.warning_threshold
                )
        
        else:
            # Reset violation counter if threshold is not violated
            self.violation_counters[threshold_name] = 0
            
            # Auto-resolve related alerts
            self._auto_resolve_alerts(alert_type)
    
    def _check_threshold_inverted(self, threshold_name: str, current_value: float,
                                alert_type: AlertType, alert_title: str):
        """Check an inverted threshold (lower values are worse)"""
        threshold = self.thresholds[threshold_name]
        
        # Skip if current value is None or invalid
        if current_value is None or not isinstance(current_value, (int, float)):
            return
        
        # For inverted thresholds, critical is lower than warning
        if current_value <= threshold.critical_threshold:
            self.violation_counters[threshold_name] += 1
            
            if self.violation_counters[threshold_name] >= threshold.consecutive_violations_for_alert:
                self._generate_alert(
                    alert_type, AlertSeverity.CRITICAL, alert_title,
                    f"{alert_title}: {current_value:.1f}{threshold.unit} (Critical threshold: {threshold.critical_threshold}{threshold.unit})",
                    current_value, threshold.critical_threshold
                )
        
        elif current_value <= threshold.warning_threshold:
            self.violation_counters[threshold_name] += 1
            
            if self.violation_counters[threshold_name] >= threshold.consecutive_violations_for_alert:
                self._generate_alert(
                    alert_type, AlertSeverity.HIGH, alert_title,
                    f"{alert_title}: {current_value:.1f}{threshold.unit} (Warning threshold: {threshold.warning_threshold}{threshold.unit})",
                    current_value, threshold.warning_threshold
                )
        
        else:
            # Reset violation counter
            self.violation_counters[threshold_name] = 0
            
            # Auto-resolve related alerts
            self._auto_resolve_alerts(alert_type)
    
    def _generate_alert(self, alert_type: AlertType, severity: AlertSeverity,
                       title: str, description: str, current_value: float,
                       threshold_value: float):
        """Generate a performance alert"""
        try:
            # Check if similar alert already exists
            existing_alert = None
            for alert in self.active_alerts:
                if alert.alert_type == alert_type and not alert.resolved:
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.current_value = current_value
                existing_alert.timestamp = datetime.now()
                existing_alert.description = description
                return
            
            # Create new alert
            alert_id = f"{alert_type.value}_{int(time.time())}"
            
            alert = PerformanceAlert(
                id=alert_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                description=description,
                current_value=current_value,
                threshold_value=threshold_value,
                timestamp=datetime.now(),
                metadata={
                    "consecutive_violations": self.violation_counters.get(alert_type.value, 0)
                }
            )
            
            # Store alert
            self._store_alert(alert)
            self.active_alerts.append(alert)
            
            # Trigger callbacks
            self._trigger_alert_callbacks(alert)
            
            logger.warning(f"Generated performance alert: {title} - {description}")
            
        except Exception as e:
            logger.error(f"Error generating alert: {str(e)}")
    
    def _auto_resolve_alerts(self, alert_type: AlertType):
        """Auto-resolve alerts when conditions improve"""
        try:
            for alert in self.active_alerts:
                if alert.alert_type == alert_type and not alert.resolved:
                    alert.resolved = True
                    alert.auto_resolved = True
                    alert.resolution_time = datetime.now()
                    
                    # Update in database
                    self._update_alert_resolution(alert)
                    
                    logger.info(f"Auto-resolved alert: {alert.title}")
                    
        except Exception as e:
            logger.error(f"Error auto-resolving alerts: {str(e)}")
    
    def _analyze_performance_trends(self):
        """Analyze performance trends over time"""
        try:
            if len(self.metrics_history) < 10:
                return
            
            # Analyze trends for key metrics
            recent_metrics = self.metrics_history[-10:]  # Last 10 data points
            
            trends = {}
            
            # CPU usage trend
            cpu_values = [m.cpu_usage_percent for m in recent_metrics]
            trends["cpu_usage"] = self._calculate_trend(cpu_values)
            
            # Memory usage trend
            memory_values = [m.memory_usage_percent for m in recent_metrics]
            trends["memory_usage"] = self._calculate_trend(memory_values)
            
            # RAG query time trend
            rag_values = [m.rag_query_response_time_avg for m in recent_metrics]
            trends["rag_query_time"] = self._calculate_trend(rag_values)
            
            # Generate predictive alerts for concerning trends
            for metric, trend in trends.items():
                if trend["direction"] == "increasing" and trend["strength"] > 0.7:
                    if metric in ["cpu_usage", "memory_usage", "rag_query_time"]:
                        self._generate_trend_alert(metric, trend)
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and strength"""
        if len(values) < 3:
            return {"direction": "stable", "strength": 0}
        
        # Simple linear regression
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return {"direction": "stable", "strength": 0}
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine direction and strength
        if abs(slope) < 0.1:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        strength = min(1.0, abs(slope) / max(values) if max(values) > 0 else 0)
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": slope
        }
    
    def _generate_trend_alert(self, metric: str, trend: Dict[str, Any]):
        """Generate alert for concerning performance trends"""
        try:
            alert_id = f"trend_{metric}_{int(time.time())}"
            
            alert = PerformanceAlert(
                id=alert_id,
                alert_type=AlertType.SYSTEM_OVERLOAD,
                severity=AlertSeverity.MEDIUM,
                title=f"Performance Trend Alert: {metric.replace('_', ' ').title()}",
                description=f"{metric.replace('_', ' ').title()} showing {trend['direction']} trend with strength {trend['strength']:.2f}",
                current_value=trend["slope"],
                threshold_value=0.5,
                timestamp=datetime.now(),
                metadata={
                    "trend_analysis": trend,
                    "metric": metric,
                    "predictive": True
                }
            )
            
            # Store and process alert
            self._store_alert(alert)
            self.active_alerts.append(alert)
            self._trigger_alert_callbacks(alert)
            
            logger.info(f"Generated trend alert for {metric}: {trend}")
            
        except Exception as e:
            logger.error(f"Error generating trend alert: {str(e)}")
    
    def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.monitoring_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO performance_alerts
                (id, alert_type, severity, title, description, current_value,
                 threshold_value, timestamp, resolved, resolution_time, 
                 auto_resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.alert_type.value, alert.severity.value,
                alert.title, alert.description, alert.current_value,
                alert.threshold_value, alert.timestamp, alert.resolved,
                alert.resolution_time, alert.auto_resolved,
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
    
    def _update_alert_resolution(self, alert: PerformanceAlert):
        """Update alert resolution in database"""
        try:
            conn = sqlite3.connect(self.monitoring_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE performance_alerts 
                SET resolved = ?, resolution_time = ?, auto_resolved = ?
                WHERE id = ?
            ''', (alert.resolved, alert.resolution_time, alert.auto_resolved, alert.id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating alert resolution: {str(e)}")
    
    def _trigger_alert_callbacks(self, alert: PerformanceAlert):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Keep only last 1000 metrics in memory
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
            
            # Clean database periodically
            if len(self.metrics_history) % 100 == 0:
                conn = sqlite3.connect(self.monitoring_db_path)
                cursor = conn.cursor()
                
                # Remove metrics older than 30 days
                cursor.execute('''
                    DELETE FROM system_metrics 
                    WHERE datetime(timestamp) < datetime('now', '-30 days')
                ''')
                
                # Remove resolved alerts older than 7 days
                cursor.execute('''
                    DELETE FROM performance_alerts 
                    WHERE resolved = 1 AND datetime(resolution_time) < datetime('now', '-7 days')
                ''')
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
            
            # Calculate performance trends
            trends = {}
            if len(self.metrics_history) >= 10:
                recent_metrics = self.metrics_history[-10:]
                
                cpu_values = [m.cpu_usage_percent for m in recent_metrics]
                memory_values = [m.memory_usage_percent for m in recent_metrics]
                
                trends = {
                    "cpu_usage": self._calculate_trend(cpu_values),
                    "memory_usage": self._calculate_trend(memory_values)
                }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "current_metrics": {
                    "cpu_usage_percent": latest_metrics.cpu_usage_percent if latest_metrics else 0,
                    "memory_usage_percent": latest_metrics.memory_usage_percent if latest_metrics else 0,
                    "disk_usage_percent": latest_metrics.disk_usage_percent if latest_metrics else 0,
                    "cache_hit_rate_percent": latest_metrics.cache_hit_rate_percent if latest_metrics else 0,
                    "error_rate_percent": latest_metrics.error_rate_percent if latest_metrics else 0
                },
                "alerts": {
                    "active_count": len(active_alerts),
                    "critical_count": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                    "high_count": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                    "recent_alerts": [
                        {
                            "id": alert.id,
                            "type": alert.alert_type.value,
                            "severity": alert.severity.value,
                            "title": alert.title,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
                    ]
                },
                "trends": trends,
                "cache_statistics": rag_cache.get_cache_statistics(),
                "thresholds": {
                    name: {
                        "warning": threshold.warning_threshold,
                        "critical": threshold.critical_threshold,
                        "unit": threshold.unit
                    }
                    for name, threshold in self.thresholds.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Email alert callback
def email_alert_callback(alert: PerformanceAlert):
    """Send email notification for critical alerts"""
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
        try:
            # This would be configured with actual email settings
            logger.info(f"Would send email alert: {alert.title}")
            # Actual email sending would be implemented here
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")

# Register default alert callback
performance_monitor.add_alert_callback(email_alert_callback)