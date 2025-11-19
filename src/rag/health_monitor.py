"""
Health Monitoring System for RAG System Reliability
Provides continuous health monitoring, trend analysis, and proactive alerting.
"""

import os
import sqlite3
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics

# Import diagnostic engine
try:
    from .diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentHealth, DiagnosticSeverity
except ImportError:
    from src.rag.diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentHealth, DiagnosticSeverity

logger = logging.getLogger(__name__)

class HealthTrend(Enum):
    """Health trend directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    CRITICAL_DECLINE = "critical_decline"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthSnapshot:
    """Point-in-time health snapshot"""
    timestamp: datetime
    overall_score: int
    component_scores: Dict[str, int]
    issue_count: int
    critical_issue_count: int
    performance_metrics: Dict[str, float]
    
@dataclass
class HealthTrendAnalysis:
    """Analysis of health trends over time"""
    component: str
    trend: HealthTrend
    score_change: float
    time_period: timedelta
    confidence: float  # 0-1
    prediction: Optional[str] = None

@dataclass
class HealthAlert:
    """Health monitoring alert"""
    id: str
    level: AlertLevel
    component: str
    title: str
    description: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringConfig:
    """Configuration for health monitoring"""
    check_interval_seconds: int = 300  # 5 minutes
    trend_analysis_window_hours: int = 24
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "overall_score": {"warning": 70, "critical": 50, "emergency": 30},
        "component_score": {"warning": 60, "critical": 40, "emergency": 20},
        "memory_usage": {"warning": 80, "critical": 90, "emergency": 95},
        "disk_usage": {"warning": 85, "critical": 95, "emergency": 98},
        "cpu_usage": {"warning": 80, "critical": 90, "emergency": 95}
    })
    enable_continuous_monitoring: bool = True
    max_history_days: int = 30
    enable_predictive_alerts: bool = True

class RAGHealthMonitor:
    """
    Continuous health monitoring system for RAG reliability
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None, 
                 health_db_path: str = "rag_health_monitoring.db"):
        self.config = config or MonitoringConfig()
        self.health_db_path = health_db_path
        self.monitoring_active = False
        self.monitoring_thread = None
        self.health_history: List[HealthSnapshot] = []
        self.active_alerts: List[HealthAlert] = []
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Initialize health monitoring database
        self._initialize_health_database()
        
        # Load existing history
        self._load_health_history()
    
    def _initialize_health_database(self):
        """Initialize the health monitoring database"""
        try:
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            # Health snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    overall_score INTEGER NOT NULL,
                    component_scores TEXT NOT NULL,
                    issue_count INTEGER NOT NULL,
                    critical_issue_count INTEGER NOT NULL,
                    performance_metrics TEXT NOT NULL
                )
            ''')
            
            # Health alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_alerts (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    acknowledged BOOLEAN DEFAULT 0,
                    resolved BOOLEAN DEFAULT 0,
                    resolution_time TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Trend analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trend_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    trend TEXT NOT NULL,
                    score_change REAL NOT NULL,
                    time_period_hours REAL NOT NULL,
                    confidence REAL NOT NULL,
                    prediction TEXT,
                    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Health monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing health monitoring database: {str(e)}")
    
    def start_continuous_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
        
        if not self.config.enable_continuous_monitoring:
            logger.info("Continuous monitoring is disabled in configuration")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started continuous health monitoring (interval: {self.config.check_interval_seconds}s)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped continuous health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run health check
                self.perform_health_check()
                
                # Sleep for the configured interval
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying on error
    
    def perform_health_check(self) -> HealthSnapshot:
        """
        Perform a single health check and update monitoring data
        
        Returns:
            HealthSnapshot with current health status
        """
        try:
            # Run comprehensive diagnostics
            diagnostic_result = diagnostic_engine.run_comprehensive_diagnostics()
            
            # Create health snapshot
            snapshot = HealthSnapshot(
                timestamp=datetime.now(),
                overall_score=diagnostic_result.overall_score,
                component_scores={name: comp.score for name, comp in diagnostic_result.components.items()},
                issue_count=len(diagnostic_result.issues),
                critical_issue_count=len([i for i in diagnostic_result.issues if i.severity == DiagnosticSeverity.CRITICAL]),
                performance_metrics={
                    "memory_usage_mb": diagnostic_result.performance_metrics.memory_usage_mb,
                    "cpu_usage_percent": diagnostic_result.performance_metrics.cpu_usage_percent,
                    "disk_usage_mb": diagnostic_result.performance_metrics.disk_usage_mb,
                    "database_query_time": diagnostic_result.performance_metrics.database_query_time,
                    "dependency_check_time": diagnostic_result.performance_metrics.dependency_check_time
                }
            )
            
            # Store snapshot
            self._store_health_snapshot(snapshot)
            self.health_history.append(snapshot)
            
            # Analyze trends
            trend_analysis = self.analyze_health_trends()
            
            # Check for alerts
            self._check_and_generate_alerts(snapshot, diagnostic_result, trend_analysis)
            
            # Clean old data
            self._cleanup_old_data()
            
            logger.debug(f"Health check completed - Score: {snapshot.overall_score}, Issues: {snapshot.issue_count}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            # Create error snapshot
            return HealthSnapshot(
                timestamp=datetime.now(),
                overall_score=0,
                component_scores={},
                issue_count=1,
                critical_issue_count=1,
                performance_metrics={}
            )
    
    def analyze_health_trends(self) -> List[HealthTrendAnalysis]:
        """
        Analyze health trends over the configured time window
        
        Returns:
            List of trend analyses for each component
        """
        if len(self.health_history) < 2:
            return []
        
        try:
            # Get snapshots within the trend analysis window
            cutoff_time = datetime.now() - timedelta(hours=self.config.trend_analysis_window_hours)
            recent_snapshots = [s for s in self.health_history if s.timestamp >= cutoff_time]
            
            if len(recent_snapshots) < 2:
                return []
            
            trend_analyses = []
            
            # Analyze overall score trend
            overall_trend = self._analyze_score_trend(
                "overall", 
                [s.overall_score for s in recent_snapshots],
                [s.timestamp for s in recent_snapshots]
            )
            if overall_trend:
                trend_analyses.append(overall_trend)
            
            # Analyze component trends
            all_components = set()
            for snapshot in recent_snapshots:
                all_components.update(snapshot.component_scores.keys())
            
            for component in all_components:
                scores = []
                timestamps = []
                for snapshot in recent_snapshots:
                    if component in snapshot.component_scores:
                        scores.append(snapshot.component_scores[component])
                        timestamps.append(snapshot.timestamp)
                
                if len(scores) >= 2:
                    component_trend = self._analyze_score_trend(component, scores, timestamps)
                    if component_trend:
                        trend_analyses.append(component_trend)
            
            # Store trend analysis
            self._store_trend_analysis(trend_analyses)
            
            return trend_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing health trends: {str(e)}")
            return []
    
    def _analyze_score_trend(self, component: str, scores: List[int], 
                           timestamps: List[datetime]) -> Optional[HealthTrendAnalysis]:
        """Analyze trend for a specific component's scores"""
        if len(scores) < 2:
            return None
        
        try:
            # Calculate score change
            score_change = scores[-1] - scores[0]
            time_period = timestamps[-1] - timestamps[0]
            
            # Calculate trend direction
            if len(scores) >= 3:
                # Use linear regression for better trend detection
                recent_scores = scores[-min(5, len(scores)):]  # Last 5 data points
                if len(recent_scores) >= 3:
                    # Simple slope calculation
                    x_values = list(range(len(recent_scores)))
                    slope = self._calculate_slope(x_values, recent_scores)
                    
                    if slope > 2:
                        trend = HealthTrend.IMPROVING
                    elif slope < -5:
                        trend = HealthTrend.CRITICAL_DECLINE
                    elif slope < -2:
                        trend = HealthTrend.DECLINING
                    else:
                        trend = HealthTrend.STABLE
                else:
                    # Fallback to simple comparison
                    if score_change > 5:
                        trend = HealthTrend.IMPROVING
                    elif score_change < -10:
                        trend = HealthTrend.CRITICAL_DECLINE
                    elif score_change < -5:
                        trend = HealthTrend.DECLINING
                    else:
                        trend = HealthTrend.STABLE
            else:
                # Simple two-point comparison
                if score_change > 5:
                    trend = HealthTrend.IMPROVING
                elif score_change < -10:
                    trend = HealthTrend.CRITICAL_DECLINE
                elif score_change < -5:
                    trend = HealthTrend.DECLINING
                else:
                    trend = HealthTrend.STABLE
            
            # Calculate confidence based on data points and consistency
            confidence = min(1.0, len(scores) / 10.0)  # More data points = higher confidence
            
            # Generate prediction if trend is concerning
            prediction = None
            if trend == HealthTrend.CRITICAL_DECLINE:
                prediction = f"Component {component} may fail within hours if trend continues"
            elif trend == HealthTrend.DECLINING:
                prediction = f"Component {component} performance may degrade significantly"
            
            return HealthTrendAnalysis(
                component=component,
                trend=trend,
                score_change=score_change,
                time_period=time_period,
                confidence=confidence,
                prediction=prediction
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {component}: {str(e)}")
            return None
    
    def _calculate_slope(self, x_values: List[int], y_values: List[int]) -> float:
        """Calculate slope using simple linear regression"""
        n = len(x_values)
        if n < 2:
            return 0
        
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _check_and_generate_alerts(self, snapshot: HealthSnapshot, 
                                 diagnostic_result: SystemDiagnosticResult,
                                 trend_analysis: List[HealthTrendAnalysis]):
        """Check conditions and generate alerts as needed"""
        try:
            new_alerts = []
            
            # Check overall score alerts
            overall_alerts = self._check_threshold_alerts(
                "overall_score", 
                snapshot.overall_score, 
                "Overall System Health"
            )
            new_alerts.extend(overall_alerts)
            
            # Check component score alerts
            for component, score in snapshot.component_scores.items():
                component_alerts = self._check_threshold_alerts(
                    "component_score",
                    score,
                    f"{component.title()} Component Health",
                    component
                )
                new_alerts.extend(component_alerts)
            
            # Check performance metric alerts
            perf_metrics = snapshot.performance_metrics
            if "memory_usage_mb" in perf_metrics:
                # Convert to percentage (assuming 8GB total for example)
                memory_percent = (perf_metrics["memory_usage_mb"] / (8 * 1024)) * 100
                memory_alerts = self._check_threshold_alerts(
                    "memory_usage",
                    memory_percent,
                    "Memory Usage",
                    "performance"
                )
                new_alerts.extend(memory_alerts)
            
            if "cpu_usage_percent" in perf_metrics:
                cpu_alerts = self._check_threshold_alerts(
                    "cpu_usage",
                    perf_metrics["cpu_usage_percent"],
                    "CPU Usage",
                    "performance"
                )
                new_alerts.extend(cpu_alerts)
            
            # Check trend-based alerts
            for trend in trend_analysis:
                if trend.trend == HealthTrend.CRITICAL_DECLINE:
                    alert = HealthAlert(
                        id=f"trend_critical_{trend.component}_{int(time.time())}",
                        level=AlertLevel.CRITICAL,
                        component=trend.component,
                        title=f"Critical Decline in {trend.component.title()}",
                        description=f"Component showing critical decline: {trend.score_change:.1f} point drop",
                        timestamp=datetime.now(),
                        metadata={
                            "trend": trend.trend.value,
                            "score_change": trend.score_change,
                            "confidence": trend.confidence,
                            "prediction": trend.prediction
                        }
                    )
                    new_alerts.append(alert)
                elif trend.trend == HealthTrend.DECLINING and trend.confidence > 0.7:
                    alert = HealthAlert(
                        id=f"trend_warning_{trend.component}_{int(time.time())}",
                        level=AlertLevel.WARNING,
                        component=trend.component,
                        title=f"Declining Performance in {trend.component.title()}",
                        description=f"Component performance declining: {trend.score_change:.1f} point drop",
                        timestamp=datetime.now(),
                        metadata={
                            "trend": trend.trend.value,
                            "score_change": trend.score_change,
                            "confidence": trend.confidence
                        }
                    )
                    new_alerts.append(alert)
            
            # Check for critical diagnostic issues
            for issue in diagnostic_result.issues:
                if issue.severity == DiagnosticSeverity.CRITICAL:
                    alert = HealthAlert(
                        id=f"diagnostic_critical_{issue.component}_{int(time.time())}",
                        level=AlertLevel.CRITICAL,
                        component=issue.component,
                        title=issue.title,
                        description=issue.description,
                        timestamp=datetime.now(),
                        metadata={
                            "recommendations": issue.recommendations,
                            "error_details": issue.error_details,
                            "auto_fixable": issue.auto_fixable
                        }
                    )
                    new_alerts.append(alert)
            
            # Store and process new alerts
            for alert in new_alerts:
                self._store_alert(alert)
                self.active_alerts.append(alert)
                self._trigger_alert_callbacks(alert)
            
            if new_alerts:
                logger.info(f"Generated {len(new_alerts)} new health alerts")
            
        except Exception as e:
            logger.error(f"Error checking and generating alerts: {str(e)}")
    
    def _check_threshold_alerts(self, metric_type: str, value: float, 
                              title_prefix: str, component: str = "system") -> List[HealthAlert]:
        """Check if a metric value crosses alert thresholds"""
        alerts = []
        
        if metric_type not in self.config.alert_thresholds:
            return alerts
        
        thresholds = self.config.alert_thresholds[metric_type]
        
        # Check emergency threshold
        if "emergency" in thresholds and value <= thresholds["emergency"]:
            alert = HealthAlert(
                id=f"{metric_type}_emergency_{component}_{int(time.time())}",
                level=AlertLevel.EMERGENCY,
                component=component,
                title=f"EMERGENCY: {title_prefix}",
                description=f"{title_prefix} at critical level: {value:.1f}",
                timestamp=datetime.now(),
                metadata={"metric_type": metric_type, "value": value, "threshold": thresholds["emergency"]}
            )
            alerts.append(alert)
        
        # Check critical threshold
        elif "critical" in thresholds and value <= thresholds["critical"]:
            alert = HealthAlert(
                id=f"{metric_type}_critical_{component}_{int(time.time())}",
                level=AlertLevel.CRITICAL,
                component=component,
                title=f"CRITICAL: {title_prefix}",
                description=f"{title_prefix} below critical threshold: {value:.1f}",
                timestamp=datetime.now(),
                metadata={"metric_type": metric_type, "value": value, "threshold": thresholds["critical"]}
            )
            alerts.append(alert)
        
        # Check warning threshold
        elif "warning" in thresholds and value <= thresholds["warning"]:
            alert = HealthAlert(
                id=f"{metric_type}_warning_{component}_{int(time.time())}",
                level=AlertLevel.WARNING,
                component=component,
                title=f"WARNING: {title_prefix}",
                description=f"{title_prefix} below warning threshold: {value:.1f}",
                timestamp=datetime.now(),
                metadata={"metric_type": metric_type, "value": value, "threshold": thresholds["warning"]}
            )
            alerts.append(alert)
        
        return alerts
    
    def get_health_status_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health status report
        
        Returns:
            Dictionary with current health status and trends
        """
        try:
            # Get latest snapshot
            latest_snapshot = self.health_history[-1] if self.health_history else None
            
            # Get recent trend analysis
            recent_trends = self.analyze_health_trends()
            
            # Get active alerts
            active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
            
            # Calculate health score trend
            if len(self.health_history) >= 2:
                score_trend = self.health_history[-1].overall_score - self.health_history[-2].overall_score
            else:
                score_trend = 0
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "current_health": {
                    "overall_score": latest_snapshot.overall_score if latest_snapshot else 0,
                    "component_scores": latest_snapshot.component_scores if latest_snapshot else {},
                    "issue_count": latest_snapshot.issue_count if latest_snapshot else 0,
                    "critical_issue_count": latest_snapshot.critical_issue_count if latest_snapshot else 0,
                    "score_trend": score_trend
                },
                "trends": {
                    "analysis_count": len(recent_trends),
                    "improving_components": [t.component for t in recent_trends if t.trend == HealthTrend.IMPROVING],
                    "declining_components": [t.component for t in recent_trends if t.trend in [HealthTrend.DECLINING, HealthTrend.CRITICAL_DECLINE]],
                    "stable_components": [t.component for t in recent_trends if t.trend == HealthTrend.STABLE]
                },
                "alerts": {
                    "active_count": len(active_alerts),
                    "emergency_count": len([a for a in active_alerts if a.level == AlertLevel.EMERGENCY]),
                    "critical_count": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                    "warning_count": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                    "recent_alerts": [
                        {
                            "id": alert.id,
                            "level": alert.level.value,
                            "component": alert.component,
                            "title": alert.title,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
                    ]
                },
                "monitoring_stats": {
                    "total_snapshots": len(self.health_history),
                    "monitoring_duration_hours": (
                        (self.health_history[-1].timestamp - self.health_history[0].timestamp).total_seconds() / 3600
                        if len(self.health_history) >= 2 else 0
                    ),
                    "check_interval_seconds": self.config.check_interval_seconds,
                    "last_check": latest_snapshot.timestamp.isoformat() if latest_snapshot else None
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health status report: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active
            }
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add a callback function to be called when new alerts are generated"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert_callbacks(self, alert: HealthAlert):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            # Update in memory
            for alert in self.active_alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    break
            
            # Update in database
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE health_alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Alert {alert_id} acknowledged")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        try:
            # Update in memory
            for alert in self.active_alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    break
            
            # Update in database
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE health_alerts SET resolved = 1, resolution_time = ? WHERE id = ?", 
                (datetime.now(), alert_id)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Alert {alert_id} resolved")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {str(e)}")
            return False
    
    # Database operations
    
    def _store_health_snapshot(self, snapshot: HealthSnapshot):
        """Store health snapshot in database"""
        try:
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_snapshots 
                (timestamp, overall_score, component_scores, issue_count, critical_issue_count, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp,
                snapshot.overall_score,
                json.dumps(snapshot.component_scores),
                snapshot.issue_count,
                snapshot.critical_issue_count,
                json.dumps(snapshot.performance_metrics)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing health snapshot: {str(e)}")
    
    def _store_alert(self, alert: HealthAlert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO health_alerts 
                (id, level, component, title, description, timestamp, acknowledged, resolved, resolution_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id,
                alert.level.value,
                alert.component,
                alert.title,
                alert.description,
                alert.timestamp,
                alert.acknowledged,
                alert.resolved,
                alert.resolution_time,
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
    
    def _store_trend_analysis(self, trend_analyses: List[HealthTrendAnalysis]):
        """Store trend analysis results in database"""
        try:
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            for trend in trend_analyses:
                cursor.execute('''
                    INSERT INTO trend_analysis 
                    (component, trend, score_change, time_period_hours, confidence, prediction)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    trend.component,
                    trend.trend.value,
                    trend.score_change,
                    trend.time_period.total_seconds() / 3600,
                    trend.confidence,
                    trend.prediction
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trend analysis: {str(e)}")
    
    def _load_health_history(self):
        """Load recent health history from database"""
        try:
            if not os.path.exists(self.health_db_path):
                return
            
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            # Load recent snapshots
            cutoff_time = datetime.now() - timedelta(days=self.config.max_history_days)
            cursor.execute('''
                SELECT timestamp, overall_score, component_scores, issue_count, critical_issue_count, performance_metrics
                FROM health_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp
            ''', (cutoff_time,))
            
            for row in cursor.fetchall():
                snapshot = HealthSnapshot(
                    timestamp=datetime.fromisoformat(row[0]),
                    overall_score=row[1],
                    component_scores=json.loads(row[2]),
                    issue_count=row[3],
                    critical_issue_count=row[4],
                    performance_metrics=json.loads(row[5])
                )
                self.health_history.append(snapshot)
            
            # Load active alerts
            cursor.execute('''
                SELECT id, level, component, title, description, timestamp, acknowledged, resolved, resolution_time, metadata
                FROM health_alerts 
                WHERE resolved = 0
                ORDER BY timestamp DESC
            ''')
            
            for row in cursor.fetchall():
                alert = HealthAlert(
                    id=row[0],
                    level=AlertLevel(row[1]),
                    component=row[2],
                    title=row[3],
                    description=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    acknowledged=bool(row[6]),
                    resolved=bool(row[7]),
                    resolution_time=datetime.fromisoformat(row[8]) if row[8] else None,
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                self.active_alerts.append(alert)
            
            conn.close()
            logger.info(f"Loaded {len(self.health_history)} health snapshots and {len(self.active_alerts)} active alerts")
            
        except Exception as e:
            logger.error(f"Error loading health history: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.config.max_history_days)
            
            # Clean in-memory history
            self.health_history = [s for s in self.health_history if s.timestamp >= cutoff_time]
            
            # Clean database
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM health_snapshots WHERE timestamp < ?", (cutoff_time,))
            cursor.execute("DELETE FROM trend_analysis WHERE analysis_time < ?", (cutoff_time,))
            cursor.execute("DELETE FROM health_alerts WHERE resolved = 1 AND resolution_time < ?", (cutoff_time,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")

# Global instance
health_monitor = RAGHealthMonitor()