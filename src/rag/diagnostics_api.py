"""
Diagnostics and Health Monitoring API for RAG System Reliability
Provides unified interface for diagnostics, health monitoring, and system status reporting.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import json

# Import diagnostic and monitoring components
try:
    from .diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentHealth, DiagnosticSeverity
    from .health_monitor import health_monitor, HealthAlert, AlertLevel, MonitoringConfig
except ImportError:
    from src.rag.diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentHealth, DiagnosticSeverity
    from src.rag.health_monitor import health_monitor, HealthAlert, AlertLevel, MonitoringConfig

logger = logging.getLogger(__name__)

class RAGDiagnosticsAPI:
    """
    Unified API for RAG system diagnostics and health monitoring
    """
    
    def __init__(self):
        self.diagnostic_engine = diagnostic_engine
        self.health_monitor = health_monitor
    
    # Diagnostic Operations
    
    def run_system_diagnostics(self) -> Dict[str, Any]:
        """
        Run comprehensive system diagnostics
        
        Returns:
            Dictionary with diagnostic results and recommendations
        """
        try:
            logger.info("Running comprehensive system diagnostics via API")
            
            # Run diagnostics
            result = self.diagnostic_engine.run_comprehensive_diagnostics()
            
            # Convert to API response format
            response = {
                "status": "success",
                "timestamp": result.timestamp.isoformat(),
                "diagnostic_duration": result.diagnostic_duration,
                "overall_status": result.overall_status.value,
                "overall_score": result.overall_score,
                "components": {
                    name: {
                        "status": comp.status.value,
                        "score": comp.score,
                        "issues": [
                            {
                                "severity": issue.severity.value,
                                "title": issue.title,
                                "description": issue.description,
                                "recommendations": issue.recommendations,
                                "auto_fixable": issue.auto_fixable,
                                "error_details": issue.error_details
                            }
                            for issue in comp.issues
                        ],
                        "metrics": comp.metrics,
                        "last_check": comp.last_check.isoformat()
                    }
                    for name, comp in result.components.items()
                },
                "performance_metrics": {
                    "database_query_time": result.performance_metrics.database_query_time,
                    "dependency_check_time": result.performance_metrics.dependency_check_time,
                    "memory_usage_mb": result.performance_metrics.memory_usage_mb,
                    "disk_usage_mb": result.performance_metrics.disk_usage_mb,
                    "cpu_usage_percent": result.performance_metrics.cpu_usage_percent,
                    "active_connections": result.performance_metrics.active_connections,
                    "vector_index_size": result.performance_metrics.vector_index_size,
                    "document_count": result.performance_metrics.document_count
                },
                "issues": [
                    {
                        "component": issue.component,
                        "severity": issue.severity.value,
                        "title": issue.title,
                        "description": issue.description,
                        "recommendations": issue.recommendations,
                        "auto_fixable": issue.auto_fixable,
                        "timestamp": issue.timestamp.isoformat()
                    }
                    for issue in result.issues
                ],
                "recommendations": result.recommendations
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error running system diagnostics: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """
        Get summary of latest diagnostic results
        
        Returns:
            Dictionary with diagnostic summary
        """
        try:
            summary = self.diagnostic_engine.get_diagnostic_summary()
            return {
                "status": "success",
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error getting diagnostic summary: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Get detailed health information for a specific component
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            Dictionary with component health details
        """
        try:
            # Run diagnostics to get latest component status
            result = self.diagnostic_engine.run_comprehensive_diagnostics()
            
            if component_name not in result.components:
                return {
                    "status": "error",
                    "error": f"Component '{component_name}' not found"
                }
            
            component = result.components[component_name]
            
            return {
                "status": "success",
                "component": {
                    "name": component_name,
                    "status": component.status.value,
                    "score": component.score,
                    "issues": [
                        {
                            "severity": issue.severity.value,
                            "title": issue.title,
                            "description": issue.description,
                            "recommendations": issue.recommendations,
                            "auto_fixable": issue.auto_fixable
                        }
                        for issue in component.issues
                    ],
                    "metrics": component.metrics,
                    "last_check": component.last_check.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting component health for {component_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Health Monitoring Operations
    
    def start_health_monitoring(self) -> Dict[str, Any]:
        """
        Start continuous health monitoring
        
        Returns:
            Dictionary with monitoring start status
        """
        try:
            self.health_monitor.start_continuous_monitoring()
            return {
                "status": "success",
                "message": "Health monitoring started",
                "monitoring_active": True,
                "check_interval_seconds": self.health_monitor.config.check_interval_seconds
            }
        except Exception as e:
            logger.error(f"Error starting health monitoring: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def stop_health_monitoring(self) -> Dict[str, Any]:
        """
        Stop continuous health monitoring
        
        Returns:
            Dictionary with monitoring stop status
        """
        try:
            self.health_monitor.stop_continuous_monitoring()
            return {
                "status": "success",
                "message": "Health monitoring stopped",
                "monitoring_active": False
            }
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status and monitoring information
        
        Returns:
            Dictionary with health status report
        """
        try:
            report = self.health_monitor.get_health_status_report()
            return {
                "status": "success",
                "health_report": report
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform a single health check
        
        Returns:
            Dictionary with health check results
        """
        try:
            snapshot = self.health_monitor.perform_health_check()
            
            return {
                "status": "success",
                "health_snapshot": {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "overall_score": snapshot.overall_score,
                    "component_scores": snapshot.component_scores,
                    "issue_count": snapshot.issue_count,
                    "critical_issue_count": snapshot.critical_issue_count,
                    "performance_metrics": snapshot.performance_metrics
                }
            }
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_health_trends(self) -> Dict[str, Any]:
        """
        Get health trend analysis
        
        Returns:
            Dictionary with trend analysis results
        """
        try:
            trends = self.health_monitor.analyze_health_trends()
            
            return {
                "status": "success",
                "trends": [
                    {
                        "component": trend.component,
                        "trend": trend.trend.value,
                        "score_change": trend.score_change,
                        "time_period_hours": trend.time_period.total_seconds() / 3600,
                        "confidence": trend.confidence,
                        "prediction": trend.prediction
                    }
                    for trend in trends
                ]
            }
        except Exception as e:
            logger.error(f"Error getting health trends: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Alert Management
    
    def get_active_alerts(self) -> Dict[str, Any]:
        """
        Get all active alerts
        
        Returns:
            Dictionary with active alerts
        """
        try:
            active_alerts = [alert for alert in self.health_monitor.active_alerts if not alert.resolved]
            
            return {
                "status": "success",
                "alerts": [
                    {
                        "id": alert.id,
                        "level": alert.level.value,
                        "component": alert.component,
                        "title": alert.title,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "acknowledged": alert.acknowledged,
                        "metadata": alert.metadata
                    }
                    for alert in active_alerts
                ]
            }
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            Dictionary with acknowledgment status
        """
        try:
            success = self.health_monitor.acknowledge_alert(alert_id)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Alert {alert_id} acknowledged"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Failed to acknowledge alert {alert_id}"
                }
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """
        Resolve an alert
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            Dictionary with resolution status
        """
        try:
            success = self.health_monitor.resolve_alert(alert_id)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Alert {alert_id} resolved"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Failed to resolve alert {alert_id}"
                }
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Configuration Management
    
    def update_monitoring_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update monitoring configuration
        
        Args:
            config_updates: Dictionary with configuration updates
            
        Returns:
            Dictionary with update status
        """
        try:
            # Update configuration
            if "check_interval_seconds" in config_updates:
                self.health_monitor.config.check_interval_seconds = config_updates["check_interval_seconds"]
            
            if "trend_analysis_window_hours" in config_updates:
                self.health_monitor.config.trend_analysis_window_hours = config_updates["trend_analysis_window_hours"]
            
            if "enable_continuous_monitoring" in config_updates:
                self.health_monitor.config.enable_continuous_monitoring = config_updates["enable_continuous_monitoring"]
            
            if "alert_thresholds" in config_updates:
                self.health_monitor.config.alert_thresholds.update(config_updates["alert_thresholds"])
            
            return {
                "status": "success",
                "message": "Monitoring configuration updated",
                "current_config": {
                    "check_interval_seconds": self.health_monitor.config.check_interval_seconds,
                    "trend_analysis_window_hours": self.health_monitor.config.trend_analysis_window_hours,
                    "enable_continuous_monitoring": self.health_monitor.config.enable_continuous_monitoring,
                    "alert_thresholds": self.health_monitor.config.alert_thresholds
                }
            }
        except Exception as e:
            logger.error(f"Error updating monitoring configuration: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get current monitoring configuration
        
        Returns:
            Dictionary with current configuration
        """
        try:
            return {
                "status": "success",
                "config": {
                    "check_interval_seconds": self.health_monitor.config.check_interval_seconds,
                    "trend_analysis_window_hours": self.health_monitor.config.trend_analysis_window_hours,
                    "enable_continuous_monitoring": self.health_monitor.config.enable_continuous_monitoring,
                    "alert_thresholds": self.health_monitor.config.alert_thresholds,
                    "max_history_days": self.health_monitor.config.max_history_days,
                    "enable_predictive_alerts": self.health_monitor.config.enable_predictive_alerts
                }
            }
        except Exception as e:
            logger.error(f"Error getting monitoring configuration: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Utility Methods
    
    def get_system_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive system overview combining diagnostics and monitoring
        
        Returns:
            Dictionary with complete system overview
        """
        try:
            # Get diagnostic results
            diagnostic_result = self.diagnostic_engine.run_comprehensive_diagnostics()
            
            # Get health status
            health_report = self.health_monitor.get_health_status_report()
            
            # Get active alerts
            active_alerts = [alert for alert in self.health_monitor.active_alerts if not alert.resolved]
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "system_overview": {
                    "overall_health": {
                        "status": diagnostic_result.overall_status.value,
                        "score": diagnostic_result.overall_score,
                        "trend": health_report.get("current_health", {}).get("score_trend", 0)
                    },
                    "components": {
                        name: {
                            "status": comp.status.value,
                            "score": comp.score,
                            "issue_count": len(comp.issues),
                            "critical_issues": len([i for i in comp.issues if i.severity == DiagnosticSeverity.CRITICAL])
                        }
                        for name, comp in diagnostic_result.components.items()
                    },
                    "monitoring": {
                        "active": health_report.get("monitoring_active", False),
                        "total_snapshots": health_report.get("monitoring_stats", {}).get("total_snapshots", 0),
                        "last_check": health_report.get("monitoring_stats", {}).get("last_check")
                    },
                    "alerts": {
                        "total_active": len(active_alerts),
                        "emergency": len([a for a in active_alerts if a.level == AlertLevel.EMERGENCY]),
                        "critical": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                        "warning": len([a for a in active_alerts if a.level == AlertLevel.WARNING])
                    },
                    "performance": {
                        "memory_usage_mb": diagnostic_result.performance_metrics.memory_usage_mb,
                        "cpu_usage_percent": diagnostic_result.performance_metrics.cpu_usage_percent,
                        "database_query_time": diagnostic_result.performance_metrics.database_query_time,
                        "document_count": diagnostic_result.performance_metrics.document_count
                    },
                    "recommendations": diagnostic_result.recommendations[:5]  # Top 5 recommendations
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def export_diagnostic_report(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Export comprehensive diagnostic report
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Dictionary with complete diagnostic report
        """
        try:
            # Get current diagnostics
            diagnostic_result = self.diagnostic_engine.run_comprehensive_diagnostics()
            
            # Get health status
            health_report = self.health_monitor.get_health_status_report()
            
            # Get trends
            trends = self.health_monitor.analyze_health_trends()
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "comprehensive_diagnostic_report",
                    "include_history": include_history
                },
                "current_diagnostics": {
                    "overall_status": diagnostic_result.overall_status.value,
                    "overall_score": diagnostic_result.overall_score,
                    "diagnostic_duration": diagnostic_result.diagnostic_duration,
                    "components": {
                        name: {
                            "status": comp.status.value,
                            "score": comp.score,
                            "issues": [
                                {
                                    "severity": issue.severity.value,
                                    "title": issue.title,
                                    "description": issue.description,
                                    "recommendations": issue.recommendations
                                }
                                for issue in comp.issues
                            ],
                            "metrics": comp.metrics
                        }
                        for name, comp in diagnostic_result.components.items()
                    },
                    "performance_metrics": asdict(diagnostic_result.performance_metrics),
                    "recommendations": diagnostic_result.recommendations
                },
                "health_monitoring": health_report,
                "trend_analysis": [
                    {
                        "component": trend.component,
                        "trend": trend.trend.value,
                        "score_change": trend.score_change,
                        "confidence": trend.confidence,
                        "prediction": trend.prediction
                    }
                    for trend in trends
                ]
            }
            
            if include_history:
                # Add historical data
                report["historical_data"] = {
                    "health_snapshots": [
                        {
                            "timestamp": snapshot.timestamp.isoformat(),
                            "overall_score": snapshot.overall_score,
                            "component_scores": snapshot.component_scores,
                            "issue_count": snapshot.issue_count
                        }
                        for snapshot in self.health_monitor.health_history[-50:]  # Last 50 snapshots
                    ],
                    "diagnostic_history": [
                        {
                            "timestamp": result.timestamp.isoformat(),
                            "overall_status": result.overall_status.value,
                            "overall_score": result.overall_score,
                            "issue_count": len(result.issues)
                        }
                        for result in self.diagnostic_engine.diagnostic_history[-20:]  # Last 20 diagnostics
                    ]
                }
            
            return {
                "status": "success",
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Error exporting diagnostic report: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global API instance
diagnostics_api = RAGDiagnosticsAPI()