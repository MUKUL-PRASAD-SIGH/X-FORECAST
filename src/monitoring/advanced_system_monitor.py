"""
Advanced System Configuration and Monitoring Framework
Centralized configuration management and comprehensive system observability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import json
import yaml
import os
from pathlib import Path
import psutil
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class SystemAlert:
    """System alert definition"""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Health check definition"""
    check_id: str
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetric:
    """System metric definition"""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class ConfigurationItem:
    """Configuration item definition"""
    key: str
    value: Any
    category: str
    description: str
    data_type: str
    default_value: Any
    validation_rules: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = "system"

class ConfigurationManager:
    """
    Centralized configuration management system
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configurations = {}
        self.watchers = {}
        self.validation_rules = {}
        
        # Default configurations
        self._initialize_default_configurations()
    
    def _initialize_default_configurations(self):
        """Initialize default system configurations"""
        
        default_configs = {
            # Forecasting configurations
            "forecasting.default_horizon": ConfigurationItem(
                key="forecasting.default_horizon",
                value=12,
                category="forecasting",
                description="Default forecast horizon in periods",
                data_type="int",
                default_value=12,
                validation_rules=["min:1", "max:52"]
            ),
            "forecasting.reconciliation_method": ConfigurationItem(
                key="forecasting.reconciliation_method",
                value="mint",
                category="forecasting",
                description="Default reconciliation method",
                data_type="str",
                default_value="mint",
                validation_rules=["in:mint,ols,wls,bottom_up,top_down"]
            ),
            "forecasting.confidence_level": ConfigurationItem(
                key="forecasting.confidence_level",
                value=0.95,
                category="forecasting",
                description="Confidence level for prediction intervals",
                data_type="float",
                default_value=0.95,
                validation_rules=["min:0.5", "max:0.99"]
            ),
            
            # FVA configurations
            "fva.alert_threshold": ConfigurationItem(
                key="fva.alert_threshold",
                value=0.7,
                category="fva",
                description="FVA score threshold for alerts",
                data_type="float",
                default_value=0.7,
                validation_rules=["min:0.0", "max:1.0"]
            ),
            "fva.analysis_window_days": ConfigurationItem(
                key="fva.analysis_window_days",
                value=30,
                category="fva",
                description="Analysis window for FVA calculations",
                data_type="int",
                default_value=30,
                validation_rules=["min:7", "max:365"]
            ),
            
            # FQI configurations
            "fqi.accuracy_weight": ConfigurationItem(
                key="fqi.accuracy_weight",
                value=0.4,
                category="fqi",
                description="Weight for accuracy component in FQI",
                data_type="float",
                default_value=0.4,
                validation_rules=["min:0.0", "max:1.0"]
            ),
            "fqi.bias_weight": ConfigurationItem(
                key="fqi.bias_weight",
                value=0.3,
                category="fqi",
                description="Weight for bias component in FQI",
                data_type="float",
                default_value=0.3,
                validation_rules=["min:0.0", "max:1.0"]
            ),
            "fqi.coverage_weight": ConfigurationItem(
                key="fqi.coverage_weight",
                value=0.2,
                category="fqi",
                description="Weight for coverage component in FQI",
                data_type="float",
                default_value=0.2,
                validation_rules=["min:0.0", "max:1.0"]
            ),
            "fqi.coherence_weight": ConfigurationItem(
                key="fqi.coherence_weight",
                value=0.1,
                category="fqi",
                description="Weight for coherence component in FQI",
                data_type="float",
                default_value=0.1,
                validation_rules=["min:0.0", "max:1.0"]
            ),
            
            # OTIF configurations
            "otif.time_tolerance_hours": ConfigurationItem(
                key="otif.time_tolerance_hours",
                value=24.0,
                category="otif",
                description="Time tolerance for on-time delivery",
                data_type="float",
                default_value=24.0,
                validation_rules=["min:0.0", "max:168.0"]
            ),
            "otif.quantity_tolerance_pct": ConfigurationItem(
                key="otif.quantity_tolerance_pct",
                value=0.02,
                category="otif",
                description="Quantity tolerance for in-full delivery",
                data_type="float",
                default_value=0.02,
                validation_rules=["min:0.0", "max:0.1"]
            ),
            
            # System configurations
            "system.max_concurrent_forecasts": ConfigurationItem(
                key="system.max_concurrent_forecasts",
                value=10,
                category="system",
                description="Maximum concurrent forecast executions",
                data_type="int",
                default_value=10,
                validation_rules=["min:1", "max:100"]
            ),
            "system.cache_ttl_minutes": ConfigurationItem(
                key="system.cache_ttl_minutes",
                value=60,
                category="system",
                description="Cache time-to-live in minutes",
                data_type="int",
                default_value=60,
                validation_rules=["min:1", "max:1440"]
            ),
            "system.log_level": ConfigurationItem(
                key="system.log_level",
                value="INFO",
                category="system",
                description="System logging level",
                data_type="str",
                default_value="INFO",
                validation_rules=["in:DEBUG,INFO,WARNING,ERROR,CRITICAL"]
            )
        }
        
        for key, config in default_configs.items():
            self.configurations[key] = config
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if key in self.configurations:
            return self.configurations[key].value
        return default
    
    def set_config(self, key: str, value: Any, updated_by: str = "system") -> bool:
        """Set configuration value with validation"""
        
        if key not in self.configurations:
            logger.warning(f"Configuration key not found: {key}")
            return False
        
        config_item = self.configurations[key]
        
        # Validate the new value
        if not self._validate_config_value(config_item, value):
            logger.error(f"Configuration validation failed for {key}: {value}")
            return False
        
        # Update configuration
        old_value = config_item.value
        config_item.value = value
        config_item.last_updated = datetime.utcnow()
        config_item.updated_by = updated_by
        
        # Notify watchers
        self._notify_config_watchers(key, old_value, value)
        
        # Persist configuration
        self._persist_configuration(key, config_item)
        
        logger.info(f"Configuration updated: {key} = {value}")
        return True
    
    def _validate_config_value(self, config_item: ConfigurationItem, value: Any) -> bool:
        """Validate configuration value against rules"""
        
        # Type validation
        if config_item.data_type == "int" and not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                return False
        elif config_item.data_type == "float" and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False
        elif config_item.data_type == "str" and not isinstance(value, str):
            value = str(value)
        
        # Rule validation
        for rule in config_item.validation_rules:
            if not self._apply_validation_rule(rule, value):
                return False
        
        return True
    
    def _apply_validation_rule(self, rule: str, value: Any) -> bool:
        """Apply a single validation rule"""
        
        if rule.startswith("min:"):
            min_val = float(rule.split(":")[1])
            return value >= min_val
        elif rule.startswith("max:"):
            max_val = float(rule.split(":")[1])
            return value <= max_val
        elif rule.startswith("in:"):
            allowed_values = rule.split(":")[1].split(",")
            return str(value) in allowed_values
        elif rule == "required":
            return value is not None and value != ""
        
        return True
    
    def watch_config(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Watch for configuration changes"""
        if key not in self.watchers:
            self.watchers[key] = []
        self.watchers[key].append(callback)
    
    def _notify_config_watchers(self, key: str, old_value: Any, new_value: Any):
        """Notify configuration watchers"""
        if key in self.watchers:
            for callback in self.watchers[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Configuration watcher callback failed: {e}")
    
    def _persist_configuration(self, key: str, config_item: ConfigurationItem):
        """Persist configuration to file"""
        config_file = self.config_dir / f"{config_item.category}.yaml"
        
        # Load existing configurations for this category
        category_configs = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                category_configs = yaml.safe_load(f) or {}
        
        # Update configuration
        category_configs[key] = {
            'value': config_item.value,
            'description': config_item.description,
            'data_type': config_item.data_type,
            'default_value': config_item.default_value,
            'validation_rules': config_item.validation_rules,
            'last_updated': config_item.last_updated.isoformat(),
            'updated_by': config_item.updated_by
        }
        
        # Save to file
        with open(config_file, 'w') as f:
            yaml.dump(category_configs, f, default_flow_style=False)
    
    def load_configurations(self):
        """Load configurations from files"""
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    category_configs = yaml.safe_load(f) or {}
                
                for key, config_data in category_configs.items():
                    config_item = ConfigurationItem(
                        key=key,
                        value=config_data['value'],
                        category=config_file.stem,
                        description=config_data.get('description', ''),
                        data_type=config_data.get('data_type', 'str'),
                        default_value=config_data.get('default_value'),
                        validation_rules=config_data.get('validation_rules', []),
                        last_updated=datetime.fromisoformat(config_data.get('last_updated', datetime.utcnow().isoformat())),
                        updated_by=config_data.get('updated_by', 'system')
                    )
                    
                    self.configurations[key] = config_item
                    
            except Exception as e:
                logger.error(f"Failed to load configuration file {config_file}: {e}")
    
    def get_configurations_by_category(self, category: str) -> Dict[str, ConfigurationItem]:
        """Get all configurations for a category"""
        return {
            key: config for key, config in self.configurations.items()
            if config.category == category
        }
    
    def export_configurations(self) -> Dict[str, Any]:
        """Export all configurations"""
        return {
            key: {
                'value': config.value,
                'category': config.category,
                'description': config.description,
                'data_type': config.data_type,
                'default_value': config.default_value,
                'validation_rules': config.validation_rules,
                'last_updated': config.last_updated.isoformat(),
                'updated_by': config.updated_by
            }
            for key, config in self.configurations.items()
        }

class MetricsCollector:
    """
    System metrics collection and aggregation
    """
    
    def __init__(self):
        self.metrics = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.prometheus_metrics = {}
        
        # Initialize Prometheus metrics
        self._initialize_prometheus_metrics()
        
        # Start metrics collection
        self.collection_thread = None
        self.collection_interval = 30  # seconds
        self.running = False
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        
        # System metrics
        self.prometheus_metrics['cpu_usage'] = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.prometheus_metrics['memory_usage'] = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        self.prometheus_metrics['disk_usage'] = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage'
        )
        
        # Application metrics
        self.prometheus_metrics['forecast_requests'] = Counter(
            'forecast_requests_total',
            'Total number of forecast requests',
            ['method', 'status']
        )
        
        self.prometheus_metrics['forecast_duration'] = Histogram(
            'forecast_duration_seconds',
            'Forecast generation duration',
            ['method']
        )
        
        self.prometheus_metrics['active_forecasts'] = Gauge(
            'active_forecasts_count',
            'Number of active forecast executions'
        )
        
        self.prometheus_metrics['fva_score'] = Gauge(
            'fva_score',
            'Current FVA score',
            ['user_id', 'product_id']
        )
        
        self.prometheus_metrics['fqi_score'] = Gauge(
            'fqi_score',
            'Current FQI score'
        )
        
        self.prometheus_metrics['otif_rate'] = Gauge(
            'otif_rate',
            'Current OTIF rate',
            ['location', 'customer']
        )
    
    def start_collection(self):
        """Start metrics collection"""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Metrics collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        timestamp = datetime.utcnow()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric('system.cpu_usage', cpu_percent, timestamp)
        self.prometheus_metrics['cpu_usage'].set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.record_metric('system.memory_usage', memory_percent, timestamp)
        self.prometheus_metrics['memory_usage'].set(memory_percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric('system.disk_usage', disk_percent, timestamp)
        self.prometheus_metrics['disk_usage'].set(disk_percent)
        
        # Network I/O
        network = psutil.net_io_counters()
        self.record_metric('system.network_bytes_sent', network.bytes_sent, timestamp)
        self.record_metric('system.network_bytes_recv', network.bytes_recv, timestamp)
    
    def record_metric(self, metric_name: str, value: float, 
                     timestamp: Optional[datetime] = None,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        metric = SystemMetric(
            metric_name=metric_name,
            metric_type=MetricType.GAUGE,
            value=value,
            timestamp=timestamp,
            labels=labels or {}
        )
        
        self.metrics[metric_name] = metric
        self.metric_history[metric_name].append(metric)
    
    def get_metric(self, metric_name: str) -> Optional[SystemMetric]:
        """Get current metric value"""
        return self.metrics.get(metric_name)
    
    def get_metric_history(self, metric_name: str, 
                          hours: int = 24) -> List[SystemMetric]:
        """Get metric history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            metric for metric in self.metric_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    def increment_counter(self, metric_name: str, 
                         labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if metric_name in self.prometheus_metrics:
            if labels:
                self.prometheus_metrics[metric_name].labels(**labels).inc()
            else:
                self.prometheus_metrics[metric_name].inc()
    
    def observe_histogram(self, metric_name: str, value: float,
                         labels: Optional[Dict[str, str]] = None):
        """Observe a histogram metric"""
        if metric_name in self.prometheus_metrics:
            if labels:
                self.prometheus_metrics[metric_name].labels(**labels).observe(value)
            else:
                self.prometheus_metrics[metric_name].observe(value)

class HealthMonitor:
    """
    System health monitoring and alerting
    """
    
    def __init__(self, config_manager: ConfigurationManager, 
                 metrics_collector: MetricsCollector):
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self.alerts = {}
        self.alert_handlers = []
        
        # Health check thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time_ms': 5000.0
        }
        
        # Initialize health checks
        self._initialize_health_checks()
    
    def _initialize_health_checks(self):
        """Initialize system health checks"""
        
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'database_connection': self._check_database_connection,
            'cache_connection': self._check_cache_connection,
            'api_endpoints': self._check_api_endpoints,
            'forecast_accuracy': self._check_forecast_accuracy,
            'fva_performance': self._check_fva_performance,
            'fqi_scores': self._check_fqi_scores,
            'otif_performance': self._check_otif_performance
        }
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        for check_name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check_function()
                response_time = (time.time() - start_time) * 1000
                
                health_check = HealthCheck(
                    check_id=f"{check_name}_{int(time.time())}",
                    component=check_name,
                    status=result.get('status', HealthStatus.UNKNOWN),
                    message=result.get('message', ''),
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    metadata=result.get('metadata', {})
                )
                
                results[check_name] = health_check
                
                # Generate alerts if needed
                if health_check.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
                    await self._generate_alert(health_check)
                
            except Exception as e:
                logger.error(f"Health check failed for {check_name}: {e}")
                
                results[check_name] = HealthCheck(
                    check_id=f"{check_name}_{int(time.time())}",
                    component=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0
                )
        
        return results
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        
        cpu_metric = self.metrics_collector.get_metric('system.cpu_usage')
        memory_metric = self.metrics_collector.get_metric('system.memory_usage')
        disk_metric = self.metrics_collector.get_metric('system.disk_usage')
        
        issues = []
        
        if cpu_metric and cpu_metric.value > self.thresholds['cpu_usage']:
            issues.append(f"High CPU usage: {cpu_metric.value:.1f}%")
        
        if memory_metric and memory_metric.value > self.thresholds['memory_usage']:
            issues.append(f"High memory usage: {memory_metric.value:.1f}%")
        
        if disk_metric and disk_metric.value > self.thresholds['disk_usage']:
            issues.append(f"High disk usage: {disk_metric.value:.1f}%")
        
        if issues:
            return {
                'status': HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY,
                'message': '; '.join(issues),
                'metadata': {
                    'cpu_usage': cpu_metric.value if cpu_metric else 0,
                    'memory_usage': memory_metric.value if memory_metric else 0,
                    'disk_usage': disk_metric.value if disk_metric else 0
                }
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'System resources within normal limits',
            'metadata': {
                'cpu_usage': cpu_metric.value if cpu_metric else 0,
                'memory_usage': memory_metric.value if memory_metric else 0,
                'disk_usage': disk_metric.value if disk_metric else 0
            }
        }
    
    async def _check_database_connection(self) -> Dict[str, Any]:
        """Check database connectivity"""
        # Mock database check
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Database connection healthy',
            'metadata': {'connection_pool_size': 10}
        }
    
    async def _check_cache_connection(self) -> Dict[str, Any]:
        """Check cache connectivity"""
        # Mock cache check
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Cache connection healthy',
            'metadata': {'cache_hit_rate': 0.85}
        }
    
    async def _check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoint health"""
        # Mock API check
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'All API endpoints responding',
            'metadata': {'average_response_time_ms': 150}
        }
    
    async def _check_forecast_accuracy(self) -> Dict[str, Any]:
        """Check forecast accuracy performance"""
        # Mock forecast accuracy check
        accuracy = 0.92
        
        if accuracy < 0.8:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Forecast accuracy below threshold: {accuracy:.2%}',
                'metadata': {'accuracy': accuracy}
            }
        elif accuracy < 0.85:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f'Forecast accuracy degraded: {accuracy:.2%}',
                'metadata': {'accuracy': accuracy}
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': f'Forecast accuracy healthy: {accuracy:.2%}',
            'metadata': {'accuracy': accuracy}
        }
    
    async def _check_fva_performance(self) -> Dict[str, Any]:
        """Check FVA performance"""
        # Mock FVA check
        fva_score = 0.78
        threshold = self.config_manager.get_config('fva.alert_threshold', 0.7)
        
        if fva_score < threshold:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f'FVA score below threshold: {fva_score:.2f}',
                'metadata': {'fva_score': fva_score, 'threshold': threshold}
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': f'FVA performance healthy: {fva_score:.2f}',
            'metadata': {'fva_score': fva_score}
        }
    
    async def _check_fqi_scores(self) -> Dict[str, Any]:
        """Check FQI scores"""
        # Mock FQI check
        fqi_score = 0.89
        
        if fqi_score < 0.7:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'FQI score critically low: {fqi_score:.2f}',
                'metadata': {'fqi_score': fqi_score}
            }
        elif fqi_score < 0.8:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f'FQI score degraded: {fqi_score:.2f}',
                'metadata': {'fqi_score': fqi_score}
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': f'FQI score healthy: {fqi_score:.2f}',
            'metadata': {'fqi_score': fqi_score}
        }
    
    async def _check_otif_performance(self) -> Dict[str, Any]:
        """Check OTIF performance"""
        # Mock OTIF check
        otif_rate = 0.94
        
        if otif_rate < 0.9:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f'OTIF rate below target: {otif_rate:.2%}',
                'metadata': {'otif_rate': otif_rate}
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': f'OTIF performance healthy: {otif_rate:.2%}',
            'metadata': {'otif_rate': otif_rate}
        }
    
    async def _generate_alert(self, health_check: HealthCheck):
        """Generate alert for health check failure"""
        
        severity = AlertSeverity.WARNING
        if health_check.status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.ERROR
        
        alert = SystemAlert(
            alert_id=f"alert_{int(time.time())}_{health_check.component}",
            severity=severity,
            component=health_check.component,
            message=health_check.message,
            timestamp=health_check.timestamp,
            metadata=health_check.metadata
        )
        
        self.alerts[alert.alert_id] = alert
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[SystemAlert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get active alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.utcnow()

class AdvancedSystemMonitor:
    """
    Advanced system monitoring and configuration management
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_manager = ConfigurationManager(config_dir)
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(self.config_manager, self.metrics_collector)
        
        # Load configurations
        self.config_manager.load_configurations()
        
        # Start Prometheus metrics server
        self.prometheus_port = 8001
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 60  # seconds
    
    async def start_monitoring(self):
        """Start system monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            
            # Start metrics collection
            self.metrics_collector.start_collection()
            
            # Start Prometheus metrics server
            start_http_server(self.prometheus_port)
            
            # Start health monitoring loop
            asyncio.create_task(self._health_monitoring_loop())
            
            logger.info("Advanced system monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        self.metrics_collector.stop_collection()
        logger.info("Advanced system monitoring stopped")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.monitoring_active:
            try:
                await self.health_monitor.run_health_checks()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get latest health checks
        # This would normally be stored and retrieved from the health monitor
        health_status = {
            'overall_status': 'healthy',
            'components': {
                'system_resources': 'healthy',
                'database_connection': 'healthy',
                'cache_connection': 'healthy',
                'api_endpoints': 'healthy',
                'forecast_accuracy': 'healthy',
                'fva_performance': 'healthy',
                'fqi_scores': 'healthy',
                'otif_performance': 'healthy'
            }
        }
        
        # Get system metrics
        cpu_metric = self.metrics_collector.get_metric('system.cpu_usage')
        memory_metric = self.metrics_collector.get_metric('system.memory_usage')
        disk_metric = self.metrics_collector.get_metric('system.disk_usage')
        
        system_metrics = {
            'cpu_usage': cpu_metric.value if cpu_metric else 0,
            'memory_usage': memory_metric.value if memory_metric else 0,
            'disk_usage': disk_metric.value if disk_metric else 0,
            'uptime_hours': (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600
        }
        
        # Get active alerts
        active_alerts = self.health_monitor.get_active_alerts()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'health_status': health_status,
            'system_metrics': system_metrics,
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            'monitoring_active': self.monitoring_active
        }
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        
        configs_by_category = {}
        for category in ['forecasting', 'fva', 'fqi', 'otif', 'system']:
            configs_by_category[category] = {
                key.split('.', 1)[1]: config.value
                for key, config in self.config_manager.get_configurations_by_category(category).items()
            }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'configurations': configs_by_category,
            'total_configurations': len(self.config_manager.configurations)
        }
    
    async def update_configuration(self, key: str, value: Any, updated_by: str = "api") -> bool:
        """Update system configuration"""
        return self.config_manager.set_config(key, value, updated_by)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary"""
        
        metrics_summary = {}
        
        for metric_name in ['system.cpu_usage', 'system.memory_usage', 'system.disk_usage']:
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            
            if history:
                values = [m.value for m in history]
                metrics_summary[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'data_points': len(values)
                }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'metrics': metrics_summary
        }