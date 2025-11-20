"""
Live System Health Monitoring and Circuit Breaker Protection
Real-time service health checker with live status indicators and automatic recovery detection
"""

import asyncio
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import psutil
from contextlib import asynccontextmanager

from .error_handling import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, service_health_monitor
from .performance_monitor import performance_monitor, AlertSeverity, AlertType

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheckType(Enum):
    """Types of health checks"""
    HTTP_ENDPOINT = "http_endpoint"
    DATABASE_CONNECTION = "database_connection"
    FILE_SYSTEM = "file_system"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_SPACE = "disk_space"
    CUSTOM = "custom"

@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    name: str
    check_type: HealthCheckType
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    status: ServiceStatus
    response_time_ms: float
    timestamp: datetime
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class ServiceHealthInfo:
    """Comprehensive service health information"""
    service_name: str
    status: ServiceStatus
    health_score: float  # 0.0 to 1.0
    last_check: datetime
    consecutive_failures: int
    consecutive_successes: int
    uptime_percentage: float
    average_response_time: float
    circuit_breaker_state: CircuitBreakerState
    recent_checks: List[HealthCheckResult] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

class LiveHealthMonitor:
    """
    Real-time system health monitoring with circuit breaker protection
    """
    
    def __init__(self, db_path: str = "health_monitoring.db"):
        self.db_path = db_path
        self.health_checks: Dict[str, HealthCheckConfig] = {}
        self.health_checkers: Dict[str, Callable] = {}
        self.service_health: Dict[str, ServiceHealthInfo] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.health_callbacks: List[Callable[[str, ServiceHealthInfo], None]] = []
        self.recovery_callbacks: List[Callable[[str, ServiceHealthInfo], None]] = []
        
        # Initialize database
        self._init_database()
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _init_database(self):
        """Initialize health monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Health check results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_check_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message TEXT,
                    details TEXT,
                    error TEXT
                )
            ''')
            
            # Service health summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS service_health_summary (
                    service_name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    health_score REAL,
                    last_check TIMESTAMP,
                    consecutive_failures INTEGER DEFAULT 0,
                    consecutive_successes INTEGER DEFAULT 0,
                    uptime_percentage REAL DEFAULT 100.0,
                    average_response_time REAL DEFAULT 0.0,
                    circuit_breaker_state TEXT DEFAULT 'closed',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Health events table for notifications
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Health monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing health monitoring database: {e}")
    
    def _register_default_health_checks(self):
        """Register default system health checks"""
        # System resource checks
        self.register_health_check(
            HealthCheckConfig(
                name="system_memory",
                check_type=HealthCheckType.MEMORY_USAGE,
                interval_seconds=30,
                metadata={"warning_threshold": 80, "critical_threshold": 95}
            ),
            self._check_memory_usage
        )
        
        self.register_health_check(
            HealthCheckConfig(
                name="system_cpu",
                check_type=HealthCheckType.CPU_USAGE,
                interval_seconds=30,
                metadata={"warning_threshold": 80, "critical_threshold": 95}
            ),
            self._check_cpu_usage
        )
        
        self.register_health_check(
            HealthCheckConfig(
                name="system_disk",
                check_type=HealthCheckType.DISK_SPACE,
                interval_seconds=60,
                metadata={"warning_threshold": 85, "critical_threshold": 95}
            ),
            self._check_disk_space
        )
        
        # Application-specific checks
        self.register_health_check(
            HealthCheckConfig(
                name="parameter_detection_service",
                check_type=HealthCheckType.CUSTOM,
                interval_seconds=30,
                failure_threshold=3,
                recovery_threshold=2
            ),
            self._check_parameter_detection_service
        )
        
        self.register_health_check(
            HealthCheckConfig(
                name="ensemble_initialization_service",
                check_type=HealthCheckType.CUSTOM,
                interval_seconds=45,
                failure_threshold=3,
                recovery_threshold=2
            ),
            self._check_ensemble_initialization_service
        )
        
        self.register_health_check(
            HealthCheckConfig(
                name="data_processing_service",
                check_type=HealthCheckType.CUSTOM,
                interval_seconds=30,
                failure_threshold=3,
                recovery_threshold=2
            ),
            self._check_data_processing_service
        )
    
    def register_health_check(self, config: HealthCheckConfig, checker: Callable) -> None:
        """Register a health check"""
        self.health_checks[config.name] = config
        self.health_checkers[config.name] = checker
        
        # Initialize service health info
        self.service_health[config.name] = ServiceHealthInfo(
            service_name=config.name,
            status=ServiceStatus.UNKNOWN,
            health_score=0.0,
            last_check=datetime.now(),
            consecutive_failures=0,
            consecutive_successes=0,
            uptime_percentage=100.0,
            average_response_time=0.0,
            circuit_breaker_state=CircuitBreakerState.CLOSED
        )
        
        # Create circuit breaker for the service
        cb_config = CircuitBreakerConfig(
            name=config.name,
            failure_threshold=config.failure_threshold,
            recovery_timeout=60.0
        )
        self.circuit_breakers[config.name] = CircuitBreaker(cb_config)
        
        # Register with global service health monitor
        service_health_monitor.register_service(
            config.name,
            lambda: self._get_service_health_status(config.name),
            cb_config
        )
        
        logger.info(f"Registered health check for {config.name}")
    
    async def start_monitoring(self):
        """Start live health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks for each service
        for service_name, config in self.health_checks.items():
            if config.enabled:
                task = asyncio.create_task(self._monitor_service(service_name))
                self.monitoring_tasks.append(task)
        
        # Start system health aggregation task
        aggregation_task = asyncio.create_task(self._aggregate_system_health())
        self.monitoring_tasks.append(aggregation_task)
        
        logger.info(f"Started health monitoring for {len(self.health_checks)} services")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Stopped health monitoring")
    
    async def _monitor_service(self, service_name: str):
        """Monitor a specific service"""
        config = self.health_checks[service_name]
        checker = self.health_checkers[service_name]
        
        while self.monitoring_active:
            try:
                # Perform health check
                start_time = time.time()
                
                try:
                    # Execute health check with timeout
                    result = await asyncio.wait_for(
                        self._execute_health_check(service_name, checker),
                        timeout=config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    result = HealthCheckResult(
                        service_name=service_name,
                        status=ServiceStatus.UNHEALTHY,
                        response_time_ms=(time.time() - start_time) * 1000,
                        timestamp=datetime.now(),
                        error="Health check timed out"
                    )
                
                # Update service health
                await self._update_service_health(service_name, result)
                
                # Store result in database
                self._store_health_check_result(result)
                
                # Wait for next check
                await asyncio.sleep(config.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring service {service_name}: {e}")
                await asyncio.sleep(config.interval_seconds)
    
    async def _execute_health_check(self, service_name: str, checker: Callable) -> HealthCheckResult:
        """Execute a health check function"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(checker):
                result = await checker()
            else:
                result = checker()
            
            response_time = (time.time() - start_time) * 1000
            
            # Handle different result types
            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, bool):
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.HEALTHY if result else ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message="Health check completed"
                )
            elif isinstance(result, dict):
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.HEALTHY if result.get('healthy', False) else ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message=result.get('message', 'Health check completed'),
                    details=result
                )
            else:
                return HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message=str(result)
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _update_service_health(self, service_name: str, result: HealthCheckResult):
        """Update service health information"""
        health_info = self.service_health[service_name]
        old_status = health_info.status
        
        # Update basic info
        health_info.last_check = result.timestamp
        
        # Update consecutive counters
        if result.status == ServiceStatus.HEALTHY:
            health_info.consecutive_successes += 1
            health_info.consecutive_failures = 0
        else:
            health_info.consecutive_failures += 1
            health_info.consecutive_successes = 0
        
        # Determine new status based on thresholds
        config = self.health_checks[service_name]
        
        if result.status == ServiceStatus.HEALTHY and health_info.consecutive_successes >= config.recovery_threshold:
            health_info.status = ServiceStatus.HEALTHY
        elif result.status != ServiceStatus.HEALTHY and health_info.consecutive_failures >= config.failure_threshold:
            health_info.status = ServiceStatus.UNHEALTHY
        elif health_info.consecutive_failures > 0 and health_info.consecutive_failures < config.failure_threshold:
            health_info.status = ServiceStatus.DEGRADED
        
        # Update health score (0.0 to 1.0)
        if health_info.status == ServiceStatus.HEALTHY:
            health_info.health_score = 1.0
        elif health_info.status == ServiceStatus.DEGRADED:
            health_info.health_score = 0.5
        else:
            health_info.health_score = 0.0
        
        # Update average response time
        health_info.recent_checks.append(result)
        if len(health_info.recent_checks) > 10:
            health_info.recent_checks = health_info.recent_checks[-10:]
        
        if health_info.recent_checks:
            health_info.average_response_time = sum(
                check.response_time_ms for check in health_info.recent_checks
            ) / len(health_info.recent_checks)
        
        # Update circuit breaker state
        circuit_breaker = self.circuit_breakers[service_name]
        health_info.circuit_breaker_state = circuit_breaker.get_state()
        
        # Record circuit breaker actions
        if result.status == ServiceStatus.HEALTHY:
            await circuit_breaker.record_success()
        else:
            await circuit_breaker.record_failure()
        
        # Calculate uptime percentage (last 24 hours)
        health_info.uptime_percentage = self._calculate_uptime_percentage(service_name)
        
        # Check for status changes and trigger callbacks
        if old_status != health_info.status:
            await self._handle_status_change(service_name, old_status, health_info.status)
        
        # Update database
        self._update_service_health_summary(health_info)
    
    def _calculate_uptime_percentage(self, service_name: str) -> float:
        """Calculate service uptime percentage for the last 24 hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get checks from last 24 hours
            cursor.execute('''
                SELECT status FROM health_check_results 
                WHERE service_name = ? AND timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
            ''', (service_name,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return 100.0
            
            healthy_count = sum(1 for (status,) in results if status == ServiceStatus.HEALTHY.value)
            return (healthy_count / len(results)) * 100.0
            
        except Exception as e:
            logger.error(f"Error calculating uptime for {service_name}: {e}")
            return 0.0
    
    async def _handle_status_change(self, service_name: str, old_status: ServiceStatus, new_status: ServiceStatus):
        """Handle service status changes"""
        health_info = self.service_health[service_name]
        
        # Log status change
        logger.info(f"Service {service_name} status changed: {old_status.value} -> {new_status.value}")
        
        # Store event in database
        self._store_health_event(service_name, "status_change", old_status.value, new_status.value)
        
        # Trigger callbacks
        if new_status == ServiceStatus.HEALTHY and old_status != ServiceStatus.HEALTHY:
            # Service recovered
            for callback in self.recovery_callbacks:
                try:
                    callback(service_name, health_info)
                except Exception as e:
                    logger.error(f"Error in recovery callback: {e}")
        
        # Trigger general health callbacks
        for callback in self.health_callbacks:
            try:
                callback(service_name, health_info)
            except Exception as e:
                logger.error(f"Error in health callback: {e}")
        
        # Generate performance alerts if needed
        if new_status == ServiceStatus.UNHEALTHY:
            try:
                performance_monitor._generate_alert(
                    AlertType.SYSTEM_OVERLOAD,
                    AlertSeverity.HIGH,
                    f"Service Unhealthy: {service_name}",
                    f"Service {service_name} is unhealthy with {health_info.consecutive_failures} consecutive failures",
                    health_info.consecutive_failures,
                    self.health_checks[service_name].failure_threshold
                )
            except Exception as e:
                logger.error(f"Error generating alert: {e}")
    
    async def _aggregate_system_health(self):
        """Aggregate overall system health"""
        while self.monitoring_active:
            try:
                # Calculate overall system health
                if not self.service_health:
                    await asyncio.sleep(30)
                    continue
                
                total_score = sum(info.health_score for info in self.service_health.values())
                overall_score = total_score / len(self.service_health)
                
                # Determine overall status
                if overall_score >= 0.9:
                    overall_status = ServiceStatus.HEALTHY
                elif overall_score >= 0.6:
                    overall_status = ServiceStatus.DEGRADED
                else:
                    overall_status = ServiceStatus.UNHEALTHY
                
                # Update system-wide metrics
                system_metrics = {
                    'overall_health_score': overall_score,
                    'overall_status': overall_status.value,
                    'healthy_services': len([s for s in self.service_health.values() if s.status == ServiceStatus.HEALTHY]),
                    'degraded_services': len([s for s in self.service_health.values() if s.status == ServiceStatus.DEGRADED]),
                    'unhealthy_services': len([s for s in self.service_health.values() if s.status == ServiceStatus.UNHEALTHY]),
                    'total_services': len(self.service_health),
                    'circuit_breakers_open': len([cb for cb in self.circuit_breakers.values() if cb.get_state() == CircuitBreakerState.OPEN]),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store system health summary
                self._store_system_health_summary(system_metrics)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system health aggregation: {e}")
                await asyncio.sleep(30)
    
    # Health check implementations
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            config = self.health_checks["system_memory"]
            
            warning_threshold = config.metadata.get("warning_threshold", 80)
            critical_threshold = config.metadata.get("critical_threshold", 95)
            
            if memory.percent >= critical_threshold:
                status = ServiceStatus.UNHEALTHY
                message = f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent >= warning_threshold:
                status = ServiceStatus.DEGRADED
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = ServiceStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return HealthCheckResult(
                service_name="system_memory",
                status=status,
                response_time_ms=1.0,
                timestamp=datetime.now(),
                message=message,
                details={
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="system_memory",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=1.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check system CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            config = self.health_checks["system_cpu"]
            
            warning_threshold = config.metadata.get("warning_threshold", 80)
            critical_threshold = config.metadata.get("critical_threshold", 95)
            
            if cpu_percent >= critical_threshold:
                status = ServiceStatus.UNHEALTHY
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent >= warning_threshold:
                status = ServiceStatus.DEGRADED
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = ServiceStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                service_name="system_cpu",
                status=status,
                response_time_ms=1000.0,  # CPU check takes ~1 second
                timestamp=datetime.now(),
                message=message,
                details={
                    "percent": cpu_percent,
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="system_cpu",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=1000.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check system disk space"""
        try:
            disk = psutil.disk_usage('.')
            config = self.health_checks["system_disk"]
            
            usage_percent = (disk.used / disk.total) * 100
            warning_threshold = config.metadata.get("warning_threshold", 85)
            critical_threshold = config.metadata.get("critical_threshold", 95)
            
            if usage_percent >= critical_threshold:
                status = ServiceStatus.UNHEALTHY
                message = f"Critical disk usage: {usage_percent:.1f}%"
            elif usage_percent >= warning_threshold:
                status = ServiceStatus.DEGRADED
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = ServiceStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"
            
            return HealthCheckResult(
                service_name="system_disk",
                status=status,
                response_time_ms=5.0,
                timestamp=datetime.now(),
                message=message,
                details={
                    "percent": usage_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="system_disk",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=5.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _check_parameter_detection_service(self) -> HealthCheckResult:
        """Check parameter detection service health"""
        try:
            # Simulate parameter detection health check
            # In a real implementation, this would test the actual service
            start_time = time.time()
            
            # Simulate service call
            await asyncio.sleep(0.1)
            
            response_time = (time.time() - start_time) * 1000
            
            # For demo purposes, randomly simulate service health
            import random
            if random.random() > 0.1:  # 90% success rate
                return HealthCheckResult(
                    service_name="parameter_detection_service",
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message="Parameter detection service is responsive",
                    details={"endpoint": "/api/v1/ensemble/column-mapping"}
                )
            else:
                return HealthCheckResult(
                    service_name="parameter_detection_service",
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    error="Parameter detection service timeout"
                )
                
        except Exception as e:
            return HealthCheckResult(
                service_name="parameter_detection_service",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=5000.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _check_ensemble_initialization_service(self) -> HealthCheckResult:
        """Check ensemble initialization service health"""
        try:
            start_time = time.time()
            
            # Simulate ensemble initialization health check
            await asyncio.sleep(0.2)
            
            response_time = (time.time() - start_time) * 1000
            
            # For demo purposes, randomly simulate service health
            import random
            if random.random() > 0.15:  # 85% success rate
                return HealthCheckResult(
                    service_name="ensemble_initialization_service",
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message="Ensemble initialization service is responsive",
                    details={"endpoint": "/api/v1/ensemble/model-initialization"}
                )
            else:
                return HealthCheckResult(
                    service_name="ensemble_initialization_service",
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    error="Ensemble initialization service failed"
                )
                
        except Exception as e:
            return HealthCheckResult(
                service_name="ensemble_initialization_service",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=10000.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _check_data_processing_service(self) -> HealthCheckResult:
        """Check data processing service health"""
        try:
            start_time = time.time()
            
            # Simulate data processing health check
            await asyncio.sleep(0.05)
            
            response_time = (time.time() - start_time) * 1000
            
            # For demo purposes, randomly simulate service health
            import random
            if random.random() > 0.05:  # 95% success rate
                return HealthCheckResult(
                    service_name="data_processing_service",
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    message="Data processing service is responsive",
                    details={"endpoint": "/api/v1/ensemble/data-quality"}
                )
            else:
                return HealthCheckResult(
                    service_name="data_processing_service",
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    error="Data processing service overloaded"
                )
                
        except Exception as e:
            return HealthCheckResult(
                service_name="data_processing_service",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=5000.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _get_service_health_status(self, service_name: str) -> bool:
        """Get simple health status for a service"""
        if service_name not in self.service_health:
            return False
        return self.service_health[service_name].status == ServiceStatus.HEALTHY
    
    # Database operations
    def _store_health_check_result(self, result: HealthCheckResult):
        """Store health check result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_check_results 
                (service_name, status, response_time_ms, timestamp, message, details, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.service_name,
                result.status.value,
                result.response_time_ms,
                result.timestamp,
                result.message,
                json.dumps(result.details),
                result.error
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing health check result: {e}")
    
    def _update_service_health_summary(self, health_info: ServiceHealthInfo):
        """Update service health summary in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO service_health_summary
                (service_name, status, health_score, last_check, consecutive_failures,
                 consecutive_successes, uptime_percentage, average_response_time,
                 circuit_breaker_state, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                health_info.service_name,
                health_info.status.value,
                health_info.health_score,
                health_info.last_check,
                health_info.consecutive_failures,
                health_info.consecutive_successes,
                health_info.uptime_percentage,
                health_info.average_response_time,
                health_info.circuit_breaker_state.value,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating service health summary: {e}")
    
    def _store_health_event(self, service_name: str, event_type: str, 
                           old_status: str = None, new_status: str = None, message: str = None):
        """Store health event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_events
                (service_name, event_type, old_status, new_status, message)
                VALUES (?, ?, ?, ?, ?)
            ''', (service_name, event_type, old_status, new_status, message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing health event: {e}")
    
    def _store_system_health_summary(self, metrics: Dict[str, Any]):
        """Store system-wide health summary"""
        try:
            # This could be stored in a separate table or cache
            # For now, we'll just log it
            logger.debug(f"System health summary: {metrics}")
            
        except Exception as e:
            logger.error(f"Error storing system health summary: {e}")
    
    # Public API methods
    def get_service_health(self, service_name: str) -> Optional[ServiceHealthInfo]:
        """Get health information for a specific service"""
        return self.service_health.get(service_name)
    
    def get_all_services_health(self) -> Dict[str, ServiceHealthInfo]:
        """Get health information for all services"""
        return self.service_health.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not self.service_health:
            return {
                'overall_status': ServiceStatus.UNKNOWN.value,
                'overall_health_score': 0.0,
                'services': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate overall metrics
        total_score = sum(info.health_score for info in self.service_health.values())
        overall_score = total_score / len(self.service_health)
        
        # Determine overall status
        if overall_score >= 0.9:
            overall_status = ServiceStatus.HEALTHY
        elif overall_score >= 0.6:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.UNHEALTHY
        
        # Count services by status
        status_counts = {
            ServiceStatus.HEALTHY.value: 0,
            ServiceStatus.DEGRADED.value: 0,
            ServiceStatus.UNHEALTHY.value: 0,
            ServiceStatus.UNKNOWN.value: 0
        }
        
        for info in self.service_health.values():
            status_counts[info.status.value] += 1
        
        # Get circuit breaker information
        circuit_breaker_info = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_info[name] = cb.get_stats()
        
        return {
            'overall_status': overall_status.value,
            'overall_health_score': round(overall_score, 3),
            'service_counts': status_counts,
            'total_services': len(self.service_health),
            'services': {
                name: {
                    'status': info.status.value,
                    'health_score': info.health_score,
                    'uptime_percentage': info.uptime_percentage,
                    'average_response_time': info.average_response_time,
                    'consecutive_failures': info.consecutive_failures,
                    'circuit_breaker_state': info.circuit_breaker_state.value,
                    'last_check': info.last_check.isoformat()
                }
                for name, info in self.service_health.items()
            },
            'circuit_breakers': circuit_breaker_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def add_health_callback(self, callback: Callable[[str, ServiceHealthInfo], None]):
        """Add callback for health status changes"""
        self.health_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str, ServiceHealthInfo], None]):
        """Add callback for service recovery events"""
        self.recovery_callbacks.append(callback)
    
    async def force_health_check(self, service_name: str) -> Optional[HealthCheckResult]:
        """Force an immediate health check for a service"""
        if service_name not in self.health_checkers:
            return None
        
        checker = self.health_checkers[service_name]
        result = await self._execute_health_check(service_name, checker)
        await self._update_service_health(service_name, result)
        self._store_health_check_result(result)
        
        return result
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}
    
    async def start_background_monitoring(self):
        """Start background monitoring if not already running"""
        if not self.monitoring_active:
            await self.start_monitoring()
    
    def is_monitoring_active(self) -> bool:
        """Check if monitoring is currently active"""
        return self.monitoring_active

# Global health monitor instance
live_health_monitor = LiveHealthMonitor()

# Callback functions for integration with other systems
def health_change_callback(service_name: str, health_info: ServiceHealthInfo):
    """Callback for health status changes"""
    logger.info(f"Health status changed for {service_name}: {health_info.status.value}")

def service_recovery_callback(service_name: str, health_info: ServiceHealthInfo):
    """Callback for service recovery"""
    logger.info(f"Service {service_name} has recovered! Health score: {health_info.health_score}")

# Register default callbacks
live_health_monitor.add_health_callback(health_change_callback)
live_health_monitor.add_recovery_callback(service_recovery_callback)