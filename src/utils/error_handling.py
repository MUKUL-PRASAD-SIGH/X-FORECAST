"""
Enhanced Error Handling System for Data Upload Reliability
Provides comprehensive error classification, retry logic, and recovery strategies
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import json
import traceback
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    FILE_FORMAT = "file_format"
    DATA_QUALITY = "data_quality"
    INTERNAL_SERVER = "internal_server"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    REFRESH_AUTH = "refresh_auth"
    REDIRECT_LOGIN = "redirect_login"
    FALLBACK_MODE = "fallback_mode"
    CIRCUIT_BREAKER = "circuit_breaker"
    USER_INTERVENTION = "user_intervention"

@dataclass
class ErrorClassification:
    """Detailed error classification"""
    category: ErrorCategory
    severity: ErrorSeverity
    retryable: bool
    user_message: str
    technical_message: str
    recovery_actions: List[RecoveryAction]
    retry_delay: float = 1.0
    max_retries: int = 3
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.1
    retry_on_exceptions: List[type] = field(default_factory=list)
    retry_on_status_codes: List[int] = field(default_factory=lambda: [500, 502, 503, 504, 408, 429])

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    name: str = "default"

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    next_attempt_time: Optional[datetime] = None

class ErrorClassifier:
    """Intelligent error classification system"""
    
    def __init__(self):
        self.classification_rules = self._initialize_classification_rules()
    
    def _initialize_classification_rules(self) -> Dict[str, ErrorClassification]:
        """Initialize error classification rules"""
        return {
            # Network errors
            "connection_error": ErrorClassification(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                user_message="Connection failed. Please check your internet connection and try again.",
                technical_message="Network connection error",
                recovery_actions=[RecoveryAction.RETRY],
                retry_delay=1.0,  # Use classification retry delay, not base delay
                max_retries=3
            ),
            "timeout_error": ErrorClassification(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                user_message="Request timed out. The server may be busy. Please try again.",
                technical_message="Request timeout",
                recovery_actions=[RecoveryAction.RETRY],
                retry_delay=5.0,
                max_retries=2
            ),
            "dns_error": ErrorClassification(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                user_message="Cannot reach the server. Please check your network settings or contact support.",
                technical_message="DNS resolution failed",
                recovery_actions=[RecoveryAction.USER_INTERVENTION],
                max_retries=0
            ),
            
            # Authentication errors
            "auth_token_expired": ErrorClassification(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                user_message="Your session has expired. Please log in again.",
                technical_message="Authentication token expired",
                recovery_actions=[RecoveryAction.REFRESH_AUTH, RecoveryAction.REDIRECT_LOGIN],
                retry_delay=0.5,
                max_retries=1
            ),
            "auth_invalid_credentials": ErrorClassification(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                user_message="Invalid credentials. Please check your login information.",
                technical_message="Invalid authentication credentials",
                recovery_actions=[RecoveryAction.REDIRECT_LOGIN],
                max_retries=0
            ),
            "auth_permission_denied": ErrorClassification(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                retryable=False,
                user_message="You don't have permission to perform this action. Please contact your administrator.",
                technical_message="Permission denied",
                recovery_actions=[RecoveryAction.USER_INTERVENTION],
                max_retries=0
            ),
            
            # Validation errors
            "file_format_invalid": ErrorClassification(
                category=ErrorCategory.FILE_FORMAT,
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                user_message="Invalid file format. Please upload CSV, Excel, or PDF files only.",
                technical_message="Unsupported file format",
                recovery_actions=[RecoveryAction.USER_INTERVENTION],
                max_retries=0
            ),
            "file_size_exceeded": ErrorClassification(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                user_message="File size exceeds the 50MB limit. Please upload a smaller file.",
                technical_message="File size validation failed",
                recovery_actions=[RecoveryAction.USER_INTERVENTION],
                max_retries=0
            ),
            "data_validation_failed": ErrorClassification(
                category=ErrorCategory.DATA_QUALITY,
                severity=ErrorSeverity.MEDIUM,
                retryable=False,
                user_message="Data validation failed. Please check your file format and content.",
                technical_message="Data validation error",
                recovery_actions=[RecoveryAction.FALLBACK_MODE, RecoveryAction.USER_INTERVENTION],
                max_retries=0
            ),
            
            # Service errors
            "service_unavailable": ErrorClassification(
                category=ErrorCategory.SERVICE_UNAVAILABLE,
                severity=ErrorSeverity.HIGH,
                retryable=True,
                user_message="Service is temporarily unavailable. Please try again in a few moments.",
                technical_message="Service unavailable",
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAKER],
                retry_delay=10.0,
                max_retries=2
            ),
            "rate_limit_exceeded": ErrorClassification(
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                user_message="Too many requests. Please wait a moment before trying again.",
                technical_message="Rate limit exceeded",
                recovery_actions=[RecoveryAction.RETRY],
                retry_delay=30.0,
                max_retries=2
            ),
            "internal_server_error": ErrorClassification(
                category=ErrorCategory.INTERNAL_SERVER,
                severity=ErrorSeverity.HIGH,
                retryable=True,
                user_message="An internal server error occurred. Our team has been notified.",
                technical_message="Internal server error",
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
                retry_delay=5.0,
                max_retries=2
            ),
            
            # Default fallback
            "unknown_error": ErrorClassification(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                retryable=True,
                user_message="An unexpected error occurred. Please try again or contact support if the problem persists.",
                technical_message="Unknown error",
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
                retry_delay=3.0,
                max_retries=2
            )
        }
    
    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorClassification:
        """Classify an error and return appropriate handling strategy"""
        try:
            error_message = str(error).lower()
            error_type = type(error).__name__.lower()
            
            # Network-related errors
            if (any(keyword in error_message for keyword in ['connection', 'network', 'unreachable', 'failed to fetch']) or
                error_type in ['connectionerror', 'networkerror']):
                if 'timeout' in error_message or error_type == 'timeouterror':
                    classification = self.classification_rules["timeout_error"]
                elif 'dns' in error_message or 'name resolution' in error_message:
                    classification = self.classification_rules["dns_error"]
                else:
                    classification = self.classification_rules["connection_error"]
            
            # Timeout errors (check separately to catch TimeoutError type)
            elif 'timeout' in error_message or error_type == 'timeouterror':
                classification = self.classification_rules["timeout_error"]
            
            # Authentication errors
            elif any(keyword in error_message for keyword in ['unauthorized', '401', 'authentication']):
                if 'expired' in error_message or 'token' in error_message:
                    classification = self.classification_rules["auth_token_expired"]
                elif 'permission' in error_message or 'forbidden' in error_message or '403' in error_message:
                    classification = self.classification_rules["auth_permission_denied"]
                else:
                    classification = self.classification_rules["auth_invalid_credentials"]
            
            # File format and validation errors
            elif any(keyword in error_message for keyword in ['file format', 'invalid format', 'unsupported']):
                classification = self.classification_rules["file_format_invalid"]
            elif any(keyword in error_message for keyword in ['file size', 'too large', 'exceeds limit']):
                classification = self.classification_rules["file_size_exceeded"]
            elif any(keyword in error_message for keyword in ['validation', 'invalid data']):
                classification = self.classification_rules["data_validation_failed"]
            
            # Service errors
            elif any(keyword in error_message for keyword in ['503', 'service unavailable', 'server unavailable']):
                classification = self.classification_rules["service_unavailable"]
            elif any(keyword in error_message for keyword in ['429', 'rate limit', 'too many requests']):
                classification = self.classification_rules["rate_limit_exceeded"]
            elif any(keyword in error_message for keyword in ['500', '502', '504', 'internal server']):
                classification = self.classification_rules["internal_server_error"]
            elif 'timeout' in error_message:
                classification = self.classification_rules["timeout_error"]
            
            # Default fallback
            else:
                classification = self.classification_rules["unknown_error"]
            
            # Add context information
            if context:
                classification.context.update(context)
            
            # Add error details
            classification.context.update({
                'original_error': str(error),
                'error_type': type(error).__name__,
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            })
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classification failed: {e}")
            # Return safe fallback
            return self.classification_rules["unknown_error"]
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        classification = self.classify_error(error)
        return classification.retryable
    
    def get_user_message(self, error: Exception) -> str:
        """Get user-friendly error message"""
        classification = self.classify_error(error)
        return classification.user_message
    
    def get_recovery_actions(self, error: Exception) -> List[RecoveryAction]:
        """Get recommended recovery actions"""
        classification = self.classify_error(error)
        return classification.recovery_actions

class RetryManager:
    """Smart retry manager with exponential backoff and jitter"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.error_classifier = ErrorClassifier()
    
    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute operation with intelligent retry logic"""
        config = retry_config or self.config
        attempt = 0
        last_error = None
        
        while attempt < config.max_attempts:
            try:
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success - reset any circuit breaker state
                if hasattr(operation, '__circuit_breaker__'):
                    operation.__circuit_breaker__.record_success()
                
                return result
                
            except Exception as error:
                attempt += 1
                last_error = error
                
                # Classify the error
                classification = self.error_classifier.classify_error(error, context)
                
                # Log the error
                logger.warning(
                    f"Operation failed (attempt {attempt}/{config.max_attempts}): {classification.technical_message}",
                    extra={
                        'error_category': classification.category.value,
                        'error_severity': classification.severity.value,
                        'retryable': classification.retryable,
                        'context': context or {}
                    }
                )
                
                # Check if we should retry
                if not classification.retryable or attempt >= config.max_attempts:
                    # Record failure in circuit breaker if applicable
                    if hasattr(operation, '__circuit_breaker__'):
                        operation.__circuit_breaker__.record_failure()
                    
                    # Enhance error with classification info
                    enhanced_error = EnhancedError(
                        original_error=error,
                        classification=classification,
                        attempt_count=attempt,
                        context=context or {}
                    )
                    raise enhanced_error
                
                # Calculate delay with exponential backoff and jitter
                # Use the smaller of classification delay and config base delay for first attempt
                base_delay = min(classification.retry_delay, config.base_delay)
                delay = min(
                    base_delay * (config.backoff_multiplier ** (attempt - 1)),
                    config.max_delay
                )
                
                # Add jitter to prevent thundering herd
                jitter = delay * config.jitter * (0.5 - asyncio.get_event_loop().time() % 1)
                total_delay = delay + jitter
                
                logger.info(f"Retrying in {total_delay:.2f} seconds...")
                await asyncio.sleep(total_delay)
        
        # All retries exhausted
        if hasattr(operation, '__circuit_breaker__'):
            operation.__circuit_breaker__.record_failure()
        
        enhanced_error = EnhancedError(
            original_error=last_error,
            classification=self.error_classifier.classify_error(last_error, context),
            attempt_count=attempt,
            context=context or {}
        )
        raise enhanced_error

class CircuitBreaker:
    """Circuit breaker implementation for service protection"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation through circuit breaker"""
        async with self._lock:
            # Check circuit breaker state
            if self.stats.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.stats.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.config.name} is OPEN. "
                        f"Next attempt allowed at {self.stats.next_attempt_time}"
                    )
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Record success
            await self.record_success()
            return result
            
        except Exception as error:
            # Record failure
            await self.record_failure()
            raise
    
    async def record_success(self):
        """Record successful operation"""
        async with self._lock:
            self.stats.success_count += 1
            
            if self.stats.state == CircuitBreakerState.HALF_OPEN:
                # Reset circuit breaker on successful half-open attempt
                self.stats.state = CircuitBreakerState.CLOSED
                self.stats.failure_count = 0
                logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
    
    async def record_failure(self):
        """Record failed operation"""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.last_failure_time = datetime.now()
            
            if self.stats.failure_count >= self.config.failure_threshold:
                self.stats.state = CircuitBreakerState.OPEN
                self.stats.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
                logger.warning(f"Circuit breaker {self.config.name} opened due to {self.stats.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.stats.next_attempt_time is None:
            return True
        return datetime.now() >= self.stats.next_attempt_time
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self.stats.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'state': self.stats.state.value,
            'failure_count': self.stats.failure_count,
            'success_count': self.stats.success_count,
            'last_failure_time': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            'next_attempt_time': self.stats.next_attempt_time.isoformat() if self.stats.next_attempt_time else None
        }

class EnhancedError(Exception):
    """Enhanced error with classification and context"""
    
    def __init__(self, original_error: Exception, classification: ErrorClassification, 
                 attempt_count: int, context: Dict[str, Any]):
        self.original_error = original_error
        self.classification = classification
        self.attempt_count = attempt_count
        self.context = context
        
        super().__init__(classification.user_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'error_type': 'enhanced_error',
            'category': self.classification.category.value,
            'severity': self.classification.severity.value,
            'user_message': self.classification.user_message,
            'technical_message': self.classification.technical_message,
            'retryable': self.classification.retryable,
            'recovery_actions': [action.value for action in self.classification.recovery_actions],
            'attempt_count': self.attempt_count,
            'context': self.context,
            'timestamp': datetime.now().isoformat()
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

# Decorator for adding retry logic to functions
def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to functions"""
    def decorator(func):
        retry_manager = RetryManager(config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_manager.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    return decorator

# Decorator for adding circuit breaker to functions
def with_circuit_breaker(config: CircuitBreakerConfig):
    """Decorator to add circuit breaker to functions"""
    def decorator(func):
        circuit_breaker = CircuitBreaker(config)
        func.__circuit_breaker__ = circuit_breaker
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

# Global instances for easy access
error_classifier = ErrorClassifier()
retry_manager = RetryManager()

# Health monitoring for services
class ServiceHealthMonitor:
    """Monitor service health and manage circuit breakers"""
    
    def __init__(self):
        self.services: Dict[str, CircuitBreaker] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_status: Dict[str, bool] = {}
    
    def register_service(self, service_name: str, health_check: Callable, 
                        circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        """Register a service for health monitoring"""
        if circuit_breaker_config is None:
            circuit_breaker_config = CircuitBreakerConfig(name=service_name)
        
        self.services[service_name] = CircuitBreaker(circuit_breaker_config)
        self.health_checks[service_name] = health_check
        self.health_status[service_name] = True
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service"""
        if service_name not in self.health_checks:
            return False
        
        try:
            health_check = self.health_checks[service_name]
            if asyncio.iscoroutinefunction(health_check):
                result = await health_check()
            else:
                result = health_check()
            
            self.health_status[service_name] = bool(result)
            self.last_health_check[service_name] = datetime.now()
            return self.health_status[service_name]
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            self.health_status[service_name] = False
            return False
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get comprehensive service status"""
        if service_name not in self.services:
            return {'error': 'Service not registered'}
        
        circuit_breaker = self.services[service_name]
        health_status = await self.check_service_health(service_name)
        
        return {
            'service_name': service_name,
            'healthy': health_status,
            'circuit_breaker': circuit_breaker.get_stats(),
            'last_health_check': self.last_health_check.get(service_name, datetime.now()).isoformat()
        }
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        service_statuses = {}
        overall_healthy = True
        
        for service_name in self.services:
            status = await self.get_service_status(service_name)
            service_statuses[service_name] = status
            if not status.get('healthy', False):
                overall_healthy = False
        
        return {
            'overall_healthy': overall_healthy,
            'services': service_statuses,
            'timestamp': datetime.now().isoformat()
        }

# Global service health monitor
service_health_monitor = ServiceHealthMonitor()