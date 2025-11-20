"""
Comprehensive Logging and Monitoring Configuration
Provides centralized logging for authentication, file processing, RAG queries, and security events
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    # Authentication Events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    REGISTRATION_SUCCESS = "registration_success"
    REGISTRATION_FAILURE = "registration_failure"
    TOKEN_VALIDATION_SUCCESS = "token_validation_success"
    TOKEN_VALIDATION_FAILURE = "token_validation_failure"
    
    # File Upload and Processing Events
    FILE_UPLOAD_START = "file_upload_start"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILURE = "file_upload_failure"
    PDF_PROCESSING_START = "pdf_processing_start"
    PDF_PROCESSING_SUCCESS = "pdf_processing_success"
    PDF_PROCESSING_FAILURE = "pdf_processing_failure"
    
    # RAG System Events
    RAG_INITIALIZATION_START = "rag_initialization_start"
    RAG_INITIALIZATION_SUCCESS = "rag_initialization_success"
    RAG_INITIALIZATION_FAILURE = "rag_initialization_failure"
    RAG_QUERY_START = "rag_query_start"
    RAG_QUERY_SUCCESS = "rag_query_success"
    RAG_QUERY_FAILURE = "rag_query_failure"
    
    # Security Events
    UNAUTHORIZED_ACCESS_ATTEMPT = "unauthorized_access_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

@dataclass
class LogEvent:
    """Structured log event"""
    timestamp: datetime
    event_type: EventType
    level: LogLevel
    user_id: Optional[str] = None
    company_name: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None

class ComprehensiveLogger:
    """
    Comprehensive logging and monitoring system for multi-tenant authentication
    """
    
    def __init__(self, log_db_path: str = "logs/system_logs.db"):
        self.log_db_path = log_db_path
        self._init_logging_database()
        self._setup_file_handlers()
        
        # Create logger instances
        self.auth_logger = logging.getLogger("auth")
        self.file_logger = logging.getLogger("file_processing")
        self.rag_logger = logging.getLogger("rag_system")
        self.security_logger = logging.getLogger("security")
        
        # Configure log levels
        self._configure_loggers()
    
    def _init_logging_database(self):
        """Initialize SQLite database for structured logging"""
        os.makedirs(os.path.dirname(self.log_db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        # Main events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                level TEXT NOT NULL,
                user_id TEXT,
                company_name TEXT,
                ip_address TEXT,
                session_id TEXT,
                details TEXT,
                error_message TEXT,
                duration_ms REAL,
                file_path TEXT,
                file_size INTEGER
            )
        ''')
        
        # Authentication events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user_id TEXT,
                company_name TEXT,
                ip_address TEXT,
                success BOOLEAN,
                failure_reason TEXT
            )
        ''')
        
        # File processing events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                company_name TEXT,
                file_name TEXT,
                file_type TEXT,
                file_size INTEGER,
                processing_duration_ms REAL,
                success BOOLEAN,
                error_message TEXT
            )
        ''')
        
        # RAG performance events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user_id TEXT,
                company_name TEXT,
                response_time_ms REAL,
                success BOOLEAN,
                error_message TEXT
            )
        ''')
        
        # Security events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                ip_address TEXT,
                user_id TEXT,
                endpoint TEXT,
                threat_level TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log types"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handlers
        auth_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/auth.log", maxBytes=10*1024*1024, backupCount=5
        )
        file_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/file_processing.log", maxBytes=10*1024*1024, backupCount=5
        )
        rag_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/rag_system.log", maxBytes=10*1024*1024, backupCount=5
        )
        security_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/security.log", maxBytes=10*1024*1024, backupCount=5
        )
        
        # Set formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        for handler in [auth_handler, file_handler, rag_handler, security_handler]:
            handler.setFormatter(formatter)
        
        # Store handlers for later use
        self.handlers = {
            'auth': auth_handler,
            'file': file_handler,
            'rag': rag_handler,
            'security': security_handler
        }
    
    def _configure_loggers(self):
        """Configure individual loggers"""
        loggers = {
            'auth': self.auth_logger,
            'file': self.file_logger,
            'rag': self.rag_logger,
            'security': self.security_logger
        }
        
        for name, logger in loggers.items():
            logger.setLevel(logging.INFO)
            logger.addHandler(self.handlers[name])
            logger.propagate = False
    
    def log_event(self, event: LogEvent):
        """Log a structured event to database and files"""
        try:
            # Log to database
            self._log_to_database(event)
            
            # Log to appropriate file logger
            self._log_to_file(event)
            
        except Exception as e:
            # Fallback logging to prevent logging failures from breaking the system
            logging.error(f"Failed to log event: {e}")
    
    def _log_to_database(self, event: LogEvent):
        """Log event to SQLite database"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        # Insert into main events table
        cursor.execute('''
            INSERT INTO log_events 
            (timestamp, event_type, level, user_id, company_name, ip_address, 
             session_id, details, error_message, duration_ms, file_path, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.event_type.value, event.level.value,
            event.user_id, event.company_name, event.ip_address,
            event.session_id, json.dumps(event.details) if event.details else None,
            event.error_message, event.duration_ms, event.file_path, event.file_size
        ))
        
        # Insert into specific tables based on event type
        if event.event_type.value.startswith(('login', 'registration', 'token')):
            self._log_auth_event(cursor, event)
        elif event.event_type.value.startswith(('file_upload', 'pdf_processing')):
            self._log_file_event(cursor, event)
        elif event.event_type.value.startswith('rag_'):
            self._log_rag_event(cursor, event)
        elif event.event_type.value in ['unauthorized_access_attempt', 'suspicious_activity']:
            self._log_security_event(cursor, event)
        
        conn.commit()
        conn.close()
    
    def _log_auth_event(self, cursor, event: LogEvent):
        """Log authentication-specific event"""
        success = 'success' in event.event_type.value
        failure_reason = event.error_message if not success else None
        
        cursor.execute('''
            INSERT INTO auth_events 
            (timestamp, event_type, user_id, company_name, ip_address, success, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.event_type.value, event.user_id,
            event.company_name, event.ip_address, success, failure_reason
        ))
    
    def _log_file_event(self, cursor, event: LogEvent):
        """Log file processing event"""
        success = 'success' in event.event_type.value
        file_type = None
        if event.file_path:
            file_type = event.file_path.split('.')[-1].lower()
        
        cursor.execute('''
            INSERT INTO file_events 
            (timestamp, event_type, user_id, company_name, file_name, file_type, 
             file_size, processing_duration_ms, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.event_type.value, event.user_id,
            event.company_name, event.file_path, file_type,
            event.file_size, event.duration_ms, success, event.error_message
        ))
    
    def _log_rag_event(self, cursor, event: LogEvent):
        """Log RAG system event"""
        success = 'success' in event.event_type.value
        
        cursor.execute('''
            INSERT INTO rag_events 
            (timestamp, event_type, user_id, company_name, response_time_ms, 
             success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.event_type.value, event.user_id,
            event.company_name, event.duration_ms, success, event.error_message
        ))
    
    def _log_security_event(self, cursor, event: LogEvent):
        """Log security event"""
        threat_level = "high" if event.event_type == EventType.UNAUTHORIZED_ACCESS_ATTEMPT else "medium"
        
        cursor.execute('''
            INSERT INTO security_events 
            (timestamp, event_type, ip_address, user_id, threat_level, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.event_type.value, event.ip_address,
            event.user_id, threat_level, json.dumps(event.details) if event.details else None
        ))
    
    def _log_to_file(self, event: LogEvent):
        """Log event to appropriate file logger"""
        message = f"[{event.event_type.value}] User: {event.user_id or 'N/A'} | Company: {event.company_name or 'N/A'}"
        
        if event.error_message:
            message += f" | Error: {event.error_message}"
        
        if event.duration_ms:
            message += f" | Duration: {event.duration_ms:.2f}ms"
        
        # Route to appropriate logger
        if event.event_type.value.startswith(('login', 'registration', 'token')):
            if event.level == LogLevel.ERROR:
                self.auth_logger.error(message)
            elif event.level == LogLevel.WARNING:
                self.auth_logger.warning(message)
            else:
                self.auth_logger.info(message)
        
        elif event.event_type.value.startswith(('file_upload', 'pdf_processing')):
            if event.level == LogLevel.ERROR:
                self.file_logger.error(message)
            elif event.level == LogLevel.WARNING:
                self.file_logger.warning(message)
            else:
                self.file_logger.info(message)
        
        elif event.event_type.value.startswith('rag_'):
            if event.level == LogLevel.ERROR:
                self.rag_logger.error(message)
            elif event.level == LogLevel.WARNING:
                self.rag_logger.warning(message)
            else:
                self.rag_logger.info(message)
        
        elif event.event_type.value in ['unauthorized_access_attempt', 'suspicious_activity']:
            if event.level == LogLevel.ERROR or event.level == LogLevel.CRITICAL:
                self.security_logger.error(message)
            elif event.level == LogLevel.WARNING:
                self.security_logger.warning(message)
            else:
                self.security_logger.info(message)
    
    # Authentication logging methods
    def log_login_success(self, user_id: str, email: str, company_name: str, ip_address: str, session_id: str = None):
        """Log successful login"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.LOGIN_SUCCESS,
            level=LogLevel.INFO,
            user_id=user_id,
            company_name=company_name,
            ip_address=ip_address,
            session_id=session_id,
            details={"email": email}
        )
        self.log_event(event)
    
    def log_login_failure(self, email: str, ip_address: str, reason: str, session_id: str = None):
        """Log failed login attempt"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.LOGIN_FAILURE,
            level=LogLevel.WARNING,
            ip_address=ip_address,
            session_id=session_id,
            error_message=reason,
            details={"email": email}
        )
        self.log_event(event)
    
    def log_registration_success(self, user_id: str, email: str, company_name: str, ip_address: str):
        """Log successful registration"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.REGISTRATION_SUCCESS,
            level=LogLevel.INFO,
            user_id=user_id,
            company_name=company_name,
            ip_address=ip_address,
            details={"email": email}
        )
        self.log_event(event)
    
    def log_registration_failure(self, email: str, company_name: str, ip_address: str, reason: str):
        """Log failed registration attempt"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.REGISTRATION_FAILURE,
            level=LogLevel.WARNING,
            company_name=company_name,
            ip_address=ip_address,
            error_message=reason,
            details={"email": email}
        )
        self.log_event(event)
    
    # File processing logging methods
    def log_file_upload_start(self, user_id: str, company_name: str, file_path: str, file_size: int):
        """Log file upload start"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.FILE_UPLOAD_START,
            level=LogLevel.INFO,
            user_id=user_id,
            company_name=company_name,
            file_path=file_path,
            file_size=file_size
        )
        self.log_event(event)
    
    def log_pdf_processing_success(self, user_id: str, company_name: str, file_path: str, duration_ms: float, pages_processed: int = None):
        """Log successful PDF processing"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.PDF_PROCESSING_SUCCESS,
            level=LogLevel.INFO,
            user_id=user_id,
            company_name=company_name,
            file_path=file_path,
            duration_ms=duration_ms,
            details={"pages_processed": pages_processed} if pages_processed else None
        )
        self.log_event(event)
    
    def log_pdf_processing_failure(self, user_id: str, company_name: str, file_path: str, error_message: str):
        """Log failed PDF processing"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.PDF_PROCESSING_FAILURE,
            level=LogLevel.ERROR,
            user_id=user_id,
            company_name=company_name,
            file_path=file_path,
            error_message=error_message
        )
        self.log_event(event)
    
    # RAG system logging methods
    def log_rag_query_success(self, user_id: str, company_name: str, query: str, response_time_ms: float, relevance_score: float = None):
        """Log successful RAG query"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.RAG_QUERY_SUCCESS,
            level=LogLevel.INFO,
            user_id=user_id,
            company_name=company_name,
            duration_ms=response_time_ms,
            details={"query_length": len(query), "relevance_score": relevance_score}
        )
        self.log_event(event)
    
    def log_rag_query_failure(self, user_id: str, company_name: str, query: str, error_message: str):
        """Log failed RAG query"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.RAG_QUERY_FAILURE,
            level=LogLevel.ERROR,
            user_id=user_id,
            company_name=company_name,
            error_message=error_message,
            details={"query_length": len(query)}
        )
        self.log_event(event)
    
    def log_rag_initialization_success(self, user_id: str, company_name: str, duration_ms: float):
        """Log successful RAG initialization"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.RAG_INITIALIZATION_SUCCESS,
            level=LogLevel.INFO,
            user_id=user_id,
            company_name=company_name,
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_rag_initialization_failure(self, user_id: str, company_name: str, error_message: str):
        """Log failed RAG initialization"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.RAG_INITIALIZATION_FAILURE,
            level=LogLevel.ERROR,
            user_id=user_id,
            company_name=company_name,
            error_message=error_message
        )
        self.log_event(event)
    
    # Security logging methods
    def log_unauthorized_access(self, ip_address: str, endpoint: str, user_id: str = None, details: Dict = None):
        """Log unauthorized access attempt"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.UNAUTHORIZED_ACCESS_ATTEMPT,
            level=LogLevel.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            details={"endpoint": endpoint, **details} if details else {"endpoint": endpoint}
        )
        self.log_event(event)
    
    def log_suspicious_activity(self, ip_address: str, activity_type: str, user_id: str = None, details: Dict = None):
        """Log suspicious activity"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.SUSPICIOUS_ACTIVITY,
            level=LogLevel.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            details={"activity_type": activity_type, **details} if details else {"activity_type": activity_type}
        )
        self.log_event(event)
    
    def log_rate_limit_exceeded(self, ip_address: str, endpoint: str, user_id: str = None):
        """Log rate limit exceeded"""
        event = LogEvent(
            timestamp=datetime.now(),
            event_type=EventType.RATE_LIMIT_EXCEEDED,
            level=LogLevel.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            details={"endpoint": endpoint}
        )
        self.log_event(event)
    
    # Analytics and reporting methods
    def get_auth_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get authentication statistics for the last N days"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Get login statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_logins,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_logins,
                COUNT(DISTINCT user_id) as unique_users
            FROM auth_events 
            WHERE event_type = 'login_success' OR event_type = 'login_failure'
            AND timestamp >= ?
        ''', (since_date,))
        
        stats = cursor.fetchone()
        
        # Get registration statistics
        cursor.execute('''
            SELECT COUNT(*) as registrations
            FROM auth_events 
            WHERE event_type = 'registration_success'
            AND timestamp >= ?
        ''', (since_date,))
        
        registrations = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "period_days": days,
            "total_login_attempts": stats[0] or 0,
            "successful_logins": stats[1] or 0,
            "failed_logins": stats[2] or 0,
            "unique_users": stats[3] or 0,
            "new_registrations": registrations or 0,
            "success_rate": (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
        }
    
    def get_file_processing_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get file processing statistics"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                file_type,
                COUNT(*) as total_files,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_files,
                AVG(processing_duration_ms) as avg_processing_time,
                SUM(file_size) as total_size_bytes
            FROM file_events 
            WHERE timestamp >= ?
            GROUP BY file_type
        ''', (since_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        stats = {}
        for row in results:
            file_type, total, successful, avg_time, total_size = row
            stats[file_type] = {
                "total_files": total,
                "successful_files": successful,
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "avg_processing_time_ms": avg_time or 0,
                "total_size_mb": (total_size or 0) / (1024 * 1024)
            }
        
        return {
            "period_days": days,
            "by_file_type": stats
        }
    
    def get_security_events(self, days: int = 7) -> Dict[str, Any]:
        """Get security events summary"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                event_type,
                COUNT(*) as count,
                COUNT(DISTINCT ip_address) as unique_ips
            FROM security_events 
            WHERE timestamp >= ?
            GROUP BY event_type
        ''', (since_date,))
        
        results = cursor.fetchall()
        
        # Get top offending IPs
        cursor.execute('''
            SELECT 
                ip_address,
                COUNT(*) as violation_count
            FROM security_events 
            WHERE timestamp >= ?
            GROUP BY ip_address
            ORDER BY violation_count DESC
            LIMIT 10
        ''', (since_date,))
        
        top_ips = cursor.fetchall()
        conn.close()
        
        events_by_type = {}
        for event_type, count, unique_ips in results:
            events_by_type[event_type] = {
                "count": count,
                "unique_ips": unique_ips
            }
        
        return {
            "period_days": days,
            "events_by_type": events_by_type,
            "top_offending_ips": [{"ip": ip, "violations": count} for ip, count in top_ips]
        }

# Global logger instance
comprehensive_logger = ComprehensiveLogger()