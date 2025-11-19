"""
RAG Recovery Manager with Automatic Recovery Mechanisms
Implements automatic recovery mechanisms with exponential backoff for failed RAG initializations
and provides comprehensive recovery status tracking and reporting.
"""

import logging
import time
import sqlite3
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RecoveryStatus(Enum):
    """Recovery attempt status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY_EXHAUSTED = "retry_exhausted"

class FailureType(Enum):
    """Types of failures that can be recovered from"""
    DEPENDENCY_MISSING = "dependency_missing"
    INITIALIZATION_FAILED = "initialization_failed"
    SCHEMA_ERROR = "schema_error"
    STORAGE_ERROR = "storage_error"
    PERFORMANCE_DEGRADED = "performance_degraded"
    UNKNOWN = "unknown"

@dataclass
class RecoveryAttempt:
    """Individual recovery attempt information"""
    attempt_number: int
    status: RecoveryStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    actions_taken: List[str]
    delay_before_attempt: float

@dataclass
class RecoverySession:
    """Complete recovery session with multiple attempts"""
    session_id: str
    user_id: str
    failure_type: FailureType
    failure_context: Dict[str, Any]
    max_retries: int
    initial_delay: float
    backoff_factor: float
    start_time: datetime
    end_time: Optional[datetime]
    final_status: RecoveryStatus
    attempts: List[RecoveryAttempt]
    total_recovery_time: Optional[float]
    success_on_attempt: Optional[int]

class RAGRecoveryManager:
    """
    Manages automatic recovery mechanisms for RAG system failures with exponential backoff
    """
    
    def __init__(self, enhanced_rag_manager, db_path: str = "users.db"):
        self.enhanced_rag_manager = enhanced_rag_manager
        self.db_path = db_path
        self._initialize_recovery_database()
    
    def _initialize_recovery_database(self):
        """Initialize recovery tracking database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create recovery sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_recovery_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    failure_context TEXT NOT NULL,
                    max_retries INTEGER NOT NULL,
                    initial_delay REAL NOT NULL,
                    backoff_factor REAL NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    final_status TEXT NOT NULL,
                    total_recovery_time REAL,
                    success_on_attempt INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create recovery attempts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_recovery_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration REAL,
                    error_message TEXT,
                    actions_taken TEXT,
                    delay_before_attempt REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES rag_recovery_sessions (session_id)
                )
            ''')
            
            # Create recovery statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_recovery_statistics (
                    user_id TEXT PRIMARY KEY,
                    total_recovery_sessions INTEGER DEFAULT 0,
                    successful_recoveries INTEGER DEFAULT 0,
                    failed_recoveries INTEGER DEFAULT 0,
                    average_recovery_time REAL DEFAULT 0,
                    last_recovery_attempt TIMESTAMP,
                    recovery_success_rate REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Recovery database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recovery database: {str(e)}")
    
    def recover_with_exponential_backoff(
        self, 
        user_id: str, 
        company_name: str,
        failure_context: Dict[str, Any],
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0
    ) -> Dict[str, Any]:
        """
        Attempt recovery with exponential backoff for failed RAG initializations
        
        Args:
            user_id: User identifier
            company_name: Company name
            failure_context: Context information about the failure
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            max_delay: Maximum delay between retries
            
        Returns:
            Dictionary with recovery result and detailed attempt information
        """
        session_id = f"recovery_{user_id}_{int(datetime.now().timestamp())}"
        failure_type = self._classify_failure_type(failure_context)
        
        # Create recovery session
        recovery_session = RecoverySession(
            session_id=session_id,
            user_id=user_id,
            failure_type=failure_type,
            failure_context=failure_context,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            start_time=datetime.now(),
            end_time=None,
            final_status=RecoveryStatus.PENDING,
            attempts=[],
            total_recovery_time=None,
            success_on_attempt=None
        )
        
        logger.info(f"Starting recovery session {session_id} for user {user_id}, failure type: {failure_type.value}")
        
        try:
            # Save initial recovery session
            self._save_recovery_session(recovery_session)
            
            last_error = None
            
            for attempt_num in range(max_retries + 1):  # +1 for initial attempt
                # Calculate delay for this attempt (no delay for first attempt)
                if attempt_num > 0:
                    delay = min(initial_delay * (backoff_factor ** (attempt_num - 1)), max_delay)
                    logger.info(f"Recovery attempt {attempt_num + 1}/{max_retries + 1} for user {user_id} after {delay:.1f}s delay")
                    time.sleep(delay)
                else:
                    delay = 0.0
                
                # Create recovery attempt
                attempt = RecoveryAttempt(
                    attempt_number=attempt_num + 1,
                    status=RecoveryStatus.IN_PROGRESS,
                    start_time=datetime.now(),
                    end_time=None,
                    duration=None,
                    error_message=None,
                    actions_taken=[],
                    delay_before_attempt=delay
                )
                
                try:
                    # Attempt recovery based on failure type
                    recovery_result = self._attempt_recovery(user_id, company_name, failure_type, failure_context, attempt)
                    
                    # Update attempt with results
                    attempt.end_time = datetime.now()
                    attempt.duration = (attempt.end_time - attempt.start_time).total_seconds()
                    attempt.actions_taken = recovery_result.get("actions", [])
                    
                    if recovery_result["success"]:
                        attempt.status = RecoveryStatus.SUCCESS
                        recovery_session.attempts.append(attempt)
                        recovery_session.final_status = RecoveryStatus.SUCCESS
                        recovery_session.success_on_attempt = attempt_num + 1
                        
                        logger.info(f"Recovery successful for user {user_id} on attempt {attempt_num + 1}")
                        break
                    else:
                        attempt.status = RecoveryStatus.FAILED
                        attempt.error_message = recovery_result.get("error", "Unknown error")
                        last_error = attempt.error_message
                        
                except Exception as e:
                    attempt.end_time = datetime.now()
                    attempt.duration = (attempt.end_time - attempt.start_time).total_seconds()
                    attempt.status = RecoveryStatus.FAILED
                    attempt.error_message = str(e)
                    last_error = str(e)
                    logger.error(f"Recovery attempt {attempt_num + 1} failed for user {user_id}: {str(e)}")
                
                recovery_session.attempts.append(attempt)
                self._save_recovery_attempt(session_id, attempt)
            
            # Finalize recovery session
            recovery_session.end_time = datetime.now()
            recovery_session.total_recovery_time = (recovery_session.end_time - recovery_session.start_time).total_seconds()
            
            if recovery_session.final_status != RecoveryStatus.SUCCESS:
                recovery_session.final_status = RecoveryStatus.RETRY_EXHAUSTED
                logger.error(f"Recovery failed for user {user_id} after {max_retries + 1} attempts")
            
            # Update recovery session in database
            self._update_recovery_session(recovery_session)
            
            # Update user recovery statistics
            self._update_recovery_statistics(user_id, recovery_session)
            
            # Prepare result
            result = {
                "success": recovery_session.final_status == RecoveryStatus.SUCCESS,
                "session_id": session_id,
                "failure_type": failure_type.value,
                "total_attempts": len(recovery_session.attempts),
                "successful_attempt": recovery_session.success_on_attempt,
                "total_recovery_time": recovery_session.total_recovery_time,
                "final_status": recovery_session.final_status.value,
                "attempts": [self._attempt_to_dict(attempt) for attempt in recovery_session.attempts],
                "timestamp": datetime.now().isoformat()
            }
            
            if not result["success"]:
                result["error"] = f"Recovery failed after {max_retries + 1} attempts. Last error: {last_error}"
                result["recommendations"] = self._get_failure_recommendations(failure_type, recovery_session)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during recovery session for user {user_id}: {str(e)}")
            recovery_session.end_time = datetime.now()
            recovery_session.final_status = RecoveryStatus.FAILED
            recovery_session.total_recovery_time = (recovery_session.end_time - recovery_session.start_time).total_seconds()
            self._update_recovery_session(recovery_session)
            
            return {
                "success": False,
                "session_id": session_id,
                "error": f"Recovery system error: {str(e)}",
                "failure_type": failure_type.value,
                "total_attempts": len(recovery_session.attempts),
                "total_recovery_time": recovery_session.total_recovery_time,
                "final_status": RecoveryStatus.FAILED.value,
                "timestamp": datetime.now().isoformat(),
                "recommendations": ["Contact system administrator"]
            }
    
    def _classify_failure_type(self, failure_context: Dict[str, Any]) -> FailureType:
        """Classify the type of failure based on context"""
        error_message = failure_context.get("error_message", "").lower()
        error_type = failure_context.get("error_type", "").lower()
        
        if "dependency" in error_message or "import" in error_message or error_type == "dependency" or error_type == "dependency_missing":
            return FailureType.DEPENDENCY_MISSING
        elif "schema" in error_message or "column" in error_message or error_type == "schema" or error_type == "schema_error":
            return FailureType.SCHEMA_ERROR
        elif "storage" in error_message or "directory" in error_message or "permission" in error_message or error_type == "storage_error":
            return FailureType.STORAGE_ERROR
        elif "initialization" in error_message or error_type == "initialization" or error_type == "initialization_failed":
            return FailureType.INITIALIZATION_FAILED
        elif "performance" in error_message or "timeout" in error_message or error_type == "performance_degraded":
            return FailureType.PERFORMANCE_DEGRADED
        else:
            return FailureType.UNKNOWN
    
    def _attempt_recovery(
        self, 
        user_id: str, 
        company_name: str, 
        failure_type: FailureType, 
        failure_context: Dict[str, Any],
        attempt: RecoveryAttempt
    ) -> Dict[str, Any]:
        """Attempt recovery based on failure type"""
        try:
            if failure_type == FailureType.DEPENDENCY_MISSING:
                return self._recover_dependency_failure(user_id, failure_context, attempt)
            elif failure_type == FailureType.INITIALIZATION_FAILED:
                return self._recover_initialization_failure(user_id, company_name, failure_context, attempt)
            elif failure_type == FailureType.SCHEMA_ERROR:
                return self._recover_schema_failure(user_id, failure_context, attempt)
            elif failure_type == FailureType.STORAGE_ERROR:
                return self._recover_storage_failure(user_id, failure_context, attempt)
            elif failure_type == FailureType.PERFORMANCE_DEGRADED:
                return self._recover_performance_failure(user_id, failure_context, attempt)
            else:
                return self._recover_generic_failure(user_id, company_name, failure_context, attempt)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "actions": [f"Recovery attempt failed: {str(e)}"]
            }
    
    def _recover_dependency_failure(self, user_id: str, failure_context: Dict[str, Any], attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Recover from dependency-related failures"""
        actions = []
        
        try:
            # Re-validate dependencies
            actions.append("Re-validating system dependencies")
            dependency_result = self.enhanced_rag_manager.dependency_validator.check_all_dependencies()
            
            # If critical dependencies are now available, reinitialize RAG system
            if not dependency_result.critical_missing:
                actions.append("Critical dependencies now available - reinitializing RAG system")
                self.enhanced_rag_manager._initialize_rag_system_with_validation()
                
                if self.enhanced_rag_manager.rag_system:
                    actions.append("RAG system successfully reinitialized")
                    return {"success": True, "actions": actions}
            
            # Try fallback system
            actions.append("Attempting fallback RAG system initialization")
            self.enhanced_rag_manager._initialize_fallback_rag_system()
            
            if self.enhanced_rag_manager.rag_system:
                actions.append("Fallback RAG system initialized successfully")
                return {"success": True, "actions": actions}
            
            actions.append("Unable to initialize any RAG system - dependencies still missing")
            return {"success": False, "actions": actions, "error": "Dependencies still missing"}
            
        except Exception as e:
            actions.append(f"Dependency recovery error: {str(e)}")
            return {"success": False, "actions": actions, "error": str(e)}
    
    def _recover_initialization_failure(self, user_id: str, company_name: str, failure_context: Dict[str, Any], attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Recover from RAG initialization failures"""
        actions = []
        
        try:
            # Clear corrupted state
            actions.append("Clearing corrupted initialization state")
            self.enhanced_rag_manager._clear_user_rag_data(user_id)
            
            # Reset database status
            actions.append("Resetting database initialization status")
            self.enhanced_rag_manager._update_rag_status(user_id, "not_initialized")
            
            # Recreate storage directories
            actions.append("Recreating storage directories")
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            # Wait a moment before retry (additional to exponential backoff)
            if attempt.attempt_number > 1:
                time.sleep(0.5)
                actions.append("Applied additional delay before retry")
            
            # Attempt fresh initialization
            actions.append("Attempting fresh RAG initialization")
            init_result = self.enhanced_rag_manager._enhanced_rag_initialization(user_id, company_name, force_reinit=True)
            
            if init_result["success"]:
                actions.append("Fresh initialization successful")
                return {"success": True, "actions": actions}
            else:
                actions.append(f"Fresh initialization failed: {init_result.get('error', 'Unknown error')}")
                return {"success": False, "actions": actions, "error": init_result.get('error', 'Unknown error')}
            
        except Exception as e:
            actions.append(f"Initialization recovery error: {str(e)}")
            return {"success": False, "actions": actions, "error": str(e)}
    
    def _recover_schema_failure(self, user_id: str, failure_context: Dict[str, Any], attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Recover from database schema failures"""
        actions = []
        
        try:
            # Attempt schema migration
            actions.append("Attempting database schema migration")
            
            if self.enhanced_rag_manager.schema_migrator:
                migration_result = self.enhanced_rag_manager.schema_migrator.execute_automatic_migration()
                
                migration_success = True
                for table_name, result in migration_result.items():
                    if result.status.value == "failed":
                        migration_success = False
                        break
                
                if migration_success:
                    actions.append("Schema migration completed successfully")
                    
                    # Test database connectivity
                    actions.append("Testing database connectivity")
                    db_check = self.enhanced_rag_manager._check_database_connectivity(user_id)
                    
                    if db_check["status"] == "pass":
                        actions.append("Database connectivity restored")
                        return {"success": True, "actions": actions}
                    else:
                        actions.append(f"Database connectivity test failed: {db_check.get('message', 'Unknown error')}")
                        return {"success": False, "actions": actions, "error": db_check.get('message', 'Database connectivity failed')}
                else:
                    actions.append("Schema migration failed")
                    return {"success": False, "actions": actions, "error": "Schema migration failed"}
            else:
                actions.append("Schema migrator not available")
                return {"success": False, "actions": actions, "error": "Schema migrator not available"}
            
        except Exception as e:
            actions.append(f"Schema recovery error: {str(e)}")
            return {"success": False, "actions": actions, "error": str(e)}
    
    def _recover_storage_failure(self, user_id: str, failure_context: Dict[str, Any], attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Recover from storage-related failures"""
        actions = []
        
        try:
            # Recreate storage directories
            actions.append("Recreating storage directories")
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    actions.append(f"Created directory: {dir_path}")
                except Exception as e:
                    actions.append(f"Failed to create directory {dir_path}: {str(e)}")
                    return {"success": False, "actions": actions, "error": f"Cannot create directory {dir_path}"}
            
            # Test write permissions
            actions.append("Testing write permissions")
            try:
                test_file = os.path.join(storage_path, "rag", f".recovery_test_{attempt.attempt_number}")
                with open(test_file, 'w') as f:
                    f.write(f"recovery test attempt {attempt.attempt_number}")
                os.remove(test_file)
                actions.append("Write permissions verified")
            except Exception as e:
                actions.append(f"Write permission test failed: {str(e)}")
                return {"success": False, "actions": actions, "error": f"Write permission test failed: {str(e)}"}
            
            return {"success": True, "actions": actions}
            
        except Exception as e:
            actions.append(f"Storage recovery error: {str(e)}")
            return {"success": False, "actions": actions, "error": str(e)}
    
    def _recover_performance_failure(self, user_id: str, failure_context: Dict[str, Any], attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Recover from performance-related failures"""
        actions = []
        
        try:
            # Clear any cached data that might be causing performance issues
            actions.append("Clearing performance-related caches")
            
            if self.enhanced_rag_manager.rag_system:
                # Clear user-specific caches if available
                try:
                    self.enhanced_rag_manager.rag_system._clear_user_cache(user_id)
                    actions.append("Cleared user-specific RAG cache")
                except AttributeError:
                    actions.append("User cache clearing not available")
                except Exception as e:
                    actions.append(f"Cache clearing failed: {str(e)}")
            
            # Wait for system to stabilize
            if attempt.attempt_number > 1:
                stabilization_time = min(2.0 * attempt.attempt_number, 10.0)
                time.sleep(stabilization_time)
                actions.append(f"Applied {stabilization_time:.1f}s stabilization delay")
            
            # Test performance
            actions.append("Testing RAG system performance")
            perf_check = self.enhanced_rag_manager._check_rag_performance(user_id)
            
            if perf_check["status"] == "pass":
                actions.append("Performance test passed")
                return {"success": True, "actions": actions}
            else:
                actions.append(f"Performance test failed: {perf_check.get('message', 'Unknown error')}")
                return {"success": False, "actions": actions, "error": perf_check.get('message', 'Performance test failed')}
            
        except Exception as e:
            actions.append(f"Performance recovery error: {str(e)}")
            return {"success": False, "actions": actions, "error": str(e)}
    
    def _recover_generic_failure(self, user_id: str, company_name: str, failure_context: Dict[str, Any], attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Generic recovery attempt for unknown failure types"""
        actions = []
        
        try:
            # Run comprehensive system validation
            actions.append("Running comprehensive system validation")
            validation_result = self.enhanced_rag_manager.startup_validation()
            
            if validation_result.overall_status == "healthy":
                actions.append("System validation passed - attempting reinitialization")
                
                # Attempt reinitialization
                init_result = self.enhanced_rag_manager.safe_initialize_rag(user_id, company_name, force_reinit=True)
                
                if init_result["success"]:
                    actions.append("Reinitialization successful")
                    return {"success": True, "actions": actions}
                else:
                    actions.append(f"Reinitialization failed: {init_result.get('error', 'Unknown error')}")
                    return {"success": False, "actions": actions, "error": init_result.get('error', 'Unknown error')}
            else:
                actions.append(f"System validation failed: {validation_result.overall_status}")
                actions.extend(validation_result.issues[:3])  # Add first 3 issues
                return {"success": False, "actions": actions, "error": f"System validation failed: {validation_result.overall_status}"}
            
        except Exception as e:
            actions.append(f"Generic recovery error: {str(e)}")
            return {"success": False, "actions": actions, "error": str(e)}
    
    def _save_recovery_session(self, session: RecoverySession):
        """Save recovery session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO rag_recovery_sessions 
                (session_id, user_id, failure_type, failure_context, max_retries, 
                 initial_delay, backoff_factor, start_time, end_time, final_status, 
                 total_recovery_time, success_on_attempt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id, session.user_id, session.failure_type.value,
                str(session.failure_context), session.max_retries, session.initial_delay,
                session.backoff_factor, session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.final_status.value, session.total_recovery_time,
                session.success_on_attempt
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving recovery session {session.session_id}: {str(e)}")
    
    def _update_recovery_session(self, session: RecoverySession):
        """Update recovery session in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE rag_recovery_sessions 
                SET end_time = ?, final_status = ?, total_recovery_time = ?, success_on_attempt = ?
                WHERE session_id = ?
            ''', (
                session.end_time.isoformat() if session.end_time else None,
                session.final_status.value, session.total_recovery_time,
                session.success_on_attempt, session.session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating recovery session {session.session_id}: {str(e)}")
    
    def _save_recovery_attempt(self, session_id: str, attempt: RecoveryAttempt):
        """Save recovery attempt to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rag_recovery_attempts 
                (session_id, attempt_number, status, start_time, end_time, duration, 
                 error_message, actions_taken, delay_before_attempt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, attempt.attempt_number, attempt.status.value,
                attempt.start_time.isoformat(),
                attempt.end_time.isoformat() if attempt.end_time else None,
                attempt.duration, attempt.error_message,
                '; '.join(attempt.actions_taken), attempt.delay_before_attempt
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving recovery attempt for session {session_id}: {str(e)}")
    
    def _update_recovery_statistics(self, user_id: str, session: RecoverySession):
        """Update recovery statistics for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current statistics
            cursor.execute('''
                SELECT total_recovery_sessions, successful_recoveries, failed_recoveries, 
                       average_recovery_time
                FROM rag_recovery_statistics 
                WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if result:
                total_sessions, successful, failed, avg_time = result
                total_sessions += 1
                
                if session.final_status == RecoveryStatus.SUCCESS:
                    successful += 1
                else:
                    failed += 1
                
                # Update average recovery time
                if session.total_recovery_time:
                    if avg_time:
                        avg_time = (avg_time * (total_sessions - 1) + session.total_recovery_time) / total_sessions
                    else:
                        avg_time = session.total_recovery_time
                
                success_rate = (successful / total_sessions * 100) if total_sessions > 0 else 0
                
                cursor.execute('''
                    UPDATE rag_recovery_statistics 
                    SET total_recovery_sessions = ?, successful_recoveries = ?, 
                        failed_recoveries = ?, average_recovery_time = ?, 
                        last_recovery_attempt = ?, recovery_success_rate = ?, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (total_sessions, successful, failed, avg_time, 
                      session.start_time.isoformat(), success_rate, user_id))
            else:
                # Insert new statistics record
                successful = 1 if session.final_status == RecoveryStatus.SUCCESS else 0
                failed = 0 if session.final_status == RecoveryStatus.SUCCESS else 1
                success_rate = 100.0 if successful else 0.0
                
                cursor.execute('''
                    INSERT INTO rag_recovery_statistics 
                    (user_id, total_recovery_sessions, successful_recoveries, 
                     failed_recoveries, average_recovery_time, last_recovery_attempt, 
                     recovery_success_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, 1, successful, failed, session.total_recovery_time,
                      session.start_time.isoformat(), success_rate))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating recovery statistics for user {user_id}: {str(e)}")
    
    def get_recovery_status_report(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive recovery status report for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with recovery status and statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recovery statistics
            cursor.execute('''
                SELECT total_recovery_sessions, successful_recoveries, failed_recoveries,
                       average_recovery_time, last_recovery_attempt, recovery_success_rate
                FROM rag_recovery_statistics 
                WHERE user_id = ?
            ''', (user_id,))
            
            stats_result = cursor.fetchone()
            
            if stats_result:
                total_sessions, successful, failed, avg_time, last_attempt, success_rate = stats_result
                statistics = {
                    "total_recovery_sessions": total_sessions,
                    "successful_recoveries": successful,
                    "failed_recoveries": failed,
                    "average_recovery_time": round(avg_time or 0, 2),
                    "last_recovery_attempt": last_attempt,
                    "recovery_success_rate": round(success_rate or 0, 2)
                }
            else:
                statistics = {
                    "total_recovery_sessions": 0,
                    "successful_recoveries": 0,
                    "failed_recoveries": 0,
                    "average_recovery_time": 0,
                    "last_recovery_attempt": None,
                    "recovery_success_rate": 0
                }
            
            # Get recent recovery sessions
            cursor.execute('''
                SELECT session_id, failure_type, start_time, end_time, final_status,
                       total_recovery_time, success_on_attempt, max_retries
                FROM rag_recovery_sessions 
                WHERE user_id = ?
                ORDER BY start_time DESC
                LIMIT 10
            ''', (user_id,))
            
            recent_sessions = []
            for row in cursor.fetchall():
                session_id, failure_type, start_time, end_time, final_status, recovery_time, success_attempt, max_retries = row
                recent_sessions.append({
                    "session_id": session_id,
                    "failure_type": failure_type,
                    "start_time": start_time,
                    "end_time": end_time,
                    "final_status": final_status,
                    "total_recovery_time": recovery_time,
                    "success_on_attempt": success_attempt,
                    "max_retries": max_retries
                })
            
            conn.close()
            
            return {
                "user_id": user_id,
                "statistics": statistics,
                "recent_sessions": recent_sessions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting recovery status report for user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "statistics": {
                    "total_recovery_sessions": 0,
                    "successful_recoveries": 0,
                    "failed_recoveries": 0,
                    "average_recovery_time": 0,
                    "last_recovery_attempt": None,
                    "recovery_success_rate": 0
                },
                "recent_sessions": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _attempt_to_dict(self, attempt: RecoveryAttempt) -> Dict[str, Any]:
        """Convert RecoveryAttempt to dictionary"""
        return {
            "attempt_number": attempt.attempt_number,
            "status": attempt.status.value,
            "start_time": attempt.start_time.isoformat(),
            "end_time": attempt.end_time.isoformat() if attempt.end_time else None,
            "duration": attempt.duration,
            "error_message": attempt.error_message,
            "actions_taken": attempt.actions_taken,
            "delay_before_attempt": attempt.delay_before_attempt
        }
    
    def _get_failure_recommendations(self, failure_type: FailureType, session: RecoverySession) -> List[str]:
        """Get recommendations based on failure type and recovery session results"""
        recommendations = []
        
        if failure_type == FailureType.DEPENDENCY_MISSING:
            recommendations.extend([
                "Install missing critical dependencies (sentence_transformers, faiss-cpu)",
                "Check Python environment and package manager configuration",
                "Verify system requirements are met"
            ])
        elif failure_type == FailureType.INITIALIZATION_FAILED:
            recommendations.extend([
                "Check user data integrity and permissions",
                "Verify storage directory accessibility",
                "Review RAG system logs for detailed error information"
            ])
        elif failure_type == FailureType.SCHEMA_ERROR:
            recommendations.extend([
                "Run manual database schema migration",
                "Check database file permissions and integrity",
                "Verify database configuration"
            ])
        elif failure_type == FailureType.STORAGE_ERROR:
            recommendations.extend([
                "Check disk space availability",
                "Verify directory permissions",
                "Ensure storage path is accessible"
            ])
        elif failure_type == FailureType.PERFORMANCE_DEGRADED:
            recommendations.extend([
                "Check system resource usage (CPU, memory)",
                "Clear system caches and temporary files",
                "Consider system optimization or hardware upgrade"
            ])
        else:
            recommendations.extend([
                "Contact system administrator for assistance",
                "Review system logs for detailed error information",
                "Check overall system health and configuration"
            ])
        
        # Add session-specific recommendations
        if session.total_recovery_time and session.total_recovery_time > 60:
            recommendations.append("Consider increasing system resources for faster recovery")
        
        if len(session.attempts) > 2:
            recommendations.append("Multiple recovery attempts failed - manual intervention may be required")
        
        return recommendations
    
    def get_system_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get system-wide recovery statistics
        
        Returns:
            Dictionary with system recovery statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall recovery statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN final_status = 'success' THEN 1 ELSE 0 END) as successful_sessions,
                    AVG(total_recovery_time) as avg_recovery_time,
                    COUNT(DISTINCT user_id) as users_with_recovery_attempts
                FROM rag_recovery_sessions
                WHERE start_time >= datetime('now', '-30 days')
            ''')
            
            result = cursor.fetchone()
            total_sessions, successful_sessions, avg_recovery_time, users_with_attempts = result or (0, 0, 0, 0)
            
            # Get failure type distribution
            cursor.execute('''
                SELECT failure_type, COUNT(*) as count
                FROM rag_recovery_sessions
                WHERE start_time >= datetime('now', '-30 days')
                GROUP BY failure_type
                ORDER BY count DESC
            ''')
            
            failure_distribution = {}
            for row in cursor.fetchall():
                failure_type, count = row
                failure_distribution[failure_type] = count
            
            # Get recent recovery trends
            cursor.execute('''
                SELECT 
                    DATE(start_time) as recovery_date,
                    COUNT(*) as sessions_count,
                    SUM(CASE WHEN final_status = 'success' THEN 1 ELSE 0 END) as successful_count
                FROM rag_recovery_sessions
                WHERE start_time >= datetime('now', '-7 days')
                GROUP BY DATE(start_time)
                ORDER BY recovery_date DESC
            ''')
            
            daily_trends = []
            for row in cursor.fetchall():
                recovery_date, sessions_count, successful_count = row
                success_rate = (successful_count / sessions_count * 100) if sessions_count > 0 else 0
                daily_trends.append({
                    "date": recovery_date,
                    "total_sessions": sessions_count,
                    "successful_sessions": successful_count,
                    "success_rate": round(success_rate, 2)
                })
            
            conn.close()
            
            overall_success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            return {
                "period": "Last 30 days",
                "overall_statistics": {
                    "total_recovery_sessions": total_sessions,
                    "successful_recoveries": successful_sessions,
                    "overall_success_rate": round(overall_success_rate, 2),
                    "average_recovery_time": round(avg_recovery_time or 0, 2),
                    "users_with_recovery_attempts": users_with_attempts
                },
                "failure_type_distribution": failure_distribution,
                "daily_trends": daily_trends,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system recovery statistics: {str(e)}")
            return {
                "period": "Last 30 days",
                "overall_statistics": {
                    "total_recovery_sessions": 0,
                    "successful_recoveries": 0,
                    "overall_success_rate": 0,
                    "average_recovery_time": 0,
                    "users_with_recovery_attempts": 0
                },
                "failure_type_distribution": {},
                "daily_trends": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup_old_recovery_data(self, days_to_keep: int = 90) -> Dict[str, Any]:
        """
        Clean up old recovery data to prevent database bloat
        
        Args:
            days_to_keep: Number of days of recovery data to keep
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count records to be deleted
            cursor.execute('''
                SELECT COUNT(*) FROM rag_recovery_sessions
                WHERE start_time < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            sessions_to_delete = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM rag_recovery_attempts
                WHERE session_id IN (
                    SELECT session_id FROM rag_recovery_sessions
                    WHERE start_time < datetime('now', '-{} days')
                )
            '''.format(days_to_keep))
            
            attempts_to_delete = cursor.fetchone()[0]
            
            # Delete old recovery attempts first (foreign key constraint)
            cursor.execute('''
                DELETE FROM rag_recovery_attempts
                WHERE session_id IN (
                    SELECT session_id FROM rag_recovery_sessions
                    WHERE start_time < datetime('now', '-{} days')
                )
            '''.format(days_to_keep))
            
            # Delete old recovery sessions
            cursor.execute('''
                DELETE FROM rag_recovery_sessions
                WHERE start_time < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {sessions_to_delete} recovery sessions and {attempts_to_delete} recovery attempts older than {days_to_keep} days")
            
            return {
                "success": True,
                "sessions_deleted": sessions_to_delete,
                "attempts_deleted": attempts_to_delete,
                "days_kept": days_to_keep,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old recovery data: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }