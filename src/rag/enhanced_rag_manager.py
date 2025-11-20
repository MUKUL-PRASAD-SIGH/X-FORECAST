"""
Enhanced RAG Manager with Integrated Reliability Improvements
Combines all reliability enhancements including startup validation, dependency management,
health monitoring, diagnostics, and comprehensive error handling.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import all reliability components
try:
    from .startup_validator import startup_validator, StartupValidationResult, ValidationStatus
    from .dependency_validator import dependency_validator, DependencyValidationResult
    from .health_monitor import health_monitor, HealthSnapshot, HealthAlert, AlertLevel
    from .diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentStatus, DiagnosticSeverity
    from src.database.schema_migrator import schema_migrator, MigrationResult, MigrationStatus
    from .rag_manager import RAGManager, RAGHealthStatus
except ImportError as e:
    logging.error(f"Failed to import reliability components: {e}")
    # Fallback imports for backward compatibility
    try:
        from src.rag.startup_validator import startup_validator, StartupValidationResult, ValidationStatus
        from src.rag.dependency_validator import dependency_validator, DependencyValidationResult
        from src.rag.health_monitor import health_monitor, HealthSnapshot, HealthAlert, AlertLevel
        from src.rag.diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentStatus, DiagnosticSeverity
        from src.database.schema_migrator import schema_migrator, MigrationResult, MigrationStatus
        from src.rag.rag_manager import RAGManager, RAGHealthStatus
    except ImportError:
        logging.error("Critical: Cannot import required RAG reliability components")
        raise

logger = logging.getLogger(__name__)

class RAGSystemStatus(Enum):
    """Enhanced RAG system status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    INITIALIZING = "initializing"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

@dataclass
class EnhancedRAGStatus:
    """Enhanced RAG status with reliability metrics"""
    user_id: str
    company_name: str
    system_status: RAGSystemStatus
    health_score: int  # 0-100
    is_initialized: bool
    startup_validated: bool
    dependencies_healthy: bool
    monitoring_active: bool
    last_health_check: Optional[datetime]
    active_alerts: List[HealthAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGOperationResult:
    """Result of RAG operations with enhanced error handling"""
    success: bool
    operation: str
    user_id: str
    message: str
    status: RAGSystemStatus
    duration: float
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recommendations: List[str] = field(default_factory=list)
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedRAGManager:
    """
    Enhanced RAG Manager with integrated reliability improvements
    """
    
    def __init__(self, 
                 users_db_path: str = "users.db",
                 rag_db_path: str = "rag_vector_db.db",
                 enable_monitoring: bool = True,
                 auto_recovery: bool = True):
        self.users_db_path = users_db_path
        self.rag_db_path = rag_db_path
        self.enable_monitoring = enable_monitoring
        self.auto_recovery = auto_recovery
        
        # Initialize base RAG manager
        self.base_rag_manager = RAGManager(users_db_path, rag_db_path)
        
        # System state
        self.system_validated = False
        self.last_validation_time = None
        self.operation_history: List[RAGOperationResult] = []
        
        # Initialize reliability components
        self._initialize_reliability_components()
    
    def _initialize_reliability_components(self):
        """Initialize all reliability components"""
        try:
            # Set up health monitoring callbacks
            if self.enable_monitoring:
                health_monitor.add_alert_callback(self._handle_health_alert)
            
            logger.info("Enhanced RAG Manager initialized with reliability components")
            
        except Exception as e:
            logger.error(f"Error initializing reliability components: {str(e)}")
    
    def startup_system_validation(self, force_validation: bool = False) -> StartupValidationResult:
        """
        Run comprehensive startup validation before any RAG operations
        
        Args:
            force_validation: Force validation even if recently validated
            
        Returns:
            StartupValidationResult with validation status
        """
        try:
            # Check if validation is needed
            if (not force_validation and 
                self.system_validated and 
                self.last_validation_time and 
                (datetime.now() - self.last_validation_time).total_seconds() < 3600):  # 1 hour
                
                logger.info("System validation skipped - recently validated")
                return StartupValidationResult(
                    overall_status=ValidationStatus.COMPLETED,
                    total_duration=0.0,
                    phase_results={},
                    system_health_score=100,
                    critical_issues=[],
                    warnings=[],
                    recommendations=["System recently validated"],
                    ready_for_operation=True
                )
            
            logger.info("🚀 Starting enhanced RAG system validation")
            
            # Run comprehensive startup validation
            validation_result = startup_validator.run_startup_validation()
            
            # Update system state
            self.system_validated = validation_result.ready_for_operation
            self.last_validation_time = datetime.now()
            
            # Start health monitoring if validation successful and monitoring enabled
            if (validation_result.ready_for_operation and 
                self.enable_monitoring and 
                not health_monitor.monitoring_active):
                health_monitor.start_continuous_monitoring()
            
            # Log validation results
            if validation_result.ready_for_operation:
                logger.info(f"✅ System validation completed successfully (Score: {validation_result.system_health_score}/100)")
            else:
                logger.error(f"❌ System validation failed - {len(validation_result.critical_issues)} critical issues")
                for issue in validation_result.critical_issues[:3]:
                    logger.error(f"   • {issue}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during startup validation: {str(e)}")
            return StartupValidationResult(
                overall_status=ValidationStatus.FAILED,
                total_duration=0.0,
                phase_results={},
                system_health_score=0,
                critical_issues=[f"Validation error: {str(e)}"],
                warnings=[],
                recommendations=["Check system configuration", "Review logs"],
                ready_for_operation=False
            )
    
    def initialize_rag_for_user(self, user_id: str, company_name: str, 
                               force_reinit: bool = False,
                               validate_system: bool = True) -> RAGOperationResult:
        """
        Initialize RAG system for user with enhanced reliability
        
        Args:
            user_id: User identifier
            company_name: Company name
            force_reinit: Force reinitialization
            validate_system: Run system validation before initialization
            
        Returns:
            RAGOperationResult with initialization status
        """
        start_time = time.time()
        operation = "initialize_rag"
        
        try:
            logger.info(f"🔧 Initializing RAG for user {user_id} ({company_name})")
            
            # Step 1: System validation (if requested)
            if validate_system:
                validation_result = self.startup_system_validation()
                if not validation_result.ready_for_operation:
                    return RAGOperationResult(
                        success=False,
                        operation=operation,
                        user_id=user_id,
                        message="System validation failed - cannot initialize RAG",
                        status=RAGSystemStatus.FAILED,
                        duration=time.time() - start_time,
                        error_details="System not ready for operation",
                        recommendations=validation_result.recommendations[:3]
                    )
            
            # Step 2: Pre-initialization checks
            pre_check_result = self._run_pre_initialization_checks(user_id, company_name)
            if not pre_check_result["success"]:
                # Attempt automatic recovery
                if self.auto_recovery:
                    recovery_result = self._attempt_initialization_recovery(user_id, pre_check_result)
                    if not recovery_result["success"]:
                        return RAGOperationResult(
                            success=False,
                            operation=operation,
                            user_id=user_id,
                            message="Pre-initialization checks failed and recovery unsuccessful",
                            status=RAGSystemStatus.FAILED,
                            duration=time.time() - start_time,
                            recovery_attempted=True,
                            recovery_successful=False,
                            error_details=pre_check_result.get("error"),
                            recommendations=pre_check_result.get("recommendations", [])
                        )
            
            # Step 3: Initialize RAG using base manager with enhanced error handling
            try:
                base_result = self.base_rag_manager.initialize_rag_for_user(user_id, company_name, force_reinit)
                
                if not base_result["success"]:
                    # Attempt recovery if auto-recovery is enabled
                    if self.auto_recovery:
                        recovery_result = self._attempt_initialization_recovery(user_id, base_result)
                        if recovery_result["success"]:
                            # Retry initialization after recovery
                            base_result = self.base_rag_manager.initialize_rag_for_user(user_id, company_name, True)
                
            except Exception as e:
                logger.error(f"RAG initialization failed for user {user_id}: {str(e)}")
                base_result = {
                    "success": False,
                    "error": str(e),
                    "status": "initialization_error"
                }
            
            # Step 4: Post-initialization validation
            if base_result["success"]:
                post_check_result = self._run_post_initialization_checks(user_id)
                if not post_check_result["success"]:
                    logger.warning(f"Post-initialization checks failed for user {user_id}: {post_check_result.get('error')}")
            
            # Step 5: Update health monitoring
            if base_result["success"] and self.enable_monitoring:
                try:
                    health_snapshot = health_monitor.perform_health_check()
                    logger.info(f"Health check after initialization - Score: {health_snapshot.overall_score}/100")
                except Exception as e:
                    logger.warning(f"Health check failed after initialization: {str(e)}")
            
            # Step 6: Generate result
            duration = time.time() - start_time
            
            if base_result["success"]:
                status = RAGSystemStatus.HEALTHY
                message = f"RAG system initialized successfully for {company_name}"
                recommendations = ["Upload documents to build knowledge base", "Test RAG functionality"]
            else:
                status = RAGSystemStatus.FAILED
                message = f"RAG initialization failed: {base_result.get('error', 'Unknown error')}"
                recommendations = ["Check system logs", "Verify dependencies", "Contact administrator"]
            
            result = RAGOperationResult(
                success=base_result["success"],
                operation=operation,
                user_id=user_id,
                message=message,
                status=status,
                duration=duration,
                recovery_attempted=self.auto_recovery and not base_result["success"],
                recommendations=recommendations,
                error_details=base_result.get("error"),
                metadata={
                    "company_name": company_name,
                    "force_reinit": force_reinit,
                    "system_validated": validate_system
                }
            )
            
            # Store operation history
            self.operation_history.append(result)
            
            # Log result
            if result.success:
                logger.info(f"✅ RAG initialization completed for user {user_id} in {duration:.2f}s")
            else:
                logger.error(f"❌ RAG initialization failed for user {user_id} in {duration:.2f}s: {result.error_details}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Critical error during RAG initialization for user {user_id}: {str(e)}")
            
            result = RAGOperationResult(
                success=False,
                operation=operation,
                user_id=user_id,
                message=f"Critical initialization error: {str(e)}",
                status=RAGSystemStatus.FAILED,
                duration=duration,
                error_details=str(e),
                recommendations=["Check system configuration", "Review error logs", "Contact administrator"]
            )
            
            self.operation_history.append(result)
            return result
    
    def get_enhanced_rag_status(self, user_id: str) -> EnhancedRAGStatus:
        """
        Get comprehensive RAG status with reliability metrics
        
        Args:
            user_id: User identifier
            
        Returns:
            EnhancedRAGStatus with complete status information
        """
        try:
            # Get base RAG status
            base_health = self.base_rag_manager.get_rag_health_status(user_id)
            
            # Get system diagnostics
            diagnostic_result = diagnostic_engine.run_comprehensive_diagnostics()
            
            # Get dependency status
            dependency_result = dependency_validator.check_all_dependencies()
            
            # Get active health alerts
            active_alerts = [alert for alert in health_monitor.active_alerts 
                           if not alert.resolved and alert.component in [user_id, "system", "rag_system"]]
            
            # Calculate enhanced health score
            health_score = self._calculate_enhanced_health_score(
                base_health, diagnostic_result, dependency_result, active_alerts
            )
            
            # Determine system status
            system_status = self._determine_system_status(
                base_health, diagnostic_result, dependency_result, active_alerts
            )
            
            # Generate recommendations
            recommendations = self._generate_status_recommendations(
                base_health, diagnostic_result, dependency_result, active_alerts
            )
            
            # Get performance metrics
            performance_metrics = self._collect_performance_metrics(user_id)
            
            return EnhancedRAGStatus(
                user_id=user_id,
                company_name=base_health.company_name,
                system_status=system_status,
                health_score=health_score,
                is_initialized=base_health.is_initialized,
                startup_validated=self.system_validated,
                dependencies_healthy=dependency_result.overall_status in ["healthy", "degraded"],
                monitoring_active=health_monitor.monitoring_active,
                last_health_check=base_health.last_updated,
                active_alerts=active_alerts,
                recommendations=recommendations,
                error_message=base_health.error_message,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error getting enhanced RAG status for user {user_id}: {str(e)}")
            
            return EnhancedRAGStatus(
                user_id=user_id,
                company_name="Unknown",
                system_status=RAGSystemStatus.UNKNOWN,
                health_score=0,
                is_initialized=False,
                startup_validated=False,
                dependencies_healthy=False,
                monitoring_active=False,
                last_health_check=None,
                error_message=str(e),
                recommendations=["Check system status", "Review error logs"]
            )
    
    def run_comprehensive_diagnostics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics with enhanced reporting
        
        Args:
            user_id: Optional user ID for user-specific diagnostics
            
        Returns:
            Dictionary with comprehensive diagnostic results
        """
        try:
            logger.info("🔍 Running comprehensive RAG system diagnostics")
            
            # System-wide diagnostics
            system_diagnostics = diagnostic_engine.run_comprehensive_diagnostics()
            
            # Dependency validation
            dependency_status = dependency_validator.get_system_status_report()
            
            # Health monitoring status
            health_status = health_monitor.get_health_status_report()
            
            # Startup validation status
            validation_report = startup_validator.get_validation_report()
            
            # User-specific diagnostics (if user_id provided)
            user_diagnostics = None
            if user_id:
                user_diagnostics = self.base_rag_manager.run_rag_diagnostics(user_id)
            
            # System-wide RAG status
            system_rag_status = self.base_rag_manager.get_system_wide_rag_status()
            
            # Compile comprehensive report
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_diagnostics": {
                    "overall_status": system_diagnostics.overall_status.value,
                    "overall_score": system_diagnostics.overall_score,
                    "components": {name: comp.status.value for name, comp in system_diagnostics.components.items()},
                    "critical_issues": len([i for i in system_diagnostics.issues if i.severity == DiagnosticSeverity.CRITICAL]),
                    "total_issues": len(system_diagnostics.issues),
                    "recommendations": system_diagnostics.recommendations[:5]
                },
                "dependency_status": {
                    "overall_status": dependency_status["overall_status"],
                    "available_dependencies": dependency_status["summary"]["available"],
                    "critical_missing": dependency_status["summary"]["critical_missing"],
                    "optional_missing": dependency_status["summary"]["optional_missing"],
                    "recommendations": dependency_status["recommendations"][:3]
                },
                "health_monitoring": {
                    "monitoring_active": health_status["monitoring_active"],
                    "current_health_score": health_status["current_health"]["overall_score"],
                    "active_alerts": health_status["alerts"]["active_count"],
                    "critical_alerts": health_status["alerts"]["critical_count"],
                    "monitoring_duration_hours": health_status["monitoring_stats"]["monitoring_duration_hours"]
                },
                "startup_validation": {
                    "last_validation_status": validation_report.get("overall_status", "not_run"),
                    "system_health_score": validation_report.get("system_health_score", 0),
                    "ready_for_operation": validation_report.get("ready_for_operation", False),
                    "critical_issues": len(validation_report.get("critical_issues", [])),
                    "recommendations": validation_report.get("top_recommendations", [])
                },
                "rag_system_status": {
                    "total_users": system_rag_status.get("total_users", 0),
                    "initialized_users": system_rag_status.get("initialized_users", 0),
                    "initialization_rate": system_rag_status.get("initialization_rate", 0),
                    "system_health": system_rag_status.get("system_health", "unknown")
                }
            }
            
            # Add user-specific diagnostics if available
            if user_diagnostics:
                report["user_diagnostics"] = {
                    "user_id": user_id,
                    "overall_status": user_diagnostics["overall_status"],
                    "checks_passed": len([c for c in user_diagnostics["checks"].values() if c.get("status") == "pass"]),
                    "checks_failed": len([c for c in user_diagnostics["checks"].values() if c.get("status") == "fail"]),
                    "recommendations": user_diagnostics["recommendations"][:3]
                }
            
            # Generate overall assessment
            report["overall_assessment"] = self._generate_overall_assessment(report)
            
            logger.info(f"📊 Comprehensive diagnostics completed - Overall: {report['overall_assessment']['status']}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error running comprehensive diagnostics: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "overall_assessment": {
                    "status": "failed",
                    "message": "Diagnostics failed to complete",
                    "recommendations": ["Check system configuration", "Review error logs"]
                }
            }
    
    def handle_system_recovery(self, recovery_type: str = "auto", 
                             target_user: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle system recovery operations
        
        Args:
            recovery_type: Type of recovery ("auto", "manual", "full")
            target_user: Specific user for targeted recovery
            
        Returns:
            Dictionary with recovery results
        """
        try:
            logger.info(f"🔧 Starting system recovery - Type: {recovery_type}")
            
            recovery_results = {
                "recovery_type": recovery_type,
                "timestamp": datetime.now().isoformat(),
                "operations": [],
                "success": True,
                "errors": []
            }
            
            # Schema migration recovery
            try:
                migration_result = schema_migrator.execute_automatic_migration()
                recovery_results["operations"].append({
                    "operation": "schema_migration",
                    "success": all(r.status.value != "failed" for r in migration_result.values()),
                    "details": {name: r.status.value for name, r in migration_result.items()}
                })
            except Exception as e:
                recovery_results["errors"].append(f"Schema migration failed: {str(e)}")
                recovery_results["success"] = False
            
            # Dependency recovery (guidance only)
            try:
                dep_result = dependency_validator.check_all_dependencies()
                if dep_result.critical_missing:
                    recovery_results["operations"].append({
                        "operation": "dependency_check",
                        "success": False,
                        "missing_dependencies": dep_result.critical_missing,
                        "installation_instructions": dep_result.installation_instructions
                    })
                    recovery_results["success"] = False
                else:
                    recovery_results["operations"].append({
                        "operation": "dependency_check",
                        "success": True,
                        "message": "All critical dependencies available"
                    })
            except Exception as e:
                recovery_results["errors"].append(f"Dependency check failed: {str(e)}")
            
            # Health monitoring recovery
            try:
                if not health_monitor.monitoring_active and self.enable_monitoring:
                    health_monitor.start_continuous_monitoring()
                    recovery_results["operations"].append({
                        "operation": "health_monitoring",
                        "success": True,
                        "message": "Health monitoring restarted"
                    })
            except Exception as e:
                recovery_results["errors"].append(f"Health monitoring recovery failed: {str(e)}")
            
            # User-specific recovery
            if target_user and recovery_type in ["manual", "full"]:
                try:
                    user_recovery = self._recover_user_rag_system(target_user)
                    recovery_results["operations"].append({
                        "operation": "user_rag_recovery",
                        "user_id": target_user,
                        "success": user_recovery["success"],
                        "details": user_recovery
                    })
                    if not user_recovery["success"]:
                        recovery_results["success"] = False
                except Exception as e:
                    recovery_results["errors"].append(f"User RAG recovery failed: {str(e)}")
                    recovery_results["success"] = False
            
            # System validation after recovery
            if recovery_type == "full":
                try:
                    validation_result = self.startup_system_validation(force_validation=True)
                    recovery_results["operations"].append({
                        "operation": "system_validation",
                        "success": validation_result.ready_for_operation,
                        "health_score": validation_result.system_health_score,
                        "critical_issues": len(validation_result.critical_issues)
                    })
                    if not validation_result.ready_for_operation:
                        recovery_results["success"] = False
                except Exception as e:
                    recovery_results["errors"].append(f"System validation failed: {str(e)}")
            
            # Generate recovery summary
            successful_ops = len([op for op in recovery_results["operations"] if op["success"]])
            total_ops = len(recovery_results["operations"])
            
            recovery_results["summary"] = {
                "successful_operations": successful_ops,
                "total_operations": total_ops,
                "success_rate": (successful_ops / total_ops * 100) if total_ops > 0 else 0,
                "overall_success": recovery_results["success"]
            }
            
            if recovery_results["success"]:
                logger.info(f"✅ System recovery completed successfully ({successful_ops}/{total_ops} operations)")
            else:
                logger.error(f"❌ System recovery completed with errors ({successful_ops}/{total_ops} operations)")
            
            return recovery_results
            
        except Exception as e:
            logger.error(f"Critical error during system recovery: {str(e)}")
            return {
                "recovery_type": recovery_type,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
                "operations": [],
                "summary": {"overall_success": False}
            }
    
    # Private helper methods
    
    def _run_pre_initialization_checks(self, user_id: str, company_name: str) -> Dict[str, Any]:
        """Run pre-initialization validation checks"""
        try:
            checks = {
                "dependencies": dependency_validator.check_critical_dependencies(),
                "schema": schema_migrator.validate_schema(),
                "storage": self._check_user_storage(user_id)
            }
            
            issues = []
            recommendations = []
            
            # Check dependencies
            if checks["dependencies"].critical_missing:
                issues.append(f"Critical dependencies missing: {', '.join(checks['dependencies'].critical_missing)}")
                recommendations.extend([f"Install {dep}" for dep in checks["dependencies"].critical_missing[:2]])
            
            # Check schema
            for table_name, validation in checks["schema"].items():
                if not validation.is_valid:
                    issues.append(f"Schema issues in {table_name}: {', '.join(validation.missing_columns)}")
                    recommendations.append("Run database migration")
            
            # Check storage
            if not checks["storage"]["success"]:
                issues.append(f"Storage issues: {checks['storage']['error']}")
                recommendations.append("Check storage permissions")
            
            return {
                "success": len(issues) == 0,
                "issues": issues,
                "recommendations": recommendations,
                "checks": checks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": ["Check system configuration"]
            }
    
    def _run_post_initialization_checks(self, user_id: str) -> Dict[str, Any]:
        """Run post-initialization validation checks"""
        try:
            # Check if user RAG is actually initialized
            status = self.base_rag_manager.check_rag_initialization_status(user_id)
            
            if not status["success"] or not status["is_initialized"]:
                return {
                    "success": False,
                    "error": "RAG initialization verification failed",
                    "recommendations": ["Check initialization logs", "Retry initialization"]
                }
            
            # Check if storage directories were created
            storage_check = self._check_user_storage(user_id)
            if not storage_check["success"]:
                return {
                    "success": False,
                    "error": "Storage verification failed",
                    "recommendations": ["Check directory permissions"]
                }
            
            return {"success": True, "message": "Post-initialization checks passed"}
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": ["Check system logs"]
            }
    
    def _attempt_initialization_recovery(self, user_id: str, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt automatic recovery for initialization failures"""
        try:
            logger.info(f"🔧 Attempting automatic recovery for user {user_id}")
            
            recovery_actions = []
            
            # Try schema migration
            try:
                migration_result = schema_migrator.execute_automatic_migration()
                recovery_actions.append("schema_migration")
            except Exception as e:
                logger.warning(f"Schema migration recovery failed: {str(e)}")
            
            # Try creating storage directories
            try:
                storage_result = self._ensure_user_storage(user_id)
                if storage_result["success"]:
                    recovery_actions.append("storage_creation")
            except Exception as e:
                logger.warning(f"Storage creation recovery failed: {str(e)}")
            
            # Try clearing corrupted data
            try:
                self.base_rag_manager._clear_user_rag_data(user_id)
                recovery_actions.append("data_cleanup")
            except Exception as e:
                logger.warning(f"Data cleanup recovery failed: {str(e)}")
            
            success = len(recovery_actions) > 0
            
            return {
                "success": success,
                "recovery_actions": recovery_actions,
                "message": f"Recovery attempted: {', '.join(recovery_actions)}" if success else "No recovery actions successful"
            }
            
        except Exception as e:
            logger.error(f"Recovery attempt failed for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_user_storage(self, user_id: str) -> Dict[str, Any]:
        """Check user storage directories"""
        try:
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                if not os.path.exists(dir_path):
                    return {
                        "success": False,
                        "error": f"Missing directory: {dir_name}",
                        "missing_directory": dir_name
                    }
            
            return {"success": True, "message": "All storage directories exist"}
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _ensure_user_storage(self, user_id: str) -> Dict[str, Any]:
        """Ensure user storage directories exist"""
        try:
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            created_dirs = []
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    created_dirs.append(dir_name)
            
            return {
                "success": True,
                "created_directories": created_dirs,
                "message": f"Created {len(created_dirs)} directories" if created_dirs else "All directories already exist"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_enhanced_health_score(self, base_health: RAGHealthStatus,
                                       diagnostic_result: SystemDiagnosticResult,
                                       dependency_result: DependencyValidationResult,
                                       active_alerts: List[HealthAlert]) -> int:
        """Calculate enhanced health score from multiple sources"""
        try:
            # Start with base health score
            if base_health.status == "healthy":
                base_score = 100
            elif base_health.status == "degraded":
                base_score = 70
            elif base_health.status == "not_initialized":
                base_score = 50
            else:
                base_score = 20
            
            # Adjust based on system diagnostics
            diagnostic_adjustment = (diagnostic_result.overall_score - 100) * 0.3
            
            # Adjust based on dependencies
            if dependency_result.overall_status == "healthy":
                dependency_adjustment = 0
            elif dependency_result.overall_status == "degraded":
                dependency_adjustment = -15
            else:
                dependency_adjustment = -30
            
            # Adjust based on active alerts
            alert_adjustment = 0
            for alert in active_alerts:
                if alert.level == AlertLevel.CRITICAL:
                    alert_adjustment -= 20
                elif alert.level == AlertLevel.WARNING:
                    alert_adjustment -= 10
                elif alert.level == AlertLevel.EMERGENCY:
                    alert_adjustment -= 30
            
            # Calculate final score
            final_score = base_score + diagnostic_adjustment + dependency_adjustment + alert_adjustment
            
            return max(0, min(100, int(final_score)))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced health score: {str(e)}")
            return 0
    
    def _determine_system_status(self, base_health: RAGHealthStatus,
                               diagnostic_result: SystemDiagnosticResult,
                               dependency_result: DependencyValidationResult,
                               active_alerts: List[HealthAlert]) -> RAGSystemStatus:
        """Determine overall system status"""
        try:
            # Check for critical failures
            if (diagnostic_result.overall_status == ComponentStatus.FAILED or
                dependency_result.overall_status == "failed" or
                any(alert.level == AlertLevel.EMERGENCY for alert in active_alerts)):
                return RAGSystemStatus.FAILED
            
            # Check for degraded status
            if (base_health.status in ["degraded", "not_initialized"] or
                diagnostic_result.overall_status == ComponentStatus.DEGRADED or
                dependency_result.overall_status == "degraded" or
                any(alert.level == AlertLevel.CRITICAL for alert in active_alerts)):
                return RAGSystemStatus.DEGRADED
            
            # Check if initializing
            if not base_health.is_initialized:
                return RAGSystemStatus.INITIALIZING
            
            # Default to healthy
            return RAGSystemStatus.HEALTHY
            
        except Exception as e:
            logger.error(f"Error determining system status: {str(e)}")
            return RAGSystemStatus.UNKNOWN
    
    def _generate_status_recommendations(self, base_health: RAGHealthStatus,
                                       diagnostic_result: SystemDiagnosticResult,
                                       dependency_result: DependencyValidationResult,
                                       active_alerts: List[HealthAlert]) -> List[str]:
        """Generate recommendations based on system status"""
        recommendations = []
        
        try:
            # Base health recommendations
            if not base_health.is_initialized:
                recommendations.append("Initialize RAG system for this user")
            elif base_health.total_documents == 0:
                recommendations.append("Upload documents to build knowledge base")
            
            # Diagnostic recommendations
            recommendations.extend(diagnostic_result.recommendations[:2])
            
            # Dependency recommendations
            if dependency_result.critical_missing:
                recommendations.append(f"Install critical dependencies: {', '.join(dependency_result.critical_missing[:2])}")
            
            # Alert-based recommendations
            for alert in active_alerts[:2]:  # Top 2 alerts
                if alert.metadata.get("recommendations"):
                    recommendations.extend(alert.metadata["recommendations"][:1])
            
            # Remove duplicates and limit
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen and len(unique_recommendations) < 5:
                    seen.add(rec)
                    unique_recommendations.append(rec)
            
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Check system status", "Review error logs"]
    
    def _collect_performance_metrics(self, user_id: str) -> Dict[str, Any]:
        """Collect performance metrics for user"""
        try:
            metrics = {}
            
            # Get document counts
            doc_counts = self.base_rag_manager._get_document_counts(user_id)
            metrics.update(doc_counts)
            
            # Get last update time
            last_update = self.base_rag_manager._get_last_update_time(user_id)
            if last_update:
                metrics["last_update"] = last_update.isoformat()
                metrics["days_since_update"] = (datetime.now() - last_update).days
            
            # Get system performance metrics
            if hasattr(diagnostic_engine, 'diagnostic_history') and diagnostic_engine.diagnostic_history:
                latest_diagnostic = diagnostic_engine.diagnostic_history[-1]
                metrics.update({
                    "database_query_time": latest_diagnostic.performance_metrics.database_query_time,
                    "memory_usage_mb": latest_diagnostic.performance_metrics.memory_usage_mb,
                    "cpu_usage_percent": latest_diagnostic.performance_metrics.cpu_usage_percent
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
            return {}
    
    def _recover_user_rag_system(self, user_id: str) -> Dict[str, Any]:
        """Recover RAG system for specific user"""
        try:
            # Get user info
            status = self.base_rag_manager.check_rag_initialization_status(user_id)
            if not status["success"]:
                return {"success": False, "error": "Cannot get user status"}
            
            company_name = status.get("company_name", "Unknown")
            
            # Reset and reinitialize
            reset_result = self.base_rag_manager.reset_rag_system(user_id, company_name)
            
            return {
                "success": reset_result["success"],
                "message": reset_result.get("message", "User RAG system recovery completed"),
                "error": reset_result.get("error")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall system assessment from diagnostic report"""
        try:
            # Collect status indicators
            system_healthy = report["system_diagnostics"]["overall_status"] == "healthy"
            deps_healthy = report["dependency_status"]["overall_status"] in ["healthy", "degraded"]
            monitoring_active = report["health_monitoring"]["monitoring_active"]
            validation_ready = report["startup_validation"]["ready_for_operation"]
            
            # Calculate overall score
            scores = [
                report["system_diagnostics"]["overall_score"],
                report["health_monitoring"]["current_health_score"],
                report["startup_validation"]["system_health_score"]
            ]
            avg_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
            
            # Determine overall status
            if system_healthy and deps_healthy and validation_ready and avg_score >= 80:
                status = "healthy"
                message = "All systems operational"
            elif avg_score >= 60 and deps_healthy:
                status = "degraded"
                message = "System operational with some issues"
            else:
                status = "critical"
                message = "System requires immediate attention"
            
            # Generate top recommendations
            all_recommendations = []
            all_recommendations.extend(report["system_diagnostics"]["recommendations"])
            all_recommendations.extend(report["dependency_status"]["recommendations"])
            all_recommendations.extend(report["startup_validation"]["recommendations"])
            
            # Remove duplicates and get top 3
            seen = set()
            top_recommendations = []
            for rec in all_recommendations:
                if rec not in seen and len(top_recommendations) < 3:
                    seen.add(rec)
                    top_recommendations.append(rec)
            
            return {
                "status": status,
                "overall_score": int(avg_score),
                "message": message,
                "system_ready": validation_ready,
                "monitoring_active": monitoring_active,
                "top_recommendations": top_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating overall assessment: {str(e)}")
            return {
                "status": "unknown",
                "overall_score": 0,
                "message": "Assessment failed",
                "system_ready": False,
                "monitoring_active": False,
                "top_recommendations": ["Check system configuration"]
            }
    
    def _handle_health_alert(self, alert: HealthAlert):
        """Handle health monitoring alerts"""
        try:
            logger.warning(f"🚨 Health Alert: {alert.level.value.upper()} - {alert.title}")
            
            # Auto-recovery for certain alert types
            if self.auto_recovery and alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                if "dependency" in alert.component.lower():
                    logger.info("Attempting dependency recovery...")
                    # Could trigger dependency reinstallation guidance
                elif "database" in alert.component.lower():
                    logger.info("Attempting database recovery...")
                    # Could trigger schema migration
                elif "performance" in alert.component.lower():
                    logger.info("Performance alert - monitoring...")
                    # Could trigger performance optimization
            
        except Exception as e:
            logger.error(f"Error handling health alert: {str(e)}")

# Global instance
enhanced_rag_manager = EnhancedRAGManager()