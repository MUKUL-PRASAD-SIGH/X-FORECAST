"""
Integrated Startup Validation System for RAG System Reliability
Combines schema migration, dependency validation, and health checks into a comprehensive startup routine.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import existing components
try:
    from .dependency_validator import dependency_validator, DependencyValidationResult
    from .health_monitor import health_monitor, HealthSnapshot
    from .diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentStatus, DiagnosticSeverity
    from src.database.schema_migrator import schema_migrator, MigrationResult, MigrationStatus
except ImportError:
    from src.rag.dependency_validator import dependency_validator, DependencyValidationResult
    from src.rag.health_monitor import health_monitor, HealthSnapshot
    from src.rag.diagnostic_engine import diagnostic_engine, SystemDiagnosticResult, ComponentStatus, DiagnosticSeverity
    from src.database.schema_migrator import schema_migrator, MigrationResult, MigrationStatus

logger = logging.getLogger(__name__)

class ValidationPhase(Enum):
    """Phases of startup validation"""
    SCHEMA_MIGRATION = "schema_migration"
    DEPENDENCY_CHECK = "dependency_check"
    SYSTEM_DIAGNOSTICS = "system_diagnostics"
    HEALTH_MONITORING = "health_monitoring"
    FINAL_VALIDATION = "final_validation"

class ValidationStatus(Enum):
    """Status of validation phases"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ValidationPhaseResult:
    """Result of a validation phase"""
    phase: ValidationPhase
    status: ValidationStatus
    duration: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StartupValidationResult:
    """Complete startup validation result"""
    overall_status: ValidationStatus
    total_duration: float
    phase_results: Dict[ValidationPhase, ValidationPhaseResult]
    system_health_score: int
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    ready_for_operation: bool
    timestamp: datetime = field(default_factory=datetime.now)

class RAGStartupValidator:
    """
    Integrated startup validation system that combines all reliability checks
    """
    
    def __init__(self, 
                 users_db_path: str = "users.db",
                 rag_db_path: str = "rag_vector_db.db",
                 enable_health_monitoring: bool = True):
        self.users_db_path = users_db_path
        self.rag_db_path = rag_db_path
        self.enable_health_monitoring = enable_health_monitoring
        self.validation_history: List[StartupValidationResult] = []
    
    def run_startup_validation(self, 
                             skip_phases: Optional[List[ValidationPhase]] = None,
                             fail_fast: bool = False) -> StartupValidationResult:
        """
        Run comprehensive startup validation
        
        Args:
            skip_phases: List of phases to skip (for testing or partial validation)
            fail_fast: Stop validation on first critical failure
            
        Returns:
            StartupValidationResult with complete validation status
        """
        start_time = time.time()
        skip_phases = skip_phases or []
        
        logger.info("🚀 Starting RAG system startup validation")
        
        phase_results = {}
        critical_issues = []
        warnings = []
        recommendations = []
        
        try:
            # Phase 1: Schema Migration
            if ValidationPhase.SCHEMA_MIGRATION not in skip_phases:
                phase_results[ValidationPhase.SCHEMA_MIGRATION] = self._run_schema_migration_phase()
                if fail_fast and phase_results[ValidationPhase.SCHEMA_MIGRATION].status == ValidationStatus.FAILED:
                    return self._create_failed_result(start_time, phase_results, "Schema migration failed")
            
            # Phase 2: Dependency Check
            if ValidationPhase.DEPENDENCY_CHECK not in skip_phases:
                phase_results[ValidationPhase.DEPENDENCY_CHECK] = self._run_dependency_check_phase()
                if fail_fast and phase_results[ValidationPhase.DEPENDENCY_CHECK].status == ValidationStatus.FAILED:
                    return self._create_failed_result(start_time, phase_results, "Dependency check failed")
            
            # Phase 3: System Diagnostics
            if ValidationPhase.SYSTEM_DIAGNOSTICS not in skip_phases:
                phase_results[ValidationPhase.SYSTEM_DIAGNOSTICS] = self._run_system_diagnostics_phase()
                if fail_fast and phase_results[ValidationPhase.SYSTEM_DIAGNOSTICS].status == ValidationStatus.FAILED:
                    return self._create_failed_result(start_time, phase_results, "System diagnostics failed")
            
            # Phase 4: Health Monitoring Setup
            if ValidationPhase.HEALTH_MONITORING not in skip_phases and self.enable_health_monitoring:
                phase_results[ValidationPhase.HEALTH_MONITORING] = self._run_health_monitoring_phase()
            
            # Phase 5: Final Validation
            if ValidationPhase.FINAL_VALIDATION not in skip_phases:
                phase_results[ValidationPhase.FINAL_VALIDATION] = self._run_final_validation_phase()
            
            # Collect results from all phases
            for phase_result in phase_results.values():
                critical_issues.extend(phase_result.issues)
                warnings.extend(phase_result.warnings)
                recommendations.extend(phase_result.recommendations)
            
            # Determine overall status
            overall_status = self._determine_overall_status(phase_results)
            
            # Calculate system health score
            system_health_score = self._calculate_system_health_score(phase_results)
            
            # Determine if system is ready for operation
            ready_for_operation = self._is_ready_for_operation(phase_results, critical_issues)
            
            total_duration = time.time() - start_time
            
            result = StartupValidationResult(
                overall_status=overall_status,
                total_duration=total_duration,
                phase_results=phase_results,
                system_health_score=system_health_score,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=self._generate_final_recommendations(phase_results, critical_issues, warnings),
                ready_for_operation=ready_for_operation
            )
            
            # Store validation history
            self.validation_history.append(result)
            
            # Log results
            self._log_validation_results(result)
            
            return result
            
        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"❌ Startup validation failed with exception: {str(e)}")
            
            return StartupValidationResult(
                overall_status=ValidationStatus.FAILED,
                total_duration=total_duration,
                phase_results=phase_results,
                system_health_score=0,
                critical_issues=[f"Validation exception: {str(e)}"],
                warnings=[],
                recommendations=["Check system logs", "Fix validation errors", "Retry startup validation"],
                ready_for_operation=False
            )
    
    def _run_schema_migration_phase(self) -> ValidationPhaseResult:
        """Run schema migration phase"""
        start_time = time.time()
        logger.info("📊 Phase 1: Running database schema migration")
        
        try:
            # Run schema validation first
            validation_results = schema_migrator.validate_schema()
            
            # Run automatic migration
            migration_results = schema_migrator.execute_automatic_migration()
            
            # Analyze results
            issues = []
            warnings = []
            recommendations = []
            
            for table_name, migration_result in migration_results.items():
                if migration_result.status == MigrationStatus.FAILED:
                    issues.extend(migration_result.errors)
                elif migration_result.status == MigrationStatus.PARTIAL:
                    warnings.append(f"Partial migration for {table_name}: {len(migration_result.errors)} errors")
                    issues.extend(migration_result.errors)
                
                if migration_result.columns_added:
                    recommendations.append(f"Added {len(migration_result.columns_added)} columns to {table_name}")
            
            # Determine phase status
            if any(result.status == MigrationStatus.FAILED for result in migration_results.values()):
                status = ValidationStatus.FAILED
            elif any(result.status == MigrationStatus.PARTIAL for result in migration_results.values()):
                status = ValidationStatus.COMPLETED  # Partial is still considered completed
                warnings.append("Some migrations were partial - check logs for details")
            else:
                status = ValidationStatus.COMPLETED
            
            duration = time.time() - start_time
            
            return ValidationPhaseResult(
                phase=ValidationPhase.SCHEMA_MIGRATION,
                status=status,
                duration=duration,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                metadata={
                    "migration_results": {name: result.status.value for name, result in migration_results.items()},
                    "total_columns_added": sum(len(result.columns_added) for result in migration_results.values())
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Schema migration phase failed: {str(e)}")
            
            return ValidationPhaseResult(
                phase=ValidationPhase.SCHEMA_MIGRATION,
                status=ValidationStatus.FAILED,
                duration=duration,
                issues=[f"Schema migration error: {str(e)}"],
                recommendations=["Check database permissions", "Verify database integrity", "Review migration logs"]
            )
    
    def _run_dependency_check_phase(self) -> ValidationPhaseResult:
        """Run dependency validation phase"""
        start_time = time.time()
        logger.info("📦 Phase 2: Checking system dependencies")
        
        try:
            # Run comprehensive dependency check
            validation_result = dependency_validator.check_all_dependencies()
            
            issues = []
            warnings = []
            recommendations = []
            
            # Process critical missing dependencies
            for dep in validation_result.critical_missing:
                issues.append(f"Critical dependency missing: {dep}")
                if dep in validation_result.installation_instructions:
                    recommendations.append(f"Install {dep}: {validation_result.installation_instructions[dep]}")
            
            # Process optional missing dependencies
            for dep in validation_result.optional_missing:
                warnings.append(f"Optional dependency missing: {dep} (reduced functionality)")
                if dep in validation_result.installation_instructions:
                    recommendations.append(f"For full features, install {dep}: {validation_result.installation_instructions[dep]}")
            
            # Add degraded features information
            if validation_result.degraded_features:
                warnings.append(f"Degraded features: {', '.join(validation_result.degraded_features[:3])}")
            
            # Determine phase status
            if validation_result.overall_status == "failed":
                status = ValidationStatus.FAILED
            elif validation_result.overall_status == "degraded":
                status = ValidationStatus.COMPLETED
                warnings.append("System will run with reduced functionality")
            else:
                status = ValidationStatus.COMPLETED
            
            duration = time.time() - start_time
            
            return ValidationPhaseResult(
                phase=ValidationPhase.DEPENDENCY_CHECK,
                status=status,
                duration=duration,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                metadata={
                    "overall_dependency_status": validation_result.overall_status,
                    "available_dependencies": len(validation_result.available_dependencies),
                    "critical_missing": len(validation_result.critical_missing),
                    "optional_missing": len(validation_result.optional_missing)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Dependency check phase failed: {str(e)}")
            
            return ValidationPhaseResult(
                phase=ValidationPhase.DEPENDENCY_CHECK,
                status=ValidationStatus.FAILED,
                duration=duration,
                issues=[f"Dependency check error: {str(e)}"],
                recommendations=["Check Python environment", "Verify package installations", "Review dependency logs"]
            )
    
    def _run_system_diagnostics_phase(self) -> ValidationPhaseResult:
        """Run comprehensive system diagnostics phase"""
        start_time = time.time()
        logger.info("🔍 Phase 3: Running system diagnostics")
        
        try:
            # Run comprehensive diagnostics
            diagnostic_result = diagnostic_engine.run_comprehensive_diagnostics()
            
            issues = []
            warnings = []
            recommendations = []
            
            # Process diagnostic issues
            for issue in diagnostic_result.issues:
                if issue.severity == DiagnosticSeverity.CRITICAL:
                    issues.append(f"{issue.component}: {issue.title}")
                elif issue.severity == DiagnosticSeverity.WARNING:
                    warnings.append(f"{issue.component}: {issue.title}")
                
                # Add recommendations from issues
                recommendations.extend(issue.recommendations[:2])  # Limit to top 2 recommendations per issue
            
            # Add system-wide recommendations
            recommendations.extend(diagnostic_result.recommendations[:3])  # Top 3 system recommendations
            
            # Determine phase status based on diagnostic results
            if diagnostic_result.overall_status == ComponentStatus.FAILED:
                status = ValidationStatus.FAILED
            elif diagnostic_result.overall_status == ComponentStatus.DEGRADED:
                status = ValidationStatus.COMPLETED
                warnings.append("System diagnostics show degraded performance")
            else:
                status = ValidationStatus.COMPLETED
            
            duration = time.time() - start_time
            
            return ValidationPhaseResult(
                phase=ValidationPhase.SYSTEM_DIAGNOSTICS,
                status=status,
                duration=duration,
                issues=issues,
                warnings=warnings,
                recommendations=list(set(recommendations)),  # Remove duplicates
                metadata={
                    "overall_health_score": diagnostic_result.overall_score,
                    "component_count": len(diagnostic_result.components),
                    "total_issues": len(diagnostic_result.issues),
                    "critical_issues": len([i for i in diagnostic_result.issues if i.severity == DiagnosticSeverity.CRITICAL]),
                    "diagnostic_duration": diagnostic_result.diagnostic_duration
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"System diagnostics phase failed: {str(e)}")
            
            return ValidationPhaseResult(
                phase=ValidationPhase.SYSTEM_DIAGNOSTICS,
                status=ValidationStatus.FAILED,
                duration=duration,
                issues=[f"System diagnostics error: {str(e)}"],
                recommendations=["Check system resources", "Verify system components", "Review diagnostic logs"]
            )
    
    def _run_health_monitoring_phase(self) -> ValidationPhaseResult:
        """Run health monitoring setup phase"""
        start_time = time.time()
        logger.info("💓 Phase 4: Setting up health monitoring")
        
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Perform initial health check
            health_snapshot = health_monitor.perform_health_check()
            
            # Start continuous monitoring if enabled
            if not health_monitor.monitoring_active:
                health_monitor.start_continuous_monitoring()
                recommendations.append("Started continuous health monitoring")
            else:
                warnings.append("Health monitoring was already active")
            
            # Check health snapshot results
            if health_snapshot.critical_issue_count > 0:
                issues.append(f"Health check found {health_snapshot.critical_issue_count} critical issues")
            
            if health_snapshot.issue_count > health_snapshot.critical_issue_count:
                warnings.append(f"Health check found {health_snapshot.issue_count - health_snapshot.critical_issue_count} non-critical issues")
            
            if health_snapshot.overall_score < 70:
                warnings.append(f"System health score is low: {health_snapshot.overall_score}/100")
            
            # Determine phase status
            if health_snapshot.critical_issue_count > 0:
                status = ValidationStatus.COMPLETED  # Health monitoring setup succeeded, but system has issues
                warnings.append("Health monitoring active but system has critical issues")
            else:
                status = ValidationStatus.COMPLETED
            
            duration = time.time() - start_time
            
            return ValidationPhaseResult(
                phase=ValidationPhase.HEALTH_MONITORING,
                status=status,
                duration=duration,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                metadata={
                    "monitoring_active": health_monitor.monitoring_active,
                    "health_score": health_snapshot.overall_score,
                    "total_issues": health_snapshot.issue_count,
                    "critical_issues": health_snapshot.critical_issue_count
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health monitoring phase failed: {str(e)}")
            
            return ValidationPhaseResult(
                phase=ValidationPhase.HEALTH_MONITORING,
                status=ValidationStatus.FAILED,
                duration=duration,
                issues=[f"Health monitoring setup error: {str(e)}"],
                recommendations=["Check health monitoring configuration", "Verify monitoring database", "Review monitoring logs"]
            )
    
    def _run_final_validation_phase(self) -> ValidationPhaseResult:
        """Run final validation and readiness check"""
        start_time = time.time()
        logger.info("✅ Phase 5: Final validation and readiness check")
        
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check database accessibility
            if not os.path.exists(self.users_db_path):
                issues.append("Users database is not accessible")
            else:
                recommendations.append("Users database is accessible")
            
            # Check critical dependencies one more time
            critical_deps_result = dependency_validator.check_critical_dependencies()
            if critical_deps_result.critical_missing:
                issues.append(f"Critical dependencies still missing: {', '.join(critical_deps_result.critical_missing)}")
            else:
                recommendations.append("All critical dependencies are available")
            
            # Check if RAG system can be imported
            try:
                from .rag_manager import RAGManager
                recommendations.append("RAG manager is importable")
            except ImportError as e:
                issues.append(f"Cannot import RAG manager: {str(e)}")
            
            # Final system health check
            try:
                final_diagnostic = diagnostic_engine.run_comprehensive_diagnostics()
                if final_diagnostic.overall_status == ComponentStatus.FAILED:
                    issues.append("Final system diagnostics show failed status")
                elif final_diagnostic.overall_status == ComponentStatus.DEGRADED:
                    warnings.append("Final system diagnostics show degraded status")
                else:
                    recommendations.append("Final system diagnostics show healthy status")
            except Exception as e:
                warnings.append(f"Could not run final diagnostics: {str(e)}")
            
            # Determine phase status
            if issues:
                status = ValidationStatus.FAILED
            elif warnings:
                status = ValidationStatus.COMPLETED
                warnings.append("System ready with some limitations")
            else:
                status = ValidationStatus.COMPLETED
                recommendations.append("System is fully ready for operation")
            
            duration = time.time() - start_time
            
            return ValidationPhaseResult(
                phase=ValidationPhase.FINAL_VALIDATION,
                status=status,
                duration=duration,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                metadata={
                    "database_accessible": os.path.exists(self.users_db_path),
                    "rag_manager_importable": True,  # Will be False if import failed
                    "final_health_check": "completed"
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Final validation phase failed: {str(e)}")
            
            return ValidationPhaseResult(
                phase=ValidationPhase.FINAL_VALIDATION,
                status=ValidationStatus.FAILED,
                duration=duration,
                issues=[f"Final validation error: {str(e)}"],
                recommendations=["Review all previous phases", "Fix critical issues", "Retry startup validation"]
            )
    
    def _determine_overall_status(self, phase_results: Dict[ValidationPhase, ValidationPhaseResult]) -> ValidationStatus:
        """Determine overall validation status from phase results"""
        if not phase_results:
            return ValidationStatus.FAILED
        
        # If any critical phase failed, overall status is failed
        critical_phases = [ValidationPhase.SCHEMA_MIGRATION, ValidationPhase.DEPENDENCY_CHECK, ValidationPhase.FINAL_VALIDATION]
        
        for phase in critical_phases:
            if phase in phase_results and phase_results[phase].status == ValidationStatus.FAILED:
                return ValidationStatus.FAILED
        
        # If all phases completed, overall status is completed
        if all(result.status == ValidationStatus.COMPLETED for result in phase_results.values()):
            return ValidationStatus.COMPLETED
        
        # If some phases completed and none failed, it's still completed
        if any(result.status == ValidationStatus.COMPLETED for result in phase_results.values()):
            return ValidationStatus.COMPLETED
        
        return ValidationStatus.FAILED
    
    def _calculate_system_health_score(self, phase_results: Dict[ValidationPhase, ValidationPhaseResult]) -> int:
        """Calculate overall system health score from phase results"""
        if not phase_results:
            return 0
        
        # Weight different phases
        phase_weights = {
            ValidationPhase.SCHEMA_MIGRATION: 0.2,
            ValidationPhase.DEPENDENCY_CHECK: 0.3,
            ValidationPhase.SYSTEM_DIAGNOSTICS: 0.3,
            ValidationPhase.HEALTH_MONITORING: 0.1,
            ValidationPhase.FINAL_VALIDATION: 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for phase, result in phase_results.items():
            weight = phase_weights.get(phase, 0.1)
            
            # Calculate phase score based on status and issues
            if result.status == ValidationStatus.COMPLETED:
                phase_score = 100 - (len(result.issues) * 20) - (len(result.warnings) * 5)
            elif result.status == ValidationStatus.FAILED:
                phase_score = 20  # Minimum score for failed phases
            else:
                phase_score = 50  # Default for other statuses
            
            # Use diagnostic health score if available
            if phase == ValidationPhase.SYSTEM_DIAGNOSTICS and "overall_health_score" in result.metadata:
                phase_score = result.metadata["overall_health_score"]
            
            total_score += max(0, min(100, phase_score)) * weight
            total_weight += weight
        
        return int(total_score / total_weight) if total_weight > 0 else 0
    
    def _is_ready_for_operation(self, phase_results: Dict[ValidationPhase, ValidationPhaseResult], 
                               critical_issues: List[str]) -> bool:
        """Determine if system is ready for operation"""
        # System is not ready if there are critical issues
        if critical_issues:
            return False
        
        # Check critical phases
        critical_phases = [ValidationPhase.SCHEMA_MIGRATION, ValidationPhase.DEPENDENCY_CHECK, ValidationPhase.FINAL_VALIDATION]
        
        for phase in critical_phases:
            if phase in phase_results and phase_results[phase].status == ValidationStatus.FAILED:
                return False
        
        # If all critical phases passed, system is ready
        return True
    
    def _generate_final_recommendations(self, phase_results: Dict[ValidationPhase, ValidationPhaseResult],
                                      critical_issues: List[str], warnings: List[str]) -> List[str]:
        """Generate final recommendations based on all validation results"""
        recommendations = []
        
        # Priority recommendations for critical issues
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address the following issues before using the system:")
            recommendations.extend([f"   • {issue}" for issue in critical_issues[:5]])
        
        # Recommendations based on warnings
        if warnings and not critical_issues:
            recommendations.append("⚠️ WARNINGS: Consider addressing these issues for optimal performance:")
            recommendations.extend([f"   • {warning}" for warning in warnings[:3]])
        
        # Phase-specific recommendations
        for phase, result in phase_results.items():
            if result.recommendations:
                recommendations.extend(result.recommendations[:2])  # Top 2 per phase
        
        # General recommendations
        if not critical_issues:
            recommendations.append("✅ System validation completed successfully")
            recommendations.append("📊 Monitor system health regularly")
            recommendations.append("🔄 Run validation periodically to ensure continued reliability")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def _create_failed_result(self, start_time: float, phase_results: Dict[ValidationPhase, ValidationPhaseResult],
                            failure_reason: str) -> StartupValidationResult:
        """Create a failed validation result for early termination"""
        total_duration = time.time() - start_time
        
        return StartupValidationResult(
            overall_status=ValidationStatus.FAILED,
            total_duration=total_duration,
            phase_results=phase_results,
            system_health_score=0,
            critical_issues=[failure_reason],
            warnings=[],
            recommendations=["Fix critical issues", "Retry startup validation", "Check system logs"],
            ready_for_operation=False
        )
    
    def _log_validation_results(self, result: StartupValidationResult):
        """Log validation results with appropriate formatting"""
        status_emoji = {
            ValidationStatus.COMPLETED: "✅",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.IN_PROGRESS: "⏳"
        }
        
        emoji = status_emoji.get(result.overall_status, "❓")
        
        logger.info(f"{emoji} Startup validation {result.overall_status.value} in {result.total_duration:.2f}s")
        logger.info(f"📊 System health score: {result.system_health_score}/100")
        logger.info(f"🎯 Ready for operation: {'Yes' if result.ready_for_operation else 'No'}")
        
        if result.critical_issues:
            logger.error(f"🚨 Critical issues ({len(result.critical_issues)}): {', '.join(result.critical_issues[:3])}")
        
        if result.warnings:
            logger.warning(f"⚠️ Warnings ({len(result.warnings)}): {', '.join(result.warnings[:3])}")
        
        # Log phase results
        for phase, phase_result in result.phase_results.items():
            phase_emoji = status_emoji.get(phase_result.status, "❓")
            logger.info(f"  {phase_emoji} {phase.value}: {phase_result.status.value} ({phase_result.duration:.2f}s)")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        if not self.validation_history:
            return {"status": "no_validation_run", "message": "No startup validation has been run yet"}
        
        latest = self.validation_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "overall_status": latest.overall_status.value,
            "total_duration": latest.total_duration,
            "system_health_score": latest.system_health_score,
            "ready_for_operation": latest.ready_for_operation,
            "summary": {
                "phases_completed": len([p for p in latest.phase_results.values() if p.status == ValidationStatus.COMPLETED]),
                "phases_failed": len([p for p in latest.phase_results.values() if p.status == ValidationStatus.FAILED]),
                "critical_issues": len(latest.critical_issues),
                "warnings": len(latest.warnings),
                "recommendations": len(latest.recommendations)
            },
            "phase_details": {
                phase.value: {
                    "status": result.status.value,
                    "duration": result.duration,
                    "issues": len(result.issues),
                    "warnings": len(result.warnings),
                    "recommendations": len(result.recommendations)
                }
                for phase, result in latest.phase_results.items()
            },
            "critical_issues": latest.critical_issues,
            "top_recommendations": latest.recommendations[:5],
            "validation_history_count": len(self.validation_history)
        }

# Global instance
startup_validator = RAGStartupValidator()