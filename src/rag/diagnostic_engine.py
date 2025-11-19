"""
Comprehensive Diagnostic Engine for RAG System Reliability
Provides detailed diagnostics, performance monitoring, and actionable recommendations.
"""

import os
import sqlite3
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Import existing components
try:
    from .dependency_validator import dependency_validator, DependencyValidationResult
except ImportError:
    from src.rag.dependency_validator import dependency_validator, DependencyValidationResult

try:
    from src.database.schema_migrator import schema_migrator, SchemaValidationResult
except ImportError:
    from src.database.schema_migrator import schema_migrator, SchemaValidationResult

logger = logging.getLogger(__name__)

class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic issues"""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"
    SUCCESS = "success"

class ComponentStatus(Enum):
    """Status of system components"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class DiagnosticIssue:
    """Represents a diagnostic issue found in the system"""
    component: str
    severity: DiagnosticSeverity
    title: str
    description: str
    recommendations: List[str] = field(default_factory=list)
    error_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    auto_fixable: bool = False

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    name: str
    status: ComponentStatus
    score: int  # 0-100
    issues: List[DiagnosticIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """Performance metrics for the RAG system"""
    database_query_time: float
    dependency_check_time: float
    memory_usage_mb: float
    disk_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    vector_index_size: Optional[int] = None
    document_count: Optional[int] = None

@dataclass
class SystemDiagnosticResult:
    """Complete system diagnostic result"""
    overall_status: ComponentStatus
    overall_score: int  # 0-100
    components: Dict[str, ComponentHealth]
    performance_metrics: PerformanceMetrics
    issues: List[DiagnosticIssue]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    diagnostic_duration: float = 0.0

class RAGDiagnosticEngine:
    """
    Comprehensive diagnostic engine for RAG system reliability
    """
    
    def __init__(self, users_db_path: str = "users.db", rag_db_path: str = "rag_vector_db.db"):
        self.users_db_path = users_db_path
        self.rag_db_path = rag_db_path
        self.diagnostic_history = []
        
    def run_comprehensive_diagnostics(self) -> SystemDiagnosticResult:
        """
        Run complete system diagnostics
        
        Returns:
            SystemDiagnosticResult with complete diagnostic information
        """
        start_time = time.time()
        logger.info("Starting comprehensive RAG system diagnostics")
        
        try:
            # Initialize result containers
            components = {}
            all_issues = []
            all_recommendations = []
            
            # Run individual component diagnostics
            components["database"] = self._diagnose_database_health()
            components["dependencies"] = self._diagnose_dependencies()
            components["rag_system"] = self._diagnose_rag_system()
            components["performance"] = self._diagnose_performance()
            components["initialization"] = self._diagnose_initialization_status()
            
            # Collect all issues and recommendations
            for component in components.values():
                all_issues.extend(component.issues)
                
            # Generate performance metrics
            performance_metrics = self._collect_performance_metrics()
            
            # Calculate overall status and score
            overall_status, overall_score = self._calculate_overall_health(components)
            
            # Generate system-wide recommendations
            all_recommendations = self._generate_system_recommendations(components, all_issues)
            
            diagnostic_duration = time.time() - start_time
            
            result = SystemDiagnosticResult(
                overall_status=overall_status,
                overall_score=overall_score,
                components=components,
                performance_metrics=performance_metrics,
                issues=all_issues,
                recommendations=all_recommendations,
                diagnostic_duration=diagnostic_duration
            )
            
            # Store diagnostic history
            self.diagnostic_history.append(result)
            
            logger.info(f"Comprehensive diagnostics completed in {diagnostic_duration:.2f}s - Status: {overall_status.value}")
            return result
            
        except Exception as e:
            diagnostic_duration = time.time() - start_time
            logger.error(f"Error during comprehensive diagnostics: {str(e)}")
            
            # Return error result
            error_issue = DiagnosticIssue(
                component="diagnostic_engine",
                severity=DiagnosticSeverity.CRITICAL,
                title="Diagnostic Engine Failure",
                description=f"Failed to complete diagnostics: {str(e)}",
                recommendations=["Check system logs", "Restart diagnostic engine"],
                error_details=str(e)
            )
            
            return SystemDiagnosticResult(
                overall_status=ComponentStatus.FAILED,
                overall_score=0,
                components={},
                performance_metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0),
                issues=[error_issue],
                recommendations=["Fix diagnostic engine errors before proceeding"],
                diagnostic_duration=diagnostic_duration
            )
    
    def _diagnose_database_health(self) -> ComponentHealth:
        """Diagnose database health and schema integrity"""
        issues = []
        metrics = {}
        
        try:
            start_time = time.time()
            
            # Check database file existence
            if not os.path.exists(self.users_db_path):
                issues.append(DiagnosticIssue(
                    component="database",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Users Database Missing",
                    description=f"Users database file not found: {self.users_db_path}",
                    recommendations=["Create users database", "Run database initialization"],
                    auto_fixable=True
                ))
            
            # Check schema validation
            schema_results = schema_migrator.validate_schema()
            for table_name, validation in schema_results.items():
                if not validation.is_valid:
                    issues.append(DiagnosticIssue(
                        component="database",
                        severity=DiagnosticSeverity.WARNING,
                        title=f"Schema Issues in {table_name}",
                        description=f"Missing columns: {', '.join(validation.missing_columns)}",
                        recommendations=validation.recommendations,
                        auto_fixable=True
                    ))
            
            # Test database connectivity and performance
            if os.path.exists(self.users_db_path):
                conn_start = time.time()
                try:
                    conn = sqlite3.connect(self.users_db_path)
                    cursor = conn.cursor()
                    
                    # Test query performance
                    cursor.execute("SELECT COUNT(*) FROM users")
                    user_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM business_profiles")
                    profile_count = cursor.fetchone()[0]
                    
                    conn.close()
                    
                    query_time = time.time() - conn_start
                    metrics.update({
                        "user_count": user_count,
                        "profile_count": profile_count,
                        "query_response_time": query_time,
                        "database_size_mb": os.path.getsize(self.users_db_path) / (1024 * 1024)
                    })
                    
                    # Performance warnings
                    if query_time > 1.0:
                        issues.append(DiagnosticIssue(
                            component="database",
                            severity=DiagnosticSeverity.WARNING,
                            title="Slow Database Performance",
                            description=f"Database queries taking {query_time:.2f}s",
                            recommendations=["Optimize database indexes", "Check database size", "Consider database maintenance"]
                        ))
                    
                except Exception as e:
                    issues.append(DiagnosticIssue(
                        component="database",
                        severity=DiagnosticSeverity.CRITICAL,
                        title="Database Connection Failed",
                        description=f"Cannot connect to database: {str(e)}",
                        recommendations=["Check database permissions", "Verify database integrity"],
                        error_details=str(e)
                    ))
            
            # Check RAG vector database
            if os.path.exists(self.rag_db_path):
                metrics["rag_db_size_mb"] = os.path.getsize(self.rag_db_path) / (1024 * 1024)
            else:
                issues.append(DiagnosticIssue(
                    component="database",
                    severity=DiagnosticSeverity.INFO,
                    title="RAG Vector Database Not Found",
                    description="RAG vector database will be created when needed",
                    recommendations=["Initialize RAG system for a user to create vector database"]
                ))
            
            # Calculate health score
            score = 100
            for issue in issues:
                if issue.severity == DiagnosticSeverity.CRITICAL:
                    score -= 30
                elif issue.severity == DiagnosticSeverity.WARNING:
                    score -= 15
            
            status = ComponentStatus.HEALTHY if score >= 80 else (
                ComponentStatus.DEGRADED if score >= 50 else ComponentStatus.FAILED
            )
            
            return ComponentHealth(
                name="database",
                status=status,
                score=max(0, score),
                issues=issues,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error diagnosing database health: {str(e)}")
            return ComponentHealth(
                name="database",
                status=ComponentStatus.FAILED,
                score=0,
                issues=[DiagnosticIssue(
                    component="database",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Database Diagnostic Failed",
                    description=str(e),
                    error_details=str(e)
                )]
            )
    
    def _diagnose_dependencies(self) -> ComponentHealth:
        """Diagnose dependency availability and status"""
        issues = []
        metrics = {}
        
        try:
            # Run dependency validation
            validation_result = dependency_validator.check_all_dependencies()
            
            # Convert validation results to diagnostic issues
            for dep_name in validation_result.critical_missing:
                issues.append(DiagnosticIssue(
                    component="dependencies",
                    severity=DiagnosticSeverity.CRITICAL,
                    title=f"Critical Dependency Missing: {dep_name}",
                    description=f"Required dependency {dep_name} is not available",
                    recommendations=[
                        f"Install {dep_name}: {validation_result.installation_instructions.get(dep_name, 'pip install ' + dep_name)}",
                        "Restart system after installation"
                    ],
                    error_details=validation_result.error_details.get(dep_name),
                    auto_fixable=False
                ))
            
            for dep_name in validation_result.optional_missing:
                issues.append(DiagnosticIssue(
                    component="dependencies",
                    severity=DiagnosticSeverity.WARNING,
                    title=f"Optional Dependency Missing: {dep_name}",
                    description=f"Optional dependency {dep_name} is not available - some features may be limited",
                    recommendations=[
                        f"Install {dep_name}: {validation_result.installation_instructions.get(dep_name, 'pip install ' + dep_name)}",
                        "Features will work with reduced functionality"
                    ],
                    auto_fixable=False
                ))
            
            # Add metrics
            metrics.update({
                "total_dependencies": len(dependency_validator.dependencies),
                "available_dependencies": len(validation_result.available_dependencies),
                "critical_missing": len(validation_result.critical_missing),
                "optional_missing": len(validation_result.optional_missing),
                "degraded_features": validation_result.degraded_features
            })
            
            # Calculate health score based on dependency status
            if validation_result.overall_status == "healthy":
                score = 100
                status = ComponentStatus.HEALTHY
            elif validation_result.overall_status == "degraded":
                score = 70
                status = ComponentStatus.DEGRADED
            else:  # failed
                score = 20
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                name="dependencies",
                status=status,
                score=score,
                issues=issues,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error diagnosing dependencies: {str(e)}")
            return ComponentHealth(
                name="dependencies",
                status=ComponentStatus.FAILED,
                score=0,
                issues=[DiagnosticIssue(
                    component="dependencies",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Dependency Diagnostic Failed",
                    description=str(e),
                    error_details=str(e)
                )]
            )
    
    def _diagnose_rag_system(self) -> ComponentHealth:
        """Diagnose RAG system components and functionality"""
        issues = []
        metrics = {}
        
        try:
            # Check RAG manager availability
            try:
                from .rag_manager import RAGManager
                metrics["rag_manager_available"] = True
            except ImportError as e:
                issues.append(DiagnosticIssue(
                    component="rag_system",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="RAG Manager Import Failed",
                    description=f"Cannot import RAG manager: {str(e)}",
                    recommendations=["Check RAG module structure", "Fix import dependencies"],
                    error_details=str(e)
                ))
                metrics["rag_manager_available"] = False
            
            # Check vector database status
            if os.path.exists(self.rag_db_path):
                try:
                    # Try to connect and get basic info
                    import sqlite3
                    conn = sqlite3.connect(self.rag_db_path)
                    cursor = conn.cursor()
                    
                    # Get table info
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    metrics.update({
                        "vector_db_tables": tables,
                        "vector_db_exists": True
                    })
                    
                    # Try to get document count if documents table exists
                    if 'documents' in tables:
                        cursor.execute("SELECT COUNT(*) FROM documents")
                        doc_count = cursor.fetchone()[0]
                        metrics["document_count"] = doc_count
                        
                        if doc_count == 0:
                            issues.append(DiagnosticIssue(
                                component="rag_system",
                                severity=DiagnosticSeverity.INFO,
                                title="No Documents in RAG System",
                                description="RAG vector database exists but contains no documents",
                                recommendations=["Upload documents to enable RAG functionality"]
                            ))
                    
                    conn.close()
                    
                except Exception as e:
                    issues.append(DiagnosticIssue(
                        component="rag_system",
                        severity=DiagnosticSeverity.WARNING,
                        title="Vector Database Access Issues",
                        description=f"Cannot access vector database properly: {str(e)}",
                        recommendations=["Check vector database integrity", "Reinitialize RAG system if needed"],
                        error_details=str(e)
                    ))
            else:
                metrics["vector_db_exists"] = False
                issues.append(DiagnosticIssue(
                    component="rag_system",
                    severity=DiagnosticSeverity.INFO,
                    title="Vector Database Not Initialized",
                    description="RAG vector database has not been created yet",
                    recommendations=["Initialize RAG system for a user to create vector database"]
                ))
            
            # Check for common RAG issues
            self._check_rag_initialization_issues(issues, metrics)
            
            # Calculate health score
            score = 100
            for issue in issues:
                if issue.severity == DiagnosticSeverity.CRITICAL:
                    score -= 40
                elif issue.severity == DiagnosticSeverity.WARNING:
                    score -= 20
                elif issue.severity == DiagnosticSeverity.INFO:
                    score -= 5
            
            status = ComponentStatus.HEALTHY if score >= 80 else (
                ComponentStatus.DEGRADED if score >= 50 else ComponentStatus.FAILED
            )
            
            return ComponentHealth(
                name="rag_system",
                status=status,
                score=max(0, score),
                issues=issues,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error diagnosing RAG system: {str(e)}")
            return ComponentHealth(
                name="rag_system",
                status=ComponentStatus.FAILED,
                score=0,
                issues=[DiagnosticIssue(
                    component="rag_system",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="RAG System Diagnostic Failed",
                    description=str(e),
                    error_details=str(e)
                )]
            )
    
    def _diagnose_performance(self) -> ComponentHealth:
        """Diagnose system performance and resource usage"""
        issues = []
        metrics = {}
        
        try:
            # Get system resource usage
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('.')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            metrics.update({
                "memory_total_gb": memory_info.total / (1024**3),
                "memory_available_gb": memory_info.available / (1024**3),
                "memory_usage_percent": memory_info.percent,
                "disk_total_gb": disk_info.total / (1024**3),
                "disk_free_gb": disk_info.free / (1024**3),
                "disk_usage_percent": (disk_info.used / disk_info.total) * 100,
                "cpu_usage_percent": cpu_percent
            })
            
            # Check for performance issues
            if memory_info.percent > 90:
                issues.append(DiagnosticIssue(
                    component="performance",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="High Memory Usage",
                    description=f"Memory usage at {memory_info.percent:.1f}%",
                    recommendations=["Close unnecessary applications", "Consider adding more RAM", "Restart system if needed"]
                ))
            elif memory_info.percent > 80:
                issues.append(DiagnosticIssue(
                    component="performance",
                    severity=DiagnosticSeverity.WARNING,
                    title="Elevated Memory Usage",
                    description=f"Memory usage at {memory_info.percent:.1f}%",
                    recommendations=["Monitor memory usage", "Close unnecessary applications"]
                ))
            
            if (disk_info.used / disk_info.total) * 100 > 90:
                issues.append(DiagnosticIssue(
                    component="performance",
                    severity=DiagnosticSeverity.WARNING,
                    title="Low Disk Space",
                    description=f"Disk usage at {(disk_info.used / disk_info.total) * 100:.1f}%",
                    recommendations=["Free up disk space", "Clean temporary files", "Archive old data"]
                ))
            
            if cpu_percent > 90:
                issues.append(DiagnosticIssue(
                    component="performance",
                    severity=DiagnosticSeverity.WARNING,
                    title="High CPU Usage",
                    description=f"CPU usage at {cpu_percent:.1f}%",
                    recommendations=["Check for resource-intensive processes", "Consider system optimization"]
                ))
            
            # Test database performance
            if os.path.exists(self.users_db_path):
                db_start = time.time()
                try:
                    conn = sqlite3.connect(self.users_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM users")
                    cursor.fetchone()
                    conn.close()
                    db_time = time.time() - db_start
                    metrics["db_query_time"] = db_time
                    
                    if db_time > 2.0:
                        issues.append(DiagnosticIssue(
                            component="performance",
                            severity=DiagnosticSeverity.WARNING,
                            title="Slow Database Performance",
                            description=f"Database query took {db_time:.2f}s",
                            recommendations=["Optimize database", "Check for database locks", "Consider indexing"]
                        ))
                except Exception as e:
                    issues.append(DiagnosticIssue(
                        component="performance",
                        severity=DiagnosticSeverity.WARNING,
                        title="Database Performance Test Failed",
                        description=f"Cannot test database performance: {str(e)}",
                        error_details=str(e)
                    ))
            
            # Calculate performance score
            score = 100
            if memory_info.percent > 90:
                score -= 30
            elif memory_info.percent > 80:
                score -= 15
            
            if (disk_info.used / disk_info.total) * 100 > 90:
                score -= 20
            
            if cpu_percent > 90:
                score -= 25
            
            status = ComponentStatus.HEALTHY if score >= 80 else (
                ComponentStatus.DEGRADED if score >= 60 else ComponentStatus.FAILED
            )
            
            return ComponentHealth(
                name="performance",
                status=status,
                score=max(0, score),
                issues=issues,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error diagnosing performance: {str(e)}")
            return ComponentHealth(
                name="performance",
                status=ComponentStatus.FAILED,
                score=0,
                issues=[DiagnosticIssue(
                    component="performance",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Performance Diagnostic Failed",
                    description=str(e),
                    error_details=str(e)
                )]
            )
    
    def _diagnose_initialization_status(self) -> ComponentHealth:
        """Diagnose RAG initialization status for users"""
        issues = []
        metrics = {}
        
        try:
            if not os.path.exists(self.users_db_path):
                issues.append(DiagnosticIssue(
                    component="initialization",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Cannot Check Initialization Status",
                    description="Users database not available",
                    recommendations=["Create users database first"]
                ))
                return ComponentHealth(
                    name="initialization",
                    status=ComponentStatus.FAILED,
                    score=0,
                    issues=issues
                )
            
            conn = sqlite3.connect(self.users_db_path)
            cursor = conn.cursor()
            
            # Check if RAG columns exist
            cursor.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]
            
            rag_columns = ['rag_initialized', 'rag_initialization_error', 'rag_last_health_check', 'rag_error_count']
            missing_columns = [col for col in rag_columns if col not in columns]
            
            if missing_columns:
                issues.append(DiagnosticIssue(
                    component="initialization",
                    severity=DiagnosticSeverity.WARNING,
                    title="RAG Initialization Columns Missing",
                    description=f"Missing columns: {', '.join(missing_columns)}",
                    recommendations=["Run database migration", "Update database schema"],
                    auto_fixable=True
                ))
            else:
                # Get initialization statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_users,
                        SUM(CASE WHEN rag_initialized = 1 THEN 1 ELSE 0 END) as initialized_users,
                        SUM(CASE WHEN rag_initialization_error IS NOT NULL THEN 1 ELSE 0 END) as error_users,
                        AVG(rag_error_count) as avg_error_count
                    FROM users
                """)
                
                stats = cursor.fetchone()
                if stats:
                    total_users, initialized_users, error_users, avg_error_count = stats
                    
                    metrics.update({
                        "total_users": total_users or 0,
                        "initialized_users": initialized_users or 0,
                        "error_users": error_users or 0,
                        "avg_error_count": avg_error_count or 0,
                        "initialization_rate": (initialized_users / total_users * 100) if total_users > 0 else 0
                    })
                    
                    # Check for initialization issues
                    if total_users > 0:
                        init_rate = (initialized_users / total_users) * 100
                        
                        if init_rate < 50:
                            issues.append(DiagnosticIssue(
                                component="initialization",
                                severity=DiagnosticSeverity.WARNING,
                                title="Low RAG Initialization Rate",
                                description=f"Only {init_rate:.1f}% of users have RAG initialized",
                                recommendations=["Check initialization process", "Review error logs", "Run bulk initialization"]
                            ))
                        
                        if error_users > 0:
                            error_rate = (error_users / total_users) * 100
                            issues.append(DiagnosticIssue(
                                component="initialization",
                                severity=DiagnosticSeverity.WARNING,
                                title="RAG Initialization Errors",
                                description=f"{error_users} users ({error_rate:.1f}%) have initialization errors",
                                recommendations=["Review initialization errors", "Fix common issues", "Retry failed initializations"]
                            ))
            
            conn.close()
            
            # Calculate health score
            score = 100
            for issue in issues:
                if issue.severity == DiagnosticSeverity.CRITICAL:
                    score -= 40
                elif issue.severity == DiagnosticSeverity.WARNING:
                    score -= 20
            
            status = ComponentStatus.HEALTHY if score >= 80 else (
                ComponentStatus.DEGRADED if score >= 50 else ComponentStatus.FAILED
            )
            
            return ComponentHealth(
                name="initialization",
                status=status,
                score=max(0, score),
                issues=issues,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error diagnosing initialization status: {str(e)}")
            return ComponentHealth(
                name="initialization",
                status=ComponentStatus.FAILED,
                score=0,
                issues=[DiagnosticIssue(
                    component="initialization",
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Initialization Diagnostic Failed",
                    description=str(e),
                    error_details=str(e)
                )]
            )
    
    def _check_rag_initialization_issues(self, issues: List[DiagnosticIssue], metrics: Dict[str, Any]):
        """Check for common RAG initialization issues"""
        try:
            # Check if users database has initialization error records
            if os.path.exists(self.users_db_path):
                conn = sqlite3.connect(self.users_db_path)
                cursor = conn.cursor()
                
                # Check for recent initialization errors
                cursor.execute("""
                    SELECT COUNT(*) FROM users 
                    WHERE rag_initialization_error IS NOT NULL 
                    AND rag_initialization_error != ''
                """)
                
                error_count = cursor.fetchone()[0]
                if error_count > 0:
                    issues.append(DiagnosticIssue(
                        component="rag_system",
                        severity=DiagnosticSeverity.WARNING,
                        title="RAG Initialization Errors Detected",
                        description=f"{error_count} users have RAG initialization errors",
                        recommendations=[
                            "Review initialization error logs",
                            "Check dependency availability",
                            "Retry failed initializations"
                        ]
                    ))
                
                metrics["users_with_init_errors"] = error_count
                conn.close()
                
        except Exception as e:
            logger.warning(f"Could not check RAG initialization issues: {str(e)}")
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # Database query performance
            db_start = time.time()
            try:
                if os.path.exists(self.users_db_path):
                    conn = sqlite3.connect(self.users_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM users")
                    cursor.fetchone()
                    conn.close()
                db_query_time = time.time() - db_start
            except:
                db_query_time = -1
            
            # Dependency check performance
            dep_start = time.time()
            try:
                dependency_validator.check_critical_dependencies()
                dep_check_time = time.time() - dep_start
            except:
                dep_check_time = -1
            
            # System metrics
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('.')
            cpu_percent = psutil.cpu_percent()
            
            # Document count
            document_count = None
            if os.path.exists(self.rag_db_path):
                try:
                    conn = sqlite3.connect(self.rag_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
                    if cursor.fetchone():
                        cursor.execute("SELECT COUNT(*) FROM documents")
                        document_count = cursor.fetchone()[0]
                    conn.close()
                except:
                    pass
            
            # Vector index size
            vector_index_size = None
            if os.path.exists(self.rag_db_path):
                vector_index_size = os.path.getsize(self.rag_db_path)
            
            return PerformanceMetrics(
                database_query_time=db_query_time,
                dependency_check_time=dep_check_time,
                memory_usage_mb=memory_info.used / (1024 * 1024),
                disk_usage_mb=disk_info.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                active_connections=0,  # Would need to track this separately
                vector_index_size=vector_index_size,
                document_count=document_count
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0)
    
    def _calculate_overall_health(self, components: Dict[str, ComponentHealth]) -> Tuple[ComponentStatus, int]:
        """Calculate overall system health status and score"""
        if not components:
            return ComponentStatus.FAILED, 0
        
        # Calculate weighted average score
        weights = {
            "database": 0.3,
            "dependencies": 0.25,
            "rag_system": 0.25,
            "performance": 0.1,
            "initialization": 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for name, component in components.items():
            weight = weights.get(name, 0.1)
            total_score += component.score * weight
            total_weight += weight
        
        overall_score = int(total_score / total_weight) if total_weight > 0 else 0
        
        # Determine status based on component statuses and score
        failed_components = [c for c in components.values() if c.status == ComponentStatus.FAILED]
        critical_components = ["database", "dependencies"]
        
        # Check if any critical components failed
        critical_failed = any(
            components.get(name, ComponentHealth("", ComponentStatus.FAILED, 0)).status == ComponentStatus.FAILED
            for name in critical_components
        )
        
        if critical_failed or overall_score < 30:
            return ComponentStatus.FAILED, overall_score
        elif overall_score < 70 or any(c.status == ComponentStatus.DEGRADED for c in components.values()):
            return ComponentStatus.DEGRADED, overall_score
        else:
            return ComponentStatus.HEALTHY, overall_score
    
    def _generate_system_recommendations(self, components: Dict[str, ComponentHealth], 
                                       issues: List[DiagnosticIssue]) -> List[str]:
        """Generate system-wide recommendations based on diagnostic results"""
        recommendations = []
        
        # Priority recommendations based on critical issues
        critical_issues = [issue for issue in issues if issue.severity == DiagnosticSeverity.CRITICAL]
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address critical issues immediately:")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                recommendations.append(f"   • {issue.title}: {issue.recommendations[0] if issue.recommendations else 'See details'}")
        
        # Component-specific recommendations
        db_component = components.get("database")
        if db_component and db_component.status != ComponentStatus.HEALTHY:
            recommendations.append("🗄️ Database: Run schema migration and check database integrity")
        
        deps_component = components.get("dependencies")
        if deps_component and deps_component.status != ComponentStatus.HEALTHY:
            recommendations.append("📦 Dependencies: Install missing dependencies for full functionality")
        
        rag_component = components.get("rag_system")
        if rag_component and rag_component.status != ComponentStatus.HEALTHY:
            recommendations.append("🤖 RAG System: Initialize RAG for users and check vector database")
        
        perf_component = components.get("performance")
        if perf_component and perf_component.status != ComponentStatus.HEALTHY:
            recommendations.append("⚡ Performance: Monitor system resources and optimize if needed")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ System is healthy - continue regular monitoring")
        else:
            recommendations.append("📊 Run diagnostics regularly to monitor system health")
            recommendations.append("📝 Check logs for detailed error information")
        
        return recommendations
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get a summary of the latest diagnostic results"""
        if not self.diagnostic_history:
            return {"status": "no_diagnostics_run", "message": "No diagnostics have been run yet"}
        
        latest = self.diagnostic_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "overall_status": latest.overall_status.value,
            "overall_score": latest.overall_score,
            "component_count": len(latest.components),
            "issue_count": len(latest.issues),
            "critical_issues": len([i for i in latest.issues if i.severity == DiagnosticSeverity.CRITICAL]),
            "warning_issues": len([i for i in latest.issues if i.severity == DiagnosticSeverity.WARNING]),
            "top_recommendations": latest.recommendations[:3],
            "diagnostic_duration": latest.diagnostic_duration
        }

# Global instance
diagnostic_engine = RAGDiagnosticEngine()