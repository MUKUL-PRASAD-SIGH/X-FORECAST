"""
Enhanced RAG Manager with Dependency Validation and Graceful Degradation
Extends the base RAG manager with reliability enhancements including dependency validation,
graceful degradation, and improved error handling.
"""

import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .rag_manager import RAGManager, RAGHealthStatus
from .dependency_validator import DependencyValidator, DependencyValidationResult

logger = logging.getLogger(__name__)

@dataclass
class SystemValidationResult:
    """Result of comprehensive system validation"""
    overall_status: str  # 'healthy', 'degraded', 'failed'
    schema_status: bool
    dependency_status: str
    rag_system_status: bool
    issues: List[str]
    recommendations: List[str]
    degraded_features: List[str]
    validation_timestamp: datetime

class EnhancedRAGManager(RAGManager):
    """
    Enhanced RAG Manager with dependency validation and graceful degradation
    """
    
    def __init__(self, db_path: str = "users.db", rag_db_path: str = "rag_vector_db.db"):
        # Initialize dependency validator first
        self.dependency_validator = DependencyValidator()
        
        # Run dependency validation before initializing parent
        self.dependency_status = self.dependency_validator.check_all_dependencies()
        
        # Initialize parent with graceful handling
        super().__init__(db_path, rag_db_path)
        
        # Override RAG system initialization with dependency-aware loading
        self._initialize_rag_system_with_validation()
        
        # Initialize recovery manager
        self.recovery_manager = None
        self._initialize_recovery_manager()
    
    def _initialize_rag_system_with_validation(self):
        """Initialize RAG system with dependency validation and graceful degradation"""
        try:
            # Check if critical dependencies are available
            if not self.dependency_validator.validate_sentence_transformers():
                logger.warning("sentence_transformers not available - initializing with fallback")
                self._initialize_fallback_rag_system()
                return
            
            # Try to import and initialize full RAG system
            from .real_vector_rag import RealVectorRAG
            self.rag_system = RealVectorRAG()
            logger.info("Full RAG system initialized successfully")
            
        except ImportError as e:
            logger.error(f"RAG system import failed: {e}")
            self._initialize_fallback_rag_system()
        except Exception as e:
            logger.error(f"RAG system initialization failed: {e}")
            self._initialize_fallback_rag_system()
    
    def _initialize_fallback_rag_system(self):
        """Initialize fallback RAG system when dependencies are missing"""
        try:
            from .fallback_rag import FallbackRAGSystem
            self.rag_system = FallbackRAGSystem()
            logger.info("Fallback RAG system initialized")
        except ImportError:
            logger.warning("Fallback RAG system not available - RAG functionality disabled")
            self.rag_system = None
    
    def startup_validation(self) -> SystemValidationResult:
        """
        Comprehensive startup validation including schema, dependencies, and RAG system
        
        Returns:
            SystemValidationResult with complete validation status
        """
        try:
            issues = []
            recommendations = []
            degraded_features = []
            
            # 1. Schema validation
            schema_status = True
            if self.schema_migrator:
                try:
                    migration_result = self.run_startup_migration()
                    schema_status = migration_result["success"]
                    if not schema_status:
                        issues.append(f"Schema migration failed: {migration_result.get('error', 'Unknown error')}")
                        recommendations.append("Run database schema migration manually")
                except Exception as e:
                    schema_status = False
                    issues.append(f"Schema validation error: {str(e)}")
                    recommendations.append("Check database configuration and permissions")
            else:
                schema_status = False
                issues.append("Schema migration system not available")
                recommendations.append("Install schema migration dependencies")
            
            # 2. Dependency validation
            dependency_result = self.dependency_validator.check_all_dependencies()
            dependency_status = dependency_result.overall_status
            
            if dependency_result.critical_missing:
                issues.extend([f"Critical dependency missing: {dep}" for dep in dependency_result.critical_missing])
                recommendations.append("Install critical dependencies for full RAG functionality")
            
            if dependency_result.optional_missing:
                issues.extend([f"Optional dependency missing: {dep}" for dep in dependency_result.optional_missing])
                recommendations.append("Consider installing optional dependencies for enhanced features")
            
            degraded_features.extend(dependency_result.degraded_features)
            
            # 3. RAG system validation
            rag_system_status = self.rag_system is not None
            if not rag_system_status:
                issues.append("RAG system not available")
                recommendations.append("Install RAG system dependencies")
            
            # Determine overall status
            if not schema_status or dependency_status == "failed":
                overall_status = "failed"
            elif dependency_status == "degraded" or issues:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            return SystemValidationResult(
                overall_status=overall_status,
                schema_status=schema_status,
                dependency_status=dependency_status,
                rag_system_status=rag_system_status,
                issues=issues,
                recommendations=recommendations,
                degraded_features=degraded_features,
                validation_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Startup validation failed: {str(e)}")
            return SystemValidationResult(
                overall_status="failed",
                schema_status=False,
                dependency_status="failed",
                rag_system_status=False,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Contact system administrator"],
                degraded_features=[],
                validation_timestamp=datetime.now()
            )
    
    def pre_initialization_validation(self, user_id: str, company_name: str) -> Dict[str, Any]:
        """
        Comprehensive pre-initialization validation checks
        
        Args:
            user_id: User identifier
            company_name: Company name
            
        Returns:
            Dictionary with validation results and detailed status
        """
        try:
            validation_results = {
                "overall_status": "pass",
                "checks": {},
                "issues": [],
                "recommendations": [],
                "can_proceed": True,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # 1. User validation
            user_check = self._validate_user_exists(user_id)
            validation_results["checks"]["user_validation"] = user_check
            if not user_check["status"] == "pass":
                validation_results["overall_status"] = "fail"
                validation_results["can_proceed"] = False
                validation_results["issues"].extend(user_check.get("issues", []))
                validation_results["recommendations"].extend(user_check.get("recommendations", []))
            
            # 2. Schema validation
            schema_check = self._validate_database_schema()
            validation_results["checks"]["schema_validation"] = schema_check
            if not schema_check["status"] == "pass":
                validation_results["overall_status"] = "fail"
                validation_results["can_proceed"] = False
                validation_results["issues"].extend(schema_check.get("issues", []))
                validation_results["recommendations"].extend(schema_check.get("recommendations", []))
            
            # 3. Dependency validation
            dependency_check = self._validate_initialization_dependencies()
            validation_results["checks"]["dependency_validation"] = dependency_check
            if dependency_check["status"] == "fail":
                validation_results["overall_status"] = "fail"
                validation_results["can_proceed"] = False
            elif dependency_check["status"] == "warning":
                if validation_results["overall_status"] == "pass":
                    validation_results["overall_status"] = "warning"
            validation_results["issues"].extend(dependency_check.get("issues", []))
            validation_results["recommendations"].extend(dependency_check.get("recommendations", []))
            
            # 4. Storage validation
            storage_check = self._validate_storage_requirements(user_id)
            validation_results["checks"]["storage_validation"] = storage_check
            if not storage_check["status"] == "pass":
                validation_results["overall_status"] = "fail"
                validation_results["can_proceed"] = False
                validation_results["issues"].extend(storage_check.get("issues", []))
                validation_results["recommendations"].extend(storage_check.get("recommendations", []))
            
            # 5. System resources validation
            resource_check = self._validate_system_resources()
            validation_results["checks"]["resource_validation"] = resource_check
            if resource_check["status"] == "warning":
                if validation_results["overall_status"] == "pass":
                    validation_results["overall_status"] = "warning"
                validation_results["issues"].extend(resource_check.get("issues", []))
                validation_results["recommendations"].extend(resource_check.get("recommendations", []))
            
            logger.info(f"Pre-initialization validation completed for user {user_id}: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Pre-initialization validation failed for user {user_id}: {str(e)}")
            return {
                "overall_status": "fail",
                "checks": {},
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Contact system administrator"],
                "can_proceed": False,
                "validation_timestamp": datetime.now().isoformat()
            }
    
    def safe_initialize_rag(self, user_id: str, company_name: str, force_reinit: bool = False) -> Dict[str, Any]:
        """
        Enhanced safe RAG initialization with comprehensive validation and error handling
        
        Args:
            user_id: User identifier
            company_name: Company name
            force_reinit: Whether to force reinitialization
            
        Returns:
            Dictionary with initialization result including detailed status tracking
        """
        initialization_start_time = datetime.now()
        
        try:
            # Track initialization status in database
            self._update_initialization_status(user_id, "starting", "Pre-initialization validation")
            
            # Step 1: Pre-initialization validation
            logger.info(f"Starting enhanced RAG initialization for user {user_id} ({company_name})")
            validation_result = self.pre_initialization_validation(user_id, company_name)
            
            if not validation_result["can_proceed"]:
                self._update_initialization_status(user_id, "failed", f"Validation failed: {'; '.join(validation_result['issues'])}")
                return {
                    "success": False,
                    "error": "Pre-initialization validation failed",
                    "status": "validation_failed",
                    "validation_results": validation_result,
                    "initialization_time": (datetime.now() - initialization_start_time).total_seconds()
                }
            
            # Step 2: Check existing initialization status
            current_status = self.check_rag_initialization_status(user_id)
            if current_status.get("is_initialized") and not force_reinit:
                self._update_initialization_status(user_id, "completed", "Already initialized")
                return {
                    "success": True,
                    "message": "RAG already initialized",
                    "status": "already_initialized",
                    "current_status": current_status,
                    "validation_results": validation_result,
                    "initialization_time": (datetime.now() - initialization_start_time).total_seconds()
                }
            
            # Step 3: Prepare initialization environment
            self._update_initialization_status(user_id, "preparing", "Setting up initialization environment")
            prep_result = self._prepare_initialization_environment(user_id, company_name)
            if not prep_result["success"]:
                self._update_initialization_status(user_id, "failed", f"Environment preparation failed: {prep_result['error']}")
                return {
                    "success": False,
                    "error": f"Environment preparation failed: {prep_result['error']}",
                    "status": "preparation_failed",
                    "validation_results": validation_result,
                    "initialization_time": (datetime.now() - initialization_start_time).total_seconds()
                }
            
            # Step 4: Initialize RAG system with enhanced error handling
            self._update_initialization_status(user_id, "initializing", "Initializing RAG system")
            init_result = self._enhanced_rag_initialization(user_id, company_name, force_reinit)
            
            # Step 5: Post-initialization validation
            if init_result["success"]:
                self._update_initialization_status(user_id, "validating", "Post-initialization validation")
                post_validation = self._post_initialization_validation(user_id)
                
                if post_validation["success"]:
                    self._update_initialization_status(user_id, "completed", "Initialization completed successfully")
                    
                    # Add comprehensive result information
                    init_result.update({
                        "validation_results": validation_result,
                        "post_validation": post_validation,
                        "initialization_time": (datetime.now() - initialization_start_time).total_seconds(),
                        "status": "initialization_complete"
                    })
                    
                    # Add degradation information if applicable
                    if validation_result["overall_status"] == "warning":
                        init_result["degraded"] = True
                        init_result["degraded_features"] = validation_result.get("degraded_features", [])
                        init_result["degradation_message"] = self._generate_degradation_message(validation_result)
                    
                    logger.info(f"RAG initialization completed successfully for user {user_id} in {init_result['initialization_time']:.2f}s")
                    return init_result
                else:
                    self._update_initialization_status(user_id, "failed", f"Post-validation failed: {post_validation['error']}")
                    return {
                        "success": False,
                        "error": f"Post-initialization validation failed: {post_validation['error']}",
                        "status": "post_validation_failed",
                        "validation_results": validation_result,
                        "initialization_time": (datetime.now() - initialization_start_time).total_seconds()
                    }
            else:
                self._update_initialization_status(user_id, "failed", f"Initialization failed: {init_result['error']}")
                return {
                    "success": False,
                    "error": init_result["error"],
                    "status": "initialization_failed",
                    "validation_results": validation_result,
                    "initialization_time": (datetime.now() - initialization_start_time).total_seconds()
                }
            
        except Exception as e:
            error_msg = f"Enhanced RAG initialization error: {str(e)}"
            logger.error(f"Error in enhanced RAG initialization for user {user_id}: {error_msg}")
            self._update_initialization_status(user_id, "failed", error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "status": "initialization_error",
                "initialization_time": (datetime.now() - initialization_start_time).total_seconds()
            }
    
    def get_system_health_with_dependencies(self) -> Dict[str, Any]:
        """
        Get comprehensive system health including dependency status
        
        Returns:
            Dictionary with complete system health information
        """
        try:
            # Get base system health
            base_health = self.get_system_wide_rag_status()
            
            # Get dependency status
            dependency_report = self.dependency_validator.get_system_status_report()
            
            # Get startup validation
            validation_result = self.startup_validation()
            
            # Combine all health information
            health_report = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": validation_result.overall_status,
                "system_health": {
                    "rag_system": base_health.get("system_health", "unknown"),
                    "dependencies": dependency_report["overall_status"],
                    "schema": "healthy" if validation_result.schema_status else "failed"
                },
                "user_statistics": {
                    "total_users": base_health.get("total_users", 0),
                    "initialized_users": base_health.get("initialized_users", 0),
                    "initialization_rate": base_health.get("initialization_rate", 0)
                },
                "dependency_status": dependency_report,
                "validation_results": {
                    "issues": validation_result.issues,
                    "recommendations": validation_result.recommendations,
                    "degraded_features": validation_result.degraded_features
                },
                "document_statistics": base_health.get("document_statistics", {}),
                "system_capabilities": self._get_system_capabilities()
            }
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error getting system health with dependencies: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "failed",
                "error": str(e)
            }
    
    def _get_system_capabilities(self) -> Dict[str, Any]:
        """
        Get current system capabilities based on available dependencies
        
        Returns:
            Dictionary describing available and unavailable capabilities
        """
        capabilities = {
            "available": [],
            "unavailable": [],
            "degraded": []
        }
        
        # Check core RAG capabilities
        if self.dependency_validator.validate_sentence_transformers():
            capabilities["available"].extend([
                "Vector embeddings generation",
                "Semantic similarity search",
                "Document retrieval"
            ])
        else:
            capabilities["unavailable"].extend([
                "Vector embeddings generation",
                "Semantic similarity search"
            ])
            capabilities["degraded"].append("Basic text search (fallback)")
        
        # Check FAISS availability
        faiss_available = self.dependency_validator.check_dependency('faiss').status.value == "available"
        if faiss_available:
            capabilities["available"].append("Fast vector search")
        else:
            capabilities["unavailable"].append("Fast vector search")
            capabilities["degraded"].append("Slow similarity search (fallback)")
        
        # Check optional capabilities
        torch_available = self.dependency_validator.check_dependency('torch').status.value == "available"
        if torch_available:
            capabilities["available"].append("GPU acceleration")
        else:
            capabilities["unavailable"].append("GPU acceleration")
        
        return capabilities
    
    def recover_from_failure(self, user_id: str, error_type: str) -> Dict[str, Any]:
        """
        Attempt to recover from various failure scenarios
        
        Args:
            user_id: User identifier
            error_type: Type of error to recover from
            
        Returns:
            Dictionary with recovery result
        """
        try:
            recovery_actions = []
            recovery_success = False
            
            if error_type == "dependency":
                # Try to reinitialize with current dependencies
                validation_result = self.startup_validation()
                if validation_result.overall_status != "failed":
                    self._initialize_rag_system_with_validation()
                    recovery_actions.append("Reinitialized RAG system with available dependencies")
                    recovery_success = True
                else:
                    recovery_actions.append("Cannot recover - critical dependencies missing")
            
            elif error_type == "initialization":
                # Try to reset and reinitialize user RAG
                current_status = self.check_rag_initialization_status(user_id)
                if current_status["success"]:
                    company_name = current_status.get("company_name", "Unknown")
                    reset_result = self.reset_rag_system(user_id, company_name)
                    recovery_actions.append(f"Reset RAG system: {reset_result.get('message', 'Unknown result')}")
                    recovery_success = reset_result.get("success", False)
                else:
                    recovery_actions.append("Cannot recover - user not found")
            
            elif error_type == "schema":
                # Try to run schema migration
                if self.schema_migrator:
                    migration_result = self.run_startup_migration()
                    recovery_actions.append(f"Schema migration: {migration_result.get('message', 'Unknown result')}")
                    recovery_success = migration_result.get("success", False)
                else:
                    recovery_actions.append("Cannot recover - schema migrator not available")
            
            else:
                recovery_actions.append(f"Unknown error type: {error_type}")
            
            return {
                "success": recovery_success,
                "error_type": error_type,
                "recovery_actions": recovery_actions,
                "timestamp": datetime.now().isoformat(),
                "recommendations": self._get_recovery_recommendations(error_type, recovery_success)
            }
            
        except Exception as e:
            logger.error(f"Error during recovery for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error_type": error_type,
                "recovery_actions": [f"Recovery failed: {str(e)}"],
                "timestamp": datetime.now().isoformat(),
                "recommendations": ["Contact system administrator"]
            }
    
    def _get_recovery_recommendations(self, error_type: str, recovery_success: bool) -> List[str]:
        """
        Get recovery recommendations based on error type and recovery result
        
        Args:
            error_type: Type of error
            recovery_success: Whether recovery was successful
            
        Returns:
            List of recommendation strings
        """
        if recovery_success:
            return ["Recovery successful - system should be operational"]
        
        recommendations = []
        
        if error_type == "dependency":
            recommendations.extend([
                "Install missing critical dependencies",
                "Check dependency installation instructions",
                "Verify Python environment configuration"
            ])
        elif error_type == "initialization":
            recommendations.extend([
                "Check user data integrity",
                "Verify storage directory permissions",
                "Review RAG system logs for detailed errors"
            ])
        elif error_type == "schema":
            recommendations.extend([
                "Check database permissions",
                "Verify database file integrity",
                "Run manual schema migration if needed"
            ])
        else:
            recommendations.append("Contact system administrator for assistance")
        
        return recommendations
    
    def get_user_feedback_message(self, user_id: str) -> str:
        """
        Generate user-friendly feedback message about RAG system status
        
        Args:
            user_id: User identifier
            
        Returns:
            User-friendly status message
        """
        try:
            # Get system validation
            validation_result = self.startup_validation()
            
            # Get user-specific status
            user_status = self.check_rag_initialization_status(user_id)
            
            if validation_result.overall_status == "failed":
                return (
                    "⚠️ RAG system is currently unavailable due to missing critical components. "
                    "Please contact your system administrator to install required dependencies."
                )
            
            if validation_result.overall_status == "degraded":
                degraded_msg = self.dependency_validator.get_graceful_degradation_message(
                    self.dependency_status.critical_missing + self.dependency_status.optional_missing
                )
                return f"⚠️ RAG system is running with limited functionality:\n\n{degraded_msg}"
            
            if not user_status.get("is_initialized", False):
                return (
                    "✅ RAG system is available and ready to be initialized for your account. "
                    "Upload your company data to start using AI-powered insights."
                )
            
            return (
                "✅ RAG system is fully operational and initialized for your account. "
                "You can upload documents and ask questions about your business data."
            )
            
        except Exception as e:
            logger.error(f"Error generating user feedback message: {str(e)}")
            return (
                "❓ Unable to determine RAG system status. "
                "Please try refreshing the page or contact support if the issue persists."
            )

    # Enhanced initialization helper methods
    
    def _validate_user_exists(self, user_id: str) -> Dict[str, Any]:
        """Validate that user exists and has required data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.user_id, u.company_name, u.email, u.is_active
                FROM users u
                WHERE u.user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {
                    "status": "fail",
                    "issues": ["User not found in database"],
                    "recommendations": ["Verify user ID is correct"]
                }
            
            user_id_db, company_name, email, is_active = result
            
            if not is_active:
                return {
                    "status": "fail",
                    "issues": ["User account is inactive"],
                    "recommendations": ["Activate user account before initializing RAG"]
                }
            
            if not company_name:
                return {
                    "status": "fail",
                    "issues": ["User has no company name set"],
                    "recommendations": ["Set company name in user profile"]
                }
            
            return {
                "status": "pass",
                "message": f"User validation passed for {email} ({company_name})"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"User validation error: {str(e)}"],
                "recommendations": ["Check database connectivity"]
            }
    
    def _validate_database_schema(self) -> Dict[str, Any]:
        """Validate database schema is ready for RAG initialization"""
        try:
            if not self.schema_migrator:
                return {
                    "status": "fail",
                    "issues": ["Schema migrator not available"],
                    "recommendations": ["Install schema migration dependencies"]
                }
            
            validation_results = self.schema_migrator.validate_schema()
            
            missing_tables = []
            missing_columns = []
            
            for table_name, validation in validation_results.items():
                if not validation.is_valid:
                    if validation.missing_columns:
                        missing_columns.extend([f"{table_name}.{col}" for col in validation.missing_columns])
                    if not validation.table_exists:
                        missing_tables.append(table_name)
            
            if missing_tables or missing_columns:
                issues = []
                if missing_tables:
                    issues.append(f"Missing tables: {', '.join(missing_tables)}")
                if missing_columns:
                    issues.append(f"Missing columns: {', '.join(missing_columns)}")
                
                return {
                    "status": "fail",
                    "issues": issues,
                    "recommendations": ["Run database schema migration"]
                }
            
            return {
                "status": "pass",
                "message": "Database schema validation passed"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"Schema validation error: {str(e)}"],
                "recommendations": ["Check database configuration"]
            }
    
    def _validate_initialization_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies required for RAG initialization"""
        try:
            dependency_result = self.dependency_validator.check_all_dependencies()
            
            if dependency_result.critical_missing:
                return {
                    "status": "fail",
                    "issues": [f"Critical dependencies missing: {', '.join(dependency_result.critical_missing)}"],
                    "recommendations": ["Install critical dependencies for RAG functionality"]
                }
            
            if dependency_result.optional_missing:
                return {
                    "status": "warning",
                    "issues": [f"Optional dependencies missing: {', '.join(dependency_result.optional_missing)}"],
                    "recommendations": ["Consider installing optional dependencies for enhanced features"]
                }
            
            return {
                "status": "pass",
                "message": "All required dependencies available"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"Dependency validation error: {str(e)}"],
                "recommendations": ["Check dependency installation"]
            }
    
    def _validate_storage_requirements(self, user_id: str) -> Dict[str, Any]:
        """Validate storage requirements for user RAG initialization"""
        try:
            import psutil
            
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            # Check if directories exist or can be created
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except Exception:
                        missing_dirs.append(dir_name)
            
            if missing_dirs:
                return {
                    "status": "fail",
                    "issues": [f"Cannot create storage directories: {', '.join(missing_dirs)}"],
                    "recommendations": ["Check directory permissions and available disk space"]
                }
            
            # Check available disk space (require at least 100MB)
            try:
                disk_usage = psutil.disk_usage(os.path.dirname(storage_path))
                available_mb = disk_usage.free / (1024 * 1024)
                
                if available_mb < 100:
                    return {
                        "status": "fail",
                        "issues": [f"Insufficient disk space: {available_mb:.1f}MB available, 100MB required"],
                        "recommendations": ["Free up disk space before initializing RAG"]
                    }
            except ImportError:
                # psutil not available, skip disk space check
                pass
            
            return {
                "status": "pass",
                "message": "Storage requirements validated"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"Storage validation error: {str(e)}"],
                "recommendations": ["Check storage configuration"]
            }
    
    def _validate_system_resources(self) -> Dict[str, Any]:
        """Validate system resources for RAG initialization"""
        try:
            import psutil
            
            # Check available memory (recommend at least 512MB)
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < 256:
                return {
                    "status": "fail",
                    "issues": [f"Insufficient memory: {available_mb:.1f}MB available, 256MB minimum required"],
                    "recommendations": ["Free up system memory before initializing RAG"]
                }
            elif available_mb < 512:
                return {
                    "status": "warning",
                    "issues": [f"Low memory: {available_mb:.1f}MB available, 512MB recommended"],
                    "recommendations": ["Consider freeing up memory for optimal performance"]
                }
            
            return {
                "status": "pass",
                "message": f"System resources adequate: {available_mb:.1f}MB memory available"
            }
            
        except ImportError:
            # psutil not available, skip resource check
            return {
                "status": "pass",
                "message": "System resource validation skipped (psutil not available)"
            }
        except Exception as e:
            return {
                "status": "warning",
                "issues": [f"Resource validation error: {str(e)}"],
                "recommendations": ["Check system resource monitoring"]
            }
    
    def _update_initialization_status(self, user_id: str, status: str, message: str):
        """Update initialization status in database with detailed tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create initialization status table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_initialization_status (
                    user_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Update or insert initialization status
            cursor.execute('''
                INSERT OR REPLACE INTO rag_initialization_status 
                (user_id, status, message, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, status, message, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated initialization status for user {user_id}: {status} - {message}")
            
        except Exception as e:
            logger.error(f"Error updating initialization status for user {user_id}: {str(e)}")
    
    def _prepare_initialization_environment(self, user_id: str, company_name: str) -> Dict[str, Any]:
        """Prepare environment for RAG initialization"""
        try:
            # Create storage directories
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            # Initialize database entries if needed
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure business profile exists
            cursor.execute('''
                INSERT OR IGNORE INTO business_profiles 
                (user_id, company_name, rag_status, created_at)
                VALUES (?, ?, 'preparing', ?)
            ''', (user_id, company_name, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": "Environment prepared successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Environment preparation failed: {str(e)}"
            }
    
    def _enhanced_rag_initialization(self, user_id: str, company_name: str, force_reinit: bool) -> Dict[str, Any]:
        """Enhanced RAG initialization with better error handling"""
        try:
            if not self.rag_system:
                return {
                    "success": False,
                    "error": "RAG system not available"
                }
            
            # Initialize RAG system with enhanced error handling
            success = self.rag_system.initialize_company_rag(user_id, company_name)
            
            if success:
                # Update database status
                self._update_rag_status(user_id, "initialized")
                
                return {
                    "success": True,
                    "message": "RAG system initialized successfully",
                    "user_id": user_id,
                    "company_name": company_name
                }
            else:
                self._update_rag_status(user_id, "failed", "RAG initialization failed")
                return {
                    "success": False,
                    "error": "RAG initialization failed"
                }
                
        except Exception as e:
            error_msg = f"RAG initialization error: {str(e)}"
            self._update_rag_status(user_id, "failed", error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _post_initialization_validation(self, user_id: str) -> Dict[str, Any]:
        """Validate RAG system after initialization"""
        try:
            # Check if RAG system is properly initialized
            status = self.check_rag_initialization_status(user_id)
            
            if not status.get("is_initialized"):
                return {
                    "success": False,
                    "error": "RAG system not properly initialized"
                }
            
            # Test basic RAG functionality
            if self.rag_system:
                try:
                    # Test query functionality
                    test_results = self.rag_system.query_user_knowledge(user_id, "test", top_k=1)
                    
                    return {
                        "success": True,
                        "message": "Post-initialization validation passed",
                        "test_query_results": len(test_results) if test_results else 0
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"RAG functionality test failed: {str(e)}"
                    }
            
            return {
                "success": True,
                "message": "Basic post-initialization validation passed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Post-initialization validation error: {str(e)}"
            }
    
    def _generate_degradation_message(self, validation_result: Dict[str, Any]) -> str:
        """Generate user-friendly degradation message"""
        try:
            issues = validation_result.get("issues", [])
            if not issues:
                return "System is running with some limitations."
            
            return f"System is running with limited functionality: {'; '.join(issues[:3])}"
            
        except Exception:
            return "System is running with some limitations."
    
    # Automatic recovery mechanisms with exponential backoff
    
    def initialize_rag_with_retry(self, user_id: str, company_name: str, max_retries: int = 3, 
                                 initial_delay: float = 1.0, backoff_factor: float = 2.0) -> Dict[str, Any]:
        """
        Initialize RAG with automatic retry and exponential backoff
        
        Args:
            user_id: User identifier
            company_name: Company name
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            
        Returns:
            Dictionary with initialization result and retry information
        """
        import time
        
        retry_attempts = []
        last_error = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Calculate delay for this attempt (no delay for first attempt)
                if attempt > 0:
                    delay = initial_delay * (backoff_factor ** (attempt - 1))
                    logger.info(f"Retrying RAG initialization for user {user_id}, attempt {attempt + 1}/{max_retries + 1} after {delay:.1f}s delay")
                    time.sleep(delay)
                
                # Track retry attempt
                attempt_start_time = datetime.now()
                
                # Attempt initialization
                result = self.safe_initialize_rag(user_id, company_name, force_reinit=attempt > 0)
                
                # Record successful attempt
                attempt_info = {
                    "attempt_number": attempt + 1,
                    "success": result["success"],
                    "duration": (datetime.now() - attempt_start_time).total_seconds(),
                    "error": result.get("error") if not result["success"] else None,
                    "timestamp": datetime.now().isoformat()
                }
                retry_attempts.append(attempt_info)
                
                if result["success"]:
                    logger.info(f"RAG initialization succeeded for user {user_id} on attempt {attempt + 1}")
                    result["retry_info"] = {
                        "total_attempts": attempt + 1,
                        "retry_attempts": retry_attempts,
                        "recovery_successful": attempt > 0
                    }
                    return result
                
                last_error = result.get("error", "Unknown error")
                
            except Exception as e:
                last_error = str(e)
                attempt_info = {
                    "attempt_number": attempt + 1,
                    "success": False,
                    "duration": (datetime.now() - attempt_start_time).total_seconds(),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                retry_attempts.append(attempt_info)
                logger.error(f"RAG initialization attempt {attempt + 1} failed for user {user_id}: {str(e)}")
        
        # All attempts failed
        logger.error(f"RAG initialization failed for user {user_id} after {max_retries + 1} attempts")
        
        return {
            "success": False,
            "error": f"Initialization failed after {max_retries + 1} attempts. Last error: {last_error}",
            "status": "retry_exhausted",
            "retry_info": {
                "total_attempts": max_retries + 1,
                "retry_attempts": retry_attempts,
                "recovery_successful": False
            }
        }
    
    def auto_recover_from_failure(self, user_id: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically attempt to recover from various failure scenarios
        
        Args:
            user_id: User identifier
            failure_context: Context information about the failure
            
        Returns:
            Dictionary with recovery result and actions taken
        """
        try:
            recovery_start_time = datetime.now()
            recovery_actions = []
            recovery_successful = False
            
            error_type = failure_context.get("error_type", "unknown")
            error_message = failure_context.get("error_message", "")
            
            logger.info(f"Starting automatic recovery for user {user_id}, error type: {error_type}")
            
            # Recovery strategy based on error type
            if error_type == "dependency_missing":
                recovery_result = self._recover_from_dependency_failure(user_id, failure_context)
                recovery_actions.extend(recovery_result["actions"])
                recovery_successful = recovery_result["success"]
                
            elif error_type == "initialization_failed":
                recovery_result = self._recover_from_initialization_failure(user_id, failure_context)
                recovery_actions.extend(recovery_result["actions"])
                recovery_successful = recovery_result["success"]
                
            elif error_type == "schema_error":
                recovery_result = self._recover_from_schema_failure(user_id, failure_context)
                recovery_actions.extend(recovery_result["actions"])
                recovery_successful = recovery_result["success"]
                
            elif error_type == "storage_error":
                recovery_result = self._recover_from_storage_failure(user_id, failure_context)
                recovery_actions.extend(recovery_result["actions"])
                recovery_successful = recovery_result["success"]
                
            else:
                # Generic recovery attempt
                recovery_result = self._generic_recovery_attempt(user_id, failure_context)
                recovery_actions.extend(recovery_result["actions"])
                recovery_successful = recovery_result["success"]
            
            recovery_time = (datetime.now() - recovery_start_time).total_seconds()
            
            # Log recovery attempt
            self._log_recovery_attempt(user_id, error_type, recovery_successful, recovery_actions, recovery_time)
            
            result = {
                "success": recovery_successful,
                "error_type": error_type,
                "recovery_actions": recovery_actions,
                "recovery_time": recovery_time,
                "timestamp": datetime.now().isoformat()
            }
            
            if recovery_successful:
                logger.info(f"Automatic recovery successful for user {user_id} in {recovery_time:.2f}s")
                result["message"] = "Automatic recovery completed successfully"
            else:
                logger.warning(f"Automatic recovery failed for user {user_id} after {recovery_time:.2f}s")
                result["recommendations"] = self._get_recovery_recommendations(error_type, False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during automatic recovery for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error_type": error_type,
                "recovery_actions": [f"Recovery system error: {str(e)}"],
                "recovery_time": (datetime.now() - recovery_start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "recommendations": ["Contact system administrator"]
            }
    
    def _recover_from_dependency_failure(self, user_id: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from dependency-related failures"""
        actions = []
        
        try:
            # Re-validate dependencies
            actions.append("Re-validating system dependencies")
            dependency_result = self.dependency_validator.check_all_dependencies()
            
            # If critical dependencies are now available, try fallback initialization
            if not dependency_result.critical_missing:
                actions.append("Critical dependencies now available - attempting initialization")
                self._initialize_rag_system_with_validation()
                
                # Test if RAG system is now available
                if self.rag_system:
                    actions.append("RAG system successfully reinitialized")
                    return {"success": True, "actions": actions}
            
            # Try fallback system
            actions.append("Attempting fallback RAG system initialization")
            self._initialize_fallback_rag_system()
            
            if self.rag_system:
                actions.append("Fallback RAG system initialized successfully")
                return {"success": True, "actions": actions}
            
            actions.append("Unable to initialize any RAG system")
            return {"success": False, "actions": actions}
            
        except Exception as e:
            actions.append(f"Dependency recovery error: {str(e)}")
            return {"success": False, "actions": actions}
    
    def _recover_from_initialization_failure(self, user_id: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from RAG initialization failures"""
        actions = []
        
        try:
            # Clear any corrupted initialization state
            actions.append("Clearing corrupted initialization state")
            self._clear_user_rag_data(user_id)
            
            # Reset database status
            actions.append("Resetting database initialization status")
            self._update_rag_status(user_id, "not_initialized")
            
            # Recreate storage directories
            actions.append("Recreating storage directories")
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            # Attempt reinitialization with fresh state
            actions.append("Attempting fresh initialization")
            current_status = self.check_rag_initialization_status(user_id)
            company_name = current_status.get("company_name", "Unknown")
            
            init_result = self._enhanced_rag_initialization(user_id, company_name, force_reinit=True)
            
            if init_result["success"]:
                actions.append("Fresh initialization successful")
                return {"success": True, "actions": actions}
            else:
                actions.append(f"Fresh initialization failed: {init_result.get('error', 'Unknown error')}")
                return {"success": False, "actions": actions}
            
        except Exception as e:
            actions.append(f"Initialization recovery error: {str(e)}")
            return {"success": False, "actions": actions}
    
    def _recover_from_schema_failure(self, user_id: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from database schema failures"""
        actions = []
        
        try:
            # Attempt schema migration
            actions.append("Attempting database schema migration")
            
            if self.schema_migrator:
                migration_result = self.schema_migrator.execute_automatic_migration()
                
                migration_success = True
                for table_name, result in migration_result.items():
                    if result.status.value == "failed":
                        migration_success = False
                        break
                
                if migration_success:
                    actions.append("Schema migration completed successfully")
                    
                    # Test database connectivity
                    actions.append("Testing database connectivity")
                    db_check = self._check_database_connectivity(user_id)
                    
                    if db_check["status"] == "pass":
                        actions.append("Database connectivity restored")
                        return {"success": True, "actions": actions}
                    else:
                        actions.append(f"Database connectivity test failed: {db_check.get('message', 'Unknown error')}")
                        return {"success": False, "actions": actions}
                else:
                    actions.append("Schema migration failed")
                    return {"success": False, "actions": actions}
            else:
                actions.append("Schema migrator not available")
                return {"success": False, "actions": actions}
            
        except Exception as e:
            actions.append(f"Schema recovery error: {str(e)}")
            return {"success": False, "actions": actions}
    
    def _recover_from_storage_failure(self, user_id: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
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
                    return {"success": False, "actions": actions}
            
            # Test write permissions
            actions.append("Testing write permissions")
            try:
                test_file = os.path.join(storage_path, "rag", ".recovery_test")
                with open(test_file, 'w') as f:
                    f.write("recovery test")
                os.remove(test_file)
                actions.append("Write permissions verified")
            except Exception as e:
                actions.append(f"Write permission test failed: {str(e)}")
                return {"success": False, "actions": actions}
            
            return {"success": True, "actions": actions}
            
        except Exception as e:
            actions.append(f"Storage recovery error: {str(e)}")
            return {"success": False, "actions": actions}
    
    def _generic_recovery_attempt(self, user_id: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic recovery attempt for unknown failure types"""
        actions = []
        
        try:
            # Run comprehensive system validation
            actions.append("Running comprehensive system validation")
            validation_result = self.startup_validation()
            
            if validation_result.overall_status == "healthy":
                actions.append("System validation passed - attempting reinitialization")
                
                # Attempt reinitialization
                current_status = self.check_rag_initialization_status(user_id)
                company_name = current_status.get("company_name", "Unknown")
                
                init_result = self.safe_initialize_rag(user_id, company_name, force_reinit=True)
                
                if init_result["success"]:
                    actions.append("Reinitialization successful")
                    return {"success": True, "actions": actions}
                else:
                    actions.append(f"Reinitialization failed: {init_result.get('error', 'Unknown error')}")
                    return {"success": False, "actions": actions}
            else:
                actions.append(f"System validation failed: {validation_result.overall_status}")
                actions.extend(validation_result.issues[:3])  # Add first 3 issues
                return {"success": False, "actions": actions}
            
        except Exception as e:
            actions.append(f"Generic recovery error: {str(e)}")
            return {"success": False, "actions": actions}
    
    def _log_recovery_attempt(self, user_id: str, error_type: str, success: bool, 
                             actions: List[str], recovery_time: float):
        """Log recovery attempt to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create recovery log table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_recovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    actions TEXT NOT NULL,
                    recovery_time REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert recovery log entry
            cursor.execute('''
                INSERT INTO rag_recovery_log 
                (user_id, error_type, success, actions, recovery_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, error_type, success, '; '.join(actions), recovery_time))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging recovery attempt for user {user_id}: {str(e)}")
    
    def get_recovery_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get recovery status and history for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with recovery status and history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent recovery attempts
            cursor.execute('''
                SELECT error_type, success, actions, recovery_time, timestamp
                FROM rag_recovery_log
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (user_id,))
            
            recovery_history = []
            for row in cursor.fetchall():
                error_type, success, actions, recovery_time, timestamp = row
                recovery_history.append({
                    "error_type": error_type,
                    "success": bool(success),
                    "actions": actions.split('; ') if actions else [],
                    "recovery_time": recovery_time,
                    "timestamp": timestamp
                })
            
            # Get recovery statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_recoveries,
                    AVG(recovery_time) as avg_recovery_time
                FROM rag_recovery_log
                WHERE user_id = ?
            ''', (user_id,))
            
            stats = cursor.fetchone()
            total_attempts, successful_recoveries, avg_recovery_time = stats or (0, 0, 0)
            
            conn.close()
            
            success_rate = (successful_recoveries / total_attempts * 100) if total_attempts > 0 else 0
            
            return {
                "user_id": user_id,
                "recovery_history": recovery_history,
                "statistics": {
                    "total_attempts": total_attempts,
                    "successful_recoveries": successful_recoveries,
                    "success_rate": round(success_rate, 2),
                    "average_recovery_time": round(avg_recovery_time or 0, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting recovery status for user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "recovery_history": [],
                "statistics": {
                    "total_attempts": 0,
                    "successful_recoveries": 0,
                    "success_rate": 0,
                    "average_recovery_time": 0
                },
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }   
 
    def _initialize_recovery_manager(self):
        """Initialize the recovery manager"""
        try:
            from .recovery_manager import RAGRecoveryManager
            self.recovery_manager = RAGRecoveryManager(self, self.db_path)
            logger.info("Recovery manager initialized successfully")
        except ImportError as e:
            logger.warning(f"Recovery manager not available: {e}")
            self.recovery_manager = None
        except Exception as e:
            logger.error(f"Error initializing recovery manager: {e}")
            self.recovery_manager = None
    
    def initialize_rag_with_automatic_recovery(
        self, 
        user_id: str, 
        company_name: str,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0
    ) -> Dict[str, Any]:
        """
        Initialize RAG with automatic recovery mechanisms and exponential backoff
        
        Args:
            user_id: User identifier
            company_name: Company name
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            max_delay: Maximum delay between retries
            
        Returns:
            Dictionary with initialization result and recovery information
        """
        try:
            # First attempt normal initialization
            logger.info(f"Attempting RAG initialization for user {user_id} with automatic recovery")
            
            result = self.safe_initialize_rag(user_id, company_name)
            
            if result["success"]:
                logger.info(f"RAG initialization successful for user {user_id} on first attempt")
                result["recovery_used"] = False
                return result
            
            # If initialization failed and recovery manager is available, use recovery
            if self.recovery_manager:
                logger.info(f"Initial RAG initialization failed for user {user_id}, starting automatic recovery")
                
                failure_context = {
                    "error_type": "initialization_failed",
                    "error_message": result.get("error", "Unknown initialization error"),
                    "initial_attempt_result": result
                }
                
                recovery_result = self.recovery_manager.recover_with_exponential_backoff(
                    user_id=user_id,
                    company_name=company_name,
                    failure_context=failure_context,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    max_delay=max_delay
                )
                
                # Combine initial result with recovery information
                result.update({
                    "recovery_used": True,
                    "recovery_result": recovery_result,
                    "final_success": recovery_result["success"]
                })
                
                # Update success status based on recovery result
                result["success"] = recovery_result["success"]
                
                if recovery_result["success"]:
                    result["message"] = f"RAG initialization successful after recovery (attempt {recovery_result.get('successful_attempt', 'unknown')})"
                else:
                    result["error"] = f"RAG initialization failed after recovery: {recovery_result.get('error', 'Unknown error')}"
                
                return result
            else:
                # No recovery manager available
                logger.warning(f"RAG initialization failed for user {user_id} and no recovery manager available")
                result["recovery_used"] = False
                result["recovery_available"] = False
                return result
                
        except Exception as e:
            logger.error(f"Error in RAG initialization with automatic recovery for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Initialization with recovery failed: {str(e)}",
                "recovery_used": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def recover_from_specific_failure(
        self, 
        user_id: str, 
        company_name: str,
        failure_type: str,
        failure_context: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Recover from a specific type of failure with automatic retry
        
        Args:
            user_id: User identifier
            company_name: Company name
            failure_type: Type of failure (dependency, schema, storage, etc.)
            failure_context: Context information about the failure
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with recovery result
        """
        try:
            if not self.recovery_manager:
                return {
                    "success": False,
                    "error": "Recovery manager not available",
                    "recovery_available": False,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"Starting specific failure recovery for user {user_id}, failure type: {failure_type}")
            
            # Prepare failure context
            enhanced_failure_context = {
                "error_type": failure_type,
                **failure_context
            }
            
            # Use recovery manager
            recovery_result = self.recovery_manager.recover_with_exponential_backoff(
                user_id=user_id,
                company_name=company_name,
                failure_context=enhanced_failure_context,
                max_retries=max_retries
            )
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"Error in specific failure recovery for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Recovery system error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recovery_status_and_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Get recovery status and recommendations for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with recovery status, statistics, and recommendations
        """
        try:
            if not self.recovery_manager:
                return {
                    "user_id": user_id,
                    "recovery_available": False,
                    "message": "Recovery manager not available",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get recovery status report
            recovery_report = self.recovery_manager.get_recovery_status_report(user_id)
            
            # Add current system status
            current_status = self.check_rag_initialization_status(user_id)
            system_health = self.get_system_health_with_dependencies()
            
            # Generate recommendations based on current status and recovery history
            recommendations = self._generate_recovery_recommendations(user_id, recovery_report, current_status, system_health)
            
            return {
                "user_id": user_id,
                "recovery_available": True,
                "current_rag_status": current_status,
                "system_health": system_health.get("overall_status", "unknown"),
                "recovery_statistics": recovery_report["statistics"],
                "recent_recovery_sessions": recovery_report["recent_sessions"],
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting recovery status for user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "recovery_available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_recovery_recommendations(
        self, 
        user_id: str, 
        recovery_report: Dict[str, Any], 
        current_status: Dict[str, Any], 
        system_health: Dict[str, Any]
    ) -> List[str]:
        """Generate recovery recommendations based on user status and history"""
        recommendations = []
        
        try:
            stats = recovery_report.get("statistics", {})
            recent_sessions = recovery_report.get("recent_sessions", [])
            
            # Check if user has RAG issues
            if not current_status.get("is_initialized", False):
                recommendations.append("RAG system is not initialized - consider running initialization with automatic recovery")
            
            # Check recovery success rate
            success_rate = stats.get("recovery_success_rate", 0)
            if success_rate < 50 and stats.get("total_recovery_sessions", 0) > 2:
                recommendations.append("Low recovery success rate - manual intervention may be required")
            
            # Check recent failures
            if recent_sessions:
                recent_failures = [s for s in recent_sessions[:3] if s.get("final_status") != "success"]
                if len(recent_failures) >= 2:
                    recommendations.append("Multiple recent recovery failures - check system configuration")
            
            # Check system health
            overall_health = system_health.get("overall_status", "unknown")
            if overall_health == "failed":
                recommendations.append("System health is critical - resolve dependency and schema issues")
            elif overall_health == "degraded":
                recommendations.append("System is running with limited functionality - consider installing missing dependencies")
            
            # Check average recovery time
            avg_time = stats.get("average_recovery_time", 0)
            if avg_time > 30:
                recommendations.append("Recovery times are high - consider system optimization")
            
            if not recommendations:
                recommendations.append("System appears healthy - no specific recovery actions needed")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recovery recommendations: {str(e)}")
            return ["Unable to generate recommendations - check system logs"]

    # Helper methods for enhanced initialization
    
    def _validate_user_exists(self, user_id: str) -> Dict[str, Any]:
        """Validate that user exists in the system"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id, company_name, is_active FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {
                    "status": "fail",
                    "issues": [f"User {user_id} not found in database"],
                    "recommendations": ["Verify user ID is correct", "Check user registration status"]
                }
            
            user_id_db, company_name, is_active = result
            
            if not is_active:
                return {
                    "status": "fail",
                    "issues": [f"User {user_id} is not active"],
                    "recommendations": ["Activate user account before initializing RAG"]
                }
            
            return {
                "status": "pass",
                "message": f"User {user_id} exists and is active",
                "company_name": company_name
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"User validation error: {str(e)}"],
                "recommendations": ["Check database connectivity"]
            }
    
    def _validate_database_schema(self) -> Dict[str, Any]:
        """Validate database schema is up to date"""
        try:
            if not self.schema_migrator:
                return {
                    "status": "fail",
                    "issues": ["Schema migrator not available"],
                    "recommendations": ["Install schema migration dependencies"]
                }
            
            validation_results = self.schema_migrator.validate_schema()
            
            schema_issues = []
            for table_name, validation in validation_results.items():
                if not validation.is_valid:
                    if validation.missing_columns:
                        schema_issues.append(f"Table {table_name} missing columns: {', '.join(validation.missing_columns)}")
            
            if schema_issues:
                return {
                    "status": "fail",
                    "issues": schema_issues,
                    "recommendations": ["Run database schema migration"]
                }
            
            return {
                "status": "pass",
                "message": "Database schema is valid"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"Schema validation error: {str(e)}"],
                "recommendations": ["Check database configuration"]
            }
    
    def _validate_initialization_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies required for initialization"""
        try:
            dependency_result = self.dependency_validator.check_all_dependencies()
            
            if dependency_result.overall_status == "failed":
                return {
                    "status": "fail",
                    "issues": [f"Critical dependencies missing: {', '.join(dependency_result.critical_missing)}"],
                    "recommendations": ["Install critical dependencies before initialization"]
                }
            elif dependency_result.overall_status == "degraded":
                return {
                    "status": "warning",
                    "issues": [f"Optional dependencies missing: {', '.join(dependency_result.optional_missing)}"],
                    "recommendations": ["Consider installing optional dependencies for full functionality"]
                }
            
            return {
                "status": "pass",
                "message": "All dependencies are available"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"Dependency validation error: {str(e)}"],
                "recommendations": ["Check dependency installation"]
            }
    
    def _validate_storage_requirements(self, user_id: str) -> Dict[str, Any]:
        """Validate storage directories and permissions"""
        try:
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            issues = []
            recommendations = []
            
            # Check if base directory exists
            if not os.path.exists(storage_path):
                try:
                    os.makedirs(storage_path, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create user storage directory: {str(e)}")
                    recommendations.append("Check directory permissions")
            
            # Check required subdirectories
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except Exception as e:
                        issues.append(f"Cannot create {dir_name} directory: {str(e)}")
                        recommendations.append(f"Check permissions for {dir_path}")
            
            # Test write permissions
            try:
                test_file = os.path.join(storage_path, "rag", ".test_write")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                issues.append(f"Storage directory not writable: {str(e)}")
                recommendations.append("Check directory write permissions")
            
            if issues:
                return {
                    "status": "fail",
                    "issues": issues,
                    "recommendations": recommendations
                }
            
            return {
                "status": "pass",
                "message": "Storage requirements validated"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "issues": [f"Storage validation error: {str(e)}"],
                "recommendations": ["Check storage configuration"]
            }
    
    def _validate_system_resources(self) -> Dict[str, Any]:
        """Validate system resources for RAG initialization"""
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            issues = []
            recommendations = []
            
            if available_gb < 1.0:  # Less than 1GB available
                issues.append(f"Low available memory: {available_gb:.1f}GB")
                recommendations.append("Close other applications to free memory")
            
            # Check disk space
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 0.5:  # Less than 500MB free
                issues.append(f"Low disk space: {free_gb:.1f}GB free")
                recommendations.append("Free up disk space before initialization")
            
            if issues:
                return {
                    "status": "warning",
                    "issues": issues,
                    "recommendations": recommendations
                }
            
            return {
                "status": "pass",
                "message": f"System resources adequate (Memory: {available_gb:.1f}GB, Disk: {free_gb:.1f}GB)"
            }
            
        except ImportError:
            # psutil not available, skip resource check
            return {
                "status": "pass",
                "message": "System resource validation skipped (psutil not available)"
            }
        except Exception as e:
            return {
                "status": "warning",
                "issues": [f"Resource validation error: {str(e)}"],
                "recommendations": ["Check system resource availability"]
            }
    
    def _update_initialization_status(self, user_id: str, status: str, message: str):
        """Update initialization status in database with detailed tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create initialization_status table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS initialization_status (
                    user_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_count INTEGER DEFAULT 0
                )
            ''')
            
            # Update or insert status
            cursor.execute('''
                INSERT OR REPLACE INTO initialization_status 
                (user_id, status, message, last_updated, error_count)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, 
                    CASE WHEN ? = 'failed' THEN 
                        COALESCE((SELECT error_count FROM initialization_status WHERE user_id = ?), 0) + 1
                    ELSE 0 END)
            ''', (user_id, status, message, status, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated initialization status for user {user_id}: {status} - {message}")
            
        except Exception as e:
            logger.error(f"Error updating initialization status for user {user_id}: {str(e)}")
    
    def _prepare_initialization_environment(self, user_id: str, company_name: str) -> Dict[str, Any]:
        """Prepare the environment for RAG initialization"""
        try:
            # Ensure storage directories exist
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            # Clear any existing temporary files
            temp_files = []
            for root, dirs, files in os.walk(storage_path):
                for file in files:
                    if file.startswith('.temp_') or file.startswith('.init_'):
                        temp_files.append(os.path.join(root, file))
            
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass  # Ignore errors removing temp files
            
            # Initialize user in database if not exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET 
                    rag_initialized = 0,
                    rag_initialization_error = NULL
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": "Environment prepared successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Environment preparation failed: {str(e)}"
            }
    
    def _enhanced_rag_initialization(self, user_id: str, company_name: str, force_reinit: bool) -> Dict[str, Any]:
        """Enhanced RAG initialization with better error handling"""
        try:
            if not self.rag_system:
                return {
                    "success": False,
                    "error": "RAG system not available"
                }
            
            # Initialize RAG system with enhanced error handling
            success = self.rag_system.initialize_company_rag(user_id, company_name)
            
            if success:
                # Update database status
                self._update_rag_status(user_id, "initialized")
                
                return {
                    "success": True,
                    "message": "RAG system initialized successfully",
                    "user_id": user_id,
                    "company_name": company_name
                }
            else:
                self._update_rag_status(user_id, "failed", "RAG initialization failed")
                return {
                    "success": False,
                    "error": "RAG initialization failed"
                }
                
        except Exception as e:
            error_msg = f"RAG initialization error: {str(e)}"
            self._update_rag_status(user_id, "failed", error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _post_initialization_validation(self, user_id: str) -> Dict[str, Any]:
        """Validate RAG system after initialization"""
        try:
            # Check if RAG system is properly initialized
            status = self.check_rag_initialization_status(user_id)
            
            if not status.get("is_initialized"):
                return {
                    "success": False,
                    "error": "RAG system not properly initialized"
                }
            
            # Test basic RAG functionality
            if self.rag_system:
                try:
                    # Test query functionality
                    test_results = self.rag_system.query_user_knowledge(user_id, "test", top_k=1)
                    # If no exception, RAG system is working
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"RAG system test failed: {str(e)}"
                    }
            
            return {
                "success": True,
                "message": "Post-initialization validation passed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Post-initialization validation error: {str(e)}"
            }
    
    def _generate_degradation_message(self, validation_result: Dict[str, Any]) -> str:
        """Generate user-friendly degradation message"""
        try:
            issues = validation_result.get("issues", [])
            if not issues:
                return "System is operating normally"
            
            message_parts = ["⚠️ RAG system is running with limited functionality:"]
            
            for issue in issues[:3]:  # Limit to first 3 issues
                message_parts.append(f"• {issue}")
            
            if len(issues) > 3:
                message_parts.append(f"• ... and {len(issues) - 3} more issues")
            
            message_parts.append("\nTo restore full functionality, please address the issues above.")
            
            return "\n".join(message_parts)
            
        except Exception as e:
            logger.error(f"Error generating degradation message: {str(e)}")
            return "⚠️ System is running with limited functionality. Contact administrator for details." 
   
    # Automatic Recovery Mechanisms with Exponential Backoff
    
    def initialize_rag_with_retry(self, user_id: str, company_name: str, max_retries: int = 3, 
                                 base_delay: float = 1.0, max_delay: float = 60.0) -> Dict[str, Any]:
        """
        Initialize RAG with automatic retry and exponential backoff
        
        Args:
            user_id: User identifier
            company_name: Company name
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds
            
        Returns:
            Dictionary with initialization result and retry information
        """
        import time
        import random
        
        retry_info = {
            "attempts": 0,
            "total_time": 0,
            "retry_delays": [],
            "failure_reasons": []
        }
        
        start_time = datetime.now()
        
        for attempt in range(max_retries + 1):
            retry_info["attempts"] = attempt + 1
            
            try:
                logger.info(f"RAG initialization attempt {attempt + 1}/{max_retries + 1} for user {user_id}")
                
                # Attempt initialization
                result = self.safe_initialize_rag(user_id, company_name, force_reinit=attempt > 0)
                
                if result["success"]:
                    retry_info["total_time"] = (datetime.now() - start_time).total_seconds()
                    result["retry_info"] = retry_info
                    
                    logger.info(f"RAG initialization successful for user {user_id} after {attempt + 1} attempts")
                    return result
                else:
                    # Record failure reason
                    failure_reason = result.get("error", "Unknown error")
                    retry_info["failure_reasons"].append(failure_reason)
                    
                    logger.warning(f"RAG initialization attempt {attempt + 1} failed for user {user_id}: {failure_reason}")
                    
                    # If this is the last attempt, return the failure
                    if attempt >= max_retries:
                        retry_info["total_time"] = (datetime.now() - start_time).total_seconds()
                        result["retry_info"] = retry_info
                        result["final_attempt"] = True
                        
                        logger.error(f"RAG initialization failed for user {user_id} after {max_retries + 1} attempts")
                        return result
                    
                    # Calculate exponential backoff delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    actual_delay = delay + jitter
                    
                    retry_info["retry_delays"].append(actual_delay)
                    
                    logger.info(f"Retrying RAG initialization for user {user_id} in {actual_delay:.2f} seconds")
                    
                    # Attempt recovery before retry
                    recovery_result = self._attempt_automatic_recovery(user_id, failure_reason, attempt)
                    if recovery_result.get("recovery_attempted"):
                        logger.info(f"Automatic recovery attempted for user {user_id}: {recovery_result.get('recovery_actions', [])}")
                    
                    time.sleep(actual_delay)
                    
            except Exception as e:
                error_msg = f"Unexpected error during initialization attempt {attempt + 1}: {str(e)}"
                retry_info["failure_reasons"].append(error_msg)
                logger.error(f"Error in RAG initialization attempt for user {user_id}: {error_msg}")
                
                if attempt >= max_retries:
                    retry_info["total_time"] = (datetime.now() - start_time).total_seconds()
                    return {
                        "success": False,
                        "error": error_msg,
                        "status": "retry_exhausted",
                        "retry_info": retry_info
                    }
                
                # Continue with exponential backoff even for unexpected errors
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0.1, 0.3) * delay
                actual_delay = delay + jitter
                retry_info["retry_delays"].append(actual_delay)
                
                time.sleep(actual_delay)
        
        # This should not be reached, but just in case
        retry_info["total_time"] = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "error": "Maximum retries exceeded",
            "status": "retry_exhausted",
            "retry_info": retry_info
        }
    
    def _attempt_automatic_recovery(self, user_id: str, failure_reason: str, attempt_number: int) -> Dict[str, Any]:
        """
        Attempt automatic recovery based on failure type
        
        Args:
            user_id: User identifier
            failure_reason: Reason for the failure
            attempt_number: Current attempt number
            
        Returns:
            Dictionary with recovery result
        """
        recovery_actions = []
        recovery_attempted = False
        
        try:
            # Analyze failure reason and attempt appropriate recovery
            failure_lower = failure_reason.lower()
            
            # Recovery for dependency issues
            if any(keyword in failure_lower for keyword in ['dependency', 'import', 'module']):
                recovery_attempted = True
                recovery_actions.append("Reinitializing dependency validator")
                
                # Reinitialize dependency validator
                self.dependency_validator = DependencyValidator()
                
                # Try to reinitialize RAG system with current dependencies
                self._initialize_rag_system_with_validation()
                recovery_actions.append("Reinitialized RAG system with available dependencies")
            
            # Recovery for storage issues
            elif any(keyword in failure_lower for keyword in ['storage', 'directory', 'permission']):
                recovery_attempted = True
                recovery_actions.append("Recreating storage directories")
                
                # Recreate storage directories
                storage_path = f"data/users/{user_id}"
                required_dirs = ["csv", "pdf", "rag"]
                
                for dir_name in required_dirs:
                    dir_path = os.path.join(storage_path, dir_name)
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                        recovery_actions.append(f"Created directory: {dir_path}")
                    except Exception as e:
                        recovery_actions.append(f"Failed to create directory {dir_path}: {str(e)}")
            
            # Recovery for database issues
            elif any(keyword in failure_lower for keyword in ['database', 'schema', 'migration']):
                recovery_attempted = True
                recovery_actions.append("Attempting database recovery")
                
                # Try to run schema migration
                if self.schema_migrator:
                    try:
                        migration_result = self.run_startup_migration()
                        if migration_result["success"]:
                            recovery_actions.append("Database schema migration successful")
                        else:
                            recovery_actions.append(f"Database migration failed: {migration_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        recovery_actions.append(f"Database migration error: {str(e)}")
            
            # Recovery for RAG system issues
            elif any(keyword in failure_lower for keyword in ['rag', 'initialization', 'vector']):
                recovery_attempted = True
                recovery_actions.append("Clearing RAG system cache")
                
                # Clear any cached RAG data
                try:
                    self._clear_user_rag_data(user_id)
                    recovery_actions.append("Cleared user RAG data")
                except Exception as e:
                    recovery_actions.append(f"Failed to clear RAG data: {str(e)}")
                
                # Reinitialize RAG system
                try:
                    self._initialize_rag_system_with_validation()
                    recovery_actions.append("Reinitialized RAG system")
                except Exception as e:
                    recovery_actions.append(f"Failed to reinitialize RAG system: {str(e)}")
            
            # Generic recovery for unknown issues
            else:
                if attempt_number >= 1:  # Only attempt generic recovery after first failure
                    recovery_attempted = True
                    recovery_actions.append("Performing generic system recovery")
                    
                    # Clear temporary files
                    try:
                        storage_path = f"data/users/{user_id}"
                        if os.path.exists(storage_path):
                            for root, dirs, files in os.walk(storage_path):
                                for file in files:
                                    if file.startswith('.temp_') or file.startswith('.init_'):
                                        os.remove(os.path.join(root, file))
                        recovery_actions.append("Cleared temporary files")
                    except Exception as e:
                        recovery_actions.append(f"Failed to clear temporary files: {str(e)}")
            
            # Update recovery status in database
            self._update_recovery_status(user_id, recovery_attempted, recovery_actions, failure_reason)
            
            return {
                "recovery_attempted": recovery_attempted,
                "recovery_actions": recovery_actions,
                "failure_reason": failure_reason,
                "attempt_number": attempt_number
            }
            
        except Exception as e:
            logger.error(f"Error during automatic recovery for user {user_id}: {str(e)}")
            return {
                "recovery_attempted": False,
                "recovery_actions": [f"Recovery failed: {str(e)}"],
                "failure_reason": failure_reason,
                "attempt_number": attempt_number
            }
    
    def _update_recovery_status(self, user_id: str, recovery_attempted: bool, 
                               recovery_actions: List[str], failure_reason: str):
        """Update recovery status in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create recovery_log table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    recovery_attempted BOOLEAN NOT NULL,
                    recovery_actions TEXT,
                    failure_reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert recovery log entry
            cursor.execute('''
                INSERT INTO recovery_log (user_id, recovery_attempted, recovery_actions, failure_reason)
                VALUES (?, ?, ?, ?)
            ''', (user_id, recovery_attempted, '; '.join(recovery_actions), failure_reason))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating recovery status for user {user_id}: {str(e)}")
    
    def get_recovery_status_report(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get recovery status report for user or system-wide
        
        Args:
            user_id: Optional user identifier for user-specific report
            
        Returns:
            Dictionary with recovery status information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if recovery_log table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='recovery_log'
            ''')
            
            if not cursor.fetchone():
                conn.close()
                return {
                    "success": True,
                    "message": "No recovery attempts recorded",
                    "recovery_attempts": 0,
                    "successful_recoveries": 0
                }
            
            if user_id:
                # User-specific report
                cursor.execute('''
                    SELECT recovery_attempted, recovery_actions, failure_reason, timestamp
                    FROM recovery_log
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (user_id,))
                
                recovery_logs = cursor.fetchall()
                
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN recovery_attempted = 1 THEN 1 ELSE 0 END) as recovery_attempts
                    FROM recovery_log
                    WHERE user_id = ?
                ''', (user_id,))
                
                stats = cursor.fetchone()
                
            else:
                # System-wide report
                cursor.execute('''
                    SELECT user_id, recovery_attempted, recovery_actions, failure_reason, timestamp
                    FROM recovery_log
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''')
                
                recovery_logs = cursor.fetchall()
                
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN recovery_attempted = 1 THEN 1 ELSE 0 END) as recovery_attempts,
                        COUNT(DISTINCT user_id) as affected_users
                    FROM recovery_log
                ''')
                
                stats = cursor.fetchone()
            
            conn.close()
            
            total_attempts, recovery_attempts = stats[0], stats[1]
            
            report = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "statistics": {
                    "total_failure_attempts": total_attempts,
                    "recovery_attempts": recovery_attempts,
                    "recovery_rate": (recovery_attempts / total_attempts * 100) if total_attempts > 0 else 0
                },
                "recent_recovery_logs": []
            }
            
            # Add affected users count for system-wide report
            if not user_id and len(stats) > 2:
                report["statistics"]["affected_users"] = stats[2]
            
            # Format recovery logs
            for log in recovery_logs:
                if user_id:
                    recovery_attempted, recovery_actions, failure_reason, timestamp = log
                    log_entry = {
                        "recovery_attempted": bool(recovery_attempted),
                        "recovery_actions": recovery_actions.split('; ') if recovery_actions else [],
                        "failure_reason": failure_reason,
                        "timestamp": timestamp
                    }
                else:
                    log_user_id, recovery_attempted, recovery_actions, failure_reason, timestamp = log
                    log_entry = {
                        "user_id": log_user_id,
                        "recovery_attempted": bool(recovery_attempted),
                        "recovery_actions": recovery_actions.split('; ') if recovery_actions else [],
                        "failure_reason": failure_reason,
                        "timestamp": timestamp
                    }
                
                report["recent_recovery_logs"].append(log_entry)
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting recovery status report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_recovery_logs(self, user_id: str = None, days_old: int = 30) -> Dict[str, Any]:
        """
        Clear old recovery logs
        
        Args:
            user_id: Optional user identifier to clear logs for specific user
            days_old: Clear logs older than this many days
            
        Returns:
            Dictionary with cleanup result
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if recovery_log table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='recovery_log'
            ''')
            
            if not cursor.fetchone():
                conn.close()
                return {
                    "success": True,
                    "message": "No recovery logs to clear",
                    "logs_cleared": 0
                }
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            if user_id:
                cursor.execute('''
                    DELETE FROM recovery_log
                    WHERE user_id = ? AND timestamp < ?
                ''', (user_id, cutoff_date.isoformat()))
            else:
                cursor.execute('''
                    DELETE FROM recovery_log
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
            
            logs_cleared = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {logs_cleared} recovery logs older than {days_old} days")
            
            return {
                "success": True,
                "message": f"Cleared {logs_cleared} recovery logs",
                "logs_cleared": logs_cleared,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clearing recovery logs: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Global enhanced instance
enhanced_rag_manager = EnhancedRAGManager()