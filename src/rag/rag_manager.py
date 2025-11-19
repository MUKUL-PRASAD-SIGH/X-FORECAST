"""
RAG Management Utilities
Provides utilities for managing company-specific RAG systems including initialization,
health checks, diagnostics, and migration capabilities.
"""

import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import shutil

logger = logging.getLogger(__name__)

@dataclass
class RAGHealthStatus:
    """RAG system health status"""
    user_id: str
    company_name: str
    is_initialized: bool
    status: str  # 'healthy', 'degraded', 'failed', 'not_initialized'
    total_documents: int
    csv_count: int
    pdf_count: int
    last_updated: Optional[datetime]
    error_message: Optional[str] = None
    index_version: str = "1.0"

@dataclass
class RAGMigrationResult:
    """Result of RAG migration operation"""
    success: bool
    user_id: str
    documents_migrated: int
    errors: List[str]
    migration_time: datetime

class RAGManager:
    """
    Comprehensive RAG management utilities for multi-tenant system
    """
    
    def __init__(self, db_path: str = "users.db", rag_db_path: str = "rag_vector_db.db"):
        self.db_path = db_path
        self.rag_db_path = rag_db_path
        
        # Import RAG system with dependency validation
        try:
            from .dependency_validator import dependency_validator
            
            # Check dependencies before importing RAG system
            if dependency_validator.validate_sentence_transformers():
                from .real_vector_rag import RealVectorRAG
                self.rag_system = RealVectorRAG()
                logger.info("Full RAG system loaded successfully")
            else:
                logger.warning("Critical dependencies missing - attempting fallback RAG system")
                try:
                    from .fallback_rag import FallbackRAGSystem
                    self.rag_system = FallbackRAGSystem()
                    logger.info("Fallback RAG system loaded")
                except ImportError:
                    logger.error("Fallback RAG system also not available")
                    self.rag_system = None
                    
        except ImportError as e:
            logger.error(f"RAG system not available: {e}")
            self.rag_system = None
        
        # Import schema migration system
        try:
            from src.database.schema_migrator import DatabaseSchemaMigrator
            self.schema_migrator = DatabaseSchemaMigrator(db_path, rag_db_path)
        except ImportError:
            logger.error("Schema migration system not available")
            self.schema_migrator = None
    
    def check_rag_initialization_status(self, user_id: str) -> Dict[str, Any]:
        """
        Check RAG initialization status for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with initialization status and details
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.rag_initialized, u.rag_initialization_error, u.company_name,
                       bp.rag_status, bp.rag_initialized_at
                FROM users u
                LEFT JOIN business_profiles bp ON u.user_id = bp.user_id
                WHERE u.user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {
                    "success": False,
                    "error": "User not found",
                    "status": "user_not_found"
                }
            
            rag_initialized, error_message, company_name, rag_status, initialized_at = result
            
            # Get document counts from RAG database
            document_counts = self._get_document_counts(user_id)
            
            return {
                "success": True,
                "user_id": user_id,
                "company_name": company_name,
                "is_initialized": bool(rag_initialized),
                "status": rag_status or "not_initialized",
                "initialized_at": initialized_at,
                "error_message": error_message,
                "document_counts": document_counts,
                "requires_initialization": not bool(rag_initialized)
            }
            
        except Exception as e:
            logger.error(f"Error checking RAG status for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status": "check_failed"
            }
    
    def run_startup_migration(self) -> Dict[str, Any]:
        """
        Run automatic database schema migration during RAG system startup
        
        Returns:
            Dictionary with migration results
        """
        try:
            if not self.schema_migrator:
                return {
                    "success": False,
                    "error": "Schema migration system not available",
                    "migration_skipped": True
                }
            
            logger.info("Running automatic database schema migration for RAG system startup")
            
            # Run schema validation first
            validation_results = self.schema_migrator.validate_schema()
            
            # Check if migration is needed
            migration_needed = False
            for table_name, validation in validation_results.items():
                if not validation.is_valid and validation.missing_columns:
                    migration_needed = True
                    break
            
            if not migration_needed:
                logger.info("Database schema is up to date, no migration needed")
                return {
                    "success": True,
                    "message": "Database schema is up to date",
                    "migration_needed": False,
                    "validation_results": validation_results
                }
            
            # Execute automatic migration
            migration_results = self.schema_migrator.execute_automatic_migration()
            
            # Check migration success
            migration_success = True
            total_columns_added = 0
            total_errors = 0
            
            for table_name, migration in migration_results.items():
                if migration.status.value in ["failed"]:
                    migration_success = False
                total_columns_added += len(migration.columns_added)
                total_errors += len(migration.errors)
            
            logger.info(f"Startup migration completed: {total_columns_added} columns added, {total_errors} errors")
            
            return {
                "success": migration_success,
                "message": f"Migration completed: {total_columns_added} columns added, {total_errors} errors",
                "migration_needed": True,
                "columns_added": total_columns_added,
                "errors": total_errors,
                "validation_results": validation_results,
                "migration_results": migration_results
            }
            
        except Exception as e:
            logger.error(f"Error during startup migration: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "migration_needed": True
            }

    def initialize_rag_for_user(self, user_id: str, company_name: str, force_reinit: bool = False) -> Dict[str, Any]:
        """
        Initialize or reinitialize RAG system for a user
        
        Args:
            user_id: User identifier
            company_name: Company name
            force_reinit: Whether to force reinitialization if already initialized
            
        Returns:
            Dictionary with initialization result
        """
        try:
            # Run startup migration before initializing RAG
            migration_result = self.run_startup_migration()
            if not migration_result["success"] and migration_result.get("migration_needed", False):
                logger.warning(f"Schema migration failed, but continuing with RAG initialization: {migration_result.get('error', 'Unknown error')}")
            
            if not self.rag_system:
                return {
                    "success": False,
                    "error": "RAG system not available",
                    "status": "system_unavailable"
                }
            
            # Check current status
            current_status = self.check_rag_initialization_status(user_id)
            
            if current_status.get("is_initialized") and not force_reinit:
                return {
                    "success": True,
                    "message": "RAG already initialized",
                    "status": "already_initialized",
                    "current_status": current_status
                }
            
            # Create user storage directories if they don't exist
            storage_path = f"data/users/{user_id}"
            os.makedirs(f"{storage_path}/csv", exist_ok=True)
            os.makedirs(f"{storage_path}/pdf", exist_ok=True)
            os.makedirs(f"{storage_path}/rag", exist_ok=True)
            
            # Initialize RAG system
            success = self.rag_system.initialize_company_rag(user_id, company_name)
            
            if success:
                # Update database status
                self._update_rag_status(user_id, "initialized")
                
                logger.info(f"RAG system initialized for user {user_id} ({company_name})")
                
                return {
                    "success": True,
                    "message": "RAG system initialized successfully",
                    "status": "initialized",
                    "user_id": user_id,
                    "company_name": company_name
                }
            else:
                self._update_rag_status(user_id, "failed", "RAG initialization failed")
                return {
                    "success": False,
                    "error": "RAG initialization failed",
                    "status": "initialization_failed"
                }
                
        except Exception as e:
            error_msg = f"RAG initialization error: {str(e)}"
            logger.error(f"Error initializing RAG for user {user_id}: {error_msg}")
            self._update_rag_status(user_id, "failed", error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "status": "initialization_error"
            }
    
    def reset_rag_system(self, user_id: str, company_name: str) -> Dict[str, Any]:
        """
        Reset and reinitialize RAG system for a user (clears all data)
        
        Args:
            user_id: User identifier
            company_name: Company name
            
        Returns:
            Dictionary with reset result
        """
        try:
            if not self.rag_system:
                return {
                    "success": False,
                    "error": "RAG system not available",
                    "status": "system_unavailable"
                }
            
            # Clear existing RAG data
            self._clear_user_rag_data(user_id)
            
            # Reinitialize
            result = self.initialize_rag_for_user(user_id, company_name, force_reinit=True)
            
            if result["success"]:
                result["message"] = "RAG system reset and reinitialized successfully"
                result["status"] = "reset_complete"
                logger.info(f"RAG system reset for user {user_id} ({company_name})")
            
            return result
            
        except Exception as e:
            error_msg = f"RAG reset error: {str(e)}"
            logger.error(f"Error resetting RAG for user {user_id}: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "status": "reset_failed"
            }
    
    def get_rag_health_status(self, user_id: str) -> RAGHealthStatus:
        """
        Get comprehensive health status of user's RAG system
        
        Args:
            user_id: User identifier
            
        Returns:
            RAGHealthStatus object with detailed health information
        """
        try:
            # Get basic status
            status_info = self.check_rag_initialization_status(user_id)
            
            if not status_info["success"]:
                return RAGHealthStatus(
                    user_id=user_id,
                    company_name="Unknown",
                    is_initialized=False,
                    status="failed",
                    total_documents=0,
                    csv_count=0,
                    pdf_count=0,
                    last_updated=None,
                    error_message=status_info.get("error", "Unknown error")
                )
            
            # Get document counts and last update time
            document_counts = status_info.get("document_counts", {})
            
            # Determine health status
            health_status = "healthy"
            if not status_info["is_initialized"]:
                health_status = "not_initialized"
            elif status_info.get("error_message"):
                health_status = "degraded"
            elif document_counts.get("total", 0) == 0:
                health_status = "degraded"  # No documents loaded
            
            # Get last update time from RAG database
            last_updated = self._get_last_update_time(user_id)
            
            return RAGHealthStatus(
                user_id=user_id,
                company_name=status_info.get("company_name", "Unknown"),
                is_initialized=status_info["is_initialized"],
                status=health_status,
                total_documents=document_counts.get("total", 0),
                csv_count=document_counts.get("csv", 0),
                pdf_count=document_counts.get("pdf", 0),
                last_updated=last_updated,
                error_message=status_info.get("error_message")
            )
            
        except Exception as e:
            logger.error(f"Error getting RAG health status for user {user_id}: {str(e)}")
            return RAGHealthStatus(
                user_id=user_id,
                company_name="Unknown",
                is_initialized=False,
                status="failed",
                total_documents=0,
                csv_count=0,
                pdf_count=0,
                last_updated=None,
                error_message=str(e)
            )
    
    def run_rag_diagnostics(self, user_id: str) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics on user's RAG system
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with diagnostic results
        """
        try:
            diagnostics = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "checks": {},
                "overall_status": "healthy",
                "recommendations": []
            }
            
            # Check 1: RAG initialization status
            init_status = self.check_rag_initialization_status(user_id)
            diagnostics["checks"]["initialization"] = {
                "status": "pass" if init_status["success"] and init_status["is_initialized"] else "fail",
                "details": init_status
            }
            
            if not init_status["is_initialized"]:
                diagnostics["overall_status"] = "failed"
                diagnostics["recommendations"].append("Initialize RAG system for this user")
            
            # Check 2: Storage directories
            storage_check = self._check_storage_directories(user_id)
            diagnostics["checks"]["storage"] = storage_check
            
            if not storage_check["status"] == "pass":
                diagnostics["overall_status"] = "degraded"
                diagnostics["recommendations"].extend(storage_check.get("recommendations", []))
            
            # Check 3: Database connectivity
            db_check = self._check_database_connectivity(user_id)
            diagnostics["checks"]["database"] = db_check
            
            if not db_check["status"] == "pass":
                diagnostics["overall_status"] = "failed"
                diagnostics["recommendations"].extend(db_check.get("recommendations", []))
            
            # Check 4: Document processing status
            doc_check = self._check_document_processing_status(user_id)
            diagnostics["checks"]["documents"] = doc_check
            
            if doc_check["status"] == "warning":
                if diagnostics["overall_status"] == "healthy":
                    diagnostics["overall_status"] = "degraded"
                diagnostics["recommendations"].extend(doc_check.get("recommendations", []))
            
            # Check 5: RAG system performance
            if self.rag_system and init_status["is_initialized"]:
                perf_check = self._check_rag_performance(user_id)
                diagnostics["checks"]["performance"] = perf_check
                
                if perf_check["status"] == "warning":
                    if diagnostics["overall_status"] == "healthy":
                        diagnostics["overall_status"] = "degraded"
                    diagnostics["recommendations"].extend(perf_check.get("recommendations", []))
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error running RAG diagnostics for user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "checks": {},
                "overall_status": "failed",
                "error": str(e),
                "recommendations": ["Contact system administrator"]
            }
    
    def migrate_existing_users_to_rag(self, batch_size: int = 10) -> Dict[str, Any]:
        """
        Migrate existing users who don't have RAG initialized
        
        Args:
            batch_size: Number of users to process in each batch
            
        Returns:
            Dictionary with migration results
        """
        try:
            # Find users without RAG initialization
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.user_id, u.company_name, u.email
                FROM users u
                WHERE u.rag_initialized = 0 OR u.rag_initialized IS NULL
                AND u.is_active = 1
                LIMIT ?
            ''', (batch_size,))
            
            users_to_migrate = cursor.fetchall()
            conn.close()
            
            if not users_to_migrate:
                return {
                    "success": True,
                    "message": "No users require RAG migration",
                    "users_processed": 0,
                    "users_migrated": 0,
                    "errors": []
                }
            
            migration_results = []
            successful_migrations = 0
            errors = []
            
            for user_id, company_name, email in users_to_migrate:
                try:
                    logger.info(f"Migrating user {email} ({company_name}) to RAG system")
                    
                    result = self.initialize_rag_for_user(user_id, company_name)
                    
                    migration_result = RAGMigrationResult(
                        success=result["success"],
                        user_id=user_id,
                        documents_migrated=0,  # New initialization, no existing documents
                        errors=[result.get("error")] if not result["success"] else [],
                        migration_time=datetime.now()
                    )
                    
                    migration_results.append(migration_result)
                    
                    if result["success"]:
                        successful_migrations += 1
                        logger.info(f"Successfully migrated user {email}")
                    else:
                        error_msg = f"Failed to migrate user {email}: {result.get('error', 'Unknown error')}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error migrating user {email}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            return {
                "success": True,
                "message": f"Migration completed: {successful_migrations}/{len(users_to_migrate)} users migrated",
                "users_processed": len(users_to_migrate),
                "users_migrated": successful_migrations,
                "errors": errors,
                "migration_results": migration_results
            }
            
        except Exception as e:
            logger.error(f"Error during RAG migration: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "users_processed": 0,
                "users_migrated": 0,
                "errors": [str(e)]
            }
    
    def get_system_wide_rag_status(self) -> Dict[str, Any]:
        """
        Get system-wide RAG status and statistics
        
        Returns:
            Dictionary with system-wide RAG statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_users,
                    SUM(CASE WHEN rag_initialized = 1 THEN 1 ELSE 0 END) as initialized_users,
                    SUM(CASE WHEN rag_initialized = 0 OR rag_initialized IS NULL THEN 1 ELSE 0 END) as uninitialized_users
                FROM users 
                WHERE is_active = 1
            ''')
            
            stats = cursor.fetchone()
            total_users, initialized_users, uninitialized_users = stats
            
            # Get RAG status breakdown
            cursor.execute('''
                SELECT bp.rag_status, COUNT(*) as count
                FROM business_profiles bp
                JOIN users u ON bp.user_id = u.user_id
                WHERE u.is_active = 1
                GROUP BY bp.rag_status
            ''')
            
            status_breakdown = dict(cursor.fetchall())
            
            # Get document statistics from RAG database
            doc_stats = self._get_system_document_stats()
            
            conn.close()
            
            initialization_rate = (initialized_users / total_users * 100) if total_users > 0 else 0
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "total_users": total_users,
                "initialized_users": initialized_users,
                "uninitialized_users": uninitialized_users,
                "initialization_rate": round(initialization_rate, 2),
                "status_breakdown": status_breakdown,
                "document_statistics": doc_stats,
                "system_health": "healthy" if initialization_rate > 90 else "degraded" if initialization_rate > 70 else "poor"
            }
            
        except Exception as e:
            logger.error(f"Error getting system-wide RAG status: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Private helper methods
    
    def _get_document_counts(self, user_id: str) -> Dict[str, int]:
        """Get document counts for a user from RAG database"""
        try:
            if not os.path.exists(self.rag_db_path):
                return {"total": 0, "csv": 0, "pdf": 0}
            
            conn = sqlite3.connect(self.rag_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT document_type, COUNT(*) as count
                FROM user_vectors
                WHERE user_id = ?
                GROUP BY document_type
            ''', (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            counts = {"csv": 0, "pdf": 0}
            for doc_type, count in results:
                if doc_type in counts:
                    counts[doc_type] = count
            
            counts["total"] = sum(counts.values())
            return counts
            
        except Exception as e:
            logger.error(f"Error getting document counts for user {user_id}: {str(e)}")
            return {"total": 0, "csv": 0, "pdf": 0}
    
    def _get_last_update_time(self, user_id: str) -> Optional[datetime]:
        """Get last update time for user's RAG data"""
        try:
            if not os.path.exists(self.rag_db_path):
                return None
            
            conn = sqlite3.connect(self.rag_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT MAX(created_at) as last_update
                FROM user_vectors
                WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last update time for user {user_id}: {str(e)}")
            return None
    
    def _update_rag_status(self, user_id: str, status: str, error_message: str = None):
        """Update RAG initialization status in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update users table
            cursor.execute('''
                UPDATE users SET rag_initialized = ?, rag_initialization_error = ?
                WHERE user_id = ?
            ''', (status == "initialized", error_message, user_id))
            
            # Update business_profiles table
            rag_initialized_at = datetime.now() if status == "initialized" else None
            cursor.execute('''
                UPDATE business_profiles SET rag_status = ?, rag_initialized_at = ?
                WHERE user_id = ?
            ''', (status, rag_initialized_at, user_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating RAG status for user {user_id}: {str(e)}")
    
    def _clear_user_rag_data(self, user_id: str):
        """Clear all RAG data for a user"""
        try:
            # Clear from RAG database
            if os.path.exists(self.rag_db_path):
                conn = sqlite3.connect(self.rag_db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM user_vectors WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))
                cursor.execute('DELETE FROM user_documents WHERE user_id = ?', (user_id,))
                
                conn.commit()
                conn.close()
            
            # Clear from memory if RAG system is available
            if self.rag_system:
                self.rag_system._clear_user_data(user_id)
            
            # Clear storage directories
            storage_path = f"data/users/{user_id}/rag"
            if os.path.exists(storage_path):
                shutil.rmtree(storage_path)
                os.makedirs(storage_path, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error clearing RAG data for user {user_id}: {str(e)}")
    
    def _check_storage_directories(self, user_id: str) -> Dict[str, Any]:
        """Check if user storage directories exist and are accessible"""
        try:
            storage_path = f"data/users/{user_id}"
            required_dirs = ["csv", "pdf", "rag"]
            
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = os.path.join(storage_path, dir_name)
                if not os.path.exists(dir_path):
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                return {
                    "status": "fail",
                    "message": f"Missing storage directories: {', '.join(missing_dirs)}",
                    "missing_directories": missing_dirs,
                    "recommendations": [f"Create missing directories: {', '.join(missing_dirs)}"]
                }
            
            # Check write permissions
            try:
                test_file = os.path.join(storage_path, "rag", ".test_write")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception:
                return {
                    "status": "fail",
                    "message": "Storage directories not writable",
                    "recommendations": ["Check directory permissions"]
                }
            
            return {
                "status": "pass",
                "message": "All storage directories exist and are accessible"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Storage check failed: {str(e)}",
                "recommendations": ["Check storage directory configuration"]
            }
    
    def _check_database_connectivity(self, user_id: str) -> Dict[str, Any]:
        """Check database connectivity and user data integrity"""
        try:
            # Check main user database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
            user_exists = cursor.fetchone() is not None
            conn.close()
            
            if not user_exists:
                return {
                    "status": "fail",
                    "message": "User not found in database",
                    "recommendations": ["Verify user ID is correct"]
                }
            
            # Check RAG database connectivity
            if os.path.exists(self.rag_db_path):
                conn = sqlite3.connect(self.rag_db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM user_vectors WHERE user_id = ?', (user_id,))
                conn.close()
            
            return {
                "status": "pass",
                "message": "Database connectivity verified"
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Database connectivity failed: {str(e)}",
                "recommendations": ["Check database configuration and permissions"]
            }
    
    def _check_document_processing_status(self, user_id: str) -> Dict[str, Any]:
        """Check document processing status and identify issues"""
        try:
            if not os.path.exists(self.rag_db_path):
                return {
                    "status": "warning",
                    "message": "No documents processed yet",
                    "recommendations": ["Upload documents to build knowledge base"]
                }
            
            conn = sqlite3.connect(self.rag_db_path)
            cursor = conn.cursor()
            
            # Check for failed document processing
            cursor.execute('''
                SELECT COUNT(*) as failed_count
                FROM user_documents
                WHERE user_id = ? AND processing_status = 'failed'
            ''', (user_id,))
            
            failed_count = cursor.fetchone()[0]
            
            # Check total documents
            cursor.execute('''
                SELECT COUNT(*) as total_count
                FROM user_documents
                WHERE user_id = ?
            ''', (user_id,))
            
            total_count = cursor.fetchone()[0]
            conn.close()
            
            if failed_count > 0:
                return {
                    "status": "warning",
                    "message": f"{failed_count} documents failed to process",
                    "failed_documents": failed_count,
                    "total_documents": total_count,
                    "recommendations": ["Review failed documents and retry processing"]
                }
            
            if total_count == 0:
                return {
                    "status": "warning",
                    "message": "No documents processed yet",
                    "recommendations": ["Upload CSV or PDF documents to build knowledge base"]
                }
            
            return {
                "status": "pass",
                "message": f"All {total_count} documents processed successfully",
                "total_documents": total_count
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Document processing check failed: {str(e)}",
                "recommendations": ["Check document processing system"]
            }
    
    def _check_rag_performance(self, user_id: str) -> Dict[str, Any]:
        """Check RAG system performance for user"""
        try:
            if not self.rag_system:
                return {
                    "status": "fail",
                    "message": "RAG system not available"
                }
            
            # Test query performance
            start_time = datetime.now()
            test_results = self.rag_system.query_user_knowledge(user_id, "test query", top_k=1)
            query_time = (datetime.now() - start_time).total_seconds()
            
            if query_time > 5.0:  # Query taking too long
                return {
                    "status": "warning",
                    "message": f"Slow query performance: {query_time:.2f}s",
                    "query_time": query_time,
                    "recommendations": ["Consider optimizing RAG index"]
                }
            
            return {
                "status": "pass",
                "message": f"RAG performance normal: {query_time:.2f}s",
                "query_time": query_time,
                "results_count": len(test_results)
            }
            
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Performance check failed: {str(e)}",
                "recommendations": ["Check RAG system configuration"]
            }
    
    def _get_system_document_stats(self) -> Dict[str, Any]:
        """Get system-wide document statistics"""
        try:
            if not os.path.exists(self.rag_db_path):
                return {
                    "total_documents": 0,
                    "csv_documents": 0,
                    "pdf_documents": 0,
                    "total_vectors": 0
                }
            
            conn = sqlite3.connect(self.rag_db_path)
            cursor = conn.cursor()
            
            # Get document counts by type
            cursor.execute('''
                SELECT document_type, COUNT(*) as count
                FROM user_vectors
                GROUP BY document_type
            ''')
            
            doc_counts = dict(cursor.fetchall())
            
            # Get total vector count
            cursor.execute('SELECT COUNT(*) FROM user_vectors')
            total_vectors = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_documents": sum(doc_counts.values()),
                "csv_documents": doc_counts.get("csv", 0),
                "pdf_documents": doc_counts.get("pdf", 0),
                "total_vectors": total_vectors
            }
            
        except Exception as e:
            logger.error(f"Error getting system document stats: {str(e)}")
            return {
                "total_documents": 0,
                "csv_documents": 0,
                "pdf_documents": 0,
                "total_vectors": 0
            }

# Global instance - DEPRECATED: Use enhanced_rag_manager from enhanced_rag_manager.py instead
# This is kept for backward compatibility but will be removed in future versions
rag_manager = None  # Deprecated - use enhanced_rag_manager instead