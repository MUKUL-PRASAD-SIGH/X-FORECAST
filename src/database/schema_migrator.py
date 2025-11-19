"""
Database Schema Migration System for RAG System Reliability
Handles schema validation, migration, and ensures database consistency across all RAG components.
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MigrationStatus(Enum):
    """Migration status enumeration"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"

@dataclass
class ColumnDefinition:
    """Database column definition"""
    name: str
    data_type: str
    default_value: Optional[str] = None
    nullable: bool = True
    unique: bool = False

@dataclass
class SchemaValidationResult:
    """Result of schema validation"""
    table_name: str
    is_valid: bool
    missing_columns: List[str]
    existing_columns: List[str]
    recommendations: List[str]

@dataclass
class MigrationResult:
    """Result of migration operation"""
    table_name: str
    status: MigrationStatus
    columns_added: List[str]
    errors: List[str]
    execution_time: float
    migration_id: str

class DatabaseSchemaMigrator:
    """
    Comprehensive database schema migration system for RAG reliability
    """
    
    def __init__(self, users_db_path: str = "users.db", rag_db_path: str = "rag_vector_db.db"):
        self.users_db_path = users_db_path
        self.rag_db_path = rag_db_path
        self.migration_version = "2.0.0_rag_reliability"
        
        # Define required schema for RAG system reliability
        self.required_users_columns = [
            ColumnDefinition("rag_initialized", "BOOLEAN", "0"),
            ColumnDefinition("rag_initialization_error", "TEXT", "NULL"),
            ColumnDefinition("rag_last_health_check", "TIMESTAMP", "NULL"),
            ColumnDefinition("rag_error_count", "INTEGER", "0"),
        ]
        
        self.required_business_profiles_columns = [
            ColumnDefinition("rag_status", "TEXT", "'not_initialized'"),
            ColumnDefinition("rag_initialized_at", "TIMESTAMP", "NULL"),
            ColumnDefinition("rag_last_error", "TEXT", "NULL"),
            ColumnDefinition("rag_health_score", "INTEGER", "100"),
        ]
        
        # Define initialization status tracking table
        self.initialization_status_table_schema = """
            CREATE TABLE IF NOT EXISTS rag_initialization_status (
                user_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                message TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    
    def validate_schema(self) -> Dict[str, SchemaValidationResult]:
        """
        Validate current database schema against expected schema
        
        Returns:
            Dictionary with validation results for each table
        """
        try:
            validation_results = {}
            
            # Validate users table
            users_validation = self._validate_table_schema(
                self.users_db_path, 
                "users", 
                self.required_users_columns
            )
            validation_results["users"] = users_validation
            
            # Validate business_profiles table
            bp_validation = self._validate_table_schema(
                self.users_db_path, 
                "business_profiles", 
                self.required_business_profiles_columns
            )
            validation_results["business_profiles"] = bp_validation
            
            logger.info(f"Schema validation completed for {len(validation_results)} tables")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during schema validation: {str(e)}")
            return {
                "error": SchemaValidationResult(
                    table_name="validation_error",
                    is_valid=False,
                    missing_columns=[],
                    existing_columns=[],
                    recommendations=[f"Fix validation error: {str(e)}"]
                )
            }
    
    def migrate_users_table(self) -> MigrationResult:
        """
        Migrate users table to add missing RAG-related columns
        
        Returns:
            MigrationResult with migration details
        """
        start_time = datetime.now()
        migration_id = f"users_migration_{int(start_time.timestamp())}"
        
        try:
            logger.info("Starting users table migration for RAG system reliability")
            
            # Validate current schema
            validation = self._validate_table_schema(
                self.users_db_path, 
                "users", 
                self.required_users_columns
            )
            
            if validation.is_valid:
                execution_time = (datetime.now() - start_time).total_seconds()
                return MigrationResult(
                    table_name="users",
                    status=MigrationStatus.SKIPPED,
                    columns_added=[],
                    errors=[],
                    execution_time=execution_time,
                    migration_id=migration_id
                )
            
            # Add missing columns
            columns_added = []
            errors = []
            
            conn = sqlite3.connect(self.users_db_path)
            cursor = conn.cursor()
            
            for column in self.required_users_columns:
                if column.name in validation.missing_columns:
                    try:
                        success = self._add_column_safely(cursor, "users", column)
                        if success:
                            columns_added.append(column.name)
                            logger.info(f"Added column {column.name} to users table")
                        else:
                            errors.append(f"Failed to add column {column.name}")
                    except Exception as e:
                        error_msg = f"Error adding column {column.name}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            # Log migration
            self._log_migration(cursor, migration_id, "users", columns_added, errors)
            
            conn.commit()
            conn.close()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            status = MigrationStatus.SUCCESS if not errors else (
                MigrationStatus.PARTIAL if columns_added else MigrationStatus.FAILED
            )
            
            logger.info(f"Users table migration completed: {len(columns_added)} columns added, {len(errors)} errors")
            
            return MigrationResult(
                table_name="users",
                status=status,
                columns_added=columns_added,
                errors=errors,
                execution_time=execution_time,
                migration_id=migration_id
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Critical error during users table migration: {str(e)}"
            logger.error(error_msg)
            
            return MigrationResult(
                table_name="users",
                status=MigrationStatus.FAILED,
                columns_added=[],
                errors=[error_msg],
                execution_time=execution_time,
                migration_id=migration_id
            )
    
    def migrate_business_profiles_table(self) -> MigrationResult:
        """
        Migrate business_profiles table to add missing RAG-related columns
        
        Returns:
            MigrationResult with migration details
        """
        start_time = datetime.now()
        migration_id = f"business_profiles_migration_{int(start_time.timestamp())}"
        
        try:
            logger.info("Starting business_profiles table migration for RAG system reliability")
            
            # Validate current schema
            validation = self._validate_table_schema(
                self.users_db_path, 
                "business_profiles", 
                self.required_business_profiles_columns
            )
            
            if validation.is_valid:
                execution_time = (datetime.now() - start_time).total_seconds()
                return MigrationResult(
                    table_name="business_profiles",
                    status=MigrationStatus.SKIPPED,
                    columns_added=[],
                    errors=[],
                    execution_time=execution_time,
                    migration_id=migration_id
                )
            
            # Add missing columns
            columns_added = []
            errors = []
            
            conn = sqlite3.connect(self.users_db_path)
            cursor = conn.cursor()
            
            for column in self.required_business_profiles_columns:
                if column.name in validation.missing_columns:
                    try:
                        success = self._add_column_safely(cursor, "business_profiles", column)
                        if success:
                            columns_added.append(column.name)
                            logger.info(f"Added column {column.name} to business_profiles table")
                        else:
                            errors.append(f"Failed to add column {column.name}")
                    except Exception as e:
                        error_msg = f"Error adding column {column.name}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            # Log migration
            self._log_migration(cursor, migration_id, "business_profiles", columns_added, errors)
            
            conn.commit()
            conn.close()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            status = MigrationStatus.SUCCESS if not errors else (
                MigrationStatus.PARTIAL if columns_added else MigrationStatus.FAILED
            )
            
            logger.info(f"Business_profiles table migration completed: {len(columns_added)} columns added, {len(errors)} errors")
            
            return MigrationResult(
                table_name="business_profiles",
                status=status,
                columns_added=columns_added,
                errors=errors,
                execution_time=execution_time,
                migration_id=migration_id
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Critical error during business_profiles table migration: {str(e)}"
            logger.error(error_msg)
            
            return MigrationResult(
                table_name="business_profiles",
                status=MigrationStatus.FAILED,
                columns_added=[],
                errors=[error_msg],
                execution_time=execution_time,
                migration_id=migration_id
            )
    
    def create_initialization_status_table(self) -> MigrationResult:
        """
        Create the RAG initialization status tracking table
        
        Returns:
            MigrationResult with creation details
        """
        start_time = datetime.now()
        migration_id = f"init_status_table_{int(start_time.timestamp())}"
        
        try:
            logger.info("Creating RAG initialization status table")
            
            conn = sqlite3.connect(self.users_db_path)
            cursor = conn.cursor()
            
            # Create the table
            cursor.execute(self.initialization_status_table_schema)
            
            # Log migration
            self._log_migration(cursor, migration_id, "rag_initialization_status", ["table_created"], [])
            
            conn.commit()
            conn.close()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("RAG initialization status table created successfully")
            
            return MigrationResult(
                table_name="rag_initialization_status",
                status=MigrationStatus.SUCCESS,
                columns_added=["table_created"],
                errors=[],
                execution_time=execution_time,
                migration_id=migration_id
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error creating initialization status table: {str(e)}"
            logger.error(error_msg)
            
            return MigrationResult(
                table_name="rag_initialization_status",
                status=MigrationStatus.FAILED,
                columns_added=[],
                errors=[error_msg],
                execution_time=execution_time,
                migration_id=migration_id
            )
    
    def execute_automatic_migration(self) -> Dict[str, MigrationResult]:
        """
        Execute automatic migration during RAG system startup
        
        Returns:
            Dictionary with migration results for each table
        """
        try:
            logger.info("Starting automatic database schema migration for RAG system startup")
            
            migration_results = {}
            
            # Migrate users table
            users_result = self.migrate_users_table()
            migration_results["users"] = users_result
            
            # Migrate business_profiles table
            bp_result = self.migrate_business_profiles_table()
            migration_results["business_profiles"] = bp_result
            
            # Create initialization status table
            init_status_result = self.create_initialization_status_table()
            migration_results["rag_initialization_status"] = init_status_result
            
            # Summary
            total_columns_added = sum(len(result.columns_added) for result in migration_results.values())
            total_errors = sum(len(result.errors) for result in migration_results.values())
            
            logger.info(f"Automatic migration completed: {total_columns_added} columns added, {total_errors} errors")
            
            return migration_results
            
        except Exception as e:
            logger.error(f"Error during automatic migration: {str(e)}")
            return {
                "error": MigrationResult(
                    table_name="automatic_migration",
                    status=MigrationStatus.FAILED,
                    columns_added=[],
                    errors=[str(e)],
                    execution_time=0.0,
                    migration_id="failed_automatic_migration"
                )
            }
    
    # Private helper methods
    
    def _validate_table_schema(self, db_path: str, table_name: str, 
                              required_columns: List[ColumnDefinition]) -> SchemaValidationResult:
        """Validate schema for a specific table"""
        try:
            if not os.path.exists(db_path):
                return SchemaValidationResult(
                    table_name=table_name,
                    is_valid=False,
                    missing_columns=[col.name for col in required_columns],
                    existing_columns=[],
                    recommendations=[f"Database {db_path} does not exist"]
                )
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get existing columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [column[1] for column in cursor.fetchall()]
            conn.close()
            
            # Find missing columns
            required_column_names = [col.name for col in required_columns]
            missing_columns = [col for col in required_column_names if col not in existing_columns]
            
            is_valid = len(missing_columns) == 0
            
            recommendations = []
            if missing_columns:
                recommendations.append(f"Add missing columns to {table_name}: {', '.join(missing_columns)}")
            
            return SchemaValidationResult(
                table_name=table_name,
                is_valid=is_valid,
                missing_columns=missing_columns,
                existing_columns=existing_columns,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error validating schema for table {table_name}: {str(e)}")
            return SchemaValidationResult(
                table_name=table_name,
                is_valid=False,
                missing_columns=[],
                existing_columns=[],
                recommendations=[f"Schema validation failed: {str(e)}"]
            )
    
    def _add_column_safely(self, cursor: sqlite3.Cursor, table_name: str, 
                          column: ColumnDefinition) -> bool:
        """Safely add a column to a table with proper error handling"""
        try:
            # Build ALTER TABLE statement
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column.data_type}"
            
            if column.default_value is not None:
                alter_sql += f" DEFAULT {column.default_value}"
            
            if not column.nullable:
                alter_sql += " NOT NULL"
            
            if column.unique:
                alter_sql += " UNIQUE"
            
            cursor.execute(alter_sql)
            return True
            
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info(f"Column {column.name} already exists in {table_name}")
                return True
            else:
                logger.error(f"Error adding column {column.name} to {table_name}: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error adding column {column.name} to {table_name}: {str(e)}")
            return False
    
    def _log_migration(self, cursor: sqlite3.Cursor, migration_id: str, 
                      table_name: str, columns_added: List[str], errors: List[str]):
        """Log migration details to migration_log table"""
        try:
            # Create migration_log table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_id TEXT NOT NULL,
                    migration_version TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    migration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    columns_added TEXT,
                    errors TEXT,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            # Log this migration
            success = len(errors) == 0
            cursor.execute('''
                INSERT INTO migration_log 
                (migration_id, migration_version, table_name, columns_added, errors, success)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                migration_id,
                self.migration_version,
                table_name,
                ', '.join(columns_added) if columns_added else None,
                '; '.join(errors) if errors else None,
                success
            ))
            
        except Exception as e:
            logger.error(f"Error logging migration {migration_id}: {str(e)}")
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history from migration_log table"""
        try:
            if not os.path.exists(self.users_db_path):
                return []
            
            conn = sqlite3.connect(self.users_db_path)
            cursor = conn.cursor()
            
            # Check if migration_log table exists first
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='migration_log'
            ''')
            
            if not cursor.fetchone():
                conn.close()
                return []
            
            cursor.execute('''
                SELECT migration_id, migration_version, table_name, migration_date,
                       columns_added, errors, success
                FROM migration_log
                ORDER BY migration_date DESC
            ''')
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "migration_id": row[0],
                    "migration_version": row[1],
                    "table_name": row[2],
                    "migration_date": row[3],
                    "columns_added": row[4].split(', ') if row[4] else [],
                    "errors": row[5].split('; ') if row[5] else [],
                    "success": bool(row[6])
                })
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Error getting migration history: {str(e)}")
            return []

# Global instance
schema_migrator = DatabaseSchemaMigrator()