"""
Database Migration Script for PDF Support
Adds document tracking tables and updates existing schema for multi-format support
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

class DatabaseMigration:
    """Handle database schema migrations for PDF support"""
    
    def __init__(self):
        self.migration_version = "1.1.0_pdf_support"
        self.migration_date = datetime.now()
    
    def migrate_user_management_db(self, db_path: str = "users.db") -> bool:
        """
        Migrate user management database to support document tracking
        
        Args:
            db_path: Path to user management database
            
        Returns:
            Boolean indicating migration success
        """
        try:
            logger.info(f"Starting migration of user management database: {db_path}")
            
            if not os.path.exists(db_path):
                logger.warning(f"Database {db_path} does not exist, skipping migration")
                return True
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if migration already applied
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            
            migrations_applied = []
            
            # Add rag_initialized column to users table if not exists
            if 'rag_initialized' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN rag_initialized BOOLEAN DEFAULT 0')
                migrations_applied.append("Added rag_initialized column to users table")
                logger.info("Added rag_initialized column to users table")
            
            # Add last_login column if not exists
            if 'last_login' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN last_login TIMESTAMP')
                migrations_applied.append("Added last_login column to users table")
                logger.info("Added last_login column to users table")
            
            # Check business_profiles table
            cursor.execute("PRAGMA table_info(business_profiles)")
            bp_columns = [column[1] for column in cursor.fetchall()]
            
            # Add document count columns to business_profiles
            if 'pdf_count' not in bp_columns:
                cursor.execute('ALTER TABLE business_profiles ADD COLUMN pdf_count INTEGER DEFAULT 0')
                migrations_applied.append("Added pdf_count column to business_profiles table")
                logger.info("Added pdf_count column to business_profiles table")
            
            if 'csv_count' not in bp_columns:
                cursor.execute('ALTER TABLE business_profiles ADD COLUMN csv_count INTEGER DEFAULT 0')
                migrations_applied.append("Added csv_count column to business_profiles table")
                logger.info("Added csv_count column to business_profiles table")
            
            if 'total_documents' not in bp_columns:
                cursor.execute('ALTER TABLE business_profiles ADD COLUMN total_documents INTEGER DEFAULT 0')
                migrations_applied.append("Added total_documents column to business_profiles table")
                logger.info("Added total_documents column to business_profiles table")
            
            # Create migration log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_version TEXT NOT NULL,
                    migration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            # Log this migration
            cursor.execute('''
                INSERT INTO migration_log (migration_version, description)
                VALUES (?, ?)
            ''', (self.migration_version, f"PDF support migration: {', '.join(migrations_applied)}"))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully migrated user management database. Applied: {migrations_applied}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating user management database: {str(e)}")
            return False
    
    def migrate_rag_vector_db(self, db_path: str = "rag_vector_db.db") -> bool:
        """
        Migrate RAG vector database to support PDF documents
        
        Args:
            db_path: Path to RAG vector database
            
        Returns:
            Boolean indicating migration success
        """
        try:
            logger.info(f"Starting migration of RAG vector database: {db_path}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check existing schema
            cursor.execute("PRAGMA table_info(user_vectors)")
            columns = [column[1] for column in cursor.fetchall()]
            
            migrations_applied = []
            
            # Add new columns to user_vectors table if not exists
            if 'document_type' not in columns:
                cursor.execute('ALTER TABLE user_vectors ADD COLUMN document_type TEXT DEFAULT "csv"')
                migrations_applied.append("Added document_type column to user_vectors table")
                logger.info("Added document_type column to user_vectors table")
            
            if 'source_file' not in columns:
                cursor.execute('ALTER TABLE user_vectors ADD COLUMN source_file TEXT')
                migrations_applied.append("Added source_file column to user_vectors table")
                logger.info("Added source_file column to user_vectors table")
            
            if 'page_number' not in columns:
                cursor.execute('ALTER TABLE user_vectors ADD COLUMN page_number INTEGER')
                migrations_applied.append("Added page_number column to user_vectors table")
                logger.info("Added page_number column to user_vectors table")
            
            # Create user_documents table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_documents (
                    document_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    file_size INTEGER,
                    page_count INTEGER DEFAULT 0,
                    file_hash TEXT,
                    metadata TEXT,
                    error_message TEXT
                )
            ''')
            migrations_applied.append("Created user_documents table")
            logger.info("Created user_documents table")
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_documents_user_id ON user_documents(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_documents_file_type ON user_documents(file_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_vectors_document_type ON user_vectors(document_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_vectors_source_file ON user_vectors(source_file)')
            migrations_applied.append("Created performance indexes")
            logger.info("Created performance indexes")
            
            # Create migration log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_version TEXT NOT NULL,
                    migration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            # Log this migration
            cursor.execute('''
                INSERT INTO migration_log (migration_version, description)
                VALUES (?, ?)
            ''', (self.migration_version, f"RAG PDF support migration: {', '.join(migrations_applied)}"))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully migrated RAG vector database. Applied: {migrations_applied}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating RAG vector database: {str(e)}")
            return False
    
    def migrate_multi_tenant_db(self, db_path: str = "data/master.db") -> bool:
        """
        Migrate multi-tenant master database
        
        Args:
            db_path: Path to master database
            
        Returns:
            Boolean indicating migration success
        """
        try:
            logger.info(f"Starting migration of multi-tenant database: {db_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if tables exist and get their schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            migrations_applied = []
            
            # Create companies table if not exists
            if 'companies' not in existing_tables:
                cursor.execute('''
                    CREATE TABLE companies (
                        company_id TEXT PRIMARY KEY,
                        company_name TEXT UNIQUE NOT NULL,
                        industry TEXT NOT NULL,
                        business_type TEXT NOT NULL,
                        subscription_tier TEXT DEFAULT 'basic',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        database_path TEXT NOT NULL,
                        storage_path TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                migrations_applied.append("Created companies table")
                logger.info("Created companies table")
            
            # Create users table if not exists
            if 'users' not in existing_tables:
                cursor.execute('''
                    CREATE TABLE users (
                        user_id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        company_id TEXT NOT NULL,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        rag_initialized BOOLEAN DEFAULT 0,
                        FOREIGN KEY (company_id) REFERENCES companies (company_id)
                    )
                ''')
                migrations_applied.append("Created users table")
                logger.info("Created users table")
            else:
                # Check if rag_initialized column exists
                cursor.execute("PRAGMA table_info(users)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'rag_initialized' not in columns:
                    cursor.execute('ALTER TABLE users ADD COLUMN rag_initialized BOOLEAN DEFAULT 0')
                    migrations_applied.append("Added rag_initialized column to users table")
                    logger.info("Added rag_initialized column to users table")
            
            # Create migration log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_version TEXT NOT NULL,
                    migration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            # Log this migration
            if migrations_applied:
                cursor.execute('''
                    INSERT INTO migration_log (migration_version, description)
                    VALUES (?, ?)
                ''', (self.migration_version, f"Multi-tenant migration: {', '.join(migrations_applied)}"))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully migrated multi-tenant database. Applied: {migrations_applied}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating multi-tenant database: {str(e)}")
            return False
    
    def run_all_migrations(self) -> Dict[str, bool]:
        """
        Run all database migrations
        
        Returns:
            Dictionary with migration results
        """
        results = {}
        
        logger.info("Starting comprehensive database migration for PDF support")
        
        # Migrate user management database
        results['user_management'] = self.migrate_user_management_db()
        
        # Migrate RAG vector database
        results['rag_vector'] = self.migrate_rag_vector_db()
        
        # Migrate multi-tenant database
        results['multi_tenant'] = self.migrate_multi_tenant_db()
        
        # Summary
        successful_migrations = sum(results.values())
        total_migrations = len(results)
        
        logger.info(f"Migration completed: {successful_migrations}/{total_migrations} successful")
        
        if successful_migrations == total_migrations:
            logger.info("All database migrations completed successfully!")
        else:
            logger.warning(f"Some migrations failed: {results}")
        
        return results
    
    def verify_migrations(self) -> Dict[str, bool]:
        """
        Verify that all migrations were applied correctly
        
        Returns:
            Dictionary with verification results
        """
        verification_results = {}
        
        try:
            # Verify user management database
            if os.path.exists("users.db"):
                conn = sqlite3.connect("users.db")
                cursor = conn.cursor()
                
                cursor.execute("PRAGMA table_info(users)")
                columns = [column[1] for column in cursor.fetchall()]
                
                verification_results['user_rag_initialized'] = 'rag_initialized' in columns
                verification_results['user_last_login'] = 'last_login' in columns
                
                cursor.execute("PRAGMA table_info(business_profiles)")
                bp_columns = [column[1] for column in cursor.fetchall()]
                
                verification_results['bp_pdf_count'] = 'pdf_count' in bp_columns
                verification_results['bp_csv_count'] = 'csv_count' in bp_columns
                verification_results['bp_total_documents'] = 'total_documents' in bp_columns
                
                conn.close()
            
            # Verify RAG vector database
            if os.path.exists("rag_vector_db.db"):
                conn = sqlite3.connect("rag_vector_db.db")
                cursor = conn.cursor()
                
                cursor.execute("PRAGMA table_info(user_vectors)")
                columns = [column[1] for column in cursor.fetchall()]
                
                verification_results['rag_document_type'] = 'document_type' in columns
                verification_results['rag_source_file'] = 'source_file' in columns
                verification_results['rag_page_number'] = 'page_number' in columns
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_documents'")
                verification_results['user_documents_table'] = bool(cursor.fetchone())
                
                conn.close()
            
            # Verify multi-tenant database
            if os.path.exists("data/master.db"):
                conn = sqlite3.connect("data/master.db")
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                verification_results['mt_companies_table'] = 'companies' in tables
                verification_results['mt_users_table'] = 'users' in tables
                
                if 'users' in tables:
                    cursor.execute("PRAGMA table_info(users)")
                    columns = [column[1] for column in cursor.fetchall()]
                    verification_results['mt_rag_initialized'] = 'rag_initialized' in columns
                
                conn.close()
            
            logger.info(f"Migration verification completed: {verification_results}")
            return verification_results
            
        except Exception as e:
            logger.error(f"Error verifying migrations: {str(e)}")
            return {"error": str(e)}

def run_migration():
    """Convenience function to run all migrations"""
    migration = DatabaseMigration()
    results = migration.run_all_migrations()
    
    print("\n" + "="*50)
    print("DATABASE MIGRATION RESULTS")
    print("="*50)
    
    for db_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{db_name.upper()}: {status}")
    
    print("\nVerifying migrations...")
    verification = migration.verify_migrations()
    
    print("\nVERIFICATION RESULTS:")
    print("-"*30)
    for check, passed in verification.items():
        status = "✅" if passed else "❌"
        print(f"{check}: {status}")
    
    return results

if __name__ == "__main__":
    run_migration()