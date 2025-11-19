"""
Multi-Tenant Authentication System with Company-Specific Data Isolation
Each company gets its own database and personalized chatbot experience
"""

import hashlib
import jwt
import uuid
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Company:
    """Company information"""
    company_id: str
    company_name: str
    industry: str
    business_type: str
    subscription_tier: str
    created_at: datetime
    database_path: str
    storage_path: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert company to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """Create company from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class User:
    """User information"""
    user_id: str
    email: str
    password_hash: str
    company_id: str
    first_name: str
    last_name: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    rag_initialized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_login'] = self.last_login.isoformat() if self.last_login else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['last_login']:
            data['last_login'] = datetime.fromisoformat(data['last_login'])
        return cls(**data)

print("Defining MultiTenantAuthManager class...")

class MultiTenantAuthManager:
    """
    Multi-tenant authentication manager that handles company-specific databases
    and user management with complete data isolation
    """
    
    def __init__(self, base_data_path: str = "data", secret_key: str = "x_forecast_secret_2024"):
        self.base_data_path = Path(base_data_path)
        self.secret_key = secret_key
        self.master_db_path = self.base_data_path / "master.db"
        
        # Ensure base directories exist
        self.base_data_path.mkdir(exist_ok=True)
        (self.base_data_path / "companies").mkdir(exist_ok=True)
        (self.base_data_path / "users").mkdir(exist_ok=True)
        
        self._init_master_database()
    
    def _init_master_database(self):
        """Initialize master database for company and user tracking"""
        conn = sqlite3.connect(self.master_db_path)
        cursor = conn.cursor()
        
        # Companies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
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
        
        # Users table (master record)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
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
        
        conn.commit()
        conn.close()
        logger.info("Master database initialized")

print("Creating global instance...")

# Global instance
multi_tenant_auth = MultiTenantAuthManager()

print("Module loaded successfully")