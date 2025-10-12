"""
Multi-tenant User Authentication and Management System
"""

import hashlib
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import sqlite3
import os

@dataclass
class User:
    user_id: str
    email: str
    company_name: str
    business_type: str
    subscription_tier: str
    created_at: datetime
    is_active: bool = True

@dataclass
class BusinessProfile:
    user_id: str
    company_name: str
    business_type: str
    industry: str
    data_sources: List[str]
    model_config: Dict
    storage_path: str

class UserManager:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.secret_key = "x_forecast_secret_2024"
        self._init_database()
    
    def _init_database(self):
        """Initialize user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                company_name TEXT NOT NULL,
                business_type TEXT NOT NULL,
                subscription_tier TEXT DEFAULT 'basic',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_profiles (
                user_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                business_type TEXT NOT NULL,
                industry TEXT,
                data_sources TEXT,
                model_config TEXT,
                storage_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, email: str, password: str, company_name: str, 
                     business_type: str, industry: str = "retail") -> Dict:
        """Register new business user"""
        try:
            user_id = str(uuid.uuid4())
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            storage_path = f"data/users/{user_id}"
            
            os.makedirs(storage_path, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (user_id, email, password_hash, company_name, business_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, email, password_hash, company_name, business_type))
            
            cursor.execute('''
                INSERT INTO business_profiles 
                (user_id, company_name, business_type, industry, data_sources, model_config, storage_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, company_name, business_type, industry, "[]", "{}", storage_path))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "user_id": user_id, "message": "User registered successfully"}
            
        except sqlite3.IntegrityError:
            return {"success": False, "message": "Email already exists"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user and return JWT token"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, email, company_name, business_type, subscription_tier
            FROM users WHERE email = ? AND password_hash = ? AND is_active = 1
        ''', (email, password_hash))
        
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            user_id, email, company_name, business_type, subscription_tier = user_data
            
            payload = {
                "user_id": user_id,
                "email": email,
                "company_name": company_name,
                "exp": datetime.utcnow() + timedelta(hours=24)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm="HS256")
            
            return {
                "success": True,
                "token": token,
                "user": {
                    "user_id": user_id,
                    "email": email,
                    "company_name": company_name,
                    "business_type": business_type,
                    "subscription_tier": subscription_tier
                }
            }
        
        return None
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except:
            return None
    
    def get_business_profile(self, user_id: str) -> Optional[BusinessProfile]:
        """Get business profile for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, company_name, business_type, industry, 
                   data_sources, model_config, storage_path
            FROM business_profiles WHERE user_id = ?
        ''', (user_id,))
        
        profile_data = cursor.fetchone()
        conn.close()
        
        if profile_data:
            return BusinessProfile(
                user_id=profile_data[0],
                company_name=profile_data[1],
                business_type=profile_data[2],
                industry=profile_data[3],
                data_sources=eval(profile_data[4]) if profile_data[4] else [],
                model_config=eval(profile_data[5]) if profile_data[5] else {},
                storage_path=profile_data[6]
            )
        
        return None

user_manager = UserManager()