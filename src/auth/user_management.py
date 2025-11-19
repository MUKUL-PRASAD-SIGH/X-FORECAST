"""
Multi-tenant User Authentication and Management System
"""

import hashlib
import jwt
import uuid
import secrets
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

# Import RAG system for initialization
try:
    from src.rag.real_vector_rag import real_vector_rag
except ImportError:
    logger.warning("RAG system not available - RAG initialization will be skipped")
    real_vector_rag = None

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
                is_active BOOLEAN DEFAULT 1,
                email_verified BOOLEAN DEFAULT 0,
                verification_code TEXT,
                verification_expires TIMESTAMP,
                reset_code TEXT,
                reset_expires TIMESTAMP,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                rag_initialized BOOLEAN DEFAULT 0,
                rag_initialization_error TEXT
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
                rag_status TEXT DEFAULT 'not_initialized',
                rag_initialized_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, email: str, password: str, company_name: str, 
                     business_type: str, industry: str = "retail") -> Dict:
        """Register new business user with email verification and automatic RAG initialization"""
        try:
            user_id = str(uuid.uuid4())
            # Use bcrypt for better password hashing
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            storage_path = f"data/users/{user_id}"
            
            # Generate verification code
            verification_code = secrets.token_hex(16)
            verification_expires = datetime.now() + timedelta(hours=24)
            
            # Create user storage directories
            os.makedirs(storage_path, exist_ok=True)
            os.makedirs(f"{storage_path}/csv", exist_ok=True)
            os.makedirs(f"{storage_path}/pdf", exist_ok=True)
            os.makedirs(f"{storage_path}/rag", exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users 
                (user_id, email, password_hash, company_name, business_type, 
                 verification_code, verification_expires, email_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, email, password_hash, company_name, business_type, 
                  verification_code, verification_expires, False))
            
            cursor.execute('''
                INSERT INTO business_profiles 
                (user_id, company_name, business_type, industry, data_sources, model_config, storage_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, company_name, business_type, industry, "[]", "{}", storage_path))
            
            conn.commit()
            conn.close()
            
            # Initialize RAG system for the new user
            rag_success = self._initialize_user_rag(user_id, company_name)
            
            return {
                "success": True, 
                "user_id": user_id, 
                "message": "User registered successfully",
                "verification_code": verification_code,
                "requires_verification": True,
                "rag_initialized": rag_success
            }
            
        except sqlite3.IntegrityError:
            return {"success": False, "message": "Email already exists"}
        except Exception as e:
            logger.error(f"Error registering user {email}: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user and return JWT token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if account is locked
        cursor.execute('''
            SELECT user_id, email, company_name, business_type, subscription_tier, 
                   password_hash, email_verified, failed_login_attempts, locked_until
            FROM users WHERE email = ? AND is_active = 1
        ''', (email,))
        
        user_data = cursor.fetchone()
        
        if not user_data:
            return None
        
        user_id, email, company_name, business_type, subscription_tier, stored_hash, email_verified, failed_attempts, locked_until = user_data
        
        # Check if account is locked
        if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
            return {"error": "Account temporarily locked due to too many failed login attempts"}
        
        # Verify password using bcrypt
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            # Increment failed login attempts
            failed_attempts += 1
            lock_time = None
            
            if failed_attempts >= 5:
                lock_time = datetime.now() + timedelta(minutes=30)
            
            cursor.execute('''
                UPDATE users SET failed_login_attempts = ?, locked_until = ?
                WHERE email = ?
            ''', (failed_attempts, lock_time, email))
            conn.commit()
            conn.close()
            
            return None
        
        # Check email verification
        if not email_verified:
            conn.close()
            return {"error": "Email not verified. Please check your email for verification link."}
        
        # Reset failed login attempts on successful login
        cursor.execute('''
            UPDATE users SET failed_login_attempts = 0, locked_until = NULL
            WHERE email = ?
        ''', (email,))
        conn.commit()
        conn.close()
        
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
    
    def verify_email(self, email: str, verification_code: str) -> Dict:
        """Verify user email with verification code and ensure RAG is initialized"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, verification_expires, company_name, rag_initialized FROM users 
                WHERE email = ? AND verification_code = ?
            ''', (email, verification_code))
            
            result = cursor.fetchone()
            
            if not result:
                return {"success": False, "message": "Invalid verification code"}
            
            user_id, expires_str, company_name, rag_initialized = result
            expires = datetime.fromisoformat(expires_str)
            
            if expires < datetime.now():
                return {"success": False, "message": "Verification code has expired"}
            
            # Mark email as verified
            cursor.execute('''
                UPDATE users SET email_verified = 1, verification_code = NULL, verification_expires = NULL
                WHERE email = ?
            ''', (email,))
            
            conn.commit()
            conn.close()
            
            # Initialize RAG if not already done
            rag_success = True
            if not rag_initialized:
                rag_success = self._initialize_user_rag(user_id, company_name)
            
            return {
                "success": True, 
                "message": "Email verified successfully",
                "rag_initialized": rag_success
            }
            
        except Exception as e:
            logger.error(f"Error verifying email {email}: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def request_password_reset(self, email: str) -> Dict:
        """Generate password reset code"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id FROM users WHERE email = ? AND is_active = 1', (email,))
            
            if not cursor.fetchone():
                return {"success": False, "message": "Email not found"}
            
            reset_code = secrets.token_hex(16)
            reset_expires = datetime.now() + timedelta(hours=1)
            
            cursor.execute('''
                UPDATE users SET reset_code = ?, reset_expires = ?
                WHERE email = ?
            ''', (reset_code, reset_expires, email))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True, 
                "message": "Password reset code generated",
                "reset_code": reset_code
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def reset_password(self, email: str, reset_code: str, new_password: str) -> Dict:
        """Reset password with reset code"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, reset_expires FROM users 
                WHERE email = ? AND reset_code = ?
            ''', (email, reset_code))
            
            result = cursor.fetchone()
            
            if not result:
                return {"success": False, "message": "Invalid reset code"}
            
            user_id, expires_str = result
            expires = datetime.fromisoformat(expires_str)
            
            if expires < datetime.now():
                return {"success": False, "message": "Reset code has expired"}
            
            # Hash new password
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Update password and clear reset code
            cursor.execute('''
                UPDATE users SET password_hash = ?, reset_code = NULL, reset_expires = NULL,
                                failed_login_attempts = 0, locked_until = NULL
                WHERE email = ?
            ''', (new_password_hash, email))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": "Password reset successfully"}
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _initialize_user_rag(self, user_id: str, company_name: str) -> bool:
        """Initialize RAG system for a new user"""
        try:
            if real_vector_rag is None:
                logger.warning(f"RAG system not available for user {user_id}")
                self._update_rag_status(user_id, "failed", "RAG system not available")
                return False
            
            # Initialize company-specific RAG
            success = real_vector_rag.initialize_company_rag(user_id, company_name)
            
            if success:
                self._update_rag_status(user_id, "initialized")
                logger.info(f"RAG system initialized for user {user_id} ({company_name})")
                return True
            else:
                self._update_rag_status(user_id, "failed", "RAG initialization failed")
                logger.error(f"Failed to initialize RAG for user {user_id}")
                return False
                
        except Exception as e:
            error_msg = f"RAG initialization error: {str(e)}"
            logger.error(f"Error initializing RAG for user {user_id}: {error_msg}")
            self._update_rag_status(user_id, "failed", error_msg)
            return False
    
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
    
    def get_rag_status(self, user_id: str) -> Dict:
        """Get RAG initialization status for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.rag_initialized, u.rag_initialization_error, 
                       bp.rag_status, bp.rag_initialized_at
                FROM users u
                JOIN business_profiles bp ON u.user_id = bp.user_id
                WHERE u.user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                rag_initialized, error_message, rag_status, initialized_at = result
                return {
                    "rag_initialized": bool(rag_initialized),
                    "rag_status": rag_status,
                    "initialized_at": initialized_at,
                    "error_message": error_message
                }
            else:
                return {"rag_initialized": False, "rag_status": "not_found"}
                
        except Exception as e:
            logger.error(f"Error getting RAG status for user {user_id}: {str(e)}")
            return {"rag_initialized": False, "rag_status": "error", "error_message": str(e)}
    
    def reinitialize_user_rag(self, user_id: str) -> Dict:
        """Reinitialize RAG system for a user (fallback handling)"""
        try:
            # Get user info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT company_name FROM users WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {"success": False, "message": "User not found"}
            
            company_name = result[0]
            
            # Attempt RAG initialization
            success = self._initialize_user_rag(user_id, company_name)
            
            return {
                "success": success,
                "message": "RAG reinitialized successfully" if success else "RAG reinitialization failed"
            }
            
        except Exception as e:
            logger.error(f"Error reinitializing RAG for user {user_id}: {str(e)}")
            return {"success": False, "message": str(e)}

user_manager = UserManager()