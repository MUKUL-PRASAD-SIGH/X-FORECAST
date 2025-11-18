"""
API Security and Validation for Ensemble Integration
Provides comprehensive input validation, rate limiting, authentication, and error handling
"""

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import time
import hashlib
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import jwt
from functools import wraps
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Security configuration
SECURITY_CONFIG = {
    "jwt_secret": "ensemble_forecasting_secret_key_2024",  # In production, use environment variable
    "jwt_algorithm": "HS256",
    "token_expiry_hours": 24,
    "rate_limit_requests_per_minute": 60,
    "rate_limit_burst_size": 10,
    "max_file_size_mb": 100,
    "allowed_file_types": ["csv", "xlsx", "json"],
    "max_forecast_horizon": 24,
    "min_forecast_horizon": 1
}

# Rate limiting storage
rate_limit_storage = defaultdict(lambda: deque())
rate_limit_locks = defaultdict(asyncio.Lock)

# Security models
class TokenData(BaseModel):
    """Token data structure"""
    user_id: str
    permissions: List[str]
    company_id: Optional[str] = None
    expires_at: datetime

class RateLimitInfo(BaseModel):
    """Rate limit information"""
    requests_remaining: int
    reset_time: datetime
    burst_remaining: int

class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    value: Any

class SecurityError(BaseModel):
    """Security error response"""
    error_type: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None

# Security utilities
security = HTTPBearer()

def generate_token(user_id: str, permissions: List[str], company_id: Optional[str] = None) -> str:
    """Generate JWT token for user"""
    try:
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "company_id": company_id,
            "exp": datetime.utcnow() + timedelta(hours=SECURITY_CONFIG["token_expiry_hours"]),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, SECURITY_CONFIG["jwt_secret"], algorithm=SECURITY_CONFIG["jwt_algorithm"])
        return token
    
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise HTTPException(status_code=500, detail="Token generation failed")

def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECURITY_CONFIG["jwt_secret"], algorithms=[SECURITY_CONFIG["jwt_algorithm"]])
        
        return TokenData(
            user_id=payload["user_id"],
            permissions=payload.get("permissions", []),
            company_id=payload.get("company_id"),
            expires_at=datetime.fromtimestamp(payload["exp"])
        )
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current authenticated user from token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return verify_token(credentials.credentials)

def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (injected by FastAPI dependency)
            user = None
            for key, value in kwargs.items():
                if isinstance(value, TokenData):
                    user = value
                    break
            
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check permissions
            if not any(perm in user.permissions for perm in required_permissions):
                raise HTTPException(
                    status_code=403, 
                    detail=f"Insufficient permissions. Required: {required_permissions}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting
async def check_rate_limit(request: Request, user_id: str) -> RateLimitInfo:
    """Check and enforce rate limiting"""
    client_ip = request.client.host
    rate_key = f"{user_id}:{client_ip}"
    
    async with rate_limit_locks[rate_key]:
        now = time.time()
        minute_ago = now - 60
        
        # Get request history for this user/IP
        request_times = rate_limit_storage[rate_key]
        
        # Remove old requests (older than 1 minute)
        while request_times and request_times[0] < minute_ago:
            request_times.popleft()
        
        # Check rate limit
        requests_in_minute = len(request_times)
        max_requests = SECURITY_CONFIG["rate_limit_requests_per_minute"]
        burst_size = SECURITY_CONFIG["rate_limit_burst_size"]
        
        # Check burst limit (last 10 seconds)
        ten_seconds_ago = now - 10
        recent_requests = sum(1 for req_time in request_times if req_time > ten_seconds_ago)
        
        if requests_in_minute >= max_requests:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Too many requests per minute.",
                headers={"Retry-After": "60"}
            )
        
        if recent_requests >= burst_size:
            raise HTTPException(
                status_code=429,
                detail="Burst rate limit exceeded. Too many requests in short time.",
                headers={"Retry-After": "10"}
            )
        
        # Add current request
        request_times.append(now)
        
        return RateLimitInfo(
            requests_remaining=max_requests - requests_in_minute - 1,
            reset_time=datetime.fromtimestamp(now + 60),
            burst_remaining=burst_size - recent_requests - 1
        )

def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate limiting"""
    async def _rate_limit(user: TokenData = Depends(get_current_user)):
        return await check_rate_limit(request, user.user_id)
    return _rate_limit

# Input validation
class EnsembleUploadValidator(BaseModel):
    """Validator for ensemble upload requests"""
    file_size_bytes: int = Field(..., description="File size in bytes")
    file_name: str = Field(..., description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME content type")
    
    @validator('file_size_bytes')
    def validate_file_size(cls, v):
        max_size = SECURITY_CONFIG["max_file_size_mb"] * 1024 * 1024
        if v > max_size:
            raise ValueError(f"File size exceeds maximum allowed size of {SECURITY_CONFIG['max_file_size_mb']}MB")
        return v
    
    @validator('file_name')
    def validate_file_extension(cls, v):
        if not v:
            raise ValueError("Filename is required")
        
        extension = v.split('.')[-1].lower() if '.' in v else ''
        if extension not in SECURITY_CONFIG["allowed_file_types"]:
            raise ValueError(f"File type not allowed. Supported types: {SECURITY_CONFIG['allowed_file_types']}")
        
        # Check for potentially dangerous filenames
        dangerous_patterns = ['..', '/', '\\', '<', '>', '|', ':', '*', '?', '"']
        if any(pattern in v for pattern in dangerous_patterns):
            raise ValueError("Filename contains invalid characters")
        
        return v

class ForecastRequestValidator(BaseModel):
    """Validator for forecast requests"""
    horizon_months: int = Field(..., description="Forecast horizon in months")
    confidence_levels: List[float] = Field(default=[0.1, 0.5, 0.9], description="Confidence levels")
    
    @validator('horizon_months')
    def validate_horizon(cls, v):
        min_horizon = SECURITY_CONFIG["min_forecast_horizon"]
        max_horizon = SECURITY_CONFIG["max_forecast_horizon"]
        
        if v < min_horizon or v > max_horizon:
            raise ValueError(f"Forecast horizon must be between {min_horizon} and {max_horizon} months")
        return v
    
    @validator('confidence_levels')
    def validate_confidence_levels(cls, v):
        if not v:
            raise ValueError("At least one confidence level is required")
        
        if len(v) > 10:
            raise ValueError("Maximum 10 confidence levels allowed")
        
        for level in v:
            if not 0 < level < 1:
                raise ValueError("Confidence levels must be between 0 and 1")
        
        return sorted(set(v))  # Remove duplicates and sort

class DataFrameValidator:
    """Validator for pandas DataFrame content"""
    
    @staticmethod
    def validate_ensemble_data(df, required_columns: List[str]) -> List[ValidationError]:
        """Validate DataFrame for ensemble processing"""
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append(ValidationError(
                field="dataframe",
                message="DataFrame is empty",
                value=len(df)
            ))
            return errors
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(ValidationError(
                field="columns",
                message=f"Missing required columns: {missing_columns}",
                value=list(df.columns)
            ))
        
        # Check data types and content
        for col in df.columns:
            if col in required_columns:
                # Check for excessive missing values
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 50:
                    errors.append(ValidationError(
                        field=col,
                        message=f"Column has {missing_pct:.1f}% missing values (>50% threshold)",
                        value=missing_pct
                    ))
                
                # Validate specific column types
                if 'date' in col.lower():
                    try:
                        pd.to_datetime(df[col].dropna().head(10))
                    except:
                        errors.append(ValidationError(
                            field=col,
                            message="Date column contains invalid date values",
                            value=df[col].head(3).tolist()
                        ))
                
                elif 'amount' in col.lower() or 'sales' in col.lower():
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(ValidationError(
                            field=col,
                            message="Sales/amount column must be numeric",
                            value=str(df[col].dtype)
                        ))
                    else:
                        # Check for negative values
                        negative_count = (df[col] < 0).sum()
                        if negative_count > 0:
                            errors.append(ValidationError(
                                field=col,
                                message=f"Column contains {negative_count} negative values",
                                value=negative_count
                            ))
        
        # Check minimum data requirements
        if len(df) < 12:
            errors.append(ValidationError(
                field="dataframe",
                message="Minimum 12 data points required for ensemble forecasting",
                value=len(df)
            ))
        
        # Check for data quality issues
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        if missing_pct > 30:
            errors.append(ValidationError(
                field="dataframe",
                message=f"High missing data percentage: {missing_pct:.1f}%",
                value=missing_pct
            ))
        
        return errors

# Error handling and graceful degradation
class EnsembleErrorHandler:
    """Handle ensemble-specific errors with graceful degradation"""
    
    @staticmethod
    def handle_model_failure(model_name: str, error: Exception) -> Dict[str, Any]:
        """Handle individual model failures"""
        logger.warning(f"Model {model_name} failed: {error}")
        
        return {
            "model_name": model_name,
            "status": "failed",
            "error": str(error),
            "fallback_applied": True,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_ensemble_degradation(available_models: List[str], total_models: int) -> Dict[str, Any]:
        """Handle ensemble degradation when some models fail"""
        degradation_level = 1 - (len(available_models) / total_models)
        
        if degradation_level > 0.5:
            logger.error(f"Severe ensemble degradation: {degradation_level:.1%}")
            return {
                "status": "severely_degraded",
                "available_models": available_models,
                "degradation_level": degradation_level,
                "recommendation": "Consider retraining failed models"
            }
        elif degradation_level > 0.2:
            logger.warning(f"Moderate ensemble degradation: {degradation_level:.1%}")
            return {
                "status": "moderately_degraded",
                "available_models": available_models,
                "degradation_level": degradation_level,
                "recommendation": "Monitor model performance"
            }
        else:
            return {
                "status": "healthy",
                "available_models": available_models,
                "degradation_level": degradation_level
            }
    
    @staticmethod
    def create_fallback_forecast(historical_data, horizon: int) -> Dict[str, Any]:
        """Create fallback forecast when ensemble fails"""
        try:
            # Simple moving average fallback
            if len(historical_data) >= 3:
                recent_avg = historical_data.tail(3).mean()
                fallback_values = [recent_avg] * horizon
            else:
                fallback_values = [historical_data.mean()] * horizon
            
            return {
                "forecast_type": "fallback_moving_average",
                "values": fallback_values,
                "confidence": 0.3,  # Low confidence for fallback
                "warning": "Fallback forecast used due to ensemble failure"
            }
        
        except Exception as e:
            logger.error(f"Fallback forecast creation failed: {e}")
            return {
                "forecast_type": "constant_fallback",
                "values": [0.0] * horizon,
                "confidence": 0.1,
                "error": "All forecasting methods failed"
            }

# Security middleware
async def security_middleware(request: Request, call_next):
    """Security middleware for ensemble API"""
    start_time = time.time()
    
    try:
        # Add security headers
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    except Exception as e:
        logger.error(f"Security middleware error: {e}")
        raise HTTPException(status_code=500, detail="Security processing failed")

# Utility functions for validation
def sanitize_input(input_str: str) -> str:
    """Sanitize string input to prevent injection attacks"""
    if not isinstance(input_str, str):
        return str(input_str)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
    sanitized = input_str
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Limit length
    return sanitized[:1000]

def validate_json_payload(payload: Dict[str, Any], max_depth: int = 10) -> bool:
    """Validate JSON payload structure and depth"""
    def check_depth(obj, current_depth=0):
        if current_depth > max_depth:
            return False
        
        if isinstance(obj, dict):
            return all(check_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            return all(check_depth(item, current_depth + 1) for item in obj)
        else:
            return True
    
    return check_depth(payload)

def create_request_hash(request_data: Dict[str, Any]) -> str:
    """Create hash for request deduplication"""
    # Sort keys for consistent hashing
    sorted_data = json.dumps(request_data, sort_keys=True)
    return hashlib.sha256(sorted_data.encode()).hexdigest()

# Export security dependencies for use in API endpoints
__all__ = [
    'get_current_user',
    'require_permissions',
    'rate_limit_dependency',
    'EnsembleUploadValidator',
    'ForecastRequestValidator',
    'DataFrameValidator',
    'EnsembleErrorHandler',
    'security_middleware',
    'sanitize_input',
    'validate_json_payload',
    'create_request_hash',
    'SECURITY_CONFIG'
]