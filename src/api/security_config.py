"""
Production Security Configuration Module
Handles JWT configuration, rate limiting, CORS, and security headers
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging for security events
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Create file handler for security logs
security_handler = logging.FileHandler("logs/security.log")
security_handler.setLevel(logging.INFO)

# Create formatter
security_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
security_handler.setFormatter(security_formatter)
security_logger.addHandler(security_handler)

class SecurityConfig:
    """Production security configuration"""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true" if self.environment == "development" else "false").lower() == "true"
        
        # JWT Configuration
        self.jwt_secret_key = self._get_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 24
        self.jwt_refresh_expiration_days = 7
        
        # Rate Limiting Configuration
        self.rate_limit_requests_per_minute = 60
        self.auth_rate_limit_requests_per_minute = 5
        self.upload_rate_limit_requests_per_minute = 10
        
        # File Upload Security
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_file_extensions = {'.csv', '.pdf'}
        self.upload_scan_enabled = True
        
        # CORS Configuration
        self.allowed_origins = self._get_allowed_origins()
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allowed_headers = [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "Cache-Control"
        ]
        
        # Security Headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        # Password Security
        self.min_password_length = 8
        self.require_special_chars = True
        self.require_numbers = True
        self.require_uppercase = True
        
        # Session Security
        self.session_timeout_minutes = 30
        self.max_concurrent_sessions = 3
        
    def _get_jwt_secret(self) -> str:
        """Get JWT secret key with production validation"""
        secret = os.getenv("JWT_SECRET_KEY")
        
        if not secret:
            if self.environment == "production":
                raise ValueError("JWT_SECRET_KEY must be set in production environment")
            # Generate a secure random key for development
            secret = secrets.token_urlsafe(32)
            security_logger.warning("Using generated JWT secret key for development")
        
        # Validate secret strength in production
        if self.environment == "production":
            if len(secret) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")
            
            # Check for common/weak patterns
            weak_patterns = [
                "secret", "password", "key", "jwt_secret", "admin", "test",
                "this_is_a_32_character_secret_but_weak_password",
                "CHANGE_THIS_TO_A_SECURE_32_CHAR_SECRET_KEY_IN_PRODUCTION",
                "CHANGE_THIS_TO_A_SECURE_32_CHAR_SECRET_KEY_IN_PRODUCTION_NOW"
            ]
            
            if secret.lower() in [p.lower() for p in weak_patterns]:
                raise ValueError("JWT_SECRET_KEY cannot be a common/weak value in production")
            
            # Check for simple patterns (repeated characters, sequential, etc.)
            if len(set(secret)) < 8:  # Too few unique characters
                raise ValueError("JWT_SECRET_KEY is too predictable for production use")
            
            if secret.lower() in secret or secret.upper() in secret:  # All same case
                if not any(c.isdigit() for c in secret) or not any(c in "!@#$%^&*()_+-=" for c in secret):
                    raise ValueError("JWT_SECRET_KEY should contain mixed case, numbers, and special characters")
        
        return secret
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed CORS origins based on environment"""
        if self.environment == "production":
            # Production origins from environment variables
            origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
            origins = [origin.strip() for origin in origins if origin.strip()]
            
            if not origins:
                security_logger.warning("No ALLOWED_ORIGINS set for production environment")
                return ["https://your-production-domain.com"]
            
            return origins
        else:
            # Development origins - include all common frontend ports
            return [
                "http://localhost:3000",
                "http://localhost:3001", 
                "http://localhost:3002",
                "http://localhost:3003",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
                "http://127.0.0.1:3002",
                "http://127.0.0.1:3003"
            ]
    
    def validate_file_upload(self, filename: str, file_size: int) -> Dict[str, any]:
        """Validate file upload security"""
        errors = []
        
        # Check file extension
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.allowed_file_extensions:
            errors.append(f"File type {file_ext} not allowed. Allowed types: {', '.join(self.allowed_file_extensions)}")
        
        # Check file size
        if file_size > self.max_file_size:
            errors.append(f"File size {file_size} exceeds maximum allowed size of {self.max_file_size} bytes")
        
        # Check filename for security issues
        if any(char in filename for char in ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']):
            errors.append("Filename contains invalid characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def log_security_event(self, event_type: str, user_id: str = None, ip_address: str = None, details: str = None):
        """Log security events"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details
        }
        
        security_logger.info(f"Security Event: {log_data}")

def validate_password_strength(password: str) -> Dict[str, any]:
    """Validate password strength according to security policy"""
    errors = []
    
    min_length = 8
    require_uppercase = True
    require_numbers = True
    require_special_chars = True
    
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters long")
    
    if require_uppercase and not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if require_numbers and not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one number")
    
    if require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    
    # Check for common weak passwords
    weak_passwords = ["password", "123456", "qwerty", "admin", "user"]
    if password.lower() in weak_passwords:
        errors.append("Password is too common and easily guessable")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "strength": "strong" if len(errors) == 0 else "weak"
    }

class SecurityMonitor:
    """Monitor and track security events"""
    
    def __init__(self):
        self.failed_login_attempts = {}
        self.blocked_ips = set()
        self.max_failed_attempts = 5
        self.block_duration_minutes = 15
    
    def record_failed_login(self, ip_address: str, user_id: str = None):
        """Record failed login attempt"""
        if ip_address not in self.failed_login_attempts:
            self.failed_login_attempts[ip_address] = []
        
        self.failed_login_attempts[ip_address].append({
            "timestamp": datetime.now(),
            "user_id": user_id
        })
        
        # Check if IP should be blocked
        recent_attempts = [
            attempt for attempt in self.failed_login_attempts[ip_address]
            if datetime.now() - attempt["timestamp"] < timedelta(minutes=self.block_duration_minutes)
        ]
        
        if len(recent_attempts) >= self.max_failed_attempts:
            self.blocked_ips.add(ip_address)
            # Log security event
            security_logger.info(f"IP blocked: {ip_address} after {len(recent_attempts)} failed login attempts")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            # Log security event
            security_logger.info(f"IP unblocked: {ip_address}")

# FastAPI middleware and setup functions
def setup_cors_middleware(app):
    """Setup CORS middleware with security configuration"""
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=security_config.allowed_origins,
        allow_credentials=True,
        allow_methods=security_config.allowed_methods,
        allow_headers=security_config.allowed_headers,
        expose_headers=["*"],
        max_age=3600,
    )

def setup_rate_limiting(app):
    """Setup rate limiting with security configuration"""
    if RATE_LIMITING_AVAILABLE and limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Custom rate limit exceeded handler with security logging
        async def custom_rate_limit_handler(request, exc):
            try:
                client_ip = get_remote_address(request)
                security_config.log_security_event(
                    "rate_limit_exceeded",
                    ip_address=client_ip,
                    details=f"Rate limit exceeded for {request.url.path}"
                )
            except:
                pass
            
            return _rate_limit_exceeded_handler(request, exc)
        
        app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
        security_logger.info("Rate limiting enabled with security monitoring")
        return limiter
    else:
        security_logger.warning("Rate limiting not available")
        return None

class SecurityMiddleware:
    """Security middleware for FastAPI"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add security headers to response
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Add security headers
                    for header, value in security_config.security_headers.items():
                        headers[header.encode()] = value.encode()
                    
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# Rate limiting implementation with slowapi
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from fastapi import Request
    
    # Create limiter instance
    limiter = Limiter(key_func=get_remote_address)
    
    def auth_rate_limit():
        """Rate limit for authentication endpoints"""
        def decorator(func):
            return limiter.limit(f"{security_config.auth_rate_limit_requests_per_minute}/minute")(func)
        return decorator

    def upload_rate_limit():
        """Rate limit for file upload endpoints"""
        def decorator(func):
            return limiter.limit(f"{security_config.upload_rate_limit_requests_per_minute}/minute")(func)
        return decorator

    def general_rate_limit():
        """General rate limit for API endpoints"""
        def decorator(func):
            return limiter.limit(f"{security_config.rate_limit_requests_per_minute}/minute")(func)
        return decorator
    
    RATE_LIMITING_AVAILABLE = False  # Temporarily disabled
    
except ImportError:
    # Fallback decorators when slowapi is not available
    def auth_rate_limit():
        """Rate limit for authentication endpoints (fallback)"""
        def decorator(func):
            return func
        return decorator

    def upload_rate_limit():
        """Rate limit for file upload endpoints (fallback)"""
        def decorator(func):
            return func
        return decorator

    def general_rate_limit():
        """General rate limit for API endpoints (fallback)"""
        def decorator(func):
            return func
        return decorator
    
    limiter = None
    RATE_LIMITING_AVAILABLE = False
    security_logger.warning("slowapi not available, rate limiting disabled")

# Global instances
security_config = SecurityConfig()
security_monitor = SecurityMonitor()

# Apply production security settings if in production
if security_config.environment == "production":
    # Update settings for production
    security_config.debug = False
    security_config.jwt_expiration_hours = 8
    security_config.rate_limit_requests_per_minute = 30
    security_config.auth_rate_limit_requests_per_minute = 3
    security_config.max_file_size = 25 * 1024 * 1024
    security_config.session_timeout_minutes = 15
    
    security_logger.info("Production security settings applied")