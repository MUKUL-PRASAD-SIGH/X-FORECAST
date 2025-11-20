"""
Security Configuration Validator
Validates production security settings and configurations
"""

import os
import re
import secrets
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Validates security configuration for production deployment"""
    
    def __init__(self):
        self.validation_results = []
        self.critical_issues = []
        self.warnings = []
        
    def validate_jwt_configuration(self) -> Dict[str, any]:
        """Validate JWT security configuration"""
        issues = []
        
        # Check JWT secret key
        jwt_secret = os.getenv("JWT_SECRET_KEY", "")
        
        if not jwt_secret:
            issues.append("JWT_SECRET_KEY environment variable not set")
        elif len(jwt_secret) < 32:
            issues.append("JWT_SECRET_KEY must be at least 32 characters long")
        elif jwt_secret in ["secret", "password", "key", "jwt_secret", "CHANGE_THIS_TO_A_SECURE_32_CHAR_SECRET_KEY_IN_PRODUCTION", "CHANGE_THIS_TO_A_SECURE_32_CHAR_SECRET_KEY_IN_PRODUCTION_NOW"]:
            issues.append("JWT_SECRET_KEY is using a default/weak value")
        
        # Check JWT algorithm
        jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        if jwt_algorithm not in ["HS256", "HS384", "HS512"]:
            issues.append(f"JWT_ALGORITHM '{jwt_algorithm}' is not recommended for production")
        
        # Check JWT expiration
        jwt_expiration = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        if jwt_expiration > 24:
            self.warnings.append(f"JWT_EXPIRATION_HOURS is {jwt_expiration}, consider shorter expiration for production")
        
        return {
            "component": "JWT Configuration",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_cors_configuration(self) -> Dict[str, any]:
        """Validate CORS security configuration"""
        issues = []
        
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
        
        if not allowed_origins:
            issues.append("ALLOWED_ORIGINS not configured for production")
        elif "localhost" in allowed_origins or "127.0.0.1" in allowed_origins:
            issues.append("ALLOWED_ORIGINS contains development URLs (localhost/127.0.0.1)")
        elif "*" in allowed_origins:
            issues.append("ALLOWED_ORIGINS contains wildcard (*) which is insecure for production")
        
        # Check for HTTPS
        if allowed_origins and not all(origin.startswith("https://") for origin in allowed_origins.split(",")):
            issues.append("ALLOWED_ORIGINS should use HTTPS in production")
        
        return {
            "component": "CORS Configuration",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_rate_limiting(self) -> Dict[str, any]:
        """Validate rate limiting configuration"""
        issues = []
        
        # Check rate limiting settings
        general_rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        auth_rate_limit = int(os.getenv("AUTH_RATE_LIMIT_PER_MINUTE", "5"))
        upload_rate_limit = int(os.getenv("UPLOAD_RATE_LIMIT_PER_MINUTE", "10"))
        
        if general_rate_limit > 100:
            self.warnings.append(f"General rate limit ({general_rate_limit}/min) is quite high for production")
        
        if auth_rate_limit > 10:
            issues.append(f"Authentication rate limit ({auth_rate_limit}/min) is too high for production")
        
        if upload_rate_limit > 20:
            self.warnings.append(f"Upload rate limit ({upload_rate_limit}/min) is quite high for production")
        
        # Check if slowapi is available
        try:
            import slowapi
            from src.api.security_config import RATE_LIMITING_AVAILABLE
            if not RATE_LIMITING_AVAILABLE:
                issues.append("Rate limiting is not properly configured")
        except ImportError:
            issues.append("slowapi package not installed - rate limiting will not work")
        
        return {
            "component": "Rate Limiting",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_file_upload_security(self) -> Dict[str, any]:
        """Validate file upload security settings"""
        issues = []
        
        # Check file size limits
        max_file_size = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB default
        
        if max_file_size > 100 * 1024 * 1024:  # 100MB
            self.warnings.append(f"MAX_FILE_SIZE ({max_file_size} bytes) is quite large for production")
        
        # Check allowed file extensions
        allowed_extensions = os.getenv("ALLOWED_FILE_EXTENSIONS", ".csv,.pdf")
        dangerous_extensions = [".exe", ".bat", ".sh", ".js", ".php", ".py", ".html"]
        
        for ext in dangerous_extensions:
            if ext in allowed_extensions:
                issues.append(f"Dangerous file extension '{ext}' is allowed")
        
        # Check upload scanning
        upload_scan_enabled = os.getenv("UPLOAD_SCAN_ENABLED", "true").lower() == "true"
        if not upload_scan_enabled:
            self.warnings.append("Upload scanning is disabled - consider enabling for production")
        
        return {
            "component": "File Upload Security",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_security_headers(self) -> Dict[str, any]:
        """Validate security headers configuration"""
        issues = []
        
        required_headers = {
            "X_CONTENT_TYPE_OPTIONS": "nosniff",
            "X_FRAME_OPTIONS": ["DENY", "SAMEORIGIN"],
            "X_XSS_PROTECTION": "1; mode=block",
            "STRICT_TRANSPORT_SECURITY": None,  # Should exist
            "CONTENT_SECURITY_POLICY": None,   # Should exist
            "REFERRER_POLICY": None            # Should exist
        }
        
        for header, expected_value in required_headers.items():
            env_value = os.getenv(header)
            
            if not env_value:
                issues.append(f"Security header {header} not configured")
            elif expected_value and isinstance(expected_value, list):
                if env_value not in expected_value:
                    issues.append(f"Security header {header} has unexpected value: {env_value}")
            elif expected_value and env_value != expected_value:
                issues.append(f"Security header {header} has unexpected value: {env_value}")
        
        return {
            "component": "Security Headers",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_environment_settings(self) -> Dict[str, any]:
        """Validate general environment security settings"""
        issues = []
        
        # Check environment
        environment = os.getenv("ENVIRONMENT", "development")
        if environment != "production":
            issues.append(f"ENVIRONMENT is set to '{environment}', should be 'production'")
        
        # Check debug mode
        debug = os.getenv("DEBUG", "false").lower()
        if debug == "true":
            issues.append("DEBUG mode is enabled in production")
        
        # Check HTTPS requirements
        require_https = os.getenv("REQUIRE_HTTPS", "false").lower()
        if require_https != "true":
            self.warnings.append("REQUIRE_HTTPS is not enabled")
        
        # Check secure cookies
        secure_cookies = os.getenv("SECURE_COOKIES", "false").lower()
        if secure_cookies != "true":
            self.warnings.append("SECURE_COOKIES is not enabled")
        
        return {
            "component": "Environment Settings",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_password_policy(self) -> Dict[str, any]:
        """Validate password policy configuration"""
        issues = []
        
        min_length = int(os.getenv("MIN_PASSWORD_LENGTH", "8"))
        if min_length < 8:
            issues.append(f"MIN_PASSWORD_LENGTH ({min_length}) is too short for production")
        
        require_special = os.getenv("REQUIRE_SPECIAL_CHARS", "true").lower()
        require_numbers = os.getenv("REQUIRE_NUMBERS", "true").lower()
        require_uppercase = os.getenv("REQUIRE_UPPERCASE", "true").lower()
        
        if require_special != "true":
            issues.append("Password policy should require special characters")
        if require_numbers != "true":
            issues.append("Password policy should require numbers")
        if require_uppercase != "true":
            issues.append("Password policy should require uppercase letters")
        
        return {
            "component": "Password Policy",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def validate_logging_configuration(self) -> Dict[str, any]:
        """Validate security logging configuration"""
        issues = []
        
        log_level = os.getenv("LOG_LEVEL", "INFO")
        security_log_level = os.getenv("SECURITY_LOG_LEVEL", "INFO")
        
        if log_level not in ["INFO", "WARNING", "ERROR"]:
            self.warnings.append(f"LOG_LEVEL '{log_level}' may be too verbose for production")
        
        if security_log_level not in ["INFO", "WARNING", "ERROR"]:
            self.warnings.append(f"SECURITY_LOG_LEVEL '{security_log_level}' may be too verbose for production")
        
        # Check if logs directory exists
        if not os.path.exists("logs"):
            issues.append("Logs directory does not exist")
        
        return {
            "component": "Logging Configuration",
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "issues": issues
        }
    
    def run_full_validation(self) -> Dict[str, any]:
        """Run complete security validation"""
        print("🔒 Running Production Security Validation...")
        print("=" * 50)
        
        validations = [
            self.validate_jwt_configuration(),
            self.validate_cors_configuration(),
            self.validate_rate_limiting(),
            self.validate_file_upload_security(),
            self.validate_security_headers(),
            self.validate_environment_settings(),
            self.validate_password_policy(),
            self.validate_logging_configuration()
        ]
        
        passed = 0
        failed = 0
        
        for validation in validations:
            status_icon = "✅" if validation["status"] == "PASS" else "❌"
            print(f"{status_icon} {validation['component']}: {validation['status']}")
            
            if validation["issues"]:
                for issue in validation["issues"]:
                    print(f"   ⚠️  {issue}")
            
            if validation["status"] == "PASS":
                passed += 1
            else:
                failed += 1
                self.critical_issues.extend(validation["issues"])
        
        print("\n" + "=" * 50)
        
        if self.warnings:
            print("⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()
        
        overall_status = "PASS" if failed == 0 else "FAIL"
        status_icon = "✅" if overall_status == "PASS" else "❌"
        
        print(f"{status_icon} Overall Security Status: {overall_status}")
        print(f"📊 Results: {passed} passed, {failed} failed")
        
        if failed > 0:
            print("\n🚨 Critical Issues Found:")
            for issue in self.critical_issues:
                print(f"   • {issue}")
            print("\n⚠️  Please fix these issues before deploying to production!")
        
        return {
            "overall_status": overall_status,
            "passed": passed,
            "failed": failed,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "validations": validations,
            "timestamp": datetime.now().isoformat()
        }

def generate_secure_jwt_secret() -> str:
    """Generate a cryptographically secure JWT secret key"""
    return secrets.token_urlsafe(32)

def main():
    """Run security validation"""
    validator = SecurityValidator()
    results = validator.run_full_validation()
    
    # Suggest secure JWT key if needed
    if any("JWT_SECRET_KEY" in issue for issue in results["critical_issues"]):
        print(f"\n🔑 Suggested secure JWT secret key:")
        print(f"JWT_SECRET_KEY={generate_secure_jwt_secret()}")
        print("⚠️  Save this key securely and update your .env.production file!")

if __name__ == "__main__":
    main()