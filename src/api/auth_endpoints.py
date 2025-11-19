"""
Real Multi-Tenant Authentication Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
import os
import json
import re
import secrets
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime, timedelta

# Import authentication components
from ..auth.user_management import user_manager
from ..auth.multi_tenant_auth import Company, User

router = APIRouter()

class RegisterRequest(BaseModel):
    email: str
    password: str
    confirmPassword: str
    company_name: str
    business_type: str
    industry: str = "retail"

class LoginRequest(BaseModel):
    email: str
    password: str

class RAGInitRequest(BaseModel):
    user_id: str
    company_name: str

class EmailVerificationRequest(BaseModel):
    email: str
    verification_code: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirmRequest(BaseModel):
    email: str
    reset_code: str
    new_password: str

# Email configuration (you'll need to set these environment variables)
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "email_user": os.getenv("EMAIL_USER", ""),
    "email_password": os.getenv("EMAIL_PASSWORD", ""),
    "from_email": os.getenv("FROM_EMAIL", "noreply@xforecast.com")
}

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is valid"

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email using SMTP"""
    try:
        if not EMAIL_CONFIG["email_user"] or not EMAIL_CONFIG["email_password"]:
            print("Email configuration not set. Skipping email send.")
            return False
        
        msg = MimeMultipart()
        msg['From'] = EMAIL_CONFIG["from_email"]
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MimeText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(EMAIL_CONFIG["email_user"], EMAIL_CONFIG["email_password"])
        
        text = msg.as_string()
        server.sendmail(EMAIL_CONFIG["from_email"], to_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def get_current_user(authorization: Optional[str] = Header(None)):
    """Extract and verify JWT token from Authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(" ")[1]
    user_data = user_manager.verify_token(token)
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return user_data

@router.post("/register")
async def register_user(request: RegisterRequest):
    """Register new business user with company-specific setup"""
    try:
        # Validate input
        if not validate_email(request.email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        if request.password != request.confirmPassword:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        is_valid, password_message = validate_password(request.password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=password_message)
        
        if not request.company_name.strip():
            raise HTTPException(status_code=400, detail="Company name is required")
        # Register user in database
        result = user_manager.register_user(
            email=request.email,
            password=request.password,
            company_name=request.company_name,
            business_type=request.business_type,
            industry=request.industry
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Create company-specific directory structure
        user_id = result["user_id"]
        company_dir = f"data/users/{user_id}"
        os.makedirs(f"{company_dir}/csv", exist_ok=True)
        os.makedirs(f"{company_dir}/pdf", exist_ok=True)
        os.makedirs(f"{company_dir}/rag", exist_ok=True)
        
        # Send verification email if email service is configured
        if EMAIL_CONFIG["email_user"] and result.get("verification_code"):
            verification_url = f"http://localhost:3000/verify-email?email={request.email}&code={result['verification_code']}"
            
            email_body = f"""
            <html>
            <body>
                <h2>Welcome to X-FORECAST!</h2>
                <p>Thank you for registering your company <strong>{request.company_name}</strong>.</p>
                <p>Please click the link below to verify your email address:</p>
                <p><a href="{verification_url}" style="background-color: #00d4ff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Verify Email</a></p>
                <p>Or copy and paste this link: {verification_url}</p>
                <p>This link will expire in 24 hours.</p>
                <p>Best regards,<br>X-FORECAST Team</p>
            </body>
            </html>
            """
            
            email_sent = send_email(request.email, "Verify Your X-FORECAST Account", email_body)
            if email_sent:
                return {
                    "success": True,
                    "message": "Registration successful! Please check your email to verify your account.",
                    "user_id": user_id,
                    "requires_verification": True
                }
        
        # Automatically initialize RAG system
        rag_initialized = False
        rag_message = "RAG system will be initialized on first login"
        
        try:
            from ..rag.real_vector_rag import real_vector_rag
            
            # Check for sample data based on business type
            sample_datasets = {
                "retail": "data/sample_retail_data.csv",
                "supermarket": "data/sample_supermarket_data.csv", 
                "restaurant": "data/sample_restaurant_data.csv",
                "ecommerce": "data/sample_ecommerce_data.csv",
                "wholesale": "data/sample_wholesale_data.csv"
            }
            
            sample_data_path = sample_datasets.get(request.business_type, "data/sample_retail_data.csv")
            
            if os.path.exists(sample_data_path):
                success = real_vector_rag.load_company_data(
                    user_id=user_id,
                    company_name=request.company_name,
                    dataset_path=sample_data_path
                )
                
                if success:
                    rag_initialized = True
                    rag_message = f"RAG system initialized with sample {request.business_type} data"
            
            # Create RAG structure even if no sample data
            if not rag_initialized:
                empty_kb_path = os.path.join(f"{company_dir}/rag", "knowledge_base.json")
                with open(empty_kb_path, 'w') as f:
                    json.dump({
                        "company_name": request.company_name,
                        "user_id": user_id,
                        "initialized_at": datetime.now().isoformat(),
                        "documents": [],
                        "status": "empty"
                    }, f)
                rag_message = "RAG system structure created. Upload data to activate AI features."
                
        except ImportError:
            pass  # RAG system not available
        except Exception as e:
            print(f"RAG initialization error: {e}")
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "company_name": request.company_name,
            "rag_initialized": rag_initialized,
            "rag_message": rag_message,
            "business_type": request.business_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login")
async def login_user(request: LoginRequest):
    """Authenticate user and return JWT token"""
    try:
        result = user_manager.authenticate_user(request.email, request.password)
        
        if not result:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        if "error" in result:
            raise HTTPException(status_code=401, detail=result["error"])
        
        # Check if RAG system needs initialization
        user_id = result["user"]["user_id"]
        company_name = result["user"]["company_name"]
        
        rag_status = "initialized"
        try:
            from ..rag.real_vector_rag import real_vector_rag
            
            # Check if user has any RAG data
            if user_id not in real_vector_rag.user_metadata:
                # Initialize RAG system on first login
                profile = user_manager.get_business_profile(user_id)
                if profile:
                    business_type = profile.business_type
                    sample_datasets = {
                        "retail": "data/sample_retail_data.csv",
                        "supermarket": "data/sample_supermarket_data.csv", 
                        "restaurant": "data/sample_restaurant_data.csv",
                        "ecommerce": "data/sample_ecommerce_data.csv",
                        "wholesale": "data/sample_wholesale_data.csv"
                    }
                    
                    sample_data_path = sample_datasets.get(business_type, "data/sample_retail_data.csv")
                    
                    if os.path.exists(sample_data_path):
                        success = real_vector_rag.load_company_data(
                            user_id=user_id,
                            company_name=company_name,
                            dataset_path=sample_data_path
                        )
                        if success:
                            rag_status = f"initialized_with_sample_{business_type}"
                    else:
                        # Initialize empty RAG structure
                        real_vector_rag.initialize_company_rag(user_id, company_name)
                        rag_status = "initialized_empty"
                        
        except ImportError:
            rag_status = "not_available"
        except Exception as e:
            print(f"RAG initialization error on login: {e}")
            rag_status = "error"
        
        # Add RAG status to response
        result["rag_status"] = rag_status
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize-rag")
async def initialize_user_rag(request: RAGInitRequest, current_user: dict = Depends(get_current_user)):
    """Initialize company-specific RAG system for new user"""
    try:
        # Verify user owns this company
        if current_user["user_id"] != request.user_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to user data")
        
        # Initialize RAG system for the company
        try:
            from ..rag.real_vector_rag import real_vector_rag
            
            # Check if sample data exists for this business type
            user_profile = user_manager.get_business_profile(request.user_id)
            sample_data_path = None
            
            if user_profile:
                business_type = user_profile.business_type
                # Map business types to sample datasets
                sample_datasets = {
                    "retail": "data/sample_retail_data.csv",
                    "supermarket": "data/sample_supermarket_data.csv", 
                    "restaurant": "data/sample_restaurant_data.csv",
                    "ecommerce": "data/sample_ecommerce_data.csv",
                    "wholesale": "data/sample_wholesale_data.csv"
                }
                
                sample_data_path = sample_datasets.get(business_type, "data/sample_retail_data.csv")
            
            # Initialize with sample data if available
            if sample_data_path and os.path.exists(sample_data_path):
                success = real_vector_rag.load_company_data(
                    user_id=request.user_id,
                    company_name=request.company_name,
                    dataset_path=sample_data_path
                )
                
                if success:
                    return {
                        "success": True,
                        "message": f"RAG system initialized for {request.company_name} with sample {user_profile.business_type} data",
                        "user_id": request.user_id,
                        "initialized_with_sample": True,
                        "business_type": user_profile.business_type
                    }
            
            # If no sample data, create empty RAG structure
            user_dir = f"data/users/{request.user_id}/rag"
            os.makedirs(user_dir, exist_ok=True)
            
            # Create empty knowledge base file
            empty_kb_path = os.path.join(user_dir, "knowledge_base.json")
            with open(empty_kb_path, 'w') as f:
                json.dump({
                    "company_name": request.company_name,
                    "user_id": request.user_id,
                    "initialized_at": datetime.now().isoformat(),
                    "documents": [],
                    "status": "empty"
                }, f)
            
            return {
                "success": True,
                "message": f"RAG system structure created for {request.company_name}. Upload data to activate AI features.",
                "user_id": request.user_id,
                "initialized_with_sample": False,
                "requires_data_upload": True
            }
                
        except ImportError:
            # RAG system not available, return success anyway
            return {
                "success": True,
                "message": "RAG system will be initialized on first data upload",
                "user_id": request.user_id,
                "initialized_with_sample": False,
                "requires_data_upload": True
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get authenticated user's profile"""
    try:
        profile = user_manager.get_business_profile(current_user["user_id"])
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return {
            "user_id": profile.user_id,
            "email": current_user["email"],
            "company_name": profile.company_name,
            "business_type": profile.business_type,
            "industry": profile.industry,
            "data_sources": profile.data_sources,
            "storage_path": profile.storage_path,
            "model_config": profile.model_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-email")
async def verify_email(request: EmailVerificationRequest):
    """Verify user email with verification code"""
    try:
        result = user_manager.verify_email(request.email, request.verification_code)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "success": True,
            "message": "Email verified successfully! You can now login."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forgot-password")
async def forgot_password(request: PasswordResetRequest):
    """Request password reset"""
    try:
        result = user_manager.request_password_reset(request.email)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Send password reset email if email service is configured
        if EMAIL_CONFIG["email_user"] and result.get("reset_code"):
            reset_url = f"http://localhost:3000/reset-password?email={request.email}&code={result['reset_code']}"
            
            email_body = f"""
            <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>You requested a password reset for your X-FORECAST account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="{reset_url}" style="background-color: #ff6b6b; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                <p>Or copy and paste this link: {reset_url}</p>
                <p>This link will expire in 1 hour.</p>
                <p>If you didn't request this reset, please ignore this email.</p>
                <p>Best regards,<br>X-FORECAST Team</p>
            </body>
            </html>
            """
            
            email_sent = send_email(request.email, "Reset Your X-FORECAST Password", email_body)
            if email_sent:
                return {
                    "success": True,
                    "message": "Password reset instructions sent to your email."
                }
        
        return {
            "success": True,
            "message": "If the email exists, password reset instructions have been sent."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-password")
async def reset_password(request: PasswordResetConfirmRequest):
    """Reset password with reset code"""
    try:
        # Validate new password
        is_valid, password_message = validate_password(request.new_password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=password_message)
        
        result = user_manager.reset_password(request.email, request.reset_code, request.new_password)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "success": True,
            "message": "Password reset successfully! You can now login with your new password."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-token")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """Verify JWT token validity"""
    return {
        "valid": True,
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "company_name": current_user.get("company_name"),
        "expires_at": current_user.get("exp")
    }