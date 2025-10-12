"""
Authentication and Multi-tenant API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import os

try:
    from ..auth.user_management import user_manager
    from ..data_upload.upload_engine import upload_engine
    from ..rag.personalized_rag import personalized_rag
    from ..rag.vector_rag import vector_rag
except ImportError:
    # Fallback for missing dependencies
    class MockUserManager:
        def register_user(self, **kwargs): return {"success": False, "message": "Auth system not configured"}
        def authenticate_user(self, email, password): return None
        def verify_token(self, token): return None
        def get_business_profile(self, user_id): return None
    
    class MockUploadEngine:
        def process_upload(self, *args): return type('obj', (object,), {'success': False, 'message': 'Upload not available'})
        def get_data_summary(self, user_id): return {"files": [], "total_records": 0}
    
    class MockRAG:
        def initialize_user_rag(self, *args): pass
        def update_user_data(self, *args): pass
        def get_personalized_response(self, *args): return {"response_text": "RAG not available", "is_fallback": True}
        def generate_personalized_response(self, *args): return {"response_text": "RAG not available", "is_fallback": True}
    
    user_manager = MockUserManager()
    upload_engine = MockUploadEngine()
    personalized_rag = MockRAG()
    vector_rag = MockRAG()

router = APIRouter()
security = HTTPBearer()

class RegisterRequest(BaseModel):
    email: str
    password: str
    company_name: str
    business_type: str
    industry: str = "retail"

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user data"""
    try:
        token = credentials.credentials
        user_data = user_manager.verify_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_data
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.post("/register")
async def register_user(request: RegisterRequest):
    """Register new business user"""
    try:
        result = user_manager.register_user(
            email=request.email,
            password=request.password,
            company_name=request.company_name,
            business_type=request.business_type,
            industry=request.industry
        )
        
        if result["success"]:
            # Initialize personalized RAG for new user
            personalized_rag.initialize_user_rag(result["user_id"], request.company_name)
            return {"message": "User registered successfully", "user_id": result["user_id"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=503, detail="Authentication service temporarily unavailable")

@router.post("/login")
async def login_user(request: LoginRequest):
    """Authenticate user and return JWT token"""
    try:
        result = user_manager.authenticate_user(request.email, request.password)
        
        if result:
            # Initialize RAG if not already done
            user_id = result["user"]["user_id"]
            company_name = result["user"]["company_name"]
            personalized_rag.initialize_user_rag(user_id, company_name)
            
            return result
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=503, detail="Authentication service temporarily unavailable")

@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    data_type: str = Form("sales"),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process business data"""
    user_id = current_user["user_id"]
    
    # Save uploaded file temporarily
    temp_path = f"temp_{user_id}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file
        result = upload_engine.process_upload(temp_path, user_id, data_type)
        
        # Update user's RAG model with new data
        if result.success:
            personalized_rag.update_user_data(user_id)
            
            # Add to vector RAG for real embeddings
            try:
                import pandas as pd
                df = pd.read_csv(result.file_path)
                vector_rag.add_user_data(user_id, df, data_type)
            except Exception as e:
                print(f"Vector RAG update failed: {e}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": result.success,
            "message": result.message,
            "processed_records": result.processed_records,
            "data_quality_score": result.data_quality_score,
            "detected_columns": result.detected_columns
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/data-summary")
async def get_data_summary(current_user: dict = Depends(get_current_user)):
    """Get summary of user's uploaded data"""
    user_id = current_user["user_id"]
    summary = upload_engine.get_data_summary(user_id)
    return summary

@router.post("/chat")
async def personalized_chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """Personalized chat using user's own data"""
    user_id = current_user["user_id"]
    
    # Get personalized response from vector RAG
    response = vector_rag.generate_personalized_response(user_id, request.message)
    
    # Fallback to basic RAG if vector RAG fails
    if response.get('is_fallback') or response.get('confidence', 0) < 0.5:
        fallback_response = personalized_rag.get_personalized_response(user_id, request.message)
        if not fallback_response.get('is_fallback'):
            response = fallback_response
    
    return {
        "response_id": f"chat_{user_id}_{int(datetime.now().timestamp())}",
        "response_text": response["response_text"],
        "confidence": response.get("confidence", 0.8),
        "is_personalized": not response.get("is_fallback", False),
        "company_name": current_user["company_name"],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get user business profile"""
    user_id = current_user["user_id"]
    profile = user_manager.get_business_profile(user_id)
    
    if profile:
        return {
            "user_id": profile.user_id,
            "company_name": profile.company_name,
            "business_type": profile.business_type,
            "industry": profile.industry,
            "data_sources": profile.data_sources,
            "storage_path": profile.storage_path
        }
    else:
        raise HTTPException(status_code=404, detail="Profile not found")

from datetime import datetime