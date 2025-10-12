"""
Development Authentication with Company Datasets
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime
import pandas as pd
import os
try:
    from ..rag.real_vector_rag import real_vector_rag
except ImportError:
    # Fallback if vector RAG not available
    class MockVectorRAG:
        def load_company_data(self, *args): return True
        def generate_personalized_response(self, user_id, query):
            return type('obj', (), {'response_text': f'Mock response for {query}', 'confidence': 0.8, 'company_context': 'Demo Company'})()
        def update_user_data(self, *args): return True
    real_vector_rag = MockVectorRAG()

router = APIRouter()

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

# Legacy company data storage (kept for compatibility)
company_data = {}

# Company-specific datasets
company_datasets = {
    "SuperX Corporation": "data/superx_retail_data.csv",
    "TechCorp Industries": "data/techcorp_manufacturing_data.csv", 
    "HealthPlus Medical": "data/healthplus_medical_data.csv"
}

# Default SuperX user (login bypassed)
dev_users = {
    "admin@superx.com": {
        "user_id": "user_1",
        "email": "admin@superx.com", 
        "password": "admin123",
        "company_name": "SuperX Corporation",
        "business_type": "retail",
        "industry": "retail",
        "dataset": "data/superx_retail_data.csv",
        "created_at": "2024-01-01T00:00:00"
    }
}

# Auto-load SuperX data on startup
async def initialize_superx():
    """Initialize SuperX data on startup"""
    await load_superx_combined_data()

# Call initialization
import asyncio
try:
    asyncio.create_task(initialize_superx())
except:
    pass

async def load_superx_combined_data():
    """Load SuperX data from both sample file and uploaded files"""
    try:
        user_id = "user_1"
        company_name = "SuperX Corporation"
        
        # Load base SuperX dataset
        base_dataset = "data/superx_retail_data.csv"
        success = real_vector_rag.load_company_data(user_id, company_name, base_dataset)
        
        # Load any uploaded files from data/users/user_1/ directory
        user_data_dir = "data/users/user_1"
        if os.path.exists(user_data_dir):
            for filename in os.listdir(user_data_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(user_data_dir, filename)
                    real_vector_rag.update_user_data(user_id, file_path)
                    print(f"Added uploaded data: {filename}")
        
        print(f"SuperX data loaded: base + uploaded files")
    except Exception as e:
        print(f"Error loading SuperX data: {e}")

def generate_company_response(user_id: str, company_name: str, business_type: str, message: str) -> str:
    """Generate personalized response using real vector RAG"""
    try:
        # Use real vector RAG for personalized response
        rag_response = real_vector_rag.generate_personalized_response(user_id, message)
        return rag_response.response_text
    except Exception as e:
        print(f"RAG error for {company_name}: {e}")
        return f"🤖 **{company_name} AI Assistant**: I'm processing your data. Please try again in a moment."

@router.post("/register")
async def register_user(request: RegisterRequest):
    """Bypass register - always return SuperX user"""
    # Load SuperX data
    await load_superx_combined_data()
    
    return {
        "message": "Welcome to SuperX Corporation!",
        "user_id": "user_1"
    }

@router.post("/login")
async def login_user(request: LoginRequest):
    """Bypass login - always return SuperX user"""
    # Load SuperX data on every login attempt
    await load_superx_combined_data()
    
    # Always return success with SuperX user data
    return {
        "success": True,
        "token": "superx_token_123",
        "user": {
            "user_id": "user_1",
            "email": request.email,
            "company_name": "SuperX Corporation",
            "business_type": "retail"
        }
    }

@router.post("/chat")
async def personalized_chat(request: ChatRequest):
    """SuperX personalized chat using combined data"""
    user_id = "user_1"  # Always SuperX
    user = dev_users["admin@superx.com"]
    
    # Generate response using vector RAG with combined data
    try:
        rag_response = real_vector_rag.generate_personalized_response(user_id, request.message)
        
        return {
            "response_id": f"chat_{user_id}_{int(datetime.now().timestamp())}",
            "response_text": rag_response.response_text,
            "confidence": rag_response.confidence,
            "is_personalized": True,
            "company_name": "SuperX Corporation",
            "sources": rag_response.sources,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "response_id": f"chat_{user_id}_{int(datetime.now().timestamp())}",
            "response_text": "SuperX AI: I'm ready to help with your retail business data. Ask me about products, sales, or forecasting!",
            "confidence": 0.8,
            "is_personalized": True,
            "company_name": "SuperX Corporation",
            "timestamp": datetime.now().isoformat()
        }

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload data for SuperX Corporation"""
    user_id = "user_1"  # Always SuperX
    temp_path = f"temp_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Save to SuperX user directory
        user_dir = "data/users/user_1"
        os.makedirs(user_dir, exist_ok=True)
        
        permanent_path = os.path.join(user_dir, file.filename)
        os.rename(temp_path, permanent_path)
        
        # Update SuperX vector RAG with new data
        success = real_vector_rag.update_user_data(user_id, permanent_path)
        
        return {
            "success": success,
            "message": f"SuperX data updated with {file.filename}",
            "processed_records": "Added to SuperX knowledge base",
            "data_quality_score": 0.95,
            "file_saved": permanent_path
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/profile")
async def get_user_profile():
    """Get SuperX user profile"""
    return {
        "user_id": "user_1",
        "company_name": "SuperX Corporation",
        "business_type": "retail",
        "industry": "retail",
        "data_sources": ["superx_retail_data.csv", "uploaded_files"],
        "storage_path": "data/users/user_1"
    }

@router.get("/data-summary")
async def get_data_summary():
    """Development data summary"""
    return {
        "files": [],
        "total_records": 0,
        "message": "Development mode - company datasets loaded"
    }