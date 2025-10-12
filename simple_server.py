"""
Simple FastAPI server without complex imports
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import jwt
import uuid
from datetime import datetime, timedelta

app = FastAPI(title="X-FORECAST Simple API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
users_db = {}
SECRET_KEY = "x_forecast_secret_2024"

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

@app.get("/")
async def root():
    return {"message": "X-FORECAST API", "status": "online"}

@app.post("/api/v1/auth/register")
async def register_user(request: RegisterRequest):
    if request.email in users_db:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    user_id = str(uuid.uuid4())
    password_hash = hashlib.sha256(request.password.encode()).hexdigest()
    
    users_db[request.email] = {
        "user_id": user_id,
        "email": request.email,
        "password_hash": password_hash,
        "company_name": request.company_name,
        "business_type": request.business_type,
        "industry": request.industry
    }
    
    return {"message": "User registered successfully", "user_id": user_id}

@app.post("/api/v1/auth/login")
async def login_user(request: LoginRequest):
    if request.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = users_db[request.email]
    password_hash = hashlib.sha256(request.password.encode()).hexdigest()
    
    if user["password_hash"] != password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    payload = {
        "user_id": user["user_id"],
        "email": user["email"],
        "company_name": user["company_name"],
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    return {
        "success": True,
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "email": user["email"],
            "company_name": user["company_name"],
            "business_type": user["business_type"]
        }
    }

@app.post("/api/v1/auth/chat")
async def chat_endpoint(request: ChatRequest):
    # Simple demo response
    responses = [
        f"📈 **{request.message}**: Based on demo data, I can see trends and patterns. Upload your data for personalized insights!",
        f"🤖 **AI Response**: I understand you're asking about '{request.message}'. This is a demo response - register and upload data for real insights!",
        f"📊 **Demo Mode**: Your query '{request.message}' would be analyzed with your actual business data once uploaded."
    ]
    
    import random
    response_text = random.choice(responses)
    
    return {
        "response_id": f"demo_{int(datetime.now().timestamp())}",
        "response_text": response_text,
        "confidence": 0.8,
        "is_personalized": False,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)