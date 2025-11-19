"""
RAG Management API
Provides endpoints for managing company-specific RAG systems
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..auth.user_management import user_manager
from ..rag.rag_manager import rag_manager, RAGHealthStatus, RAGMigrationResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Management"])

def get_current_user(token: str):
    """Dependency to get current user from token"""
    user_data = user_manager.verify_token(token)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_data

@router.get("/status/{user_id}")
async def get_rag_status(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get RAG initialization status for a user
    """
    try:
        # Verify user can access this data (admin or own data)
        if current_user["user_id"] != user_id and current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        status = rag_manager.check_rag_initialization_status(user_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting RAG status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize/{user_id}")
async def initialize_rag(user_id: str, force_reinit: bool = False, current_user: dict = Depends(get_current_user)):
    """
    Initialize or reinitialize RAG system for a user
    """
    try:
        # Verify user can access this data (admin or own data)
        if current_user["user_id"] != user_id and current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get company name from user data
        company_name = current_user.get("company_name", "Unknown Company")
        
        result = rag_manager.initialize_rag_for_user(user_id, company_name, force_reinit)
        return result
        
    except Exception as e:
        logger.error(f"Error initializing RAG for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset/{user_id}")
async def reset_rag(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Reset and reinitialize RAG system for a user (clears all data)
    """
    try:
        # Verify user can access this data (admin or own data)
        if current_user["user_id"] != user_id and current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get company name from user data
        company_name = current_user.get("company_name", "Unknown Company")
        
        result = rag_manager.reset_rag_system(user_id, company_name)
        return result
        
    except Exception as e:
        logger.error(f"Error resetting RAG for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/{user_id}")
async def get_rag_health(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive health status of user's RAG system
    """
    try:
        # Verify user can access this data (admin or own data)
        if current_user["user_id"] != user_id and current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        health_status = rag_manager.get_rag_health_status(user_id)
        
        # Convert dataclass to dict for JSON response
        return {
            "user_id": health_status.user_id,
            "company_name": health_status.company_name,
            "is_initialized": health_status.is_initialized,
            "status": health_status.status,
            "total_documents": health_status.total_documents,
            "csv_count": health_status.csv_count,
            "pdf_count": health_status.pdf_count,
            "last_updated": health_status.last_updated.isoformat() if health_status.last_updated else None,
            "error_message": health_status.error_message,
            "index_version": health_status.index_version
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG health for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diagnostics/{user_id}")
async def run_rag_diagnostics(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Run comprehensive diagnostics on user's RAG system
    """
    try:
        # Verify user can access this data (admin or own data)
        if current_user["user_id"] != user_id and current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        diagnostics = rag_manager.run_rag_diagnostics(user_id)
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error running RAG diagnostics for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/migrate")
async def migrate_users_to_rag(batch_size: int = 10, current_user: dict = Depends(get_current_user)):
    """
    Migrate existing users who don't have RAG initialized (admin only)
    """
    try:
        # Only admin can run migration
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = rag_manager.migrate_existing_users_to_rag(batch_size)
        return result
        
    except Exception as e:
        logger.error(f"Error during RAG migration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_rag_status(current_user: dict = Depends(get_current_user)):
    """
    Get system-wide RAG status and statistics (admin only)
    """
    try:
        # Only admin can view system-wide status
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        status = rag_manager.get_system_wide_rag_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting system RAG status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my/status")
async def get_my_rag_status(current_user: dict = Depends(get_current_user)):
    """
    Get current user's RAG status (convenience endpoint)
    """
    try:
        user_id = current_user["user_id"]
        status = rag_manager.check_rag_initialization_status(user_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting user RAG status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my/health")
async def get_my_rag_health(current_user: dict = Depends(get_current_user)):
    """
    Get current user's RAG health status (convenience endpoint)
    """
    try:
        user_id = current_user["user_id"]
        health_status = rag_manager.get_rag_health_status(user_id)
        
        # Convert dataclass to dict for JSON response
        return {
            "user_id": health_status.user_id,
            "company_name": health_status.company_name,
            "is_initialized": health_status.is_initialized,
            "status": health_status.status,
            "total_documents": health_status.total_documents,
            "csv_count": health_status.csv_count,
            "pdf_count": health_status.pdf_count,
            "last_updated": health_status.last_updated.isoformat() if health_status.last_updated else None,
            "error_message": health_status.error_message,
            "index_version": health_status.index_version
        }
        
    except Exception as e:
        logger.error(f"Error getting user RAG health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/my/initialize")
async def initialize_my_rag(force_reinit: bool = False, current_user: dict = Depends(get_current_user)):
    """
    Initialize current user's RAG system (convenience endpoint)
    """
    try:
        user_id = current_user["user_id"]
        company_name = current_user.get("company_name", "Unknown Company")
        
        result = rag_manager.initialize_rag_for_user(user_id, company_name, force_reinit)
        return result
        
    except Exception as e:
        logger.error(f"Error initializing user RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/my/reset")
async def reset_my_rag(current_user: dict = Depends(get_current_user)):
    """
    Reset current user's RAG system (convenience endpoint)
    """
    try:
        user_id = current_user["user_id"]
        company_name = current_user.get("company_name", "Unknown Company")
        
        result = rag_manager.reset_rag_system(user_id, company_name)
        return result
        
    except Exception as e:
        logger.error(f"Error resetting user RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))