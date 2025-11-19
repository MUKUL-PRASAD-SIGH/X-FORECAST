"""
PDF Processing API Endpoints
Handles PDF upload, text extraction, and integration with RAG system
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
import shutil
from datetime import datetime
import logging
import asyncio

# Import RAG system and PDF processor
from src.rag.real_vector_rag import real_vector_rag
from src.rag.pdf_processor import pdf_processor, PDFProcessingError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pdf", tags=["PDF Processing"])

# Pydantic models
class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    processing_status: str
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    error_details: Optional[Dict[str, str]] = None

class PDFProcessingStatus(BaseModel):
    document_id: str
    filename: str
    processing_status: str
    upload_date: datetime
    file_size: int
    page_count: int
    error_message: Optional[str] = None

class PDFHistoryResponse(BaseModel):
    total_documents: int
    documents: List[PDFProcessingStatus]

def get_current_user(request):
    """Get current authenticated user - placeholder for actual auth"""
    # This should be replaced with actual authentication middleware
    # For now, return a mock user for testing
    if hasattr(request.state, 'user'):
        return request.state.user
    raise HTTPException(status_code=401, detail="Authentication required")

async def process_pdf_background(file_path: str, user_id: str, company_name: str, filename: str):
    """Background task to process PDF and integrate with RAG"""
    try:
        logger.info(f"Starting background PDF processing for {filename}")
        
        # Add PDF to RAG system
        success = real_vector_rag.add_pdf_document(user_id, company_name, file_path)
        
        if success:
            logger.info(f"Successfully processed PDF {filename} for user {user_id}")
        else:
            logger.error(f"Failed to process PDF {filename} for user {user_id}")
            
    except Exception as e:
        logger.error(f"Error in background PDF processing: {str(e)}")

@router.post("/extract-preview")
async def extract_pdf_preview(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Extract preview text from PDF without full processing
    Used for file validation and preview before upload
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return {
                "success": False,
                "message": "Invalid file type. Only PDF files are supported.",
                "extraction_success": False
            }
        
        # Get user information
        user_id = current_user.get("user_id")
        company_name = current_user.get("company_name", "Unknown Company")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in authentication token")
        
        # Create temporary file for preview extraction
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text for preview
            extraction_result = pdf_processor.extract_text(temp_file_path, user_id, company_name)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if extraction_result.success:
                # Return preview information
                preview_text = extraction_result.text[:500] if extraction_result.text else ""
                
                return {
                    "success": True,
                    "extraction_success": True,
                    "page_count": extraction_result.page_count,
                    "text_length": len(extraction_result.text) if extraction_result.text else 0,
                    "preview_text": preview_text,
                    "metadata": {
                        "filename": file.filename,
                        "file_size": len(content),
                        "processing_time": extraction_result.processing_time
                    }
                }
            else:
                return {
                    "success": False,
                    "extraction_success": False,
                    "message": f"PDF text extraction failed: {extraction_result.error}",
                    "page_count": 0,
                    "text_length": 0
                }
                
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            logger.error(f"PDF preview extraction error: {str(e)}")
            return {
                "success": False,
                "extraction_success": False,
                "message": f"PDF preview extraction failed: {str(e)}",
                "page_count": 0,
                "text_length": 0
            }
        
    except Exception as e:
        logger.error(f"PDF preview error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preview extraction failed: {str(e)}")

@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload and process PDF file for RAG integration
    
    - **file**: PDF file to upload and process
    - Returns processing status and document information
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return PDFUploadResponse(
                success=False,
                message="Invalid file type. Only PDF files are supported.",
                processing_status="failed",
                error_details=pdf_processor.get_error_suggestions("unsupported_format")
            )
        
        # Get user information
        user_id = current_user.get("user_id")
        company_name = current_user.get("company_name", "Unknown Company")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in authentication token")
        
        # Create user-specific PDF directory
        pdf_dir = f"data/users/{user_id}/pdf"
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(pdf_dir, file.filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            # Create unique filename
            base_name, ext = os.path.splitext(file.filename)
            timestamp = int(datetime.now().timestamp())
            file_path = os.path.join(pdf_dir, f"{base_name}_{timestamp}{ext}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        
        # Quick validation - try to extract text to check if file is valid
        try:
            extraction_result = pdf_processor.extract_text(file_path, user_id, company_name)
            
            if not extraction_result.success:
                # Remove invalid file
                os.remove(file_path)
                
                return PDFUploadResponse(
                    success=False,
                    message=f"PDF processing failed: {extraction_result.error}",
                    processing_status="failed",
                    error_details=pdf_processor.get_error_suggestions(
                        extraction_result.error.split(':')[0] if ':' in extraction_result.error else "extraction"
                    )
                )
            
            # Schedule background processing for RAG integration
            background_tasks.add_task(
                process_pdf_background, 
                file_path, 
                user_id, 
                company_name, 
                file.filename
            )
            
            return PDFUploadResponse(
                success=True,
                message="PDF uploaded and processing started successfully",
                document_id=f"pdf_{extraction_result.metadata.file_hash}",
                filename=file.filename,
                processing_status="processing",
                file_size=file_size,
                page_count=extraction_result.page_count
            )
            
        except Exception as e:
            # Remove file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.error(f"PDF processing error: {str(e)}")
            return PDFUploadResponse(
                success=False,
                message=f"PDF processing error: {str(e)}",
                processing_status="failed",
                error_details=pdf_processor.get_error_suggestions("extraction")
            )
        
    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/history", response_model=PDFHistoryResponse)
async def get_pdf_history(current_user: dict = Depends(get_current_user)):
    """
    Get PDF processing history for the authenticated user
    
    - Returns list of all PDF documents uploaded by the user
    - Includes processing status and metadata
    """
    try:
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in authentication token")
        
        # Get document history from RAG database
        import sqlite3
        import json
        
        conn = sqlite3.connect(real_vector_rag.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT document_id, filename, processing_status, upload_date, 
                   file_size, page_count, error_message, metadata
            FROM user_documents 
            WHERE user_id = ? AND file_type = 'pdf'
            ORDER BY upload_date DESC
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in results:
            doc_id, filename, status, upload_date, file_size, page_count, error_msg, metadata_json = row
            
            # Parse upload date
            try:
                upload_dt = datetime.fromisoformat(upload_date)
            except:
                upload_dt = datetime.now()
            
            doc_status = PDFProcessingStatus(
                document_id=doc_id,
                filename=filename,
                processing_status=status,
                upload_date=upload_dt,
                file_size=file_size or 0,
                page_count=page_count or 0,
                error_message=error_msg
            )
            documents.append(doc_status)
        
        return PDFHistoryResponse(
            total_documents=len(documents),
            documents=documents
        )
        
    except Exception as e:
        logger.error(f"Error getting PDF history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve PDF history: {str(e)}")

@router.get("/status/{document_id}")
async def get_pdf_status(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get processing status for a specific PDF document
    
    - **document_id**: ID of the PDF document
    - Returns current processing status and details
    """
    try:
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in authentication token")
        
        # Get document status from database
        import sqlite3
        
        conn = sqlite3.connect(real_vector_rag.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT document_id, filename, processing_status, upload_date, 
                   file_size, page_count, error_message
            FROM user_documents 
            WHERE user_id = ? AND document_id = ? AND file_type = 'pdf'
        ''', (user_id, document_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="PDF document not found")
        
        doc_id, filename, status, upload_date, file_size, page_count, error_msg = result
        
        # Parse upload date
        try:
            upload_dt = datetime.fromisoformat(upload_date)
        except:
            upload_dt = datetime.now()
        
        return {
            "document_id": doc_id,
            "filename": filename,
            "processing_status": status,
            "upload_date": upload_dt.isoformat(),
            "file_size": file_size or 0,
            "page_count": page_count or 0,
            "error_message": error_msg,
            "rag_integrated": status == "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PDF status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get PDF status: {str(e)}")

@router.delete("/document/{document_id}")
async def delete_pdf_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a PDF document and remove it from RAG system
    
    - **document_id**: ID of the PDF document to delete
    - Removes document from both file system and vector database
    """
    try:
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in authentication token")
        
        import sqlite3
        
        # Get document information
        conn = sqlite3.connect(real_vector_rag.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, file_path FROM user_documents 
            WHERE user_id = ? AND document_id = ? AND file_type = 'pdf'
        ''', (user_id, document_id))
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="PDF document not found")
        
        filename, file_path = result
        
        # Delete from database
        cursor.execute('DELETE FROM user_documents WHERE user_id = ? AND document_id = ?', 
                      (user_id, document_id))
        
        # Delete related vector embeddings
        cursor.execute('DELETE FROM user_vectors WHERE user_id = ? AND source_file = ?', 
                      (user_id, filename))
        
        conn.commit()
        conn.close()
        
        # Delete physical file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Rebuild user index to remove deleted content
        company_name = current_user.get("company_name", "Unknown Company")
        real_vector_rag._build_user_index(user_id, company_name)
        
        return {
            "success": True,
            "message": f"PDF document '{filename}' deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting PDF document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF document: {str(e)}")

@router.get("/search")
async def search_pdf_content(
    query: str,
    limit: int = 5,
    current_user: dict = Depends(get_current_user)
):
    """
    Search through PDF content using RAG system
    
    - **query**: Search query
    - **limit**: Maximum number of results to return
    - Returns relevant PDF content with source attribution
    """
    try:
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in authentication token")
        
        # Search using enhanced RAG query
        results = real_vector_rag.query_user_knowledge_enhanced(user_id, query, top_k=limit)
        
        # Filter for PDF results only
        pdf_results = [result for result in results if result.get('document_type') == 'pdf']
        
        # Format results for API response
        formatted_results = []
        for result in pdf_results:
            formatted_result = {
                "content": result['content'],
                "relevance_score": result['score'],
                "source_file": result.get('source_file', 'Unknown'),
                "page_number": result.get('page_number'),
                "document_id": result['doc_id'],
                "source_attribution": result.get('source_attribution', '')
            }
            formatted_results.append(formatted_result)
        
        return {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error searching PDF content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF search failed: {str(e)}")