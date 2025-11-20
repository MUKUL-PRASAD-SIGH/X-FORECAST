"""
File utilities for enhanced data processing
"""

from fastapi import UploadFile, HTTPException
import os
from typing import Optional

def validate_file_format(file: UploadFile) -> str:
    """Validate and determine file format"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Get file extension
    file_extension = file.filename.lower().split('.')[-1]
    
    # Validate supported formats
    supported_formats = {
        'csv': 'csv',
        'xlsx': 'xlsx', 
        'xls': 'xlsx',
        'json': 'json',
        'pdf': 'pdf'
    }
    
    if file_extension not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Supported formats: {list(supported_formats.keys())}"
        )
    
    return supported_formats[file_extension]

def get_file_size_mb(file: UploadFile) -> float:
    """Get file size in MB"""
    
    if hasattr(file, 'size') and file.size:
        return file.size / (1024 * 1024)
    
    # Fallback: try to get size from file object
    try:
        current_position = file.file.tell()
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(current_position)  # Restore position
        return size / (1024 * 1024)
    except:
        return 0.0

def is_file_size_valid(file: UploadFile, max_size_mb: float = 50.0) -> bool:
    """Check if file size is within limits"""
    
    file_size_mb = get_file_size_mb(file)
    return file_size_mb <= max_size_mb