"""
PDF Processing Module for Multi-Tenant RAG System
Handles PDF text extraction, metadata processing, and integration with vector embeddings
"""

import PyPDF2
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PDFMetadata:
    """PDF document metadata"""
    filename: str
    upload_date: datetime
    page_count: int
    file_size: int
    company_id: str
    user_id: str
    file_hash: str
    extraction_status: str = "pending"  # pending, success, failed
    error_message: Optional[str] = None

@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction"""
    text: str
    page_count: int
    metadata: PDFMetadata
    success: bool
    error: Optional[str] = None
    page_texts: List[str] = None  # Text from individual pages

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    def __init__(self, message: str, error_type: str):
        self.message = message
        self.error_type = error_type  # 'extraction', 'encoding', 'corrupted', 'unsupported'
        super().__init__(self.message)

class PDFProcessor:
    """
    PDF text extraction and processing for RAG integration
    Handles various PDF formats and provides error handling
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
    
    def extract_text(self, file_path: str, user_id: str, company_id: str) -> PDFExtractionResult:
        """
        Extract text content from PDF file with comprehensive error handling
        
        Args:
            file_path: Path to PDF file
            user_id: User ID for tracking
            company_id: Company ID for multi-tenant isolation
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise PDFProcessingError(f"File not found: {file_path}", "file_not_found")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise PDFProcessingError(f"File too large: {file_size} bytes", "file_too_large")
            
            # Check file extension
            if not file_path.lower().endswith('.pdf'):
                raise PDFProcessingError("File is not a PDF", "unsupported_format")
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            
            # Create metadata
            filename = os.path.basename(file_path)
            metadata = PDFMetadata(
                filename=filename,
                upload_date=datetime.now(),
                page_count=0,
                file_size=file_size,
                company_id=company_id,
                user_id=user_id,
                file_hash=file_hash,
                extraction_status="processing"
            )
            
            # Extract text from PDF
            extracted_text, page_texts = self._extract_pdf_text(file_path)
            
            # Update metadata
            metadata.page_count = len(page_texts)
            metadata.extraction_status = "success"
            
            logger.info(f"Successfully extracted text from {filename}: {len(extracted_text)} characters, {len(page_texts)} pages")
            
            return PDFExtractionResult(
                text=extracted_text,
                page_count=len(page_texts),
                metadata=metadata,
                success=True,
                page_texts=page_texts
            )
            
        except PDFProcessingError as e:
            logger.error(f"PDF processing error for {file_path}: {e.message}")
            metadata.extraction_status = "failed"
            metadata.error_message = e.message
            
            return PDFExtractionResult(
                text="",
                page_count=0,
                metadata=metadata,
                success=False,
                error=e.message
            )
            
        except Exception as e:
            logger.error(f"Unexpected error processing PDF {file_path}: {str(e)}")
            metadata.extraction_status = "failed"
            metadata.error_message = f"Unexpected error: {str(e)}"
            
            return PDFExtractionResult(
                text="",
                page_count=0,
                metadata=metadata,
                success=False,
                error=str(e)
            )
    
    def _extract_pdf_text(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Extract text from PDF using PyPDF2 with fallback handling
        
        Returns:
            Tuple of (combined_text, list_of_page_texts)
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise PDFProcessingError("PDF is encrypted and cannot be processed", "encrypted")
                
                page_texts = []
                combined_text = ""
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        # Clean and normalize text
                        cleaned_text = self._clean_text(page_text)
                        
                        if cleaned_text.strip():  # Only add non-empty pages
                            page_texts.append(cleaned_text)
                            combined_text += f"\n--- Page {page_num + 1} ---\n{cleaned_text}\n"
                        
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        # Continue with other pages
                        continue
                
                if not combined_text.strip():
                    raise PDFProcessingError("No text could be extracted from PDF", "no_text_found")
                
                return combined_text.strip(), page_texts
                
        except PyPDF2.errors.PdfReadError as e:
            raise PDFProcessingError(f"PDF read error: {str(e)}", "corrupted")
        except Exception as e:
            raise PDFProcessingError(f"Text extraction failed: {str(e)}", "extraction")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file for deduplication
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def process_for_rag(self, extraction_result: PDFExtractionResult) -> List[Dict[str, Any]]:
        """
        Process extracted PDF text into documents suitable for RAG indexing
        
        Args:
            extraction_result: Result from extract_text method
            
        Returns:
            List of document dictionaries for RAG indexing
        """
        if not extraction_result.success:
            return []
        
        documents = []
        metadata = extraction_result.metadata
        
        # Create document-level summary
        doc_summary = {
            'doc_id': f"pdf_{metadata.file_hash}",
            'content': f"Document: {metadata.filename}. "
                      f"Pages: {metadata.page_count}. "
                      f"Company: {metadata.company_id}. "
                      f"Content: {extraction_result.text[:500]}...",  # First 500 chars
            'metadata': {
                'type': 'pdf_document',
                'filename': metadata.filename,
                'page_count': metadata.page_count,
                'file_size': metadata.file_size,
                'upload_date': metadata.upload_date.isoformat(),
                'company_id': metadata.company_id,
                'user_id': metadata.user_id,
                'file_hash': metadata.file_hash
            }
        }
        documents.append(doc_summary)
        
        # Create page-level documents for better granularity
        if extraction_result.page_texts:
            for page_num, page_text in enumerate(extraction_result.page_texts, 1):
                if len(page_text.strip()) > 50:  # Only index substantial pages
                    page_doc = {
                        'doc_id': f"pdf_{metadata.file_hash}_page_{page_num}",
                        'content': f"Document: {metadata.filename}, Page {page_num}. "
                                  f"Company: {metadata.company_id}. "
                                  f"Content: {page_text}",
                        'metadata': {
                            'type': 'pdf_page',
                            'filename': metadata.filename,
                            'page_number': page_num,
                            'total_pages': metadata.page_count,
                            'company_id': metadata.company_id,
                            'user_id': metadata.user_id,
                            'file_hash': metadata.file_hash,
                            'parent_doc_id': f"pdf_{metadata.file_hash}"
                        }
                    }
                    documents.append(page_doc)
        
        return documents
    
    def get_error_suggestions(self, error_type: str) -> Dict[str, str]:
        """
        Get user-friendly error messages and suggestions
        
        Args:
            error_type: Type of error encountered
            
        Returns:
            Dictionary with error message and suggestion
        """
        error_messages = {
            'file_not_found': {
                'message': 'The PDF file could not be found.',
                'suggestion': 'Please ensure the file was uploaded correctly and try again.'
            },
            'file_too_large': {
                'message': 'The PDF file is too large to process.',
                'suggestion': 'Please use a PDF file smaller than 50MB or split the document into smaller files.'
            },
            'unsupported_format': {
                'message': 'The file format is not supported.',
                'suggestion': 'Please upload a valid PDF file (.pdf extension).'
            },
            'encrypted': {
                'message': 'The PDF is password-protected and cannot be processed.',
                'suggestion': 'Please remove the password protection or provide an unencrypted version.'
            },
            'corrupted': {
                'message': 'The PDF file appears to be corrupted or damaged.',
                'suggestion': 'Please try re-saving the PDF or uploading a different version.'
            },
            'no_text_found': {
                'message': 'No text could be extracted from the PDF.',
                'suggestion': 'The PDF may contain only images. Try using a PDF with selectable text or convert images to text first.'
            },
            'extraction': {
                'message': 'An error occurred while extracting text from the PDF.',
                'suggestion': 'Please try uploading the file again or contact support if the problem persists.'
            }
        }
        
        return error_messages.get(error_type, {
            'message': 'An unknown error occurred while processing the PDF.',
            'suggestion': 'Please try uploading the file again or contact support.'
        })

# Global instance
pdf_processor = PDFProcessor()