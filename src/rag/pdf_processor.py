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
import traceback
import time
from pathlib import Path

# Import comprehensive logging system
from src.utils.logging_config import comprehensive_logger

# Configure logging for PDF processing
logger = logging.getLogger(__name__)

# Create a dedicated logger for PDF processing errors
pdf_error_logger = logging.getLogger('pdf_processing_errors')
pdf_error_logger.setLevel(logging.ERROR)

# Create file handler for PDF error logging (without sensitive data)
if not pdf_error_logger.handlers:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    error_handler = logging.FileHandler(log_dir / "pdf_processing_errors.log")
    error_handler.setLevel(logging.ERROR)
    
    # Format that excludes sensitive information
    error_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    error_handler.setFormatter(error_formatter)
    pdf_error_logger.addHandler(error_handler)

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
    def __init__(self, message: str, error_type: str, recoverable: bool = False):
        self.message = message
        self.error_type = error_type  # 'extraction', 'encoding', 'corrupted', 'unsupported', 'file_not_found', 'file_too_large', 'encrypted', 'no_text_found'
        self.recoverable = recoverable  # Whether the error can be recovered from with user action
        super().__init__(self.message)

class PDFProcessingWarning(Exception):
    """Warning for PDF processing issues that don't prevent processing"""
    def __init__(self, message: str, warning_type: str):
        self.message = message
        self.warning_type = warning_type
        super().__init__(self.message)

class PDFProcessor:
    """
    PDF text extraction and processing for RAG integration
    Handles various PDF formats and provides error handling
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        self.max_pages = 1000  # Maximum pages to process
        self.processing_timeout = 300  # 5 minutes timeout
        self.fallback_extractors = ['PyPDF2']  # Could add pdfplumber, pymupdf as fallbacks
        self.retry_attempts = 3
        self.min_text_length = 10  # Minimum text length to consider extraction successful
    
    def extract_text(self, file_path: str, user_id: str, company_id: str) -> PDFExtractionResult:
        """
        Extract text content from PDF file with comprehensive error handling and fallback mechanisms
        Enhanced with performance optimizations for large files
        
        Args:
            file_path: Path to PDF file
            user_id: User ID for tracking (sanitized for logging)
            company_id: Company ID for multi-tenant isolation (sanitized for logging)
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        filename = os.path.basename(file_path) if file_path else "unknown_file"
        
        # Use performance optimizer for large files
        try:
            from src.utils.performance_optimizer import pdf_optimizer
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
            
            # Use optimizer for files larger than 5MB or when performance monitoring is active
            if file_size_mb > 5:
                logger.info(f"Using performance optimizer for large PDF: {filename} ({file_size_mb:.1f}MB)")
                return pdf_optimizer.optimize_pdf_processing(self, file_path, user_id, company_id)
        except ImportError:
            logger.debug("Performance optimizer not available, using standard processing")
        except Exception as e:
            logger.warning(f"Performance optimizer failed, falling back to standard processing: {str(e)}")
        
        # Sanitize IDs for logging (remove sensitive information)
        sanitized_user_id = f"user_{hash(user_id) % 10000}" if user_id else "unknown_user"
        sanitized_company_id = f"company_{hash(company_id) % 10000}" if company_id else "unknown_company"
        
        # Initialize metadata with default values
        metadata = PDFMetadata(
            filename=filename,
            upload_date=datetime.now(),
            page_count=0,
            file_size=0,
            company_id=company_id,
            user_id=user_id,
            file_hash="",
            extraction_status="processing"
        )
        
        try:
            # Comprehensive file validation
            self._validate_pdf_file(file_path)
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            
            # Update metadata
            metadata.file_hash = file_hash
            metadata.file_size = file_size
            
            # Log processing start using comprehensive logger
            comprehensive_logger.log_file_upload_start(
                user_id=user_id,
                company_name=company_id,  # Using company_id as company_name for now
                file_path=file_path,
                file_size=file_size
            )
            
            # Extract text with retry mechanism
            extracted_text, page_texts, warnings = self._extract_pdf_text_with_retry(file_path)
            
            # Validate extraction results
            if not extracted_text or len(extracted_text.strip()) < self.min_text_length:
                raise PDFProcessingError(
                    "Insufficient text extracted from PDF. The document may contain only images or be corrupted.",
                    "no_text_found",
                    recoverable=True
                )
            
            # Update metadata with successful extraction
            metadata.page_count = len(page_texts)
            metadata.extraction_status = "success"
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Log successful processing using comprehensive logger
            comprehensive_logger.log_pdf_processing_success(
                user_id=user_id,
                company_name=company_id,
                file_path=file_path,
                duration_ms=processing_time,
                pages_processed=len(page_texts)
            )
            
            # Also log to standard logger
            logger.info(f"Successfully extracted text from {filename}: {len(extracted_text)} characters, {len(page_texts)} pages in {processing_time/1000:.2f}s")
            
            # Log warnings if any
            if warnings:
                for warning in warnings:
                    pdf_error_logger.warning(f"PDF processing warning - File: {filename}, Warning: {warning.message}, Type: {warning.warning_type}")
            
            return PDFExtractionResult(
                text=extracted_text,
                page_count=len(page_texts),
                metadata=metadata,
                success=True,
                page_texts=page_texts
            )
            
        except PDFProcessingError as e:
            # Log structured error using comprehensive logger
            comprehensive_logger.log_pdf_processing_failure(
                user_id=user_id,
                company_name=company_id,
                file_path=file_path,
                error_message=f"{e.error_type}: {e.message}"
            )
            
            # Also log to standard error logger
            processing_time = time.time() - start_time
            pdf_error_logger.error(f"PDF processing failed - File: {filename}, Error: {e.error_type}, Message: {e.message}, Recoverable: {e.recoverable}, ProcessingTime: {processing_time:.2f}s, User: {sanitized_user_id}, Company: {sanitized_company_id}")
            
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
            # Log unexpected errors using comprehensive logger
            error_message = f"Unexpected processing error: {str(e)}"
            comprehensive_logger.log_pdf_processing_failure(
                user_id=user_id,
                company_name=company_id,
                file_path=file_path,
                error_message=error_message
            )
            
            # Also log to standard error logger with stack trace
            processing_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            pdf_error_logger.error(f"Unexpected PDF processing error - File: {filename}, Error: {str(e)}, ProcessingTime: {processing_time:.2f}s, User: {sanitized_user_id}, Company: {sanitized_company_id}")
            pdf_error_logger.debug(f"Stack trace for {filename}: {error_trace}")
            
            metadata.extraction_status = "failed"
            metadata.error_message = error_message
            
            return PDFExtractionResult(
                text="",
                page_count=0,
                metadata=metadata,
                success=False,
                error=f"Unexpected processing error: {str(e)}"
            )
    
    def _validate_pdf_file(self, file_path: str) -> None:
        """
        Comprehensive PDF file validation
        
        Args:
            file_path: Path to PDF file
            
        Raises:
            PDFProcessingError: If validation fails
        """
        # Check file existence
        if not file_path or not os.path.exists(file_path):
            raise PDFProcessingError(
                "PDF file not found or path is invalid",
                "file_not_found",
                recoverable=True
            )
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise PDFProcessingError(
                "PDF file is empty",
                "corrupted",
                recoverable=True
            )
        
        if file_size > self.max_file_size:
            raise PDFProcessingError(
                f"PDF file is too large ({file_size:,} bytes). Maximum allowed size is {self.max_file_size:,} bytes",
                "file_too_large",
                recoverable=True
            )
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            raise PDFProcessingError(
                "File does not have a PDF extension",
                "unsupported_format",
                recoverable=True
            )
        
        # Basic file header validation
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    raise PDFProcessingError(
                        "File does not appear to be a valid PDF (invalid header)",
                        "corrupted",
                        recoverable=True
                    )
        except IOError as e:
            raise PDFProcessingError(
                f"Cannot read PDF file: {str(e)}",
                "file_access_error",
                recoverable=True
            )
    
    def _extract_pdf_text_with_retry(self, file_path: str) -> Tuple[str, List[str], List[PDFProcessingWarning]]:
        """
        Extract text from PDF with retry mechanism and fallback handling
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (combined_text, list_of_page_texts, warnings)
        """
        warnings = []
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                return self._extract_pdf_text(file_path, warnings)
            except PDFProcessingError as e:
                last_error = e
                if not e.recoverable or attempt == self.retry_attempts - 1:
                    raise e
                
                # Log retry attempt
                pdf_error_logger.warning(f"PDF extraction attempt {attempt + 1} failed for {os.path.basename(file_path)}: {e.message}. Retrying...")
                time.sleep(1)  # Brief delay before retry
            except Exception as e:
                last_error = e
                if attempt == self.retry_attempts - 1:
                    raise PDFProcessingError(f"Text extraction failed after {self.retry_attempts} attempts: {str(e)}", "extraction")
                
                pdf_error_logger.warning(f"PDF extraction attempt {attempt + 1} failed for {os.path.basename(file_path)}: {str(e)}. Retrying...")
                time.sleep(1)
        
        # If we get here, all retries failed
        if last_error:
            if isinstance(last_error, PDFProcessingError):
                raise last_error
            else:
                raise PDFProcessingError(f"Text extraction failed after {self.retry_attempts} attempts: {str(last_error)}", "extraction")
    
    def _extract_pdf_text(self, file_path: str, warnings: List[PDFProcessingWarning] = None) -> Tuple[str, List[str]]:
        """
        Extract text from PDF using PyPDF2 with comprehensive error handling and fallback mechanisms
        
        Args:
            file_path: Path to PDF file
            warnings: List to collect warnings during processing
            
        Returns:
            Tuple of (combined_text, list_of_page_texts)
        """
        if warnings is None:
            warnings = []
            
        try:
            with open(file_path, 'rb') as file:
                # Initialize PDF reader with error handling
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                except PyPDF2.errors.PdfReadError as e:
                    if "EOF marker not found" in str(e):
                        raise PDFProcessingError(
                            "PDF file appears to be truncated or corrupted (EOF marker not found)",
                            "corrupted",
                            recoverable=True
                        )
                    elif "Invalid PDF" in str(e):
                        raise PDFProcessingError(
                            "Invalid PDF format detected",
                            "corrupted",
                            recoverable=True
                        )
                    else:
                        raise PDFProcessingError(f"PDF read error: {str(e)}", "corrupted", recoverable=True)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    # Try to decrypt with empty password (some PDFs have owner password but no user password)
                    try:
                        pdf_reader.decrypt("")
                        warnings.append(PDFProcessingWarning(
                            "PDF was encrypted but successfully decrypted with empty password",
                            "encryption_bypassed"
                        ))
                    except:
                        raise PDFProcessingError(
                            "PDF is password-protected and cannot be processed. Please provide an unencrypted version.",
                            "encrypted",
                            recoverable=True
                        )
                
                # Check page count
                num_pages = len(pdf_reader.pages)
                if num_pages == 0:
                    raise PDFProcessingError(
                        "PDF contains no pages",
                        "no_content",
                        recoverable=False
                    )
                
                if num_pages > self.max_pages:
                    warnings.append(PDFProcessingWarning(
                        f"PDF has {num_pages} pages, processing only first {self.max_pages} pages",
                        "page_limit_exceeded"
                    ))
                    num_pages = self.max_pages
                
                page_texts = []
                combined_text = ""
                failed_pages = 0
                
                # Extract text from each page with individual error handling
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        # Clean and normalize text
                        cleaned_text = self._clean_text(page_text)
                        
                        if cleaned_text.strip():  # Only add non-empty pages
                            page_texts.append(cleaned_text)
                            combined_text += f"\n--- Page {page_num + 1} ---\n{cleaned_text}\n"
                        else:
                            # Page has no extractable text
                            warnings.append(PDFProcessingWarning(
                                f"Page {page_num + 1} contains no extractable text (may be image-based)",
                                "empty_page"
                            ))
                        
                    except Exception as e:
                        failed_pages += 1
                        warnings.append(PDFProcessingWarning(
                            f"Failed to extract text from page {page_num + 1}: {str(e)}",
                            "page_extraction_failed"
                        ))
                        
                        # If too many pages fail, consider it a critical error
                        if failed_pages > num_pages * 0.5:  # More than 50% of pages failed
                            raise PDFProcessingError(
                                f"Failed to extract text from {failed_pages} out of {num_pages} pages. PDF may be corrupted or image-based.",
                                "extraction_failure_threshold",
                                recoverable=True
                            )
                        continue
                
                # Validate extraction results
                if not combined_text.strip():
                    if failed_pages == num_pages:
                        raise PDFProcessingError(
                            "Could not extract text from any pages. PDF may contain only images or be corrupted.",
                            "no_text_found",
                            recoverable=True
                        )
                    else:
                        raise PDFProcessingError(
                            "No readable text found in PDF. Document may be image-based or use unsupported encoding.",
                            "no_text_found",
                            recoverable=True
                        )
                
                # Check if extraction was mostly successful
                if failed_pages > 0:
                    success_rate = ((num_pages - failed_pages) / num_pages) * 100
                    warnings.append(PDFProcessingWarning(
                        f"Partial extraction success: {success_rate:.1f}% of pages processed successfully",
                        "partial_extraction"
                    ))
                
                return combined_text.strip(), page_texts
                
        except PyPDF2.errors.PdfReadError as e:
            error_msg = str(e).lower()
            if "invalid pdf" in error_msg or "not a pdf file" in error_msg:
                raise PDFProcessingError(
                    "File is not a valid PDF or is corrupted",
                    "corrupted",
                    recoverable=True
                )
            elif "encrypted" in error_msg:
                raise PDFProcessingError(
                    "PDF is encrypted and cannot be processed",
                    "encrypted",
                    recoverable=True
                )
            else:
                raise PDFProcessingError(f"PDF read error: {str(e)}", "corrupted", recoverable=True)
                
        except MemoryError:
            raise PDFProcessingError(
                "PDF file is too large to process in available memory. Please try a smaller file.",
                "memory_error",
                recoverable=True
            )
            
        except IOError as e:
            raise PDFProcessingError(
                f"File access error: {str(e)}",
                "file_access_error",
                recoverable=True
            )
            
        except Exception as e:
            raise PDFProcessingError(f"Unexpected text extraction error: {str(e)}", "extraction", recoverable=False)
    
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
    
    def get_error_suggestions(self, error_type: str) -> Dict[str, Any]:
        """
        Get comprehensive user-friendly error messages, suggestions, and recovery options
        
        Args:
            error_type: Type of error encountered
            
        Returns:
            Dictionary with error message, suggestion, recovery options, and technical details
        """
        error_info = {
            'file_not_found': {
                'message': 'The PDF file could not be found or accessed.',
                'suggestion': 'Please ensure the file was uploaded correctly and try again.',
                'recovery_options': [
                    'Verify the file exists and is accessible',
                    'Try uploading the file again',
                    'Check file permissions'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'file_too_large': {
                'message': 'The PDF file exceeds the maximum allowed size (50MB).',
                'suggestion': 'Please reduce the file size or split the document into smaller parts.',
                'recovery_options': [
                    'Compress the PDF using online tools or PDF software',
                    'Split the document into multiple smaller files',
                    'Remove high-resolution images from the PDF',
                    'Convert to a more efficient PDF format'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'unsupported_format': {
                'message': 'The file format is not supported or the file is not a valid PDF.',
                'suggestion': 'Please upload a valid PDF file with .pdf extension.',
                'recovery_options': [
                    'Ensure the file has a .pdf extension',
                    'Convert the document to PDF format',
                    'Verify the file is not corrupted during upload'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'encrypted': {
                'message': 'The PDF is password-protected and cannot be processed.',
                'suggestion': 'Please provide an unencrypted version of the document.',
                'recovery_options': [
                    'Remove password protection using PDF software',
                    'Save/export as a new unencrypted PDF',
                    'Contact the document owner for an unprotected version'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'corrupted': {
                'message': 'The PDF file appears to be corrupted, damaged, or has an invalid format.',
                'suggestion': 'Please try re-creating or re-saving the PDF document.',
                'recovery_options': [
                    'Re-save the PDF from the original application',
                    'Try opening and re-exporting the PDF',
                    'Use PDF repair tools to fix corruption',
                    'Upload a different version of the document'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'no_text_found': {
                'message': 'No readable text could be extracted from the PDF.',
                'suggestion': 'The PDF may contain only images or use unsupported text encoding.',
                'recovery_options': [
                    'Ensure the PDF contains selectable text (not just images)',
                    'Use OCR software to convert image-based PDFs to text',
                    'Re-create the PDF with text-based content',
                    'Try a different PDF creation method'
                ],
                'severity': 'warning',
                'recoverable': True
            },
            'no_content': {
                'message': 'The PDF file contains no pages or content.',
                'suggestion': 'Please upload a PDF file that contains actual content.',
                'recovery_options': [
                    'Verify the PDF has pages with content',
                    'Check if the PDF was created correctly',
                    'Try uploading a different document'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'memory_error': {
                'message': 'The PDF file is too large to process in available memory.',
                'suggestion': 'Please try a smaller file or split the document.',
                'recovery_options': [
                    'Split the PDF into smaller documents',
                    'Reduce image quality in the PDF',
                    'Remove unnecessary pages or content',
                    'Try processing during off-peak hours'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'extraction_failure_threshold': {
                'message': 'Text extraction failed for most pages in the PDF.',
                'suggestion': 'The document may be image-based or use unsupported formatting.',
                'recovery_options': [
                    'Use OCR software to convert the PDF to searchable text',
                    'Try a different PDF creation method',
                    'Verify the PDF quality and formatting',
                    'Contact support if the issue persists'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'page_extraction_failed': {
                'message': 'Some pages could not be processed, but extraction partially succeeded.',
                'suggestion': 'The document was partially processed. Some content may be missing.',
                'recovery_options': [
                    'Review the extracted content for completeness',
                    'Try re-creating problematic pages',
                    'Split the document and process pages separately'
                ],
                'severity': 'warning',
                'recoverable': True
            },
            'file_access_error': {
                'message': 'Unable to access or read the PDF file.',
                'suggestion': 'There may be a file permission or system access issue.',
                'recovery_options': [
                    'Try uploading the file again',
                    'Check if the file is being used by another application',
                    'Verify file permissions',
                    'Contact system administrator if the issue persists'
                ],
                'severity': 'error',
                'recoverable': True
            },
            'extraction': {
                'message': 'An unexpected error occurred during text extraction.',
                'suggestion': 'Please try uploading the file again or contact support.',
                'recovery_options': [
                    'Try uploading the file again',
                    'Wait a few minutes and retry',
                    'Try a different PDF if available',
                    'Contact technical support with error details'
                ],
                'severity': 'error',
                'recoverable': False
            }
        }
        
        return error_info.get(error_type, {
            'message': 'An unknown error occurred while processing the PDF.',
            'suggestion': 'Please try uploading the file again or contact support.',
            'recovery_options': [
                'Try uploading the file again',
                'Verify the file is a valid PDF',
                'Contact technical support'
            ],
            'severity': 'error',
            'recoverable': False
        })
    
    def get_fallback_suggestions(self, file_path: str) -> Dict[str, Any]:
        """
        Provide fallback suggestions when PDF processing fails
        
        Args:
            file_path: Path to the failed PDF file
            
        Returns:
            Dictionary with alternative processing suggestions
        """
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        suggestions = {
            'alternative_formats': [
                'Convert PDF to plain text (.txt) format',
                'Export content as Microsoft Word document',
                'Save as HTML format for web-based content'
            ],
            'preprocessing_options': [
                'Use OCR software (like Adobe Acrobat Pro) to make text searchable',
                'Flatten the PDF to remove complex formatting',
                'Reduce file size by compressing images'
            ],
            'manual_alternatives': [
                'Copy and paste text content directly into the system',
                'Upload individual pages as separate files',
                'Provide a summary document with key information'
            ],
            'technical_details': {
                'filename': filename,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'supported_formats': self.supported_extensions,
                'max_file_size_mb': round(self.max_file_size / (1024 * 1024), 2),
                'max_pages': self.max_pages
            }
        }
        
        return suggestions

# Global instance
pdf_processor = PDFProcessor()