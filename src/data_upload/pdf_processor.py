"""
PDF Processing Engine for Document Upload and Text Extraction
"""

import os
import PyPDF2
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PDFMetadata:
    """Metadata extracted from PDF files"""
    filename: str
    file_size: int
    page_count: int
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    author: Optional[str]
    title: Optional[str]
    subject: Optional[str]
    producer: Optional[str]

@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction"""
    success: bool
    text: str
    metadata: Optional[PDFMetadata]
    page_texts: List[str]
    error_message: Optional[str]
    processing_time: float

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    def __init__(self, message: str, error_type: str):
        self.message = message
        self.error_type = error_type  # 'extraction', 'encoding', 'corrupted', 'access'
        super().__init__(self.message)

class PDFProcessor:
    """
    PDF text extraction and processing engine
    Handles various PDF formats and provides comprehensive error handling
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        self.max_pages = 1000  # Maximum pages to process
    
    def extract_text(self, file_path: str) -> PDFExtractionResult:
        """
        Extract text content from PDF file with comprehensive error handling
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        start_time = datetime.now()
        
        try:
            # Validate file
            self._validate_pdf_file(file_path)
            
            # Extract metadata first
            metadata = self._extract_metadata(file_path)
            
            # Extract text content
            text_content, page_texts = self._extract_text_content(file_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PDFExtractionResult(
                success=True,
                text=text_content,
                metadata=metadata,
                page_texts=page_texts,
                error_message=None,
                processing_time=processing_time
            )
            
        except PDFProcessingError as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"PDF processing error: {e.message} (Type: {e.error_type})")
            
            return PDFExtractionResult(
                success=False,
                text="",
                metadata=None,
                page_texts=[],
                error_message=e.message,
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Unexpected error processing PDF: {str(e)}")
            
            return PDFExtractionResult(
                success=False,
                text="",
                metadata=None,
                page_texts=[],
                error_message=f"Unexpected error: {str(e)}",
                processing_time=processing_time
            )
    
    def _validate_pdf_file(self, file_path: str) -> None:
        """Validate PDF file before processing"""
        if not os.path.exists(file_path):
            raise PDFProcessingError(f"File not found: {file_path}", "access")
        
        if not file_path.lower().endswith('.pdf'):
            raise PDFProcessingError("File is not a PDF", "format")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise PDFProcessingError(
                f"File too large: {file_size / (1024*1024):.1f}MB (max: {self.max_file_size / (1024*1024)}MB)",
                "size"
            )
        
        if file_size == 0:
            raise PDFProcessingError("File is empty", "corrupted")
    
    def _extract_metadata(self, file_path: str) -> PDFMetadata:
        """Extract metadata from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get file stats
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                modification_date = datetime.fromtimestamp(file_stats.st_mtime)
                
                # Get PDF metadata
                metadata_dict = pdf_reader.metadata if pdf_reader.metadata else {}
                
                # Extract creation date
                creation_date = None
                if '/CreationDate' in metadata_dict:
                    try:
                        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
                        date_str = str(metadata_dict['/CreationDate'])
                        if date_str.startswith('D:'):
                            date_str = date_str[2:16]  # Extract YYYYMMDDHHMMSS
                            creation_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                    except:
                        pass
                
                return PDFMetadata(
                    filename=os.path.basename(file_path),
                    file_size=file_size,
                    page_count=len(pdf_reader.pages),
                    creation_date=creation_date,
                    modification_date=modification_date,
                    author=metadata_dict.get('/Author', '').strip() if metadata_dict.get('/Author') else None,
                    title=metadata_dict.get('/Title', '').strip() if metadata_dict.get('/Title') else None,
                    subject=metadata_dict.get('/Subject', '').strip() if metadata_dict.get('/Subject') else None,
                    producer=metadata_dict.get('/Producer', '').strip() if metadata_dict.get('/Producer') else None
                )
                
        except Exception as e:
            logger.warning(f"Could not extract metadata: {str(e)}")
            # Return basic metadata with file stats
            file_stats = os.stat(file_path)
            return PDFMetadata(
                filename=os.path.basename(file_path),
                file_size=file_stats.st_size,
                page_count=0,
                creation_date=None,
                modification_date=datetime.fromtimestamp(file_stats.st_mtime),
                author=None,
                title=None,
                subject=None,
                producer=None
            )
    
    def _extract_text_content(self, file_path: str) -> tuple[str, List[str]]:
        """Extract text content from PDF pages"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) > self.max_pages:
                    raise PDFProcessingError(
                        f"PDF has too many pages: {len(pdf_reader.pages)} (max: {self.max_pages})",
                        "size"
                    )
                
                page_texts = []
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        # Clean and normalize text
                        page_text = self._clean_text(page_text)
                        
                        page_texts.append(page_text)
                        full_text += page_text + "\n\n"
                        
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                        page_texts.append("")
                
                # Final text cleaning
                full_text = self._clean_text(full_text)
                
                if not full_text.strip():
                    raise PDFProcessingError(
                        "No text content could be extracted. PDF may be image-based or corrupted.",
                        "extraction"
                    )
                
                return full_text, page_texts
                
        except PyPDF2.errors.PdfReadError as e:
            raise PDFProcessingError(f"PDF file is corrupted or invalid: {str(e)}", "corrupted")
        except Exception as e:
            if "extraction" in str(e).lower():
                raise PDFProcessingError(str(e), "extraction")
            else:
                raise PDFProcessingError(f"Text extraction failed: {str(e)}", "extraction")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters but keep newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def process_for_rag(self, file_path: str, user_id: str, company_id: str) -> Dict[str, Any]:
        """
        Process PDF for RAG integration
        
        Args:
            file_path: Path to the PDF file
            user_id: User identifier
            company_id: Company identifier
            
        Returns:
            Dictionary with processing results and metadata
        """
        extraction_result = self.extract_text(file_path)
        
        if not extraction_result.success:
            return {
                'success': False,
                'error': extraction_result.error_message,
                'processing_time': extraction_result.processing_time
            }
        
        # Prepare document for RAG storage
        document_data = {
            'success': True,
            'user_id': user_id,
            'company_id': company_id,
            'filename': extraction_result.metadata.filename,
            'text_content': extraction_result.text,
            'page_texts': extraction_result.page_texts,
            'metadata': {
                'file_size': extraction_result.metadata.file_size,
                'page_count': extraction_result.metadata.page_count,
                'creation_date': extraction_result.metadata.creation_date.isoformat() if extraction_result.metadata.creation_date else None,
                'modification_date': extraction_result.metadata.modification_date.isoformat() if extraction_result.metadata.modification_date else None,
                'author': extraction_result.metadata.author,
                'title': extraction_result.metadata.title,
                'subject': extraction_result.metadata.subject,
                'producer': extraction_result.metadata.producer
            },
            'processing_time': extraction_result.processing_time,
            'processed_at': datetime.now().isoformat()
        }
        
        return document_data
    
    def get_error_suggestion(self, error_type: str) -> str:
        """Get user-friendly error suggestions"""
        suggestions = {
            'extraction': 'The PDF may be image-based or have complex formatting. Try converting it to text format first.',
            'encoding': 'The PDF uses an unsupported encoding. Try saving it in a different format or using a different PDF creator.',
            'corrupted': 'The PDF file appears to be corrupted. Try re-downloading or re-creating the file.',
            'access': 'Cannot access the file. Check that the file exists and you have permission to read it.',
            'size': 'The file is too large. Try splitting it into smaller documents or compressing it.',
            'format': 'Only PDF files are supported. Please convert your document to PDF format.'
        }
        
        return suggestions.get(error_type, 'Please try uploading a different PDF file or contact support.')

# Create global instance
pdf_processor = PDFProcessor()