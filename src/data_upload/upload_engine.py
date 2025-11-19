"""
Multi-format Data Upload and Processing Engine
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
from dataclasses import dataclass
from .pdf_processor import pdf_processor, PDFExtractionResult

@dataclass
class UploadResult:
    success: bool
    message: str
    processed_records: int
    data_quality_score: float
    detected_columns: List[str]
    file_path: str

class DataUploadEngine:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.pdf']
        self.required_columns = {
            'sales': ['date', 'product', 'quantity', 'revenue'],
            'inventory': ['product', 'stock_level', 'reorder_point'],
            'customers': ['customer_id', 'transaction_date', 'amount']
        }
    
    def process_upload(self, file_path: str, user_id: str, data_type: str = 'sales') -> UploadResult:
        """Process uploaded file and store in user's directory"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                return UploadResult(False, f"Unsupported format: {file_ext}", 0, 0.0, [], "")
            
            # Handle PDF files differently
            if file_ext == '.pdf':
                return self._process_pdf_upload(file_path, user_id)
            
            # Handle structured data files (CSV, Excel, JSON)
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            
            df_cleaned = self._clean_data(df)
            quality_score = self._calculate_quality_score(df_cleaned, data_type)
            
            output_path = f"data/users/{user_id}/processed_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_cleaned.to_csv(output_path, index=False)
            
            return UploadResult(
                success=True,
                message="Data processed successfully",
                processed_records=len(df_cleaned),
                data_quality_score=quality_score,
                detected_columns=list(df_cleaned.columns),
                file_path=output_path
            )
            
        except Exception as e:
            return UploadResult(False, f"Processing error: {str(e)}", 0, 0.0, [], "")
    
    def _process_pdf_upload(self, file_path: str, user_id: str) -> UploadResult:
        """Process PDF file upload and extract text content"""
        try:
            # Extract text using PDF processor
            extraction_result = pdf_processor.extract_text(file_path)
            
            if not extraction_result.success:
                return UploadResult(
                    success=False,
                    message=f"PDF processing failed: {extraction_result.error_message}",
                    processed_records=0,
                    data_quality_score=0.0,
                    detected_columns=[],
                    file_path=""
                )
            
            # Save extracted text to user's directory
            pdf_dir = f"data/users/{user_id}/pdf"
            os.makedirs(pdf_dir, exist_ok=True)
            
            # Save the original PDF
            filename = os.path.basename(file_path)
            pdf_output_path = os.path.join(pdf_dir, filename)
            
            # Copy PDF to user directory if not already there
            if os.path.abspath(file_path) != os.path.abspath(pdf_output_path):
                import shutil
                shutil.copy2(file_path, pdf_output_path)
            
            # Save extracted text as a separate file for easy access
            text_filename = f"{os.path.splitext(filename)[0]}_extracted_text.txt"
            text_output_path = os.path.join(pdf_dir, text_filename)
            
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(extraction_result.text)
            
            # Calculate quality score based on text content and metadata
            quality_score = self._calculate_pdf_quality_score(extraction_result)
            
            return UploadResult(
                success=True,
                message=f"PDF processed successfully. Extracted {len(extraction_result.text)} characters from {extraction_result.metadata.page_count} pages.",
                processed_records=extraction_result.metadata.page_count,
                data_quality_score=quality_score,
                detected_columns=[f"Page {i+1}" for i in range(extraction_result.metadata.page_count)],
                file_path=pdf_output_path
            )
            
        except Exception as e:
            return UploadResult(False, f"PDF processing error: {str(e)}", 0, 0.0, [], "")
    
    def _calculate_pdf_quality_score(self, extraction_result: PDFExtractionResult) -> float:
        """Calculate quality score for PDF extraction"""
        if not extraction_result.success or not extraction_result.text:
            return 0.0
        
        score = 1.0
        
        # Check text content quality
        text_length = len(extraction_result.text.strip())
        if text_length < 100:
            score *= 0.5  # Very short text
        elif text_length < 500:
            score *= 0.7  # Short text
        
        # Check for successful page extraction
        successful_pages = sum(1 for page_text in extraction_result.page_texts if page_text.strip())
        if extraction_result.metadata.page_count > 0:
            page_success_rate = successful_pages / extraction_result.metadata.page_count
            score *= page_success_rate
        
        # Bonus for metadata availability
        if extraction_result.metadata.title or extraction_result.metadata.author:
            score = min(1.0, score + 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        df = df.drop_duplicates()
        df = df.dropna(thresh=len(df.columns) * 0.7)
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        for col in df.columns:
            if 'date' in col or 'time' in col:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
    
    def _calculate_quality_score(self, df: pd.DataFrame, data_type: str) -> float:
        """Calculate data quality score (0-1)"""
        score = 1.0
        
        required = self.required_columns.get(data_type, [])
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            score -= 0.3
        
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        score *= completeness
        
        if len(df) != len(df.drop_duplicates()):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_data_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's uploaded data"""
        user_dir = f"data/users/{user_id}"
        if not os.path.exists(user_dir):
            return {"files": [], "total_records": 0, "pdf_files": [], "csv_files": []}
        
        csv_files = []
        pdf_files = []
        total_records = 0
        
        # Check CSV files in main directory
        for filename in os.listdir(user_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(user_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    csv_files.append({
                        "filename": filename,
                        "type": "csv",
                        "records": len(df),
                        "columns": list(df.columns),
                        "upload_date": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
                    total_records += len(df)
                except:
                    pass
        
        # Check PDF files in pdf subdirectory
        pdf_dir = os.path.join(user_dir, "pdf")
        if os.path.exists(pdf_dir):
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(pdf_dir, filename)
                    try:
                        # Get basic file info
                        file_stats = os.stat(file_path)
                        
                        # Try to get page count from metadata if available
                        try:
                            extraction_result = pdf_processor.extract_text(file_path)
                            page_count = extraction_result.metadata.page_count if extraction_result.metadata else 0
                            text_length = len(extraction_result.text) if extraction_result.text else 0
                        except:
                            page_count = 0
                            text_length = 0
                        
                        pdf_files.append({
                            "filename": filename,
                            "type": "pdf",
                            "records": page_count,  # Use page count as "records" for PDFs
                            "file_size": file_stats.st_size,
                            "text_length": text_length,
                            "upload_date": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                        })
                        total_records += page_count
                    except:
                        pass
        
        # Combine all files
        all_files = csv_files + pdf_files
        
        return {
            "files": all_files,
            "csv_files": csv_files,
            "pdf_files": pdf_files,
            "total_records": total_records,
            "total_csv_files": len(csv_files),
            "total_pdf_files": len(pdf_files)
        }

upload_engine = DataUploadEngine()