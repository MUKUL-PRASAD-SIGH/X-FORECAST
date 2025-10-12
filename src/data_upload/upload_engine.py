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
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
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
            return {"files": [], "total_records": 0}
        
        files = []
        total_records = 0
        
        for filename in os.listdir(user_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(user_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    files.append({
                        "filename": filename,
                        "records": len(df),
                        "columns": list(df.columns),
                        "upload_date": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
                    total_records += len(df)
                except:
                    pass
        
        return {"files": files, "total_records": total_records}

upload_engine = DataUploadEngine()