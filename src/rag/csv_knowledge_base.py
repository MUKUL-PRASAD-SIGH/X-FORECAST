"""
CSV Knowledge Base with RAG (Retrieval-Augmented Generation)
Persistent storage and intelligent querying of uploaded CSV data
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about uploaded dataset"""
    dataset_id: str
    user_id: str
    filename: str
    upload_date: datetime
    columns: List[str]
    row_count: int
    data_types: Dict[str, str]
    summary_stats: Dict[str, Any]
    sample_data: List[Dict[str, Any]]
    suggested_questions: List[str]

class CSVKnowledgeBase:
    """
    Persistent CSV Knowledge Base with RAG capabilities
    """
    
    def __init__(self, storage_dir: str = "data/knowledge_base"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.storage_dir / "knowledge_base.db"
        self.init_database()
        
        # In-memory cache for quick access
        self.datasets_cache = {}
        self.load_datasets_cache()
    
    def init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create datasets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                columns TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                data_types TEXT NOT NULL,
                summary_stats TEXT NOT NULL,
                sample_data TEXT NOT NULL,
                suggested_questions TEXT NOT NULL
            )
        ''')
        
        # Create user_datasets index
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_datasets 
            ON datasets(user_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_dataset(self, user_id: str, csv_file_path: str, filename: str = None) -> DatasetInfo:
        """Add new CSV dataset to knowledge base"""
        
        try:
            # Load CSV data
            df = pd.read_csv(csv_file_path)
            
            # Generate dataset ID
            dataset_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(filename or csv_file_path) % 10000}"
            
            # Analyze dataset
            dataset_info = self._analyze_dataset(
                dataset_id=dataset_id,
                user_id=user_id,
                df=df,
                filename=filename or os.path.basename(csv_file_path)
            )
            
            # Save dataset file
            dataset_file_path = self.storage_dir / f"{dataset_id}.csv"
            df.to_csv(dataset_file_path, index=False)
            
            # Save to database
            self._save_dataset_to_db(dataset_info)
            
            # Update cache
            self.datasets_cache[dataset_id] = dataset_info
            
            logger.info(f"Added dataset {dataset_id} with {len(df)} rows")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Failed to add dataset: {e}")
            raise
    
    def _analyze_dataset(self, dataset_id: str, user_id: str, df: pd.DataFrame, filename: str) -> DatasetInfo:
        """Analyze dataset and extract metadata"""
        
        # Basic info
        columns = df.columns.tolist()
        row_count = len(df)
        data_types = {col: str(df[col].dtype) for col in columns}
        
        # Summary statistics
        summary_stats = {}
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                summary_stats[col] = {
                    'type': 'numeric',
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()) if df[col].std() == df[col].std() else 0.0  # Handle NaN
                }
            else:
                unique_values = df[col].unique()
                summary_stats[col] = {
                    'type': 'categorical',
                    'unique_count': len(unique_values),
                    'top_values': unique_values[:10].tolist() if len(unique_values) > 0 else []
                }
        
        # Sample data (first 5 rows)
        sample_data = df.head(5).to_dict('records')
        
        # Generate suggested questions
        suggested_questions = self._generate_suggested_questions(df, columns, summary_stats)
        
        return DatasetInfo(
            dataset_id=dataset_id,
            user_id=user_id,
            filename=filename,
            upload_date=datetime.now(),
            columns=columns,
            row_count=row_count,
            data_types=data_types,
            summary_stats=summary_stats,
            sample_data=sample_data,
            suggested_questions=suggested_questions
        )
    
    def _generate_suggested_questions(self, df: pd.DataFrame, columns: List[str], 
                                    summary_stats: Dict[str, Any]) -> List[str]:
        """Generate intelligent suggested questions based on data"""
        
        questions = []
        
        # Date-based questions
        date_columns = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            questions.extend([
                f"What's the trend over time for {date_columns[0]}?",
                f"Show me monthly patterns in the data",
                f"What happened in the latest period?"
            ])
        
        # Numeric columns questions
        numeric_columns = [col for col, stats in summary_stats.items() if stats['type'] == 'numeric']
        if numeric_columns:
            for col in numeric_columns[:3]:  # Top 3 numeric columns
                questions.extend([
                    f"What's the average {col}?",
                    f"Show me the distribution of {col}",
                    f"What are the top values for {col}?"
                ])
        
        # Categorical columns questions
        categorical_columns = [col for col, stats in summary_stats.items() if stats['type'] == 'categorical']
        if categorical_columns:
            for col in categorical_columns[:2]:  # Top 2 categorical columns
                questions.extend([
                    f"What are the most common {col} values?",
                    f"How many unique {col} are there?",
                    f"Show me breakdown by {col}"
                ])
        
        # Business-specific questions based on column names
        business_questions = []
        
        # Sales/Revenue related
        if any(word in ' '.join(columns).lower() for word in ['sales', 'revenue', 'amount', 'price']):
            business_questions.extend([
                "What's our total revenue?",
                "Which products/categories perform best?",
                "What's the sales trend?"
            ])
        
        # Customer related
        if any(word in ' '.join(columns).lower() for word in ['customer', 'client', 'user']):
            business_questions.extend([
                "How many customers do we have?",
                "What's the customer distribution?",
                "Who are our top customers?"
            ])
        
        # Product related
        if any(word in ' '.join(columns).lower() for word in ['product', 'item', 'sku']):
            business_questions.extend([
                "What are our top-selling products?",
                "How many products do we have?",
                "Which products need attention?"
            ])
        
        # Inventory related
        if any(word in ' '.join(columns).lower() for word in ['stock', 'inventory', 'quantity']):
            business_questions.extend([
                "What's our current stock level?",
                "Which items are low in stock?",
                "What's the inventory turnover?"
            ])
        
        questions.extend(business_questions)
        
        # Remove duplicates and limit to 15 questions
        unique_questions = list(dict.fromkeys(questions))[:15]
        
        return unique_questions
    
    def _save_dataset_to_db(self, dataset_info: DatasetInfo):
        """Save dataset info to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO datasets 
            (dataset_id, user_id, filename, upload_date, columns, row_count, 
             data_types, summary_stats, sample_data, suggested_questions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_info.dataset_id,
            dataset_info.user_id,
            dataset_info.filename,
            dataset_info.upload_date.isoformat(),
            json.dumps(dataset_info.columns),
            dataset_info.row_count,
            json.dumps(dataset_info.data_types),
            json.dumps(dataset_info.summary_stats),
            json.dumps(dataset_info.sample_data),
            json.dumps(dataset_info.suggested_questions)
        ))
        
        conn.commit()
        conn.close()
    
    def load_datasets_cache(self):
        """Load all datasets into memory cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM datasets')
        rows = cursor.fetchall()
        
        for row in rows:
            dataset_info = DatasetInfo(
                dataset_id=row[0],
                user_id=row[1],
                filename=row[2],
                upload_date=datetime.fromisoformat(row[3]),
                columns=json.loads(row[4]),
                row_count=row[5],
                data_types=json.loads(row[6]),
                summary_stats=json.loads(row[7]),
                sample_data=json.loads(row[8]),
                suggested_questions=json.loads(row[9])
            )
            self.datasets_cache[dataset_info.dataset_id] = dataset_info
        
        conn.close()
    
    def get_user_datasets(self, user_id: str) -> List[DatasetInfo]:
        """Get all datasets for a user"""
        return [
            dataset for dataset in self.datasets_cache.values()
            if dataset.user_id == user_id
        ]
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get specific dataset info"""
        return self.datasets_cache.get(dataset_id)
    
    def load_dataset_data(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load actual CSV data for a dataset"""
        dataset_file_path = self.storage_dir / f"{dataset_id}.csv"
        if dataset_file_path.exists():
            return pd.read_csv(dataset_file_path)
        return None
    
    def query_data(self, user_id: str, query: str) -> Dict[str, Any]:
        """Query user's data using natural language"""
        
        user_datasets = self.get_user_datasets(user_id)
        if not user_datasets:
            return {
                "success": False,
                "message": "No datasets found. Please upload some data first.",
                "suggestions": ["Upload your CSV data to get started!"]
            }
        
        # Find most relevant dataset
        relevant_dataset = self._find_relevant_dataset(query, user_datasets)
        
        if not relevant_dataset:
            return {
                "success": False,
                "message": "Couldn't find relevant data for your query.",
                "suggestions": self._get_all_suggestions(user_datasets)
            }
        
        # Load dataset
        df = self.load_dataset_data(relevant_dataset.dataset_id)
        if df is None:
            return {
                "success": False,
                "message": "Dataset file not found.",
                "suggestions": []
            }
        
        # Process query
        result = self._process_query(query, df, relevant_dataset)
        
        return result
    
    def _find_relevant_dataset(self, query: str, datasets: List[DatasetInfo]) -> Optional[DatasetInfo]:
        """Find most relevant dataset for the query"""
        
        query_lower = query.lower()
        
        # Score datasets based on column name matches
        best_score = 0
        best_dataset = None
        
        for dataset in datasets:
            score = 0
            
            # Check column names
            for col in dataset.columns:
                if col.lower() in query_lower:
                    score += 2
                
                # Check for partial matches
                col_words = col.lower().split('_')
                for word in col_words:
                    if word in query_lower:
                        score += 1
            
            # Check filename
            if dataset.filename.lower().replace('.csv', '') in query_lower:
                score += 3
            
            if score > best_score:
                best_score = score
                best_dataset = dataset
        
        # If no good match, return the most recent dataset
        if best_score == 0 and datasets:
            return max(datasets, key=lambda d: d.upload_date)
        
        return best_dataset
    
    def _process_query(self, query: str, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Process natural language query against dataset"""
        
        query_lower = query.lower()
        
        try:
            # Summary queries
            if any(word in query_lower for word in ['summary', 'overview', 'describe']):
                return self._generate_summary(df, dataset_info)
            
            # Count queries
            if any(word in query_lower for word in ['count', 'how many', 'total']):
                return self._handle_count_query(query_lower, df, dataset_info)
            
            # Average/mean queries
            if any(word in query_lower for word in ['average', 'mean', 'avg']):
                return self._handle_average_query(query_lower, df, dataset_info)
            
            # Top/maximum queries
            if any(word in query_lower for word in ['top', 'highest', 'maximum', 'max', 'best']):
                return self._handle_top_query(query_lower, df, dataset_info)
            
            # Trend queries
            if any(word in query_lower for word in ['trend', 'over time', 'pattern', 'change']):
                return self._handle_trend_query(query_lower, df, dataset_info)
            
            # Distribution queries
            if any(word in query_lower for word in ['distribution', 'breakdown', 'split']):
                return self._handle_distribution_query(query_lower, df, dataset_info)
            
            # Default: return basic info
            return self._generate_basic_info(df, dataset_info)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}",
                "suggestions": dataset_info.suggested_questions[:5]
            }
    
    def _generate_summary(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Generate dataset summary"""
        
        summary = {
            "success": True,
            "message": f"📊 **Dataset Summary: {dataset_info.filename}**",
            "data": {
                "total_records": len(df),
                "columns": len(df.columns),
                "date_range": None,
                "key_insights": []
            },
            "suggestions": dataset_info.suggested_questions[:5]
        }
        
        # Add date range if available
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            try:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                summary["data"]["date_range"] = {
                    "start": df[date_col].min().strftime('%Y-%m-%d'),
                    "end": df[date_col].max().strftime('%Y-%m-%d')
                }
            except:
                pass
        
        # Key insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                summary["data"]["key_insights"].append(
                    f"{col}: avg {df[col].mean():.2f}, max {df[col].max():.2f}"
                )
        
        return summary
    
    def _handle_count_query(self, query: str, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Handle count-related queries"""
        
        # Find relevant column
        for col in df.columns:
            if col.lower() in query:
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    return {
                        "success": True,
                        "message": f"📊 **Count Analysis for {col}**",
                        "data": {
                            "unique_values": unique_count,
                            "total_records": len(df),
                            "top_values": df[col].value_counts().head().to_dict()
                        },
                        "suggestions": dataset_info.suggested_questions[:5]
                    }
        
        # Default: total records
        return {
            "success": True,
            "message": f"📊 **Total Records: {len(df):,}**",
            "data": {"total_records": len(df)},
            "suggestions": dataset_info.suggested_questions[:5]
        }
    
    def _handle_average_query(self, query: str, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Handle average-related queries"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Find relevant numeric column
        for col in numeric_cols:
            if col.lower() in query:
                avg_value = df[col].mean()
                return {
                    "success": True,
                    "message": f"📊 **Average {col}: {avg_value:.2f}**",
                    "data": {
                        "column": col,
                        "average": avg_value,
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "std": df[col].std()
                    },
                    "suggestions": dataset_info.suggested_questions[:5]
                }
        
        # Default: show averages for all numeric columns
        if len(numeric_cols) > 0:
            averages = {col: df[col].mean() for col in numeric_cols}
            return {
                "success": True,
                "message": "📊 **Average Values**",
                "data": {"averages": averages},
                "suggestions": dataset_info.suggested_questions[:5]
            }
        
        return {
            "success": False,
            "message": "No numeric columns found for average calculation.",
            "suggestions": dataset_info.suggested_questions[:5]
        }
    
    def _handle_top_query(self, query: str, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Handle top/maximum queries"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Find relevant column
        for col in df.columns:
            if col.lower() in query:
                if col in numeric_cols:
                    top_value = df[col].max()
                    top_record = df[df[col] == top_value].iloc[0]
                    return {
                        "success": True,
                        "message": f"📊 **Top {col}: {top_value}**",
                        "data": {
                            "column": col,
                            "max_value": top_value,
                            "record": top_record.to_dict()
                        },
                        "suggestions": dataset_info.suggested_questions[:5]
                    }
                else:
                    top_values = df[col].value_counts().head()
                    return {
                        "success": True,
                        "message": f"📊 **Top {col} Values**",
                        "data": {
                            "column": col,
                            "top_values": top_values.to_dict()
                        },
                        "suggestions": dataset_info.suggested_questions[:5]
                    }
        
        return {
            "success": False,
            "message": "Please specify which column you want to see the top values for.",
            "suggestions": dataset_info.suggested_questions[:5]
        }
    
    def _handle_trend_query(self, query: str, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Handle trend analysis queries"""
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        if not date_columns:
            return {
                "success": False,
                "message": "No date columns found for trend analysis.",
                "suggestions": dataset_info.suggested_questions[:5]
            }
        
        try:
            date_col = date_columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Find numeric column for trend
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {
                    "success": False,
                    "message": "No numeric columns found for trend analysis.",
                    "suggestions": dataset_info.suggested_questions[:5]
                }
            
            trend_col = numeric_cols[0]  # Use first numeric column
            
            # Group by date and calculate trend
            daily_trend = df.groupby(df[date_col].dt.date)[trend_col].sum()
            
            return {
                "success": True,
                "message": f"📈 **Trend Analysis: {trend_col} over {date_col}**",
                "data": {
                    "trend_column": trend_col,
                    "date_column": date_col,
                    "trend_data": daily_trend.tail(10).to_dict(),
                    "total_change": daily_trend.iloc[-1] - daily_trend.iloc[0] if len(daily_trend) > 1 else 0
                },
                "suggestions": dataset_info.suggested_questions[:5]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error in trend analysis: {str(e)}",
                "suggestions": dataset_info.suggested_questions[:5]
            }
    
    def _handle_distribution_query(self, query: str, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Handle distribution/breakdown queries"""
        
        # Find relevant categorical column
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col.lower() in query:
                distribution = df[col].value_counts()
                return {
                    "success": True,
                    "message": f"📊 **Distribution of {col}**",
                    "data": {
                        "column": col,
                        "distribution": distribution.to_dict(),
                        "unique_count": len(distribution)
                    },
                    "suggestions": dataset_info.suggested_questions[:5]
                }
        
        # Default: show distribution of first categorical column
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            distribution = df[col].value_counts()
            return {
                "success": True,
                "message": f"📊 **Distribution of {col}**",
                "data": {
                    "column": col,
                    "distribution": distribution.to_dict()
                },
                "suggestions": dataset_info.suggested_questions[:5]
            }
        
        return {
            "success": False,
            "message": "No categorical columns found for distribution analysis.",
            "suggestions": dataset_info.suggested_questions[:5]
        }
    
    def _generate_basic_info(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Generate basic dataset information"""
        
        return {
            "success": True,
            "message": f"📊 **Dataset: {dataset_info.filename}**",
            "data": {
                "records": len(df),
                "columns": df.columns.tolist(),
                "sample": df.head(3).to_dict('records')
            },
            "suggestions": dataset_info.suggested_questions[:8]
        }
    
    def _get_all_suggestions(self, datasets: List[DatasetInfo]) -> List[str]:
        """Get all suggestions from user's datasets"""
        
        all_suggestions = []
        for dataset in datasets:
            all_suggestions.extend(dataset.suggested_questions)
        
        # Remove duplicates and return top 10
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        return unique_suggestions[:10]
    
    def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        """Delete a dataset (only by owner)"""
        
        dataset = self.get_dataset(dataset_id)
        if not dataset or dataset.user_id != user_id:
            return False
        
        try:
            # Delete file
            dataset_file_path = self.storage_dir / f"{dataset_id}.csv"
            if dataset_file_path.exists():
                dataset_file_path.unlink()
            
            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM datasets WHERE dataset_id = ?', (dataset_id,))
            conn.commit()
            conn.close()
            
            # Remove from cache
            if dataset_id in self.datasets_cache:
                del self.datasets_cache[dataset_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            return False