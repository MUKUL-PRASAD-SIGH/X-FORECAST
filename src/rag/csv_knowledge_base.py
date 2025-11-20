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

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

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
    insights: List[str]
    suggested_questions: List[str]

class CSVKnowledgeBase:
    """
    Persistent CSV Knowledge Base with RAG capabilities
    """
    
    def __init__(self, data_dir: str = "data/knowledge_base"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.data_dir / "knowledge_base.db"
        self.init_database()
        
        # In-memory cache for fast access
        self.datasets_cache = {}
        self.load_datasets_cache()
    
    def init_database(self):
        """Initialize SQLite database for metadata storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
                insights TEXT NOT NULL,
                suggested_questions TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_queries (
                query_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_date TEXT NOT NULL,
                datasets_used TEXT NOT NULL,
                response TEXT NOT NULL
            )
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
            
            # Save dataset to persistent storage
            dataset_file = self.data_dir / f"{dataset_id}.pkl"
            with open(dataset_file, 'wb') as f:
                pickle.dump(df, f)
            
            # Save metadata to database
            self._save_dataset_metadata(dataset_info)
            
            # Update cache
            self.datasets_cache[dataset_id] = dataset_info
            
            logger.info(f"Added dataset {dataset_id} with {len(df)} rows")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Failed to add dataset: {e}")
            raise
    
    def _analyze_dataset(self, dataset_id: str, user_id: str, df: pd.DataFrame, filename: str) -> DatasetInfo:
        """Analyze dataset and extract insights"""
        
        # Basic info
        columns = df.columns.tolist()
        row_count = len(df)
        data_types = {col: str(df[col].dtype) for col in columns}
        
        # Summary statistics
        summary_stats = {}
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                summary_stats[col] = {
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'missing_count': int(df[col].isna().sum())
                }
            else:
                summary_stats[col] = {
                    'unique_count': int(df[col].nunique()),
                    'most_common': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                    'missing_count': int(df[col].isna().sum())
                }
        
        # Sample data (first 5 rows)
        sample_data = df.head(5).to_dict('records')
        
        # Generate insights
        insights = self._generate_insights(df, columns, summary_stats)
        
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
            insights=insights,
            suggested_questions=suggested_questions
        )
    
    def _generate_insights(self, df: pd.DataFrame, columns: List[str], summary_stats: Dict) -> List[str]:
        """Generate insights from dataset"""
        
        insights = []
        
        # Data quality insights
        total_missing = sum(stats.get('missing_count', 0) for stats in summary_stats.values())
        if total_missing > 0:
            missing_pct = (total_missing / (len(df) * len(columns))) * 100
            insights.append(f"Dataset has {missing_pct:.1f}% missing values")
        
        # Numeric columns insights
        numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            insights.append(f"Found {len(numeric_cols)} numeric columns for analysis")
            
            # Check for potential outliers
            for col in numeric_cols[:3]:  # Check first 3 numeric columns
                if col in summary_stats and summary_stats[col]['std']:
                    mean = summary_stats[col]['mean']
                    std = summary_stats[col]['std']
                    outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
                    if len(outliers) > 0:
                        insights.append(f"Column '{col}' has {len(outliers)} potential outliers")
        
        # Categorical columns insights
        categorical_cols = [col for col in columns if df[col].dtype == 'object']
        if categorical_cols:
            insights.append(f"Found {len(categorical_cols)} categorical columns")
            
            # Check for high cardinality
            for col in categorical_cols[:3]:
                unique_count = summary_stats[col]['unique_count']
                if unique_count > len(df) * 0.8:
                    insights.append(f"Column '{col}' has high cardinality ({unique_count} unique values)")
        
        # Date columns insights
        date_cols = []
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass
        
        if date_cols:
            insights.append(f"Found {len(date_cols)} potential date columns for time series analysis")
        
        # Size insights
        if len(df) > 10000:
            insights.append("Large dataset suitable for machine learning models")
        elif len(df) > 1000:
            insights.append("Medium-sized dataset good for statistical analysis")
        else:
            insights.append("Small dataset suitable for exploratory analysis")
        
        return insights
    
    def _generate_suggested_questions(self, df: pd.DataFrame, columns: List[str], summary_stats: Dict) -> List[str]:
        """Generate suggested questions based on dataset content"""
        
        questions = []
        
        # Generic questions
        questions.extend([
            "What are the key insights from this data?",
            "Show me a summary of the dataset",
            "What columns have the most missing values?"
        ])
        
        # Numeric column questions
        numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            for col in numeric_cols[:3]:
                questions.extend([
                    f"What is the distribution of {col}?",
                    f"Show me statistics for {col}",
                    f"Are there any outliers in {col}?"
                ])
        
        # Categorical column questions
        categorical_cols = [col for col in columns if df[col].dtype == 'object']
        if categorical_cols:
            for col in categorical_cols[:3]:
                questions.extend([
                    f"What are the most common values in {col}?",
                    f"Show me the distribution of {col}",
                    f"How many unique values are in {col}?"
                ])
        
        # Date-based questions
        date_cols = []
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                    questions.extend([
                        f"Show me trends over time using {col}",
                        f"What is the date range in {col}?",
                        f"Can you forecast future values based on {col}?"
                    ])
                except:
                    pass
        
        # Relationship questions
        if len(numeric_cols) >= 2:
            questions.extend([
                f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?",
                "Show me correlations between numeric columns",
                "Which variables are most strongly related?"
            ])
        
        # Business-specific questions based on column names
        business_keywords = {
            'sales': ["What are the total sales?", "Show me sales trends", "Which products have highest sales?"],
            'revenue': ["What is the total revenue?", "Show me revenue by category", "What drives revenue growth?"],
            'price': ["What is the average price?", "Show me price distribution", "How does price affect sales?"],
            'quantity': ["What is the total quantity sold?", "Show me quantity trends", "Which items sell most?"],
            'customer': ["How many unique customers?", "Show me customer segments", "What is customer behavior?"],
            'product': ["How many products?", "Which products are popular?", "Show me product performance"],
            'inventory': ["What is the inventory level?", "Show me stock analysis", "Which items are low in stock?"]
        }
        
        for keyword, keyword_questions in business_keywords.items():
            if any(keyword in col.lower() for col in columns):
                questions.extend(keyword_questions[:2])  # Add first 2 questions for each keyword
        
        return questions[:15]  # Return top 15 questions
    
    def _save_dataset_metadata(self, dataset_info: DatasetInfo):
        """Save dataset metadata to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO datasets 
            (dataset_id, user_id, filename, upload_date, columns, row_count, 
             data_types, summary_stats, sample_data, insights, suggested_questions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_info.dataset_id,
            dataset_info.user_id,
            dataset_info.filename,
            dataset_info.upload_date.isoformat(),
            json.dumps(dataset_info.columns, cls=NumpyEncoder),
            dataset_info.row_count,
            json.dumps(dataset_info.data_types, cls=NumpyEncoder),
            json.dumps(dataset_info.summary_stats, cls=NumpyEncoder),
            json.dumps(dataset_info.sample_data, cls=NumpyEncoder),
            json.dumps(dataset_info.insights, cls=NumpyEncoder),
            json.dumps(dataset_info.suggested_questions, cls=NumpyEncoder)
        ))
        
        conn.commit()
        conn.close()
    
    def load_datasets_cache(self):
        """Load all datasets metadata into cache"""
        
        try:
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
                    insights=json.loads(row[9]),
                    suggested_questions=json.loads(row[10])
                )
                self.datasets_cache[dataset_info.dataset_id] = dataset_info
            
            conn.close()
            logger.info(f"Loaded {len(self.datasets_cache)} datasets into cache")
            
        except Exception as e:
            logger.error(f"Failed to load datasets cache: {e}")
    
    def get_user_datasets(self, user_id: str) -> List[DatasetInfo]:
        """Get all datasets for a user"""
        return [info for info in self.datasets_cache.values() if info.user_id == user_id]
    
    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage"""
        
        dataset_file = self.data_dir / f"{dataset_id}.pkl"
        if dataset_file.exists():
            try:
                with open(dataset_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
        
        return None
    
    def query_datasets(self, user_id: str, query: str) -> Dict[str, Any]:
        """Query user's datasets using natural language"""
        
        user_datasets = self.get_user_datasets(user_id)
        if not user_datasets:
            return {
                "success": False,
                "message": "No datasets found. Please upload some data first.",
                "results": []
            }
        
        # Simple keyword-based matching for now
        # In production, use more sophisticated NLP/embedding-based matching
        query_lower = query.lower()
        
        results = []
        
        for dataset_info in user_datasets:
            # Check if query matches column names
            matching_columns = [col for col in dataset_info.columns if col.lower() in query_lower]
            
            if matching_columns:
                # Load actual dataset
                df = self.get_dataset(dataset_info.dataset_id)
                if df is not None:
                    result = self._analyze_query_on_dataset(query, df, dataset_info, matching_columns)
                    results.append(result)
        
        if not results:
            # Provide suggestions based on available data
            all_suggestions = []
            for dataset_info in user_datasets:
                all_suggestions.extend(dataset_info.suggested_questions)
            
            return {
                "success": False,
                "message": "No direct matches found in your data.",
                "suggestions": all_suggestions[:10],
                "available_datasets": [
                    {
                        "filename": info.filename,
                        "columns": info.columns,
                        "row_count": info.row_count
                    }
                    for info in user_datasets
                ]
            }
        
        return {
            "success": True,
            "message": f"Found {len(results)} relevant results",
            "results": results
        }
    
    def _analyze_query_on_dataset(self, query: str, df: pd.DataFrame, 
                                 dataset_info: DatasetInfo, matching_columns: List[str]) -> Dict[str, Any]:
        """Analyze query on specific dataset"""
        
        query_lower = query.lower()
        result = {
            "dataset": dataset_info.filename,
            "matching_columns": matching_columns,
            "analysis": {}
        }
        
        # Statistical queries
        if any(word in query_lower for word in ['average', 'mean', 'avg']):
            for col in matching_columns:
                if df[col].dtype in ['int64', 'float64']:
                    result["analysis"][f"average_{col}"] = float(df[col].mean())
        
        if any(word in query_lower for word in ['sum', 'total']):
            for col in matching_columns:
                if df[col].dtype in ['int64', 'float64']:
                    result["analysis"][f"total_{col}"] = float(df[col].sum())
        
        if any(word in query_lower for word in ['max', 'maximum', 'highest']):
            for col in matching_columns:
                if df[col].dtype in ['int64', 'float64']:
                    result["analysis"][f"max_{col}"] = float(df[col].max())
        
        if any(word in query_lower for word in ['min', 'minimum', 'lowest']):
            for col in matching_columns:
                if df[col].dtype in ['int64', 'float64']:
                    result["analysis"][f"min_{col}"] = float(df[col].min())
        
        # Distribution queries
        if any(word in query_lower for word in ['distribution', 'count', 'frequency']):
            for col in matching_columns:
                if df[col].dtype == 'object':
                    value_counts = df[col].value_counts().head(5)
                    result["analysis"][f"top_values_{col}"] = value_counts.to_dict()
        
        # Trend queries
        if any(word in query_lower for word in ['trend', 'over time', 'time series']):
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and matching_columns:
                date_col = date_cols[0]
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    df_temp = df_temp.sort_values(date_col)
                    
                    for col in matching_columns:
                        if df[col].dtype in ['int64', 'float64']:
                            # Simple trend analysis
                            recent_avg = df_temp[col].tail(10).mean()
                            older_avg = df_temp[col].head(10).mean()
                            trend = "increasing" if recent_avg > older_avg else "decreasing"
                            result["analysis"][f"trend_{col}"] = {
                                "direction": trend,
                                "recent_average": float(recent_avg),
                                "older_average": float(older_avg)
                            }
                except:
                    pass
        
        return result
    
    def get_all_suggestions(self, user_id: str) -> List[str]:
        """Get all suggested questions for user's datasets"""
        
        user_datasets = self.get_user_datasets(user_id)
        all_suggestions = []
        
        for dataset_info in user_datasets:
            all_suggestions.extend(dataset_info.suggested_questions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in all_suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:20]  # Return top 20 unique suggestions
    
    def get_dataset_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of all user datasets"""
        
        user_datasets = self.get_user_datasets(user_id)
        
        if not user_datasets:
            return {
                "total_datasets": 0,
                "total_rows": 0,
                "available_columns": [],
                "insights": []
            }
        
        total_rows = sum(info.row_count for info in user_datasets)
        all_columns = []
        all_insights = []
        
        for info in user_datasets:
            all_columns.extend(info.columns)
            all_insights.extend(info.insights)
        
        # Get unique columns
        unique_columns = list(set(all_columns))
        
        return {
            "total_datasets": len(user_datasets),
            "total_rows": total_rows,
            "available_columns": unique_columns,
            "insights": all_insights,
            "datasets": [
                {
                    "filename": info.filename,
                    "rows": info.row_count,
                    "columns": len(info.columns),
                    "upload_date": info.upload_date.strftime("%Y-%m-%d %H:%M")
                }
                for info in user_datasets
            ]
        }