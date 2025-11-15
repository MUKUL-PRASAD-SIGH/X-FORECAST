"""
Company Data Manager for Sales Forecasting
Handles company-specific data upload, validation, and storage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DataRequirements:
    """Data requirements for monthly sales upload"""
    required_columns: List[str]
    optional_columns: List[str]
    date_format: str
    min_months: int
    max_file_size_mb: int
    supported_formats: List[str]

@dataclass
class CompanyProfile:
    """Company profile and configuration"""
    company_id: str
    company_name: str
    industry: str
    created_date: datetime
    data_requirements: DataRequirements
    adaptive_config: Dict[str, Any]
    total_uploads: int = 0
    last_upload_date: Optional[datetime] = None
    model_performance_history: List[Dict] = None
    
    def __post_init__(self):
        if self.model_performance_history is None:
            self.model_performance_history = []

class CompanyDataManager:
    """Manages company-specific sales data and requirements"""
    
    def __init__(self, data_dir: str = "company_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Default data requirements
        self.default_requirements = DataRequirements(
            required_columns=[
                'date',           # Date of sales (YYYY-MM-DD or YYYY-MM)
                'sales_amount',   # Total sales amount
                'product_category', # Product category or SKU
                'region'          # Sales region or location
            ],
            optional_columns=[
                'units_sold',     # Number of units sold
                'customer_count', # Number of customers
                'avg_order_value', # Average order value
                'marketing_spend', # Marketing expenditure
                'promotions',     # Promotional activities (0/1)
                'seasonality_factor', # Custom seasonality indicator
                'external_factors' # External factors (weather, events, etc.)
            ],
            date_format='YYYY-MM-DD',
            min_months=3,
            max_file_size_mb=50,
            supported_formats=['csv', 'xlsx', 'json']
        )
        
        self.companies: Dict[str, CompanyProfile] = {}
        self._load_existing_companies()
    
    def _load_existing_companies(self):
        """Load existing company profiles"""
        try:
            companies_file = self.data_dir / "companies.json"
            if companies_file.exists():
                with open(companies_file, 'r') as f:
                    companies_data = json.load(f)
                
                for company_id, data in companies_data.items():
                    # Convert datetime strings back to datetime objects
                    if 'created_date' in data:
                        data['created_date'] = datetime.fromisoformat(data['created_date'])
                    if 'last_upload_date' in data and data['last_upload_date']:
                        data['last_upload_date'] = datetime.fromisoformat(data['last_upload_date'])
                    
                    # Convert data requirements
                    if 'data_requirements' in data:
                        data['data_requirements'] = DataRequirements(**data['data_requirements'])
                    
                    self.companies[company_id] = CompanyProfile(**data)
                
                logger.info(f"Loaded {len(self.companies)} existing companies")
        
        except Exception as e:
            logger.error(f"Failed to load existing companies: {e}")
    
    def _save_companies(self):
        """Save company profiles to disk"""
        try:
            companies_data = {}
            for company_id, profile in self.companies.items():
                data = asdict(profile)
                # Convert datetime objects to strings
                if data['created_date']:
                    data['created_date'] = data['created_date'].isoformat()
                if data['last_upload_date']:
                    data['last_upload_date'] = data['last_upload_date'].isoformat()
                
                companies_data[company_id] = data
            
            companies_file = self.data_dir / "companies.json"
            with open(companies_file, 'w') as f:
                json.dump(companies_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.companies)} companies")
        
        except Exception as e:
            logger.error(f"Failed to save companies: {e}")
    
    def register_company(self, company_id: str, company_name: str, 
                        industry: str, custom_requirements: Optional[Dict] = None) -> CompanyProfile:
        """Register a new company"""
        
        if company_id in self.companies:
            raise ValueError(f"Company {company_id} already exists")
        
        # Use custom requirements or defaults
        requirements = self.default_requirements
        if custom_requirements:
            # Merge custom requirements with defaults
            req_dict = asdict(requirements)
            req_dict.update(custom_requirements)
            requirements = DataRequirements(**req_dict)
        
        # Create company profile
        profile = CompanyProfile(
            company_id=company_id,
            company_name=company_name,
            industry=industry,
            created_date=datetime.now(),
            data_requirements=requirements,
            adaptive_config={
                'adaptive_learning_enabled': True,
                'learning_window_months': 6,
                'min_model_weight': 0.05,
                'weight_update_frequency': 'monthly',
                'confidence_intervals_enabled': True,
                'pattern_detection_enabled': True
            }
        )
        
        self.companies[company_id] = profile
        
        # Create company data directory
        company_dir = self.data_dir / company_id
        company_dir.mkdir(exist_ok=True)
        
        self._save_companies()
        
        logger.info(f"Registered new company: {company_name} ({company_id})")
        return profile
    
    def get_company_profile(self, company_id: str) -> Optional[CompanyProfile]:
        """Get company profile"""
        return self.companies.get(company_id)
    
    def get_data_requirements(self, company_id: str) -> DataRequirements:
        """Get data requirements for a company"""
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        return self.companies[company_id].data_requirements
    
    def validate_upload_data(self, company_id: str, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate uploaded sales data"""
        
        if company_id not in self.companies:
            return False, [f"Company {company_id} not found"]
        
        requirements = self.companies[company_id].data_requirements
        errors = []
        
        # Check required columns
        missing_columns = set(requirements.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check data types and formats
        if 'date' in data.columns:
            try:
                # Try to parse dates
                pd.to_datetime(data['date'])
            except Exception as e:
                errors.append(f"Invalid date format. Expected {requirements.date_format}: {e}")
        
        if 'sales_amount' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['sales_amount']):
                errors.append("sales_amount must be numeric")
            elif (data['sales_amount'] < 0).any():
                errors.append("sales_amount cannot be negative")
        
        # Check minimum data requirements
        if len(data) < requirements.min_months:
            errors.append(f"Minimum {requirements.min_months} months of data required, got {len(data)}")
        
        # Check for duplicates
        if 'date' in data.columns and 'product_category' in data.columns:
            duplicates = data.duplicated(subset=['date', 'product_category', 'region'])
            if duplicates.any():
                errors.append(f"Found {duplicates.sum()} duplicate records")
        
        # Check data quality
        null_percentages = data.isnull().sum() / len(data) * 100
        high_null_cols = null_percentages[null_percentages > 50].index.tolist()
        if high_null_cols:
            errors.append(f"Columns with >50% missing data: {high_null_cols}")
        
        return len(errors) == 0, errors
    
    def save_company_data(self, company_id: str, data: pd.DataFrame, 
                         upload_metadata: Optional[Dict] = None) -> str:
        """Save validated company data"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        # Validate data first
        is_valid, errors = self.validate_upload_data(company_id, data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sales_data_{timestamp}.csv"
        
        company_dir = self.data_dir / company_id
        filepath = company_dir / filename
        
        # Save data
        data.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'upload_timestamp': datetime.now().isoformat(),
            'filename': filename,
            'records_count': len(data),
            'date_range': {
                'start': data['date'].min() if 'date' in data.columns else None,
                'end': data['date'].max() if 'date' in data.columns else None
            },
            'columns': list(data.columns),
            'data_quality': {
                'null_percentages': data.isnull().sum().to_dict(),
                'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist()
            }
        }
        
        if upload_metadata:
            metadata.update(upload_metadata)
        
        metadata_file = company_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Update company profile
        profile = self.companies[company_id]
        profile.total_uploads += 1
        profile.last_upload_date = datetime.now()
        
        self._save_companies()
        
        logger.info(f"Saved data for company {company_id}: {len(data)} records")
        return str(filepath)
    
    def get_company_data_history(self, company_id: str) -> List[Dict]:
        """Get upload history for a company"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        company_dir = self.data_dir / company_id
        if not company_dir.exists():
            return []
        
        history = []
        
        # Find all metadata files
        for metadata_file in company_dir.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                history.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load metadata {metadata_file}: {e}")
        
        # Sort by upload timestamp
        history.sort(key=lambda x: x.get('upload_timestamp', ''), reverse=True)
        
        return history
    
    def load_company_data(self, company_id: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load all or recent company data"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        company_dir = self.data_dir / company_id
        if not company_dir.exists():
            return pd.DataFrame()
        
        # Find all data files
        data_files = list(company_dir.glob("sales_data_*.csv"))
        data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if limit:
            data_files = data_files[:limit]
        
        if not data_files:
            return pd.DataFrame()
        
        # Load and combine data
        all_data = []
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                df['upload_file'] = file_path.name
                all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to load data file {file_path}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Convert date column
        if 'date' in combined_data.columns:
            combined_data['date'] = pd.to_datetime(combined_data['date'])
        
        # Remove duplicates (keep most recent upload)
        if 'date' in combined_data.columns and 'product_category' in combined_data.columns:
            combined_data = combined_data.drop_duplicates(
                subset=['date', 'product_category', 'region'], 
                keep='first'
            )
        
        # Sort by date
        if 'date' in combined_data.columns:
            combined_data = combined_data.sort_values('date')
        
        return combined_data
    
    def get_data_requirements_template(self, company_id: str) -> Dict[str, Any]:
        """Get data requirements template for company"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        requirements = self.companies[company_id].data_requirements
        
        template = {
            'data_requirements': {
                'required_columns': requirements.required_columns,
                'optional_columns': requirements.optional_columns,
                'date_format': requirements.date_format,
                'min_months': requirements.min_months,
                'max_file_size_mb': requirements.max_file_size_mb,
                'supported_formats': requirements.supported_formats
            },
            'sample_data': {
                'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
                'sales_amount': [100000, 120000, 95000],
                'product_category': ['Electronics', 'Electronics', 'Electronics'],
                'region': ['North', 'North', 'North'],
                'units_sold': [500, 600, 475],
                'customer_count': [250, 300, 238]
            },
            'validation_rules': {
                'sales_amount': 'Must be positive numeric values',
                'date': f'Must be in {requirements.date_format} format',
                'duplicates': 'No duplicate records for same date/category/region',
                'missing_data': 'Required columns cannot be empty'
            },
            'tips': [
                'Ensure consistent date formatting across all records',
                'Use consistent category and region names',
                'Include at least 3 months of historical data for initial upload',
                'Optional columns help improve forecast accuracy',
                'Upload new data monthly for best adaptive learning'
            ]
        }
        
        return template
    
    def update_company_config(self, company_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update company adaptive configuration"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        try:
            profile = self.companies[company_id]
            profile.adaptive_config.update(config_updates)
            
            self._save_companies()
            
            logger.info(f"Updated config for company {company_id}: {config_updates}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update config for company {company_id}: {e}")
            return False
    
    def get_company_stats(self, company_id: str) -> Dict[str, Any]:
        """Get company statistics and summary"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        profile = self.companies[company_id]
        
        # Load recent data for stats
        try:
            data = self.load_company_data(company_id, limit=5)  # Last 5 uploads
            
            stats = {
                'company_info': {
                    'company_id': profile.company_id,
                    'company_name': profile.company_name,
                    'industry': profile.industry,
                    'created_date': profile.created_date.isoformat(),
                    'total_uploads': profile.total_uploads,
                    'last_upload_date': profile.last_upload_date.isoformat() if profile.last_upload_date else None
                },
                'data_summary': {
                    'total_records': len(data),
                    'date_range': {
                        'start': data['date'].min().isoformat() if 'date' in data.columns and not data.empty else None,
                        'end': data['date'].max().isoformat() if 'date' in data.columns and not data.empty else None
                    },
                    'categories': data['product_category'].nunique() if 'product_category' in data.columns else 0,
                    'regions': data['region'].nunique() if 'region' in data.columns else 0,
                    'avg_monthly_sales': data['sales_amount'].mean() if 'sales_amount' in data.columns else 0
                },
                'adaptive_config': profile.adaptive_config,
                'model_performance': profile.model_performance_history[-5:] if profile.model_performance_history else []
            }
            
        except Exception as e:
            logger.error(f"Failed to generate stats for company {company_id}: {e}")
            stats = {
                'company_info': {
                    'company_id': profile.company_id,
                    'company_name': profile.company_name,
                    'industry': profile.industry,
                    'created_date': profile.created_date.isoformat(),
                    'total_uploads': profile.total_uploads,
                    'last_upload_date': profile.last_upload_date.isoformat() if profile.last_upload_date else None
                },
                'data_summary': {},
                'adaptive_config': profile.adaptive_config,
                'model_performance': []
            }
        
        return stats