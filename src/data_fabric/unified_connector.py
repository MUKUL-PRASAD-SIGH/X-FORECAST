"""
Enhanced Unified Data Connector for Cyberpunk AI Dashboard
Supports multiple data sources: CRM, ERP, Marketing, and existing CSV data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
import aiohttp
import json
from sqlalchemy import create_engine, text
import redis
from .connector import DataConnector  # Import existing connector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    type: str  # 'csv', 'api', 'database', 'crm', 'erp', 'marketing'
    connection_string: str
    credentials: Dict[str, Any]
    refresh_interval: int = 3600  # seconds
    enabled: bool = True

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    validity: float
    overall_score: float
    issues: List[str]

@dataclass
class DataSyncResult:
    """Result of data synchronization operation"""
    source: str
    success: bool
    records_processed: int
    records_updated: int
    records_failed: int
    sync_duration: float
    quality_metrics: DataQualityMetrics
    errors: List[str]

class BaseDataConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.last_sync = None
        self.cache = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from source"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection health"""
        pass
    
    def assess_data_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Assess data quality metrics"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        # Completeness: percentage of non-missing values
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # Basic quality assessment (can be enhanced with domain-specific rules)
        accuracy = 0.95  # Placeholder - would implement actual accuracy checks
        consistency = 0.90  # Placeholder - would check for consistent formats
        timeliness = 1.0 if self.last_sync and (datetime.now() - self.last_sync).seconds < self.config.refresh_interval else 0.5
        validity = 0.85  # Placeholder - would validate against business rules
        
        overall_score = (completeness + accuracy + consistency + timeliness + validity) / 5
        
        issues = []
        if completeness < 0.9:
            issues.append(f"High missing data rate: {(1-completeness)*100:.1f}%")
        if timeliness < 0.8:
            issues.append("Data may be stale")
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            overall_score=overall_score,
            issues=issues
        )

class CRMConnector(BaseDataConnector):
    """Connector for CRM systems (Salesforce, HubSpot, etc.)"""
    
    async def connect(self) -> bool:
        """Connect to CRM system"""
        try:
            # Implement CRM-specific connection logic
            logger.info(f"Connecting to CRM: {self.config.name}")
            # Placeholder for actual CRM connection
            return True
        except Exception as e:
            logger.error(f"CRM connection failed: {e}")
            return False
    
    async def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch customer data from CRM"""
        try:
            # Generate sample CRM data for demo
            np.random.seed(42)
            n_customers = 1000
            
            data = {
                'customer_id': [f'CRM_{i:06d}' for i in range(n_customers)],
                'company_name': [f'Company_{i}' for i in range(n_customers)],
                'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing'], n_customers),
                'annual_revenue': np.random.lognormal(15, 1, n_customers),
                'employees': np.random.randint(10, 10000, n_customers),
                'created_date': pd.date_range('2020-01-01', periods=n_customers, freq='D'),
                'last_activity': pd.date_range('2024-01-01', periods=n_customers, freq='H'),
                'lead_score': np.random.randint(0, 100, n_customers),
                'stage': np.random.choice(['Lead', 'Qualified', 'Opportunity', 'Customer', 'Churned'], n_customers),
                'contact_email': [f'contact_{i}@company{i}.com' for i in range(n_customers)],
                'phone': [f'+1-555-{i:04d}' for i in range(n_customers)]
            }
            
            df = pd.DataFrame(data)
            self.last_sync = datetime.now()
            logger.info(f"Fetched {len(df)} records from CRM")
            return df
            
        except Exception as e:
            logger.error(f"CRM data fetch failed: {e}")
            return pd.DataFrame()
    
    async def validate_connection(self) -> bool:
        """Validate CRM connection"""
        return await self.connect()

class ERPConnector(BaseDataConnector):
    """Connector for ERP systems (SAP, Oracle, etc.)"""
    
    async def connect(self) -> bool:
        """Connect to ERP system"""
        try:
            logger.info(f"Connecting to ERP: {self.config.name}")
            # Placeholder for actual ERP connection
            return True
        except Exception as e:
            logger.error(f"ERP connection failed: {e}")
            return False
    
    async def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch operational data from ERP"""
        try:
            # Generate sample ERP data for demo
            np.random.seed(123)
            n_transactions = 5000
            
            data = {
                'transaction_id': [f'ERP_{i:08d}' for i in range(n_transactions)],
                'customer_id': [f'CRM_{np.random.randint(0, 1000):06d}' for _ in range(n_transactions)],
                'product_id': [f'SKU_{np.random.randint(1, 500):04d}' for _ in range(n_transactions)],
                'transaction_date': pd.date_range('2023-01-01', periods=n_transactions, freq='H'),
                'quantity': np.random.randint(1, 100, n_transactions),
                'unit_price': np.random.uniform(10, 1000, n_transactions),
                'total_amount': np.random.uniform(50, 10000, n_transactions),
                'currency': 'USD',
                'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Cash', 'Check'], n_transactions),
                'sales_rep': [f'Rep_{np.random.randint(1, 50):02d}' for _ in range(n_transactions)],
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_transactions),
                'channel': np.random.choice(['Online', 'Retail', 'Partner', 'Direct'], n_transactions)
            }
            
            df = pd.DataFrame(data)
            self.last_sync = datetime.now()
            logger.info(f"Fetched {len(df)} records from ERP")
            return df
            
        except Exception as e:
            logger.error(f"ERP data fetch failed: {e}")
            return pd.DataFrame()
    
    async def validate_connection(self) -> bool:
        """Validate ERP connection"""
        return await self.connect()

class MarketingConnector(BaseDataConnector):
    """Connector for Marketing Automation systems (Marketo, Pardot, etc.)"""
    
    async def connect(self) -> bool:
        """Connect to Marketing system"""
        try:
            logger.info(f"Connecting to Marketing: {self.config.name}")
            # Placeholder for actual Marketing connection
            return True
        except Exception as e:
            logger.error(f"Marketing connection failed: {e}")
            return False
    
    async def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch marketing data"""
        try:
            # Generate sample Marketing data for demo
            np.random.seed(456)
            n_campaigns = 2000
            
            data = {
                'campaign_id': [f'CAMP_{i:06d}' for i in range(n_campaigns)],
                'customer_id': [f'CRM_{np.random.randint(0, 1000):06d}' for _ in range(n_campaigns)],
                'campaign_name': [f'Campaign_{i}' for i in range(n_campaigns)],
                'campaign_type': np.random.choice(['Email', 'Social', 'PPC', 'Content', 'Event'], n_campaigns),
                'start_date': pd.date_range('2023-01-01', periods=n_campaigns, freq='D'),
                'end_date': pd.date_range('2023-02-01', periods=n_campaigns, freq='D'),
                'budget': np.random.uniform(1000, 50000, n_campaigns),
                'impressions': np.random.randint(1000, 100000, n_campaigns),
                'clicks': np.random.randint(10, 5000, n_campaigns),
                'conversions': np.random.randint(1, 500, n_campaigns),
                'cost_per_click': np.random.uniform(0.5, 10, n_campaigns),
                'conversion_rate': np.random.uniform(0.01, 0.15, n_campaigns),
                'roi': np.random.uniform(-0.5, 3.0, n_campaigns),
                'channel': np.random.choice(['Google', 'Facebook', 'LinkedIn', 'Twitter', 'Email'], n_campaigns)
            }
            
            df = pd.DataFrame(data)
            self.last_sync = datetime.now()
            logger.info(f"Fetched {len(df)} records from Marketing")
            return df
            
        except Exception as e:
            logger.error(f"Marketing data fetch failed: {e}")
            return pd.DataFrame()
    
    async def validate_connection(self) -> bool:
        """Validate Marketing connection"""
        return await self.connect()

class UnifiedDataConnector:
    """Enhanced unified data connector supporting multiple sources"""
    
    def __init__(self, cache_config: Optional[Dict] = None):
        self.connectors: Dict[str, BaseDataConnector] = {}
        self.existing_connector = DataConnector('./data/raw')  # Use existing connector
        self.cache_enabled = cache_config is not None
        self.redis_client = None
        
        if self.cache_enabled and cache_config:
            try:
                self.redis_client = redis.Redis(**cache_config)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
                self.cache_enabled = False
    
    def add_connector(self, connector: BaseDataConnector):
        """Add a data source connector"""
        self.connectors[connector.config.name] = connector
        logger.info(f"Added connector: {connector.config.name}")
    
    def configure_default_connectors(self):
        """Configure default CRM, ERP, and Marketing connectors"""
        # CRM Configuration
        crm_config = DataSourceConfig(
            name='salesforce_crm',
            type='crm',
            connection_string='https://api.salesforce.com',
            credentials={'api_key': 'demo_key', 'secret': 'demo_secret'},
            refresh_interval=1800  # 30 minutes
        )
        self.add_connector(CRMConnector(crm_config))
        
        # ERP Configuration
        erp_config = DataSourceConfig(
            name='sap_erp',
            type='erp',
            connection_string='sap://erp.company.com',
            credentials={'username': 'demo_user', 'password': 'demo_pass'},
            refresh_interval=3600  # 1 hour
        )
        self.add_connector(ERPConnector(erp_config))
        
        # Marketing Configuration
        marketing_config = DataSourceConfig(
            name='marketo_marketing',
            type='marketing',
            connection_string='https://api.marketo.com',
            credentials={'client_id': 'demo_client', 'client_secret': 'demo_secret'},
            refresh_interval=1800  # 30 minutes
        )
        self.add_connector(MarketingConnector(marketing_config))
    
    async def sync_all_sources(self) -> List[DataSyncResult]:
        """Synchronize data from all connected sources"""
        results = []
        
        for name, connector in self.connectors.items():
            if not connector.config.enabled:
                continue
                
            start_time = datetime.now()
            try:
                # Validate connection
                if not await connector.validate_connection():
                    results.append(DataSyncResult(
                        source=name,
                        success=False,
                        records_processed=0,
                        records_updated=0,
                        records_failed=0,
                        sync_duration=0,
                        quality_metrics=DataQualityMetrics(0, 0, 0, 0, 0, 0, ['Connection failed']),
                        errors=['Connection validation failed']
                    ))
                    continue
                
                # Fetch data
                df = await connector.fetch_data()
                
                if df.empty:
                    results.append(DataSyncResult(
                        source=name,
                        success=False,
                        records_processed=0,
                        records_updated=0,
                        records_failed=0,
                        sync_duration=(datetime.now() - start_time).total_seconds(),
                        quality_metrics=DataQualityMetrics(0, 0, 0, 0, 0, 0, ['No data returned']),
                        errors=['No data returned from source']
                    ))
                    continue
                
                # Assess data quality
                quality_metrics = connector.assess_data_quality(df)
                
                # Cache data if enabled
                if self.cache_enabled and self.redis_client:
                    try:
                        cache_key = f"data:{name}:{datetime.now().strftime('%Y%m%d_%H')}"
                        self.redis_client.setex(
                            cache_key, 
                            connector.config.refresh_interval,
                            df.to_json()
                        )
                    except Exception as e:
                        logger.warning(f"Cache storage failed for {name}: {e}")
                
                # Create success result
                sync_duration = (datetime.now() - start_time).total_seconds()
                results.append(DataSyncResult(
                    source=name,
                    success=True,
                    records_processed=len(df),
                    records_updated=len(df),
                    records_failed=0,
                    sync_duration=sync_duration,
                    quality_metrics=quality_metrics,
                    errors=[]
                ))
                
                logger.info(f"Successfully synced {len(df)} records from {name} in {sync_duration:.2f}s")
                
            except Exception as e:
                sync_duration = (datetime.now() - start_time).total_seconds()
                error_msg = str(e)
                results.append(DataSyncResult(
                    source=name,
                    success=False,
                    records_processed=0,
                    records_updated=0,
                    records_failed=1,
                    sync_duration=sync_duration,
                    quality_metrics=DataQualityMetrics(0, 0, 0, 0, 0, 0, [error_msg]),
                    errors=[error_msg]
                ))
                logger.error(f"Sync failed for {name}: {e}")
        
        return results
    
    async def get_unified_customer_view(self, customer_id: str) -> Dict[str, Any]:
        """Get comprehensive customer profile from all sources"""
        customer_profile = {
            'customer_id': customer_id,
            'crm_data': {},
            'erp_data': {},
            'marketing_data': {},
            'unified_metrics': {}
        }
        
        try:
            # Fetch from CRM
            if 'salesforce_crm' in self.connectors:
                crm_connector = self.connectors['salesforce_crm']
                crm_data = await crm_connector.fetch_data()
                customer_crm = crm_data[crm_data['customer_id'] == customer_id]
                if not customer_crm.empty:
                    customer_profile['crm_data'] = customer_crm.iloc[0].to_dict()
            
            # Fetch from ERP
            if 'sap_erp' in self.connectors:
                erp_connector = self.connectors['sap_erp']
                erp_data = await erp_connector.fetch_data()
                customer_erp = erp_data[erp_data['customer_id'] == customer_id]
                if not customer_erp.empty:
                    customer_profile['erp_data'] = {
                        'total_transactions': len(customer_erp),
                        'total_revenue': customer_erp['total_amount'].sum(),
                        'avg_order_value': customer_erp['total_amount'].mean(),
                        'last_transaction': customer_erp['transaction_date'].max(),
                        'preferred_channel': customer_erp['channel'].mode().iloc[0] if not customer_erp.empty else None
                    }
            
            # Fetch from Marketing
            if 'marketo_marketing' in self.connectors:
                marketing_connector = self.connectors['marketo_marketing']
                marketing_data = await marketing_connector.fetch_data()
                customer_marketing = marketing_data[marketing_data['customer_id'] == customer_id]
                if not customer_marketing.empty:
                    customer_profile['marketing_data'] = {
                        'campaigns_engaged': len(customer_marketing),
                        'total_impressions': customer_marketing['impressions'].sum(),
                        'total_clicks': customer_marketing['clicks'].sum(),
                        'avg_conversion_rate': customer_marketing['conversion_rate'].mean(),
                        'preferred_channel': customer_marketing['channel'].mode().iloc[0] if not customer_marketing.empty else None
                    }
            
            # Calculate unified metrics
            customer_profile['unified_metrics'] = self._calculate_unified_metrics(customer_profile)
            
        except Exception as e:
            logger.error(f"Failed to get unified customer view for {customer_id}: {e}")
            customer_profile['error'] = str(e)
        
        return customer_profile
    
    def _calculate_unified_metrics(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate unified customer metrics"""
        metrics = {}
        
        try:
            # Customer lifetime value estimation
            if profile['erp_data']:
                erp = profile['erp_data']
                metrics['estimated_ltv'] = erp.get('total_revenue', 0) * 1.5  # Simple estimation
                metrics['avg_order_value'] = erp.get('avg_order_value', 0)
                metrics['transaction_frequency'] = erp.get('total_transactions', 0)
            
            # Engagement score
            engagement_score = 0
            if profile['crm_data']:
                engagement_score += profile['crm_data'].get('lead_score', 0) * 0.4
            if profile['marketing_data']:
                marketing = profile['marketing_data']
                click_rate = marketing.get('total_clicks', 0) / max(marketing.get('total_impressions', 1), 1)
                engagement_score += click_rate * 100 * 0.3
                engagement_score += marketing.get('avg_conversion_rate', 0) * 100 * 0.3
            
            metrics['engagement_score'] = min(engagement_score, 100)
            
            # Risk assessment
            risk_factors = []
            if profile['erp_data'] and profile['erp_data'].get('total_transactions', 0) == 0:
                risk_factors.append('No recent transactions')
            if profile['marketing_data'] and profile['marketing_data'].get('avg_conversion_rate', 0) < 0.02:
                risk_factors.append('Low conversion rate')
            
            metrics['risk_factors'] = risk_factors
            metrics['churn_risk'] = 'High' if len(risk_factors) > 1 else 'Medium' if len(risk_factors) == 1 else 'Low'
            
        except Exception as e:
            logger.error(f"Failed to calculate unified metrics: {e}")
            metrics['calculation_error'] = str(e)
        
        return metrics
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_quality': 0,
            'recommendations': []
        }
        
        total_quality = 0
        active_sources = 0
        
        for name, connector in self.connectors.items():
            if not connector.config.enabled:
                continue
                
            # Get cached quality metrics or assess current data
            try:
                # This would typically fetch the latest quality assessment
                # For demo, we'll use placeholder values
                quality = DataQualityMetrics(
                    completeness=0.95,
                    accuracy=0.92,
                    consistency=0.88,
                    timeliness=0.90,
                    validity=0.85,
                    overall_score=0.90,
                    issues=['Minor data inconsistencies']
                )
                
                report['sources'][name] = {
                    'quality_score': quality.overall_score,
                    'completeness': quality.completeness,
                    'accuracy': quality.accuracy,
                    'consistency': quality.consistency,
                    'timeliness': quality.timeliness,
                    'validity': quality.validity,
                    'issues': quality.issues,
                    'last_sync': connector.last_sync.isoformat() if connector.last_sync else None
                }
                
                total_quality += quality.overall_score
                active_sources += 1
                
            except Exception as e:
                report['sources'][name] = {
                    'error': str(e),
                    'quality_score': 0
                }
        
        if active_sources > 0:
            report['overall_quality'] = total_quality / active_sources
        
        # Generate recommendations
        if report['overall_quality'] < 0.8:
            report['recommendations'].append('Overall data quality is below threshold. Review data sources.')
        if active_sources < len(self.connectors):
            report['recommendations'].append('Some data sources are inactive. Check connections.')
        
        return report