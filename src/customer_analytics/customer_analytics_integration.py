"""
Customer Analytics Integration Module
Integrates customer analytics engine with company sales system and ensemble forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from .customer_analytics_engine import CustomerAnalyticsEngine, CustomerAnalytics

try:
    from ..company_sales.company_data_manager import CompanyDataManager
    COMPANY_DATA_AVAILABLE = True
except ImportError:
    # Fallback for when running as standalone
    CompanyDataManager = None
    COMPANY_DATA_AVAILABLE = False

try:
    from ..models.ensemble_forecasting_engine import EnsembleForecastingEngine
    ENSEMBLE_AVAILABLE = True
except ImportError:
    EnsembleForecastingEngine = None
    ENSEMBLE_AVAILABLE = False

logger = logging.getLogger(__name__)

class CustomerAnalyticsIntegration:
    """Integration layer for customer analytics with company sales system"""
    
    def __init__(self, data_dir: str = "company_data"):
        self.customer_analytics_engine = CustomerAnalyticsEngine()
        
        if COMPANY_DATA_AVAILABLE:
            self.company_data_manager = CompanyDataManager(data_dir)
        else:
            self.company_data_manager = None
            logger.warning("CompanyDataManager not available - running in standalone mode")
        
        self.analytics_cache = {}  # Cache for analytics results
        self.cache_duration_hours = 6  # Cache results for 6 hours
    
    def analyze_company_customers(self, company_id: str, 
                                force_refresh: bool = False) -> Optional[CustomerAnalytics]:
        """Analyze customers for a specific company"""
        try:
            # Check cache first
            cache_key = f"{company_id}_customer_analytics"
            if not force_refresh and cache_key in self.analytics_cache:
                cached_result, cache_time = self.analytics_cache[cache_key]
                if datetime.now() - cache_time < timedelta(hours=self.cache_duration_hours):
                    logger.info(f"Returning cached customer analytics for company {company_id}")
                    return cached_result
            
            # Load company data
            company_profile = self.company_data_manager.get_company_profile(company_id)
            if not company_profile:
                logger.error(f"Company {company_id} not found")
                return None
            
            # Load sales data
            sales_data = self.company_data_manager.load_company_data(company_id)
            if sales_data.empty:
                logger.warning(f"No sales data found for company {company_id}")
                return None
            
            logger.info(f"Analyzing {len(sales_data)} sales records for company {company_id}")
            
            # Perform customer analytics
            analytics_result = self.customer_analytics_engine.analyze_customers(sales_data)
            
            # Cache the result
            self.analytics_cache[cache_key] = (analytics_result, datetime.now())
            
            logger.info(f"Customer analytics completed for company {company_id}: "
                       f"{analytics_result.total_customers} customers analyzed")
            
            return analytics_result
            
        except Exception as e:
            logger.error(f"Failed to analyze customers for company {company_id}: {e}")
            return None
    
    def get_customer_insights_for_forecasting(self, company_id: str) -> Dict[str, Any]:
        """Get customer insights that can enhance forecasting models"""
        try:
            analytics = self.analyze_company_customers(company_id)
            if not analytics:
                return {}
            
            # Extract insights relevant for forecasting
            insights = {
                'customer_metrics': {
                    'total_customers': analytics.total_customers,
                    'active_customers': analytics.active_customers,
                    'retention_rate': analytics.overall_retention_rate,
                    'average_ltv': analytics.overall_ltv,
                    'churn_risk_percentage': (analytics.churned_customers / analytics.total_customers * 100) 
                                           if analytics.total_customers > 0 else 0
                },
                'segment_performance': [
                    {
                        'segment_name': segment.segment_name,
                        'customer_count': segment.customer_count,
                        'revenue_contribution': segment.revenue_contribution,
                        'growth_rate': segment.growth_rate,
                        'avg_ltv': segment.avg_ltv,
                        'health_score': segment.segment_health_score
                    }
                    for segment in analytics.customer_segments
                ],
                'cohort_trends': [
                    {
                        'cohort_month': cohort.cohort_month,
                        'cohort_size': cohort.cohort_size,
                        'avg_ltv': cohort.avg_ltv,
                        'churn_rate': cohort.churn_rate,
                        'retention_month_1': cohort.retention_rates.get(1, 0),
                        'retention_month_3': cohort.retention_rates.get(3, 0),
                        'retention_month_6': cohort.retention_rates.get(6, 0)
                    }
                    for cohort in analytics.cohort_analyses
                ],
                'churn_analysis': {
                    'high_risk_customers': len([r for r in analytics.churn_risks if r.risk_level in ['High', 'Critical']]),
                    'medium_risk_customers': len([r for r in analytics.churn_risks if r.risk_level == 'Medium']),
                    'low_risk_customers': len([r for r in analytics.churn_risks if r.risk_level == 'Low']),
                    'top_risk_factors': self._get_top_risk_factors(analytics.churn_risks)
                },
                'forecasting_adjustments': self._calculate_forecasting_adjustments(analytics)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get customer insights for forecasting: {e}")
            return {}
    
    def _get_top_risk_factors(self, churn_risks: List) -> List[str]:
        """Extract top risk factors from churn analysis"""
        try:
            all_factors = []
            for risk in churn_risks:
                all_factors.extend(risk.key_risk_factors)
            
            # Count frequency of each factor
            factor_counts = {}
            for factor in all_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            # Return top 5 factors
            sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
            return [factor for factor, count in sorted_factors[:5]]
            
        except Exception as e:
            logger.error(f"Failed to get top risk factors: {e}")
            return []
    
    def _calculate_forecasting_adjustments(self, analytics: CustomerAnalytics) -> Dict[str, float]:
        """Calculate adjustments for forecasting models based on customer analytics"""
        try:
            adjustments = {
                'retention_multiplier': 1.0,
                'growth_multiplier': 1.0,
                'seasonality_adjustment': 1.0,
                'risk_adjustment': 1.0
            }
            
            # Retention-based adjustment
            if analytics.overall_retention_rate > 0.9:
                adjustments['retention_multiplier'] = 1.1  # Boost forecast for high retention
            elif analytics.overall_retention_rate < 0.7:
                adjustments['retention_multiplier'] = 0.9  # Reduce forecast for low retention
            
            # Growth-based adjustment from segments
            if analytics.customer_segments:
                avg_growth = np.mean([seg.growth_rate for seg in analytics.customer_segments])
                if avg_growth > 10:
                    adjustments['growth_multiplier'] = 1.05
                elif avg_growth < -10:
                    adjustments['growth_multiplier'] = 0.95
            
            # Risk-based adjustment
            if analytics.total_customers > 0:
                high_risk_percentage = analytics.churned_customers / analytics.total_customers
                if high_risk_percentage > 0.3:
                    adjustments['risk_adjustment'] = 0.9  # High churn risk
                elif high_risk_percentage < 0.1:
                    adjustments['risk_adjustment'] = 1.05  # Low churn risk
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Failed to calculate forecasting adjustments: {e}")
            return {'retention_multiplier': 1.0, 'growth_multiplier': 1.0, 
                   'seasonality_adjustment': 1.0, 'risk_adjustment': 1.0}
    
    def get_customer_performance_tracking(self, company_id: str) -> Dict[str, Any]:
        """Get customer performance metrics for tracking dashboard"""
        try:
            analytics = self.analyze_company_customers(company_id)
            if not analytics:
                return {}
            
            # Calculate performance metrics
            performance_data = {
                'overview_metrics': {
                    'total_customers': analytics.total_customers,
                    'active_customers': analytics.active_customers,
                    'new_customers': analytics.new_customers_this_month,
                    'retention_rate': round(analytics.overall_retention_rate * 100, 2),
                    'average_ltv': round(analytics.overall_ltv, 2),
                    'churn_risk_count': analytics.churned_customers
                },
                'ltv_distribution': self._calculate_ltv_distribution(analytics.customer_ltvs),
                'segment_breakdown': [
                    {
                        'segment': segment.segment_name,
                        'customers': segment.customer_count,
                        'percentage': round(segment.customer_count / analytics.total_customers * 100, 1) 
                                    if analytics.total_customers > 0 else 0,
                        'avg_ltv': round(segment.avg_ltv, 2),
                        'revenue_share': round(segment.revenue_contribution, 1),
                        'health_score': round(segment.segment_health_score, 2)
                    }
                    for segment in analytics.customer_segments
                ],
                'churn_risk_breakdown': {
                    'critical': len([r for r in analytics.churn_risks if r.risk_level == 'Critical']),
                    'high': len([r for r in analytics.churn_risks if r.risk_level == 'High']),
                    'medium': len([r for r in analytics.churn_risks if r.risk_level == 'Medium']),
                    'low': len([r for r in analytics.churn_risks if r.risk_level == 'Low'])
                },
                'cohort_performance': [
                    {
                        'cohort': cohort.cohort_month,
                        'size': cohort.cohort_size,
                        'ltv': round(cohort.avg_ltv, 2),
                        'retention_1m': round(cohort.retention_rates.get(1, 0) * 100, 1),
                        'retention_3m': round(cohort.retention_rates.get(3, 0) * 100, 1),
                        'retention_6m': round(cohort.retention_rates.get(6, 0) * 100, 1)
                    }
                    for cohort in analytics.cohort_analyses[-6:]  # Last 6 cohorts
                ],
                'key_insights': analytics.key_insights,
                'recommendations': analytics.recommendations,
                'last_updated': analytics.analysis_date.isoformat()
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get customer performance tracking: {e}")
            return {}
    
    def _calculate_ltv_distribution(self, customer_ltvs: List) -> Dict[str, int]:
        """Calculate LTV distribution for visualization"""
        try:
            if not customer_ltvs:
                return {}
            
            ltv_values = [ltv.combined_ltv for ltv in customer_ltvs]
            
            # Define LTV buckets
            buckets = {
                '0-500': 0,
                '500-1000': 0,
                '1000-2500': 0,
                '2500-5000': 0,
                '5000+': 0
            }
            
            for ltv in ltv_values:
                if ltv < 500:
                    buckets['0-500'] += 1
                elif ltv < 1000:
                    buckets['500-1000'] += 1
                elif ltv < 2500:
                    buckets['1000-2500'] += 1
                elif ltv < 5000:
                    buckets['2500-5000'] += 1
                else:
                    buckets['5000+'] += 1
            
            return buckets
            
        except Exception as e:
            logger.error(f"Failed to calculate LTV distribution: {e}")
            return {}
    
    def get_actionable_customer_insights(self, company_id: str) -> Dict[str, Any]:
        """Get actionable insights for business decision making"""
        try:
            analytics = self.analyze_company_customers(company_id)
            if not analytics:
                return {}
            
            # Generate actionable insights
            actionable_insights = {
                'immediate_actions': [],
                'strategic_recommendations': [],
                'performance_alerts': [],
                'opportunities': []
            }
            
            # Immediate actions based on churn risk
            critical_customers = [r for r in analytics.churn_risks if r.risk_level == 'Critical']
            if critical_customers:
                actionable_insights['immediate_actions'].append({
                    'priority': 'Critical',
                    'action': f'Contact {len(critical_customers)} critical-risk customers immediately',
                    'impact': 'Prevent immediate churn and revenue loss',
                    'timeline': '24-48 hours'
                })
            
            high_risk_customers = [r for r in analytics.churn_risks if r.risk_level == 'High']
            if high_risk_customers:
                actionable_insights['immediate_actions'].append({
                    'priority': 'High',
                    'action': f'Launch retention campaign for {len(high_risk_customers)} high-risk customers',
                    'impact': 'Reduce churn rate by 15-25%',
                    'timeline': '1 week'
                })
            
            # Strategic recommendations based on segments
            if analytics.customer_segments:
                top_segment = max(analytics.customer_segments, key=lambda x: x.revenue_contribution)
                actionable_insights['strategic_recommendations'].append({
                    'recommendation': f'Develop VIP program for {top_segment.segment_name} segment',
                    'rationale': f'This segment contributes {top_segment.revenue_contribution:.1f}% of revenue',
                    'expected_impact': 'Increase retention by 10-15% and LTV by 20%'
                })
            
            # Performance alerts
            if analytics.overall_retention_rate < 0.7:
                actionable_insights['performance_alerts'].append({
                    'alert': 'Low retention rate detected',
                    'current_value': f'{analytics.overall_retention_rate * 100:.1f}%',
                    'benchmark': '70%+',
                    'recommended_action': 'Review customer experience and implement retention strategies'
                })
            
            # Opportunities
            new_customer_percentage = (analytics.new_customers_this_month / analytics.total_customers * 100) if analytics.total_customers > 0 else 0
            if new_customer_percentage > 20:
                actionable_insights['opportunities'].append({
                    'opportunity': 'High new customer acquisition',
                    'description': f'{new_customer_percentage:.1f}% of customers are new this month',
                    'action': 'Implement onboarding program to maximize new customer LTV'
                })
            
            return actionable_insights
            
        except Exception as e:
            logger.error(f"Failed to get actionable customer insights: {e}")
            return {}
    
    def export_customer_analytics(self, company_id: str, format: str = 'json') -> Optional[str]:
        """Export customer analytics data in specified format"""
        try:
            analytics = self.analyze_company_customers(company_id)
            if not analytics:
                return None
            
            # Prepare export data
            export_data = {
                'company_id': company_id,
                'analysis_date': analytics.analysis_date.isoformat(),
                'summary': {
                    'total_customers': analytics.total_customers,
                    'active_customers': analytics.active_customers,
                    'retention_rate': analytics.overall_retention_rate,
                    'average_ltv': analytics.overall_ltv
                },
                'customer_ltvs': [
                    {
                        'customer_id': ltv.customer_id,
                        'combined_ltv': ltv.combined_ltv,
                        'ltv_percentile': ltv.ltv_percentile,
                        'segment': ltv.segment,
                        'risk_score': ltv.risk_score
                    }
                    for ltv in analytics.customer_ltvs
                ],
                'segments': [
                    {
                        'segment_name': seg.segment_name,
                        'customer_count': seg.customer_count,
                        'avg_ltv': seg.avg_ltv,
                        'revenue_contribution': seg.revenue_contribution,
                        'health_score': seg.segment_health_score
                    }
                    for seg in analytics.customer_segments
                ],
                'churn_risks': [
                    {
                        'customer_id': risk.customer_id,
                        'churn_probability': risk.churn_probability,
                        'risk_level': risk.risk_level,
                        'recommended_actions': risk.recommended_actions
                    }
                    for risk in analytics.churn_risks
                ],
                'insights': analytics.key_insights,
                'recommendations': analytics.recommendations
            }
            
            # Create export file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"customer_analytics_{company_id}_{timestamp}.{format}"
            
            company_dir = Path("company_data") / company_id
            company_dir.mkdir(exist_ok=True)
            filepath = company_dir / filename
            
            if format.lower() == 'json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Export key metrics as CSV
                df = pd.DataFrame([
                    {
                        'customer_id': ltv.customer_id,
                        'ltv': ltv.combined_ltv,
                        'segment': ltv.segment,
                        'risk_score': ltv.risk_score,
                        'churn_probability': next((r.churn_probability for r in analytics.churn_risks 
                                                if r.customer_id == ltv.customer_id), 0)
                    }
                    for ltv in analytics.customer_ltvs
                ])
                df.to_csv(filepath, index=False)
            
            logger.info(f"Customer analytics exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to export customer analytics: {e}")
            return None
    
    def clear_analytics_cache(self, company_id: Optional[str] = None):
        """Clear analytics cache for specific company or all companies"""
        if company_id:
            cache_key = f"{company_id}_customer_analytics"
            if cache_key in self.analytics_cache:
                del self.analytics_cache[cache_key]
                logger.info(f"Cleared analytics cache for company {company_id}")
        else:
            self.analytics_cache.clear()
            logger.info("Cleared all analytics cache")