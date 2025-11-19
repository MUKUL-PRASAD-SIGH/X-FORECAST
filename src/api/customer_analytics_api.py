"""
Customer Analytics API Endpoints
Provides REST API endpoints for customer analytics functionality
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import os

from src.customer_analytics.customer_analytics_integration import CustomerAnalyticsIntegration
from src.customer_analytics.customer_analytics_engine import CustomerAnalytics

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/customer-analytics", tags=["Customer Analytics"])

# Initialize customer analytics integration
customer_analytics_integration = CustomerAnalyticsIntegration()

@router.get("/companies/{company_id}/analytics")
async def get_customer_analytics(
    company_id: str,
    force_refresh: bool = Query(False, description="Force refresh of analytics data")
) -> Dict[str, Any]:
    """Get comprehensive customer analytics for a company"""
    try:
        logger.info(f"Getting customer analytics for company {company_id}")
        
        analytics = customer_analytics_integration.analyze_company_customers(
            company_id, force_refresh=force_refresh
        )
        
        if not analytics:
            raise HTTPException(
                status_code=404, 
                detail=f"No customer analytics available for company {company_id}"
            )
        
        # Convert to API response format
        response_data = {
            'company_id': company_id,
            'analysis_date': analytics.analysis_date.isoformat(),
            'overview': {
                'total_customers': analytics.total_customers,
                'active_customers': analytics.active_customers,
                'new_customers_this_month': analytics.new_customers_this_month,
                'churned_customers': analytics.churned_customers,
                'overall_ltv': round(analytics.overall_ltv, 2),
                'overall_retention_rate': round(analytics.overall_retention_rate * 100, 2)
            },
            'customer_segments': [
                {
                    'segment_id': segment.segment_id,
                    'segment_name': segment.segment_name,
                    'customer_count': segment.customer_count,
                    'avg_ltv': round(segment.avg_ltv, 2),
                    'avg_retention_rate': round(segment.avg_retention_rate * 100, 2),
                    'revenue_contribution': round(segment.revenue_contribution, 2),
                    'growth_rate': round(segment.growth_rate, 2),
                    'health_score': round(segment.segment_health_score, 2),
                    'characteristics': segment.characteristics
                }
                for segment in analytics.customer_segments
            ],
            'churn_analysis': {
                'total_at_risk': len([r for r in analytics.churn_risks if r.risk_level in ['High', 'Critical']]),
                'risk_breakdown': {
                    'critical': len([r for r in analytics.churn_risks if r.risk_level == 'Critical']),
                    'high': len([r for r in analytics.churn_risks if r.risk_level == 'High']),
                    'medium': len([r for r in analytics.churn_risks if r.risk_level == 'Medium']),
                    'low': len([r for r in analytics.churn_risks if r.risk_level == 'Low'])
                }
            },
            'cohort_summary': [
                {
                    'cohort_month': cohort.cohort_month,
                    'cohort_size': cohort.cohort_size,
                    'avg_ltv': round(cohort.avg_ltv, 2),
                    'churn_rate': round(cohort.churn_rate * 100, 2),
                    'months_tracked': cohort.months_tracked
                }
                for cohort in analytics.cohort_analyses[-6:]  # Last 6 cohorts
            ],
            'key_insights': analytics.key_insights,
            'recommendations': analytics.recommendations
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get customer analytics for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/ltv-analysis")
async def get_ltv_analysis(
    company_id: str,
    top_n: int = Query(50, description="Number of top customers to return")
) -> Dict[str, Any]:
    """Get detailed LTV analysis for customers"""
    try:
        analytics = customer_analytics_integration.analyze_company_customers(company_id)
        
        if not analytics:
            raise HTTPException(
                status_code=404, 
                detail=f"No LTV data available for company {company_id}"
            )
        
        # Sort customers by LTV and get top N
        sorted_ltvs = sorted(analytics.customer_ltvs, key=lambda x: x.combined_ltv, reverse=True)
        top_customers = sorted_ltvs[:top_n]
        
        # Calculate LTV distribution
        ltv_values = [ltv.combined_ltv for ltv in analytics.customer_ltvs]
        
        response_data = {
            'company_id': company_id,
            'total_customers_analyzed': len(analytics.customer_ltvs),
            'ltv_statistics': {
                'average_ltv': round(sum(ltv_values) / len(ltv_values), 2) if ltv_values else 0,
                'median_ltv': round(sorted(ltv_values)[len(ltv_values)//2], 2) if ltv_values else 0,
                'max_ltv': round(max(ltv_values), 2) if ltv_values else 0,
                'min_ltv': round(min(ltv_values), 2) if ltv_values else 0
            },
            'top_customers': [
                {
                    'customer_id': ltv.customer_id,
                    'combined_ltv': round(ltv.combined_ltv, 2),
                    'historical_ltv': round(ltv.historical_ltv, 2),
                    'predicted_ltv': round(ltv.predicted_ltv, 2),
                    'ltv_percentile': round(ltv.ltv_percentile, 1),
                    'avg_order_value': round(ltv.avg_order_value, 2),
                    'purchase_frequency': round(ltv.purchase_frequency, 2),
                    'customer_lifespan_days': ltv.customer_lifespan_days,
                    'risk_score': round(ltv.risk_score, 3),
                    'segment': ltv.segment
                }
                for ltv in top_customers
            ],
            'ltv_distribution': customer_analytics_integration._calculate_ltv_distribution(analytics.customer_ltvs)
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get LTV analysis for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/churn-risks")
async def get_churn_risks(
    company_id: str,
    risk_level: Optional[str] = Query(None, description="Filter by risk level: Critical, High, Medium, Low")
) -> Dict[str, Any]:
    """Get churn risk analysis for customers"""
    try:
        analytics = customer_analytics_integration.analyze_company_customers(company_id)
        
        if not analytics:
            raise HTTPException(
                status_code=404, 
                detail=f"No churn risk data available for company {company_id}"
            )
        
        # Filter by risk level if specified
        churn_risks = analytics.churn_risks
        if risk_level:
            churn_risks = [r for r in churn_risks if r.risk_level.lower() == risk_level.lower()]
        
        # Sort by churn probability (highest first)
        churn_risks = sorted(churn_risks, key=lambda x: x.churn_probability, reverse=True)
        
        response_data = {
            'company_id': company_id,
            'total_customers_analyzed': len(analytics.churn_risks),
            'filtered_results': len(churn_risks),
            'risk_summary': {
                'critical': len([r for r in analytics.churn_risks if r.risk_level == 'Critical']),
                'high': len([r for r in analytics.churn_risks if r.risk_level == 'High']),
                'medium': len([r for r in analytics.churn_risks if r.risk_level == 'Medium']),
                'low': len([r for r in analytics.churn_risks if r.risk_level == 'Low'])
            },
            'churn_risks': [
                {
                    'customer_id': risk.customer_id,
                    'churn_probability': round(risk.churn_probability * 100, 2),
                    'risk_level': risk.risk_level,
                    'key_risk_factors': risk.key_risk_factors,
                    'days_since_last_purchase': risk.days_since_last_purchase,
                    'purchase_trend': risk.purchase_trend,
                    'recommended_actions': risk.recommended_actions,
                    'confidence_score': round(risk.confidence_score * 100, 2)
                }
                for risk in churn_risks
            ]
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get churn risks for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/cohort-analysis")
async def get_cohort_analysis(company_id: str) -> Dict[str, Any]:
    """Get detailed cohort analysis"""
    try:
        analytics = customer_analytics_integration.analyze_company_customers(company_id)
        
        if not analytics:
            raise HTTPException(
                status_code=404, 
                detail=f"No cohort data available for company {company_id}"
            )
        
        response_data = {
            'company_id': company_id,
            'total_cohorts': len(analytics.cohort_analyses),
            'cohorts': [
                {
                    'cohort_month': cohort.cohort_month,
                    'cohort_size': cohort.cohort_size,
                    'avg_ltv': round(cohort.avg_ltv, 2),
                    'churn_rate': round(cohort.churn_rate * 100, 2),
                    'months_tracked': cohort.months_tracked,
                    'retention_rates': {
                        str(month): round(rate * 100, 2) 
                        for month, rate in cohort.retention_rates.items()
                    },
                    'revenue_per_period': {
                        str(month): round(revenue, 2) 
                        for month, revenue in cohort.revenue_per_cohort.items()
                    }
                }
                for cohort in analytics.cohort_analyses
            ]
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get cohort analysis for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/performance-tracking")
async def get_performance_tracking(company_id: str) -> Dict[str, Any]:
    """Get customer performance tracking metrics for dashboard"""
    try:
        performance_data = customer_analytics_integration.get_customer_performance_tracking(company_id)
        
        if not performance_data:
            raise HTTPException(
                status_code=404, 
                detail=f"No performance data available for company {company_id}"
            )
        
        return {
            'company_id': company_id,
            **performance_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance tracking for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/actionable-insights")
async def get_actionable_insights(company_id: str) -> Dict[str, Any]:
    """Get actionable business insights based on customer analytics"""
    try:
        insights = customer_analytics_integration.get_actionable_customer_insights(company_id)
        
        if not insights:
            raise HTTPException(
                status_code=404, 
                detail=f"No actionable insights available for company {company_id}"
            )
        
        return {
            'company_id': company_id,
            'generated_at': datetime.now().isoformat(),
            **insights
        }
        
    except Exception as e:
        logger.error(f"Failed to get actionable insights for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/forecasting-insights")
async def get_forecasting_insights(company_id: str) -> Dict[str, Any]:
    """Get customer insights that can enhance forecasting models"""
    try:
        insights = customer_analytics_integration.get_customer_insights_for_forecasting(company_id)
        
        if not insights:
            raise HTTPException(
                status_code=404, 
                detail=f"No forecasting insights available for company {company_id}"
            )
        
        return {
            'company_id': company_id,
            'generated_at': datetime.now().isoformat(),
            **insights
        }
        
    except Exception as e:
        logger.error(f"Failed to get forecasting insights for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/companies/{company_id}/export")
async def export_customer_analytics(
    company_id: str,
    format: str = Query("json", description="Export format: json or csv"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, str]:
    """Export customer analytics data"""
    try:
        if format.lower() not in ['json', 'csv']:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        filepath = customer_analytics_integration.export_customer_analytics(company_id, format)
        
        if not filepath:
            raise HTTPException(
                status_code=404, 
                detail=f"No data available for export for company {company_id}"
            )
        
        return {
            'message': 'Export completed successfully',
            'filepath': filepath,
            'format': format,
            'company_id': company_id,
            'exported_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to export customer analytics for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies/{company_id}/export/{filename}")
async def download_export_file(company_id: str, filename: str):
    """Download exported customer analytics file"""
    try:
        filepath = f"company_data/{company_id}/{filename}"
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Export file not found")
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Failed to download export file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/companies/{company_id}/cache")
async def clear_analytics_cache(company_id: str) -> Dict[str, str]:
    """Clear analytics cache for a company to force refresh"""
    try:
        customer_analytics_integration.clear_analytics_cache(company_id)
        
        return {
            'message': f'Analytics cache cleared for company {company_id}',
            'company_id': company_id,
            'cleared_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for customer analytics API"""
    return {
        'status': 'healthy',
        'service': 'customer-analytics-api',
        'timestamp': datetime.now().isoformat()
    }

# Note: Exception handlers should be added to the main app, not router
# These would be added in main.py if needed

# Exception handlers removed - should be added to main app if needed