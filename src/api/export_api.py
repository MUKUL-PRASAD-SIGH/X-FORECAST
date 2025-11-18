"""
Comprehensive Export API for Ensemble Forecasting System
Provides multi-format export functionality for forecasts, model performance, and business insights
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime, timedelta
import logging
import os
import tempfile
from pathlib import Path

# Import required components
try:
    from models.ensemble_forecasting_engine import EnsembleForecastingEngine, EnsembleResult
    from models.business_insights_engine import BusinessInsightsEngine, BusinessInsightsResult
    from models.model_performance_tracker import ModelPerformanceTracker
    from company_sales.company_data_manager import CompanyDataManager
    from company_sales.company_forecasting_engine import CompanyForecastingEngine
except ImportError as e:
    logging.warning(f"Import warning in export_api: {e}")

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

router = APIRouter(prefix="/api/export", tags=["Export & Reporting"])

# Pydantic models
class ExportRequest(BaseModel):
    format: str = Field(..., description="Export format: pdf, excel, json")
    include_forecasts: bool = Field(True, description="Include forecast data")
    include_performance: bool = Field(True, description="Include model performance metrics")
    include_insights: bool = Field(True, description="Include business insights")
    include_metadata: bool = Field(True, description="Include model metadata")
    include_charts: bool = Field(False, description="Include chart data (for supported formats)")
    custom_title: Optional[str] = Field(None, description="Custom report title")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    horizon_months: Optional[int] = Field(6, description="Forecast horizon for new forecasts")

class ExportResponse(BaseModel):
    success: bool
    filename: str
    download_url: str
    file_size_bytes: int
    format: str
    generated_at: str
    expires_at: str
    metadata: Dict[str, Any]

class ReportMetadata(BaseModel):
    """Metadata for export reports"""
    report_id: str
    company_id: str
    generated_at: datetime
    report_type: str
    format: str
    data_sources: List[str]
    model_versions: Dict[str, str]
    confidence_levels: List[float]
    forecast_horizon: int
    data_quality_score: float

# Helper functions
def get_company_id_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract company ID from authorization token"""
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# Initialize components
try:
    company_data_manager = CompanyDataManager()
    company_forecasting_engine = CompanyForecastingEngine(company_data_manager)
    ensemble_engine = EnsembleForecastingEngine()
    insights_engine = BusinessInsightsEngine()
    performance_tracker = ModelPerformanceTracker()
except Exception as e:
    logger.warning(f"Component initialization warning: {e}")
    company_data_manager = None
    company_forecasting_engine = None
    ensemble_engine = None
    insights_engine = None
    performance_tracker = None

class ExportEngine:
    """Core export functionality engine"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "forecast_exports"
        self.temp_dir.mkdir(exist_ok=True)
        
        # File retention (24 hours)
        self.file_retention_hours = 24
        
    async def generate_comprehensive_report(self, 
                                          company_id: str, 
                                          export_request: ExportRequest) -> Dict[str, Any]:
        """Generate comprehensive export report with all requested components"""
        try:
            logger.info(f"Generating comprehensive report for company {company_id}")
            
            # Collect all data components
            report_data = {
                'metadata': await self._generate_metadata(company_id, export_request),
                'forecasts': None,
                'performance': None,
                'insights': None,
                'model_info': None
            }
            
            # Generate forecasts if requested
            if export_request.include_forecasts:
                report_data['forecasts'] = await self._generate_forecast_data(company_id, export_request)
            
            # Generate performance data if requested
            if export_request.include_performance:
                report_data['performance'] = await self._generate_performance_data(company_id)
            
            # Generate business insights if requested
            if export_request.include_insights:
                report_data['insights'] = await self._generate_insights_data(company_id, export_request)
            
            # Generate model metadata if requested
            if export_request.include_metadata:
                report_data['model_info'] = await self._generate_model_metadata(company_id)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")
            raise
    
    async def _generate_metadata(self, company_id: str, export_request: ExportRequest) -> Dict[str, Any]:
        """Generate report metadata"""
        try:
            return {
                'report_id': f"export_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'company_id': company_id,
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_forecast_report',
                'format': export_request.format,
                'title': export_request.custom_title or f"Ensemble Forecast Report - {company_id}",
                'export_options': export_request.dict(),
                'data_sources': ['ensemble_forecasting', 'performance_tracking', 'business_insights'],
                'model_versions': {
                    'arima': '1.0',
                    'ets': '1.0', 
                    'xgboost': '1.0',
                    'lstm': '1.0',
                    'croston': '1.0'
                },
                'confidence_levels': [0.1, 0.5, 0.9],
                'forecast_horizon': export_request.horizon_months,
                'data_quality_score': 0.85  # Would be calculated from actual data
            }
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_forecast_data(self, company_id: str, export_request: ExportRequest) -> Dict[str, Any]:
        """Generate forecast data for export"""
        try:
            forecast_data = {
                'ensemble_forecast': None,
                'individual_forecasts': {},
                'confidence_intervals': {},
                'model_weights': {},
                'forecast_accuracy': None,
                'historical_data': None
            }
            
            # Initialize historical_data variable
            historical_data = None
            
            # Get historical data
            if company_data_manager:
                try:
                    historical_data = company_data_manager.load_company_data(company_id)
                    if historical_data is not None and not historical_data.empty:
                        forecast_data['historical_data'] = {
                            'data': historical_data.to_dict('records'),
                            'summary': {
                                'total_records': len(historical_data),
                                'date_range': {
                                    'start': historical_data['date'].min().isoformat() if 'date' in historical_data.columns else None,
                                    'end': historical_data['date'].max().isoformat() if 'date' in historical_data.columns else None
                                },
                                'total_sales': float(historical_data['sales_amount'].sum()) if 'sales_amount' in historical_data.columns else 0
                            }
                        }
                except Exception as e:
                    logger.warning(f"Historical data loading failed: {e}")
            
            # Generate new forecast using ensemble engine
            if ensemble_engine:
                try:
                    if company_data_manager and historical_data is not None and not historical_data.empty:
                        ensemble_result = await ensemble_engine.process_new_data(historical_data)
                    else:
                        ensemble_result = await ensemble_engine.generate_forecast(export_request.horizon_months)
                    
                    # Extract forecast data
                    if ensemble_result:
                        forecast_data['ensemble_forecast'] = {
                            'dates': [date.isoformat() for date in ensemble_result.point_forecast.index],
                            'values': ensemble_result.point_forecast.tolist(),
                            'horizon_months': ensemble_result.forecast_horizon
                        }
                        
                        # Individual model forecasts
                        for model_name, forecast in ensemble_result.individual_forecasts.items():
                            forecast_data['individual_forecasts'][model_name] = {
                                'dates': [date.isoformat() for date in forecast.index],
                                'values': forecast.tolist()
                            }
                        
                        # Confidence intervals
                        for level, interval in ensemble_result.confidence_intervals.items():
                            forecast_data['confidence_intervals'][level] = {
                                'dates': [date.isoformat() for date in interval.index],
                                'values': interval.tolist()
                            }
                        
                        # Model weights
                        forecast_data['model_weights'] = ensemble_result.model_weights
                        
                        # Forecast accuracy
                        forecast_data['forecast_accuracy'] = {
                            'ensemble_accuracy': ensemble_result.ensemble_accuracy,
                            'data_quality_score': ensemble_result.data_quality_score,
                            'pattern_analysis': {
                                'pattern_type': ensemble_result.pattern_analysis.pattern_type,
                                'confidence': ensemble_result.pattern_analysis.confidence,
                                'seasonality_strength': ensemble_result.pattern_analysis.seasonality_strength,
                                'trend_strength': ensemble_result.pattern_analysis.trend_strength
                            }
                        }
                        
                except Exception as e:
                    logger.error(f"Ensemble forecast generation failed: {e}")
                    forecast_data['error'] = str(e)
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Forecast data generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_performance_data(self, company_id: str) -> Dict[str, Any]:
        """Generate model performance data for export"""
        try:
            performance_data = {
                'current_performance': {},
                'performance_history': [],
                'weight_evolution': [],
                'model_rankings': [],
                'accuracy_trends': {},
                'drift_detection': {}
            }
            
            # Get current performance from ensemble engine
            if ensemble_engine:
                try:
                    current_metrics = ensemble_engine.get_performance_metrics()
                    performance_data['current_performance'] = current_metrics
                except Exception as e:
                    logger.warning(f"Current performance retrieval failed: {e}")
            
            # Get performance history from company forecasting engine
            if company_forecasting_engine:
                try:
                    performance_history = company_forecasting_engine.company_performance_history.get(company_id, [])
                    performance_data['performance_history'] = [
                        {
                            'model_name': perf.model_name,
                            'mae': float(perf.mae) if pd.notna(perf.mae) else None,
                            'mape': float(perf.mape) if pd.notna(perf.mape) else None,
                            'rmse': float(perf.rmse) if pd.notna(perf.rmse) else None,
                            'r_squared': float(perf.r_squared) if pd.notna(perf.r_squared) else None,
                            'weight': perf.weight,
                            'evaluation_date': perf.evaluation_date.isoformat(),
                            'data_points': perf.data_points
                        }
                        for perf in performance_history[-50:]  # Last 50 records
                    ]
                    
                    # Weight evolution
                    weight_history = company_forecasting_engine.company_weight_history.get(company_id, [])
                    performance_data['weight_evolution'] = [
                        {
                            'update_date': record.update_date.isoformat(),
                            'old_weights': record.old_weights,
                            'new_weights': record.new_weights,
                            'trigger_reason': record.trigger_reason
                        }
                        for record in weight_history[-20:]  # Last 20 weight changes
                    ]
                    
                except Exception as e:
                    logger.warning(f"Performance history retrieval failed: {e}")
            
            # Get comprehensive performance tracking data
            if performance_tracker:
                try:
                    # Model rankings
                    model_rankings = await performance_tracker.get_model_rankings()
                    performance_data['model_rankings'] = model_rankings
                    
                    # Accuracy trends
                    for model_name in ['arima', 'ets', 'xgboost', 'lstm', 'croston']:
                        try:
                            trend_data = await performance_tracker.get_accuracy_trend(model_name)
                            performance_data['accuracy_trends'][model_name] = trend_data
                        except Exception as e:
                            logger.warning(f"Accuracy trend retrieval failed for {model_name}: {e}")
                    
                    # Drift detection results
                    for model_name in ['arima', 'ets', 'xgboost', 'lstm', 'croston']:
                        try:
                            drift_result = await performance_tracker.detect_model_drift(model_name)
                            performance_data['drift_detection'][model_name] = {
                                'drift_type': drift_result.drift_type.value,
                                'drift_score': drift_result.drift_score,
                                'confidence': drift_result.confidence,
                                'detected_at': drift_result.detected_at.isoformat(),
                                'requires_retraining': drift_result.requires_retraining
                            }
                        except Exception as e:
                            logger.warning(f"Drift detection failed for {model_name}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Comprehensive performance tracking failed: {e}")
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Performance data generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_insights_data(self, company_id: str, export_request: ExportRequest) -> Dict[str, Any]:
        """Generate business insights data for export"""
        try:
            insights_data = {
                'executive_summary': '',
                'key_findings': [],
                'performance_analysis': {},
                'growth_indicators': {},
                'category_performance': [],
                'regional_performance': [],
                'risk_assessment': {},
                'opportunities': {},
                'recommendations': [],
                'confidence_score': 0.0
            }
            
            if not insights_engine:
                insights_data['error'] = 'Business insights engine not available'
                return insights_data
            
            # Get company data for insights generation
            if company_data_manager:
                try:
                    company_data = company_data_manager.load_company_data(company_id)
                    
                    if company_data.empty:
                        insights_data['error'] = 'No company data available for insights generation'
                        return insights_data
                    
                    # Generate ensemble result for insights
                    if ensemble_engine:
                        ensemble_result = await ensemble_engine.process_new_data(company_data)
                        
                        # Create mock pattern analysis (would be from actual pattern detection)
                        from models.advanced_pattern_detection import AdvancedPatternAnalysis, TrendAnalysis, SeasonalityAnalysis, VolatilityAnalysis, AnomalyDetection
                        
                        mock_pattern_analysis = AdvancedPatternAnalysis(
                            trend_analysis=TrendAnalysis(
                                trend_type='increasing',
                                trend_strength=0.7,
                                trend_direction=1.0,
                                trend_acceleration=0.1,
                                trend_consistency=0.8,
                                trend_significance=0.9
                            ),
                            seasonality_analysis=SeasonalityAnalysis(
                                seasonal_strength=0.6,
                                seasonal_period=12,
                                seasonal_consistency=0.7,
                                seasonal_peaks=[6, 12],
                                seasonal_troughs=[2, 8],
                                seasonal_amplitude=0.3
                            ),
                            volatility_analysis=VolatilityAnalysis(
                                volatility_level='medium',
                                coefficient_of_variation=0.25,
                                volatility_trend='stable',
                                risk_score=0.4,
                                volatility_clusters=[],
                                stability_periods=[]
                            ),
                            anomaly_detection=AnomalyDetection(
                                anomaly_score=0.1,
                                anomaly_count=2,
                                anomaly_dates=[],
                                anomaly_severity='low',
                                anomaly_patterns=[],
                                anomaly_impact=0.05
                            ),
                            confidence_score=0.8
                        )
                        
                        # Generate comprehensive insights
                        insights_result = insights_engine.generate_comprehensive_insights(
                            data=company_data,
                            ensemble_result=ensemble_result,
                            pattern_analysis=mock_pattern_analysis
                        )
                        
                        # Convert insights to exportable format
                        insights_data.update({
                            'executive_summary': insights_result.executive_summary,
                            'key_findings': insights_result.key_findings,
                            'performance_analysis': {
                                'revenue_growth_rate': insights_result.performance_analysis.revenue_growth_rate,
                                'revenue_trend': insights_result.performance_analysis.revenue_trend,
                                'performance_score': insights_result.performance_analysis.performance_score,
                                'benchmark_comparison': insights_result.performance_analysis.benchmark_comparison,
                                'key_performance_indicators': insights_result.performance_analysis.key_performance_indicators,
                                'performance_drivers': insights_result.performance_analysis.performance_drivers,
                                'performance_risks': insights_result.performance_analysis.performance_risks
                            },
                            'growth_indicators': {
                                'monthly_growth_rate': insights_result.growth_indicators.monthly_growth_rate,
                                'quarterly_growth_rate': insights_result.growth_indicators.quarterly_growth_rate,
                                'year_over_year_growth': insights_result.growth_indicators.year_over_year_growth,
                                'growth_acceleration': insights_result.growth_indicators.growth_acceleration,
                                'growth_sustainability_score': insights_result.growth_indicators.growth_sustainability_score,
                                'growth_drivers': insights_result.growth_indicators.growth_drivers,
                                'growth_barriers': insights_result.growth_indicators.growth_barriers,
                                'growth_forecast': insights_result.growth_indicators.growth_forecast
                            },
                            'category_performance': [
                                {
                                    'category_name': cat.category_name,
                                    'revenue_share': cat.revenue_share,
                                    'growth_rate': cat.growth_rate,
                                    'performance_rank': cat.performance_rank,
                                    'trend_direction': cat.trend_direction,
                                    'seasonality_impact': cat.seasonality_impact,
                                    'volatility_level': cat.volatility_level,
                                    'opportunities': cat.opportunities,
                                    'risks': cat.risks
                                }
                                for cat in insights_result.category_performance
                            ],
                            'regional_performance': [
                                {
                                    'region_name': reg.region_name,
                                    'revenue_share': reg.revenue_share,
                                    'growth_rate': reg.growth_rate,
                                    'performance_rank': reg.performance_rank,
                                    'market_penetration': reg.market_penetration,
                                    'competitive_position': reg.competitive_position,
                                    'regional_trends': reg.regional_trends,
                                    'expansion_opportunities': reg.expansion_opportunities
                                }
                                for reg in insights_result.regional_performance
                            ],
                            'risk_assessment': {
                                'overall_risk_score': insights_result.risk_assessment.overall_risk_score,
                                'risk_level': insights_result.risk_assessment.risk_level,
                                'primary_risks': insights_result.risk_assessment.primary_risks,
                                'risk_mitigation_strategies': insights_result.risk_assessment.risk_mitigation_strategies,
                                'risk_monitoring_metrics': insights_result.risk_assessment.risk_monitoring_metrics,
                                'contingency_plans': insights_result.risk_assessment.contingency_plans
                            },
                            'opportunities': {
                                'opportunities': insights_result.opportunity_identification.opportunities,
                                'opportunity_score': insights_result.opportunity_identification.opportunity_score,
                                'quick_wins': insights_result.opportunity_identification.quick_wins,
                                'strategic_opportunities': insights_result.opportunity_identification.strategic_opportunities,
                                'resource_requirements': insights_result.opportunity_identification.resource_requirements,
                                'expected_roi': insights_result.opportunity_identification.expected_roi
                            },
                            'recommendations': [
                                {
                                    'insight_type': rec.insight_type,
                                    'title': rec.title,
                                    'description': rec.description,
                                    'confidence': rec.confidence,
                                    'impact_score': rec.impact_score,
                                    'supporting_data': rec.supporting_data,
                                    'recommended_actions': rec.recommended_actions,
                                    'urgency': rec.urgency
                                }
                                for rec in insights_result.actionable_recommendations
                            ],
                            'confidence_score': insights_result.confidence_score
                        })
                        
                except Exception as e:
                    logger.error(f"Insights generation failed: {e}")
                    insights_data['error'] = str(e)
            
            return insights_data
            
        except Exception as e:
            logger.error(f"Insights data generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_model_metadata(self, company_id: str) -> Dict[str, Any]:
        """Generate model metadata for export"""
        try:
            metadata = {
                'model_versions': {
                    'arima': {'version': '1.0', 'library': 'statsmodels', 'parameters': {}},
                    'ets': {'version': '1.0', 'library': 'statsmodels', 'parameters': {}},
                    'xgboost': {'version': '1.0', 'library': 'xgboost', 'parameters': {}},
                    'lstm': {'version': '1.0', 'library': 'pytorch', 'parameters': {}},
                    'croston': {'version': '1.0', 'library': 'custom', 'parameters': {}}
                },
                'ensemble_configuration': {
                    'adaptive_learning_enabled': True,
                    'min_model_weight': 0.05,
                    'max_model_weight': 0.7,
                    'weight_smoothing_factor': 0.1,
                    'performance_window_days': 30,
                    'confidence_levels': [0.1, 0.5, 0.9]
                },
                'data_requirements': {
                    'minimum_data_points': 12,
                    'required_columns': ['date', 'sales_amount'],
                    'optional_columns': ['product_category', 'region'],
                    'data_frequency': 'monthly'
                },
                'training_information': {
                    'last_training_date': datetime.now().isoformat(),
                    'training_data_points': 0,
                    'validation_method': 'time_series_split',
                    'performance_metrics': ['mae', 'mape', 'rmse', 'r_squared']
                }
            }
            
            # Get actual model status if available
            if ensemble_engine:
                try:
                    model_status = ensemble_engine.get_model_status_summary()
                    metadata['model_status'] = model_status
                except Exception as e:
                    logger.warning(f"Model status retrieval failed: {e}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Model metadata generation failed: {e}")
            return {'error': str(e)}

# Initialize export engine
export_engine = ExportEngine()

# Export format handlers
# Import the comprehensive formatter
try:
    from utils.export_formatters import comprehensive_formatter
    COMPREHENSIVE_FORMATTER_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_FORMATTER_AVAILABLE = False
    logger.warning("Comprehensive formatter not available, using basic export")

class ExportFormatHandler:
    """Handles different export formats with comprehensive formatting"""
    
    @staticmethod
    def to_json(data: Dict[str, Any], filename: str, metadata: Dict[str, Any] = None) -> str:
        """Export data to comprehensive JSON format"""
        try:
            if COMPREHENSIVE_FORMATTER_AVAILABLE:
                return comprehensive_formatter.to_json(data, filename, metadata)
            else:
                # Fallback to basic JSON export
                return ExportFormatHandler._basic_json_export(data, filename)
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise
    
    @staticmethod
    def to_excel(data: Dict[str, Any], filename: str, metadata: Dict[str, Any] = None) -> str:
        """Export data to comprehensive Excel format"""
        try:
            if COMPREHENSIVE_FORMATTER_AVAILABLE:
                return comprehensive_formatter.to_excel(data, filename, metadata)
            else:
                # Fallback to basic Excel export
                return ExportFormatHandler._basic_excel_export(data, filename)
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise
    
    @staticmethod
    def to_pdf(data: Dict[str, Any], filename: str, metadata: Dict[str, Any] = None) -> str:
        """Export data to comprehensive PDF format"""
        try:
            if COMPREHENSIVE_FORMATTER_AVAILABLE:
                return comprehensive_formatter.to_pdf(data, filename, metadata)
            else:
                # Fallback to basic PDF export
                return ExportFormatHandler._basic_pdf_export(data, filename)
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise
    
    @staticmethod
    def _basic_json_export(data: Dict[str, Any], filename: str) -> str:
        """Basic JSON export fallback"""
        file_path = export_engine.temp_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        clean_data = convert_numpy_types(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(file_path)
    
    @staticmethod
    def _basic_excel_export(data: Dict[str, Any], filename: str) -> str:
        """Basic Excel export fallback"""
        file_path = export_engine.temp_dir / filename
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Basic metadata sheet
            if 'metadata' in data and data['metadata']:
                metadata_df = pd.DataFrame([data['metadata']])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Basic forecasts sheet
            if 'forecasts' in data and data['forecasts']:
                forecasts = data['forecasts']
                if forecasts.get('ensemble_forecast'):
                    ensemble_df = pd.DataFrame({
                        'Date': forecasts['ensemble_forecast']['dates'],
                        'Ensemble_Forecast': forecasts['ensemble_forecast']['values']
                    })
                    ensemble_df.to_excel(writer, sheet_name='Ensemble_Forecast', index=False)
        
        return str(file_path)
    
    @staticmethod
    def _basic_pdf_export(data: Dict[str, Any], filename: str) -> str:
        """Basic PDF export fallback"""
        file_path = export_engine.temp_dir / filename.replace('.pdf', '.txt')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("ENSEMBLE FORECAST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if 'metadata' in data and data['metadata']:
                f.write("REPORT METADATA\n")
                f.write("-" * 20 + "\n")
                metadata = data['metadata']
                f.write(f"Report ID: {metadata.get('report_id', 'N/A')}\n")
                f.write(f"Company ID: {metadata.get('company_id', 'N/A')}\n")
                f.write(f"Generated At: {metadata.get('generated_at', 'N/A')}\n\n")
        
        return str(file_path)

# API Endpoints

@router.post("/comprehensive-report", response_model=ExportResponse)
async def export_comprehensive_report(
    export_request: ExportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Export comprehensive forecast report with all requested components and metadata"""
    
    try:
        logger.info(f"Exporting comprehensive report for company {company_id} in {export_request.format} format")
        
        start_time = datetime.now()
        
        # Generate comprehensive report data
        report_data = await export_engine.generate_comprehensive_report(company_id, export_request)
        
        # Generate comprehensive metadata
        if COMPREHENSIVE_FORMATTER_AVAILABLE:
            export_metadata = comprehensive_formatter.generate_export_metadata(
                company_id=company_id,
                export_format=export_request.format,
                export_options=export_request.dict(),
                data_summary=report_data.get('metadata', {})
            )
        else:
            export_metadata = {
                'export_id': f"export_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'company_id': company_id,
                'generated_at': datetime.now().isoformat(),
                'format': export_request.format,
                'forecast_horizon': export_request.horizon_months
            }
        
        # Generate filename with export ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_id_short = export_metadata.get('export_id', 'unknown')[-8:]  # Last 8 chars
        filename = f"ensemble_forecast_report_{company_id}_{timestamp}_{export_id_short}.{export_request.format}"
        
        # Export to requested format with metadata
        if export_request.format.lower() == 'json':
            file_path = ExportFormatHandler.to_json(report_data, filename, export_metadata)
        elif export_request.format.lower() == 'excel':
            file_path = ExportFormatHandler.to_excel(report_data, filename.replace('.excel', '.xlsx'), export_metadata)
        elif export_request.format.lower() == 'pdf':
            file_path = ExportFormatHandler.to_pdf(report_data, filename, export_metadata)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_request.format}")
        
        # Get file size and calculate generation time
        file_size = os.path.getsize(file_path)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate expiration time
        expires_at = datetime.now() + timedelta(hours=export_engine.file_retention_hours)
        
        # Enhanced response metadata with comprehensive model information
        response_metadata = {
            'export_id': export_metadata.get('export_id'),
            'company_id': company_id,
            'generation_time_seconds': generation_time,
            'components_included': {
                'forecasts': export_request.include_forecasts,
                'performance': export_request.include_performance,
                'insights': export_request.include_insights,
                'metadata': export_request.include_metadata,
                'charts': export_request.include_charts
            },
            'data_quality_score': report_data.get('metadata', {}).get('data_quality_score', 0.85),
            'forecast_horizon': export_request.horizon_months,
            'model_versions': export_metadata.get('model_versions', {}),
            'confidence_levels': export_metadata.get('confidence_levels', []),
            'data_points_included': comprehensive_formatter._count_data_points(report_data) if COMPREHENSIVE_FORMATTER_AVAILABLE else 0,
            'model_information': {
                'ensemble_models': ['ARIMA', 'ETS', 'XGBoost', 'LSTM', 'Croston'],
                'adaptive_weighting': True,
                'performance_tracking': True,
                'pattern_detection': True,
                'business_insights': export_request.include_insights,
                'real_time_updates': True
            },
            'export_features': {
                'comprehensive_formatting': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'cyberpunk_styling': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'advanced_charts': export_request.include_charts and COMPREHENSIVE_FORMATTER_AVAILABLE,
                'professional_pdf': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'advanced_excel': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'metadata_inclusion': export_request.include_metadata,
                'multi_sheet_excel': export_request.format.lower() == 'excel',
                'structured_json': export_request.format.lower() == 'json'
            },
            'system_capabilities': {
                'ensemble_forecasting': True,
                'model_performance_tracking': True,
                'business_insights_generation': True,
                'pattern_detection': True,
                'automated_training': True,
                'real_time_monitoring': True
            }
        }
        
        logger.info(f"Export completed successfully: {filename} ({file_size / 1024 / 1024:.2f} MB, {generation_time:.2f}s)")
        
        return ExportResponse(
            success=True,
            filename=os.path.basename(file_path),
            download_url=f"/api/export/download/{os.path.basename(file_path)}",
            file_size_bytes=file_size,
            format=export_request.format,
            generated_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata=response_metadata
        )
        
    except Exception as e:
        logger.error(f"Comprehensive report export failed for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/download/{filename}")
async def download_export_file(filename: str):
    """Download exported file"""
    
    try:
        file_path = export_engine.temp_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found or expired")
        
        # Determine media type based on file extension
        if filename.endswith('.json'):
            media_type = 'application/json'
        elif filename.endswith('.xlsx'):
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif filename.endswith('.txt'):
            media_type = 'text/plain'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
        
    except Exception as e:
        logger.error(f"File download failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/formats")
async def get_supported_formats():
    """Get list of supported export formats with comprehensive feature information"""
    
    base_formats = [
        {
            'format': 'json',
            'description': 'Comprehensive JSON with metadata and structured data',
            'file_extension': '.json',
            'supports_charts': False,
            'max_size_mb': 100,
            'features': {
                'metadata_inclusion': True,
                'structured_data': True,
                'machine_readable': True,
                'human_readable': True,
                'cyberpunk_styling': False
            }
        },
        {
            'format': 'excel',
            'description': 'Multi-sheet Excel workbook with professional formatting',
            'file_extension': '.xlsx',
            'supports_charts': COMPREHENSIVE_FORMATTER_AVAILABLE,
            'max_size_mb': 150,
            'features': {
                'metadata_inclusion': True,
                'multiple_sheets': True,
                'professional_styling': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'cyberpunk_colors': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'auto_formatting': COMPREHENSIVE_FORMATTER_AVAILABLE
            }
        },
        {
            'format': 'pdf',
            'description': 'Professional PDF report with comprehensive formatting',
            'file_extension': '.pdf',
            'supports_charts': COMPREHENSIVE_FORMATTER_AVAILABLE,
            'max_size_mb': 75,
            'features': {
                'metadata_inclusion': True,
                'professional_layout': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'cyberpunk_styling': COMPREHENSIVE_FORMATTER_AVAILABLE,
                'executive_summary': True,
                'recommendations': True
            }
        }
    ]
    
    return {
        'supported_formats': base_formats,
        'export_components': [
            {
                'component': 'forecasts',
                'description': 'Ensemble and individual model forecasts with confidence intervals',
                'includes': ['ensemble_forecast', 'individual_forecasts', 'confidence_intervals', 'model_weights', 'historical_data']
            },
            {
                'component': 'performance',
                'description': 'Model performance metrics and evolution tracking',
                'includes': ['current_performance', 'performance_history', 'weight_evolution', 'model_rankings', 'accuracy_trends', 'drift_detection']
            },
            {
                'component': 'insights',
                'description': 'AI-generated business insights and recommendations',
                'includes': ['executive_summary', 'key_findings', 'performance_analysis', 'growth_indicators', 'category_performance', 'regional_performance', 'risk_assessment', 'opportunities', 'recommendations']
            },
            {
                'component': 'metadata',
                'description': 'Comprehensive technical metadata and model information',
                'includes': ['model_versions', 'ensemble_configuration', 'data_requirements', 'training_information', 'system_info']
            }
        ],
        'system_capabilities': {
            'comprehensive_formatting': COMPREHENSIVE_FORMATTER_AVAILABLE,
            'cyberpunk_styling': COMPREHENSIVE_FORMATTER_AVAILABLE,
            'professional_pdf': COMPREHENSIVE_FORMATTER_AVAILABLE,
            'advanced_excel': COMPREHENSIVE_FORMATTER_AVAILABLE,
            'metadata_generation': True,
            'multi_format_support': True
        }
    }

@router.post("/forecast-only", response_model=ExportResponse)
async def export_forecast_only(
    export_request: ExportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Export only forecast data with minimal metadata"""
    
    try:
        logger.info(f"Exporting forecast-only report for company {company_id}")
        
        # Override export options to include only forecasts
        forecast_request = export_request.copy()
        forecast_request.include_forecasts = True
        forecast_request.include_performance = False
        forecast_request.include_insights = False
        forecast_request.include_metadata = True  # Keep minimal metadata
        
        # Generate forecast data only
        report_data = await export_engine.generate_comprehensive_report(company_id, forecast_request)
        
        # Filter to only forecast data
        filtered_data = {
            'metadata': report_data.get('metadata', {}),
            'forecasts': report_data.get('forecasts', {})
        }
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"forecast_only_{company_id}_{timestamp}.{export_request.format}"
        
        # Export with minimal metadata
        if COMPREHENSIVE_FORMATTER_AVAILABLE:
            export_metadata = comprehensive_formatter.generate_export_metadata(
                company_id=company_id,
                export_format=export_request.format,
                export_options=forecast_request.dict()
            )
            export_metadata['report_title'] = f"Forecast Report - {company_id}"
        else:
            export_metadata = {'company_id': company_id, 'report_type': 'forecast_only'}
        
        # Export to requested format
        if export_request.format.lower() == 'json':
            file_path = ExportFormatHandler.to_json(filtered_data, filename, export_metadata)
        elif export_request.format.lower() == 'excel':
            file_path = ExportFormatHandler.to_excel(filtered_data, filename.replace('.excel', '.xlsx'), export_metadata)
        elif export_request.format.lower() == 'pdf':
            file_path = ExportFormatHandler.to_pdf(filtered_data, filename, export_metadata)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_request.format}")
        
        file_size = os.path.getsize(file_path)
        expires_at = datetime.now() + timedelta(hours=export_engine.file_retention_hours)
        
        return ExportResponse(
            success=True,
            filename=os.path.basename(file_path),
            download_url=f"/api/export/download/{os.path.basename(file_path)}",
            file_size_bytes=file_size,
            format=export_request.format,
            generated_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata={
                'company_id': company_id,
                'report_type': 'forecast_only',
                'forecast_horizon': export_request.horizon_months
            }
        )
        
    except Exception as e:
        logger.error(f"Forecast-only export failed for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/performance-report", response_model=ExportResponse)
async def export_performance_report(
    export_request: ExportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Export comprehensive model performance report"""
    
    try:
        logger.info(f"Exporting performance report for company {company_id}")
        
        # Override export options to focus on performance
        perf_request = export_request.copy()
        perf_request.include_forecasts = False
        perf_request.include_performance = True
        perf_request.include_insights = False
        perf_request.include_metadata = True
        
        # Generate performance data
        report_data = await export_engine.generate_comprehensive_report(company_id, perf_request)
        
        # Filter to performance data
        filtered_data = {
            'metadata': report_data.get('metadata', {}),
            'performance': report_data.get('performance', {}),
            'model_info': report_data.get('model_info', {})
        }
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_report_{company_id}_{timestamp}.{export_request.format}"
        
        # Export with performance-focused metadata
        if COMPREHENSIVE_FORMATTER_AVAILABLE:
            export_metadata = comprehensive_formatter.generate_export_metadata(
                company_id=company_id,
                export_format=export_request.format,
                export_options=perf_request.dict()
            )
            export_metadata['report_title'] = f"Model Performance Report - {company_id}"
        else:
            export_metadata = {'company_id': company_id, 'report_type': 'performance_report'}
        
        # Export to requested format
        if export_request.format.lower() == 'json':
            file_path = ExportFormatHandler.to_json(filtered_data, filename, export_metadata)
        elif export_request.format.lower() == 'excel':
            file_path = ExportFormatHandler.to_excel(filtered_data, filename.replace('.excel', '.xlsx'), export_metadata)
        elif export_request.format.lower() == 'pdf':
            file_path = ExportFormatHandler.to_pdf(filtered_data, filename, export_metadata)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_request.format}")
        
        file_size = os.path.getsize(file_path)
        expires_at = datetime.now() + timedelta(hours=export_engine.file_retention_hours)
        
        return ExportResponse(
            success=True,
            filename=os.path.basename(file_path),
            download_url=f"/api/export/download/{os.path.basename(file_path)}",
            file_size_bytes=file_size,
            format=export_request.format,
            generated_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata={
                'company_id': company_id,
                'report_type': 'performance_report',
                'models_analyzed': 5
            }
        )
        
    except Exception as e:
        logger.error(f"Performance report export failed for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/insights-report", response_model=ExportResponse)
async def export_insights_report(
    export_request: ExportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Export AI-generated business insights and recommendations report"""
    
    try:
        logger.info(f"Exporting insights report for company {company_id}")
        
        # Override export options to focus on insights
        insights_request = export_request.copy()
        insights_request.include_forecasts = False
        insights_request.include_performance = False
        insights_request.include_insights = True
        insights_request.include_metadata = True
        
        # Generate insights data
        report_data = await export_engine.generate_comprehensive_report(company_id, insights_request)
        
        # Filter to insights data
        filtered_data = {
            'metadata': report_data.get('metadata', {}),
            'insights': report_data.get('insights', {})
        }
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"business_insights_{company_id}_{timestamp}.{export_request.format}"
        
        # Export with insights-focused metadata
        if COMPREHENSIVE_FORMATTER_AVAILABLE:
            export_metadata = comprehensive_formatter.generate_export_metadata(
                company_id=company_id,
                export_format=export_request.format,
                export_options=insights_request.dict()
            )
            export_metadata['report_title'] = f"Business Insights Report - {company_id}"
        else:
            export_metadata = {'company_id': company_id, 'report_type': 'insights_report'}
        
        # Export to requested format
        if export_request.format.lower() == 'json':
            file_path = ExportFormatHandler.to_json(filtered_data, filename, export_metadata)
        elif export_request.format.lower() == 'excel':
            file_path = ExportFormatHandler.to_excel(filtered_data, filename.replace('.excel', '.xlsx'), export_metadata)
        elif export_request.format.lower() == 'pdf':
            file_path = ExportFormatHandler.to_pdf(filtered_data, filename, export_metadata)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_request.format}")
        
        file_size = os.path.getsize(file_path)
        expires_at = datetime.now() + timedelta(hours=export_engine.file_retention_hours)
        
        return ExportResponse(
            success=True,
            filename=os.path.basename(file_path),
            download_url=f"/api/export/download/{os.path.basename(file_path)}",
            file_size_bytes=file_size,
            format=export_request.format,
            generated_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata={
                'company_id': company_id,
                'report_type': 'insights_report',
                'confidence_score': filtered_data.get('insights', {}).get('confidence_score', 0.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Insights report export failed for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/model-performance-detailed", response_model=ExportResponse)
async def export_detailed_model_performance(
    export_request: ExportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Export detailed model performance report with weight evolution and accuracy metrics"""
    
    try:
        logger.info(f"Exporting detailed model performance for company {company_id}")
        
        # Generate comprehensive performance data
        report_data = await export_engine.generate_comprehensive_report(company_id, export_request)
        
        # Enhance performance data with additional metrics
        enhanced_performance = report_data.get('performance', {})
        
        # Add model comparison matrix
        if enhanced_performance.get('model_rankings'):
            model_comparison = []
            for model in enhanced_performance['model_rankings']:
                model_comparison.append({
                    'model_name': model.get('model_name', 'Unknown'),
                    'current_accuracy': model.get('accuracy', 0),
                    'rank': model.get('rank', 0),
                    'weight': model.get('weight', 0),
                    'performance_trend': model.get('trend', 'stable'),
                    'last_updated': model.get('last_updated', datetime.now().isoformat())
                })
            enhanced_performance['model_comparison_matrix'] = model_comparison
        
        # Add weight evolution summary
        if enhanced_performance.get('weight_evolution'):
            weight_summary = {
                'total_weight_changes': len(enhanced_performance['weight_evolution']),
                'most_recent_change': enhanced_performance['weight_evolution'][-1] if enhanced_performance['weight_evolution'] else None,
                'weight_stability_score': 0.85,  # Would be calculated from actual data
                'adaptive_learning_active': True
            }
            enhanced_performance['weight_evolution_summary'] = weight_summary
        
        # Create enhanced report data
        enhanced_data = {
            'metadata': report_data.get('metadata', {}),
            'performance': enhanced_performance,
            'model_info': report_data.get('model_info', {}),
            'forecasts': report_data.get('forecasts', {}) if export_request.include_forecasts else None
        }
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"detailed_performance_{company_id}_{timestamp}.{export_request.format}"
        
        # Export with enhanced metadata
        if COMPREHENSIVE_FORMATTER_AVAILABLE:
            export_metadata = comprehensive_formatter.generate_export_metadata(
                company_id=company_id,
                export_format=export_request.format,
                export_options=export_request.dict()
            )
            export_metadata['report_title'] = f"Detailed Model Performance Analysis - {company_id}"
            export_metadata['report_type'] = 'detailed_performance'
        else:
            export_metadata = {'company_id': company_id, 'report_type': 'detailed_performance'}
        
        # Export to requested format
        if export_request.format.lower() == 'json':
            file_path = ExportFormatHandler.to_json(enhanced_data, filename, export_metadata)
        elif export_request.format.lower() == 'excel':
            file_path = ExportFormatHandler.to_excel(enhanced_data, filename.replace('.excel', '.xlsx'), export_metadata)
        elif export_request.format.lower() == 'pdf':
            file_path = ExportFormatHandler.to_pdf(enhanced_data, filename, export_metadata)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_request.format}")
        
        file_size = os.path.getsize(file_path)
        expires_at = datetime.now() + timedelta(hours=export_engine.file_retention_hours)
        
        return ExportResponse(
            success=True,
            filename=os.path.basename(file_path),
            download_url=f"/api/export/download/{os.path.basename(file_path)}",
            file_size_bytes=file_size,
            format=export_request.format,
            generated_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata={
                'company_id': company_id,
                'report_type': 'detailed_performance',
                'models_analyzed': 5,
                'weight_changes_tracked': enhanced_performance.get('weight_evolution_summary', {}).get('total_weight_changes', 0),
                'performance_metrics': ['MAE', 'MAPE', 'RMSE', 'R²', 'Accuracy', 'Drift Detection']
            }
        )
        
    except Exception as e:
        logger.error(f"Detailed performance export failed for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.delete("/cleanup")
async def cleanup_expired_files():
    """Clean up expired export files"""
    
    try:
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(hours=export_engine.file_retention_hours)
        
        for file_path in export_engine.temp_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
        
        return {
            'success': True,
            'cleaned_files': cleaned_count,
            'cleanup_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")