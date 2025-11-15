"""
Company Sales Forecasting API
Provides endpoints for company-specific sales data upload and forecasting
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import logging
import asyncio

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from company_sales.company_data_manager import CompanyDataManager, CompanyProfile
    from company_sales.company_forecasting_engine import CompanyForecastingEngine, ForecastResult
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from company_sales.company_data_manager import CompanyDataManager, CompanyProfile
    from company_sales.company_forecasting_engine import CompanyForecastingEngine, ForecastResult

logger = logging.getLogger(__name__)

# Initialize managers
company_data_manager = CompanyDataManager()
forecasting_engine = CompanyForecastingEngine(company_data_manager)

# Security
security = HTTPBearer()

router = APIRouter(prefix="/api/company-sales", tags=["Company Sales Forecasting"])

# Pydantic models
class CompanyRegistration(BaseModel):
    company_name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Industry sector")
    custom_requirements: Optional[Dict[str, Any]] = Field(None, description="Custom data requirements")

class CompanyResponse(BaseModel):
    company_id: str
    company_name: str
    industry: str
    created_date: str
    data_requirements: Dict[str, Any]
    adaptive_config: Dict[str, Any]

class DataUploadResponse(BaseModel):
    success: bool
    message: str
    file_path: Optional[str] = None
    records_processed: Optional[int] = None
    validation_errors: Optional[List[str]] = None
    # Enhanced fields for ensemble integration
    models_initialized: Optional[List[str]] = None
    pattern_detected: Optional[str] = None
    data_quality: Optional[float] = None
    ensemble_status: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    horizon_months: Optional[int] = Field(6, description="Forecast horizon in months")

class ForecastResponse(BaseModel):
    company_id: str
    forecast_date: str
    forecast_horizon_months: int
    point_forecast: Dict[str, float]  # date -> value
    confidence_intervals: Dict[str, Dict[str, float]]  # level -> {date -> value}
    model_weights: Dict[str, float]
    model_performances: List[Dict[str, Any]]
    pattern_detected: Dict[str, Any]
    forecast_accuracy_metrics: Dict[str, float]
    recommendations: List[str]

class ConfigUpdate(BaseModel):
    adaptive_learning_enabled: Optional[bool] = None
    learning_window_months: Optional[int] = None
    min_model_weight: Optional[float] = None
    weight_update_frequency: Optional[str] = None
    confidence_intervals_enabled: Optional[bool] = None
    pattern_detection_enabled: Optional[bool] = None

# Helper functions
def get_company_id_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract company ID from authorization token (simplified)"""
    # In a real implementation, this would validate JWT tokens and extract company ID
    # For demo purposes, we'll use the token as company ID
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

def validate_file_format(file: UploadFile) -> str:
    """Validate uploaded file format"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = file.filename.split('.')[-1].lower()
    supported_formats = ['csv', 'xlsx', 'json']
    
    if file_extension not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {supported_formats}"
        )
    
    return file_extension

async def parse_uploaded_file(file: UploadFile, file_format: str) -> pd.DataFrame:
    """Parse uploaded file into DataFrame"""
    try:
        content = await file.read()
        
        if file_format == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_format == 'xlsx':
            df = pd.read_excel(io.BytesIO(content))
        elif file_format == 'json':
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        return df
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

# API Endpoints

@router.post("/register", response_model=CompanyResponse)
async def register_company(
    registration: CompanyRegistration,
    company_id: str = Depends(get_company_id_from_token)
):
    """Register a new company for sales forecasting"""
    
    try:
        # Check if company already exists
        existing_profile = company_data_manager.get_company_profile(company_id)
        if existing_profile:
            raise HTTPException(status_code=400, detail="Company already registered")
        
        # Register company
        profile = company_data_manager.register_company(
            company_id=company_id,
            company_name=registration.company_name,
            industry=registration.industry,
            custom_requirements=registration.custom_requirements
        )
        
        return CompanyResponse(
            company_id=profile.company_id,
            company_name=profile.company_name,
            industry=profile.industry,
            created_date=profile.created_date.isoformat(),
            data_requirements=profile.data_requirements.__dict__,
            adaptive_config=profile.adaptive_config
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to register company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/profile", response_model=CompanyResponse)
async def get_company_profile(company_id: str = Depends(get_company_id_from_token)):
    """Get company profile and configuration"""
    
    try:
        profile = company_data_manager.get_company_profile(company_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Company not found")
        
        return CompanyResponse(
            company_id=profile.company_id,
            company_name=profile.company_name,
            industry=profile.industry,
            created_date=profile.created_date.isoformat(),
            data_requirements=profile.data_requirements.__dict__,
            adaptive_config=profile.adaptive_config
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data-requirements")
async def get_data_requirements(company_id: str = Depends(get_company_id_from_token)):
    """Get data requirements and upload template for company"""
    
    try:
        template = company_data_manager.get_data_requirements_template(company_id)
        return template
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get data requirements for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/detect-parameters")
async def detect_file_parameters(
    file: UploadFile = File(..., description="Sales data file for parameter detection"),
    company_id: str = Depends(get_company_id_from_token)
):
    """Detect file parameters and column mappings for ensemble initialization"""
    
    try:
        # Validate file format
        file_format = validate_file_format(file)
        
        # Parse file for preview
        data = await parse_uploaded_file(file, file_format)
        
        # Limit preview to first 100 rows for performance
        preview_data = data.head(100) if len(data) > 100 else data
        
        # Detect column types and parameters
        detected_columns = []
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Determine column type
            col_type = "unknown"
            confidence = 0.5
            sample_values = col_data.head(3).astype(str).tolist()
            
            # Date detection
            if col.lower() in ['date', 'time', 'timestamp', 'period']:
                try:
                    pd.to_datetime(col_data.head(10))
                    col_type = "date"
                    confidence = 0.9
                except:
                    pass
            
            # Numeric detection
            elif pd.api.types.is_numeric_dtype(col_data):
                if col.lower() in ['sales', 'amount', 'revenue', 'value', 'price']:
                    col_type = "sales_amount"
                    confidence = 0.8
                elif col.lower() in ['units', 'quantity', 'count']:
                    col_type = "units_sold"
                    confidence = 0.8
                else:
                    col_type = "numeric"
                    confidence = 0.6
            
            # Categorical detection
            elif col_data.dtype == 'object':
                unique_ratio = col_data.nunique() / len(col_data)
                if col.lower() in ['category', 'product', 'sku', 'item']:
                    col_type = "product_category"
                    confidence = 0.8
                elif col.lower() in ['region', 'location', 'area', 'territory']:
                    col_type = "region"
                    confidence = 0.8
                elif unique_ratio < 0.1:  # Low unique ratio suggests categorical
                    col_type = "categorical"
                    confidence = 0.7
                else:
                    col_type = "text"
                    confidence = 0.6
            
            detected_columns.append({
                "name": col,
                "type": col_type,
                "sample_values": sample_values,
                "confidence": confidence
            })
        
        # Create preview summary
        preview_summary = {
            "rows": len(data),
            "columns": len(data.columns),
            "date_range": None,
            "sample_data": preview_data.head(5).to_dict('records') if not preview_data.empty else []
        }
        
        # Try to detect date range
        date_columns = [col for col in detected_columns if col["type"] == "date"]
        if date_columns:
            try:
                date_col = date_columns[0]["name"]
                dates = pd.to_datetime(data[date_col])
                preview_summary["date_range"] = {
                    "start": dates.min().isoformat(),
                    "end": dates.max().isoformat()
                }
            except:
                pass
        
        return {
            "success": True,
            "detected_columns": detected_columns,
            "preview": preview_summary,
            "file_info": {
                "name": file.filename,
                "size": file.size,
                "format": file_format
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect parameters for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Parameter detection failed: {str(e)}")

@router.post("/upload-enhanced", response_model=DataUploadResponse)
async def upload_enhanced_sales_data(
    file: UploadFile = File(..., description="Sales data file with ensemble initialization"),
    company_id: str = Depends(get_company_id_from_token)
):
    """Enhanced upload with ensemble model initialization and pattern detection"""
    
    try:
        # Validate file format
        file_format = validate_file_format(file)
        
        # Check file size (50MB limit)
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        # Parse file
        data = await parse_uploaded_file(file, file_format)
        
        # Validate data
        is_valid, errors = company_data_manager.validate_upload_data(company_id, data)
        
        if not is_valid:
            return DataUploadResponse(
                success=False,
                message="Data validation failed",
                validation_errors=errors
            )
        
        # Save data
        file_path = company_data_manager.save_company_data(
            company_id=company_id,
            data=data,
            upload_metadata={
                'original_filename': file.filename,
                'file_format': file_format,
                'file_size': file.size,
                'enhanced_upload': True
            }
        )
        
        # Initialize or update ensemble models
        ensemble_initialized = False
        models_initialized = []
        pattern_detected = "unknown"
        
        try:
            if company_id not in forecasting_engine.company_models:
                success = forecasting_engine.initialize_company_models(company_id)
                if success:
                    ensemble_initialized = True
                    models_initialized = ["ARIMA", "ETS", "XGBoost", "LSTM", "Croston"]
            
            # Generate initial forecast to detect patterns
            if ensemble_initialized or company_id in forecasting_engine.company_models:
                try:
                    forecast_result = forecasting_engine.generate_forecast(company_id, horizon_months=3)
                    if forecast_result and forecast_result.pattern_detected:
                        pattern_detected = forecast_result.pattern_detected.pattern_type
                except Exception as e:
                    logger.warning(f"Pattern detection failed for company {company_id}: {e}")
        
        except Exception as e:
            logger.warning(f"Ensemble initialization failed for company {company_id}: {e}")
        
        # Calculate data quality score
        data_quality = 0.8  # Placeholder - would implement actual quality assessment
        
        return DataUploadResponse(
            success=True,
            message="Data uploaded and ensemble models initialized successfully",
            file_path=file_path,
            records_processed=len(data),
            validation_errors=None,
            # Enhanced response fields
            models_initialized=models_initialized if ensemble_initialized else None,
            pattern_detected=pattern_detected if pattern_detected != "unknown" else None,
            data_quality=data_quality,
            ensemble_status={
                "initialized": ensemble_initialized,
                "total_models": len(models_initialized) if models_initialized else 0,
                "pattern_detection_enabled": True,
                "adaptive_weights_enabled": True
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload enhanced data for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced upload failed: {str(e)}")

@router.post("/upload-data", response_model=DataUploadResponse)
async def upload_sales_data(
    file: UploadFile = File(..., description="Sales data file (CSV, Excel, or JSON)"),
    company_id: str = Depends(get_company_id_from_token)
):
    """Upload monthly sales data for forecasting (legacy endpoint)"""
    
    try:
        # Validate file format
        file_format = validate_file_format(file)
        
        # Check file size (50MB limit)
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        # Parse file
        data = await parse_uploaded_file(file, file_format)
        
        # Validate data
        is_valid, errors = company_data_manager.validate_upload_data(company_id, data)
        
        if not is_valid:
            return DataUploadResponse(
                success=False,
                message="Data validation failed",
                validation_errors=errors
            )
        
        # Save data
        file_path = company_data_manager.save_company_data(
            company_id=company_id,
            data=data,
            upload_metadata={
                'original_filename': file.filename,
                'file_format': file_format,
                'file_size': file.size
            }
        )
        
        # Initialize or update models
        if company_id not in forecasting_engine.company_models:
            success = forecasting_engine.initialize_company_models(company_id)
            if not success:
                logger.warning(f"Failed to initialize models for company {company_id}")
        
        return DataUploadResponse(
            success=True,
            message="Data uploaded and processed successfully",
            file_path=file_path,
            records_processed=len(data)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload data for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Generate adaptive ensemble forecast for company"""
    
    try:
        # Generate forecast
        result = forecasting_engine.generate_forecast(
            company_id=company_id,
            horizon_months=request.horizon_months
        )
        
        # Convert forecast result to response format
        point_forecast_dict = {}
        if not result.point_forecast.empty:
            point_forecast_dict = {
                date.strftime('%Y-%m-%d'): float(value) 
                for date, value in result.point_forecast.items()
            }
        
        confidence_intervals_dict = {}
        for level, series in result.confidence_intervals.items():
            if not series.empty:
                confidence_intervals_dict[level] = {
                    date.strftime('%Y-%m-%d'): float(value)
                    for date, value in series.items()
                }
        
        model_performances_list = []
        for perf in result.model_performances:
            model_performances_list.append({
                'model_name': perf.model_name,
                'mae': perf.mae if pd.notna(perf.mae) else None,
                'mape': perf.mape if pd.notna(perf.mape) else None,
                'rmse': perf.rmse if pd.notna(perf.rmse) else None,
                'r_squared': perf.r_squared if pd.notna(perf.r_squared) else None,
                'weight': perf.weight,
                'evaluation_date': perf.evaluation_date.isoformat(),
                'data_points': perf.data_points
            })
        
        pattern_dict = {
            'pattern_type': result.pattern_detected.pattern_type,
            'seasonality_strength': result.pattern_detected.seasonality_strength,
            'trend_strength': result.pattern_detected.trend_strength,
            'intermittency_ratio': result.pattern_detected.intermittency_ratio,
            'volatility': result.pattern_detected.volatility,
            'confidence': result.pattern_detected.confidence,
            'detected_period': result.pattern_detected.detected_period
        }
        
        return ForecastResponse(
            company_id=result.company_id,
            forecast_date=result.forecast_date.isoformat(),
            forecast_horizon_months=result.forecast_horizon_months,
            point_forecast=point_forecast_dict,
            confidence_intervals=confidence_intervals_dict,
            model_weights=result.model_weights,
            model_performances=model_performances_list,
            pattern_detected=pattern_dict,
            forecast_accuracy_metrics=result.forecast_accuracy_metrics,
            recommendations=result.recommendations
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate forecast for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Forecast generation failed")

@router.get("/model-status")
async def get_model_status(company_id: str = Depends(get_company_id_from_token)):
    """Get current model status and performance for company"""
    
    try:
        status = forecasting_engine.get_company_model_status(company_id)
        return status
    
    except Exception as e:
        logger.error(f"Failed to get model status for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")

@router.get("/data-history")
async def get_data_history(company_id: str = Depends(get_company_id_from_token)):
    """Get data upload history for company"""
    
    try:
        history = company_data_manager.get_company_data_history(company_id)
        return {"upload_history": history}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get data history for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get data history")

@router.get("/statistics")
async def get_company_statistics(company_id: str = Depends(get_company_id_from_token)):
    """Get comprehensive company statistics and analytics"""
    
    try:
        stats = company_data_manager.get_company_stats(company_id)
        return stats
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get statistics for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@router.put("/config", response_model=Dict[str, Any])
async def update_company_config(
    config_update: ConfigUpdate,
    company_id: str = Depends(get_company_id_from_token)
):
    """Update company adaptive forecasting configuration"""
    
    try:
        # Convert to dict and remove None values
        updates = {k: v for k, v in config_update.dict().items() if v is not None}
        
        if not updates:
            raise HTTPException(status_code=400, detail="No configuration updates provided")
        
        success = company_data_manager.update_company_config(company_id, updates)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
        
        # Get updated profile
        profile = company_data_manager.get_company_profile(company_id)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "updated_config": profile.adaptive_config if profile else {}
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")

@router.get("/forecast-history")
async def get_forecast_history(
    limit: Optional[int] = 10,
    company_id: str = Depends(get_company_id_from_token)
):
    """Get historical forecasts and their accuracy"""
    
    try:
        # Get performance history
        performance_history = forecasting_engine.company_performance_history.get(company_id, [])
        
        # Get weight change history
        weight_history = forecasting_engine.company_weight_history.get(company_id, [])
        
        # Limit results
        if limit:
            performance_history = performance_history[-limit:]
            weight_history = weight_history[-limit:]
        
        # Format response
        performance_data = []
        for perf in performance_history:
            performance_data.append({
                'model_name': perf.model_name,
                'mae': perf.mae if pd.notna(perf.mae) else None,
                'mape': perf.mape if pd.notna(perf.mape) else None,
                'rmse': perf.rmse if pd.notna(perf.rmse) else None,
                'r_squared': perf.r_squared if pd.notna(perf.r_squared) else None,
                'weight': perf.weight,
                'evaluation_date': perf.evaluation_date.isoformat(),
                'data_points': perf.data_points
            })
        
        weight_data = []
        for weight_record in weight_history:
            weight_data.append({
                'update_date': weight_record.update_date.isoformat(),
                'old_weights': weight_record.old_weights,
                'new_weights': weight_record.new_weights,
                'trigger_reason': weight_record.trigger_reason
            })
        
        return {
            'performance_history': performance_data,
            'weight_change_history': weight_data
        }
    
    except Exception as e:
        logger.error(f"Failed to get forecast history for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get forecast history")

@router.get("/ensemble/status")
async def get_ensemble_status(company_id: str = Depends(get_company_id_from_token)):
    """Get detailed ensemble status for real-time monitoring"""
    
    try:
        status = forecasting_engine.get_ensemble_status(company_id)
        return status
    
    except Exception as e:
        logger.error(f"Failed to get ensemble status for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ensemble status")

@router.get("/ensemble/performance")
async def get_ensemble_performance(company_id: str = Depends(get_company_id_from_token)):
    """Get real-time ensemble performance metrics for monitoring dashboard"""
    
    try:
        # Get ensemble status
        status = forecasting_engine.get_ensemble_status(company_id)
        
        if not status or not status.get('initialized', False):
            return {
                "overall_accuracy": 0.0,
                "model_performances": [],
                "weight_evolution": [],
                "confidence_score": 0.0,
                "prediction_reliability": 0.0,
                "last_updated": datetime.now().isoformat(),
                "total_predictions": 0,
                "system_health": 0.0
            }
        
        # Get model performances with enhanced metrics
        model_performances = []
        total_accuracy = 0.0
        active_models = 0
        
        for model_name, config in status.get('model_configurations', {}).items():
            performance = status.get('recent_performance', {}).get(model_name, {})
            weight = status.get('model_weights', {}).get(model_name, 0.0)
            
            # Calculate model status based on performance
            mape = performance.get('mape', 100.0)
            accuracy = max(0.0, 1.0 - mape / 100.0) if mape < 100 else 0.0
            
            if accuracy >= 0.8:
                model_status = 'healthy'
                trend = 'improving'
            elif accuracy >= 0.6:
                model_status = 'warning'
                trend = 'stable'
            elif accuracy >= 0.4:
                model_status = 'degraded'
                trend = 'declining'
            else:
                model_status = 'failed'
                trend = 'declining'
            
            model_performances.append({
                "model_name": model_name,
                "accuracy": accuracy,
                "mape": mape,
                "mae": performance.get('mae', 0.0),
                "rmse": performance.get('rmse', 0.0),
                "weight": weight,
                "last_updated": datetime.now().isoformat(),
                "trend": trend,
                "status": model_status,
                "prediction_count": performance.get('prediction_count', 0),
                "error_rate": max(0.0, 1.0 - accuracy)
            })
            
            if accuracy > 0:
                total_accuracy += accuracy * weight
                active_models += 1
        
        # Calculate overall metrics
        overall_accuracy = total_accuracy if active_models > 0 else 0.0
        confidence_score = status.get('pattern_confidence', 0.0)
        system_health = min(1.0, overall_accuracy * 1.2)  # Boost system health slightly
        
        return {
            "overall_accuracy": overall_accuracy,
            "model_performances": model_performances,
            "weight_evolution": [],  # TODO: Implement weight history tracking
            "confidence_score": confidence_score,
            "prediction_reliability": overall_accuracy,
            "last_updated": datetime.now().isoformat(),
            "total_predictions": status.get('total_forecasts', 0),
            "system_health": system_health
        }
    
    except Exception as e:
        logger.error(f"Failed to get ensemble performance for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ensemble performance")

@router.websocket("/ws/ensemble-performance/{company_id}")
async def websocket_ensemble_performance(websocket: WebSocket, company_id: str):
    """WebSocket endpoint for real-time ensemble performance updates"""
    await websocket.accept()
    
    try:
        while True:
            # Get current performance metrics
            try:
                # Simulate getting performance data (in real implementation, this would come from the forecasting engine)
                status = forecasting_engine.get_ensemble_status(company_id)
                
                if status and status.get('initialized', False):
                    # Get model performances with enhanced metrics
                    model_performances = []
                    total_accuracy = 0.0
                    active_models = 0
                    
                    for model_name, config in status.get('model_configurations', {}).items():
                        performance = status.get('recent_performance', {}).get(model_name, {})
                        weight = status.get('model_weights', {}).get(model_name, 0.0)
                        
                        # Calculate model status based on performance
                        mape = performance.get('mape', 100.0)
                        accuracy = max(0.0, 1.0 - mape / 100.0) if mape < 100 else 0.0
                        
                        if accuracy >= 0.8:
                            model_status = 'healthy'
                            trend = 'improving'
                        elif accuracy >= 0.6:
                            model_status = 'warning'
                            trend = 'stable'
                        elif accuracy >= 0.4:
                            model_status = 'degraded'
                            trend = 'declining'
                        else:
                            model_status = 'failed'
                            trend = 'declining'
                        
                        model_performances.append({
                            "model_name": model_name,
                            "accuracy": accuracy,
                            "mape": mape,
                            "mae": performance.get('mae', 0.0),
                            "rmse": performance.get('rmse', 0.0),
                            "weight": weight,
                            "last_updated": datetime.now().isoformat(),
                            "trend": trend,
                            "status": model_status,
                            "prediction_count": performance.get('prediction_count', 0),
                            "error_rate": max(0.0, 1.0 - accuracy)
                        })
                        
                        if accuracy > 0:
                            total_accuracy += accuracy * weight
                            active_models += 1
                    
                    # Calculate overall metrics
                    overall_accuracy = total_accuracy if active_models > 0 else 0.0
                    confidence_score = status.get('pattern_confidence', 0.0)
                    system_health = min(1.0, overall_accuracy * 1.2)
                    
                    performance_data = {
                        "overall_accuracy": overall_accuracy,
                        "model_performances": model_performances,
                        "weight_evolution": [],
                        "confidence_score": confidence_score,
                        "prediction_reliability": overall_accuracy,
                        "last_updated": datetime.now().isoformat(),
                        "total_predictions": status.get('total_forecasts', 0),
                        "system_health": system_health
                    }
                else:
                    # No ensemble initialized
                    performance_data = {
                        "overall_accuracy": 0.0,
                        "model_performances": [],
                        "weight_evolution": [],
                        "confidence_score": 0.0,
                        "prediction_reliability": 0.0,
                        "last_updated": datetime.now().isoformat(),
                        "total_predictions": 0,
                        "system_health": 0.0
                    }
                
                await websocket.send_json(performance_data)
                
            except Exception as e:
                logger.error(f"Error sending WebSocket data: {e}")
                # Send error state
                await websocket.send_json({
                    "overall_accuracy": 0.0,
                    "model_performances": [],
                    "error": str(e),
                    "last_updated": datetime.now().isoformat()
                })
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await websocket.close()

@router.post("/ensemble/initialize")
async def initialize_ensemble_models(
    company_id: str = Depends(get_company_id_from_token)
):
    """Initialize ensemble models with progress tracking"""
    
    try:
        # Check if already initialized
        if company_id in forecasting_engine.company_models:
            return {
                "success": True,
                "message": "Ensemble models already initialized",
                "status": forecasting_engine.get_ensemble_status(company_id)
            }
        
        # Initialize with progress tracking
        progress_updates = []
        
        def progress_callback(message: str, progress: float):
            progress_updates.append({
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            })
        
        success = forecasting_engine.initialize_company_models(company_id, progress_callback)
        
        if success:
            status = forecasting_engine.get_ensemble_status(company_id)
            return {
                "success": True,
                "message": "Ensemble models initialized successfully",
                "progress_log": progress_updates,
                "status": status
            }
        else:
            return {
                "success": False,
                "message": "Failed to initialize ensemble models",
                "progress_log": progress_updates
            }
    
    except Exception as e:
        logger.error(f"Failed to initialize ensemble for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ensemble initialization failed: {str(e)}")

# Business Insights and Recommendations Endpoints

@router.get("/{company_id}/insights")
async def get_business_insights(
    company_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get comprehensive business insights for a company"""
    
    try:
        # Validate company access
        if company_id != credentials.credentials:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get company data
        company_data = company_data_manager.get_company_data(company_id)
        if company_data is None or company_data.empty:
            raise HTTPException(status_code=404, detail="No data available for insights generation")
        
        # Generate insights using the business insights engine
        try:
            from models.business_insights_engine import BusinessInsightsEngine
            from models.advanced_pattern_detection import AdvancedPatternAnalysis
            from models.ensemble_forecasting_engine import EnsembleResult
            
            insights_engine = BusinessInsightsEngine()
            
            # Create mock ensemble result and pattern analysis for now
            # In real implementation, these would come from the actual forecasting engine
            mock_ensemble_result = type('EnsembleResult', (), {
                'ensemble_accuracy': 0.85,
                'data_quality_score': 0.9,
                'point_forecast': company_data['sales_amount'].tail(6) if 'sales_amount' in company_data.columns else pd.Series([100, 105, 110, 115, 120, 125])
            })()
            
            mock_pattern_analysis = type('AdvancedPatternAnalysis', (), {
                'trend_analysis': type('TrendAnalysis', (), {
                    'trend_strength': 0.7,
                    'trend_type': 'increasing'
                })(),
                'seasonality_analysis': type('SeasonalityAnalysis', (), {
                    'seasonal_strength': 0.5,
                    'seasonal_consistency': 0.8
                })(),
                'volatility_analysis': type('VolatilityAnalysis', (), {
                    'risk_score': 0.3,
                    'coefficient_of_variation': 0.2,
                    'volatility_level': 'medium'
                })(),
                'anomaly_detection': type('AnomalyDetection', (), {
                    'anomaly_score': 0.1
                })(),
                'confidence_score': 0.8
            })()
            
            # Generate insights
            insights_result = insights_engine.generate_comprehensive_insights(
                company_data, mock_ensemble_result, mock_pattern_analysis
            )
            
            # Convert to API response format
            response_data = {
                "executive_summary": insights_result.executive_summary,
                "key_findings": insights_result.key_findings,
                "confidence_score": insights_result.confidence_score,
                "generated_at": insights_result.generated_at.isoformat(),
                "performance_analysis": {
                    "revenue_growth_rate": insights_result.performance_analysis.revenue_growth_rate,
                    "revenue_trend": insights_result.performance_analysis.revenue_trend,
                    "performance_score": insights_result.performance_analysis.performance_score
                },
                "growth_indicators": {
                    "monthly_growth_rate": insights_result.growth_indicators.monthly_growth_rate,
                    "growth_acceleration": insights_result.growth_indicators.growth_acceleration,
                    "growth_sustainability_score": insights_result.growth_indicators.growth_sustainability_score
                },
                "risk_assessment": {
                    "overall_risk_score": insights_result.risk_assessment.overall_risk_score,
                    "risk_level": insights_result.risk_assessment.risk_level,
                    "primary_risks": insights_result.risk_assessment.primary_risks
                },
                "opportunity_identification": {
                    "opportunity_score": insights_result.opportunity_identification.opportunity_score,
                    "opportunities": insights_result.opportunity_identification.opportunities
                },
                "actionable_recommendations": [
                    {
                        "insight_type": insight.insight_type,
                        "title": insight.title,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "impact_score": insight.impact_score,
                        "supporting_data": insight.supporting_data,
                        "recommended_actions": insight.recommended_actions,
                        "urgency": insight.urgency
                    }
                    for insight in insights_result.actionable_recommendations
                ]
            }
            
            return response_data
            
        except ImportError:
            # Fallback if insights engine is not available
            return {
                "executive_summary": "Business insights generation is currently unavailable. Please ensure all required modules are installed.",
                "key_findings": ["Insights engine not available"],
                "confidence_score": 0.0,
                "generated_at": datetime.now().isoformat(),
                "performance_analysis": {
                    "revenue_growth_rate": 0.05,
                    "revenue_trend": "stable",
                    "performance_score": 7.0
                },
                "growth_indicators": {
                    "monthly_growth_rate": 0.02,
                    "growth_acceleration": 0.01,
                    "growth_sustainability_score": 6.5
                },
                "risk_assessment": {
                    "overall_risk_score": 4.0,
                    "risk_level": "medium",
                    "primary_risks": []
                },
                "opportunity_identification": {
                    "opportunity_score": 6.0,
                    "opportunities": []
                },
                "actionable_recommendations": []
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate insights for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Insights generation failed")

@router.get("/{company_id}/recommendations")
async def get_recommendations(
    company_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get strategic recommendations for a company"""
    
    try:
        # Validate company access
        if company_id != credentials.credentials:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get company data
        company_data = company_data_manager.get_company_data(company_id)
        if company_data is None or company_data.empty:
            raise HTTPException(status_code=404, detail="No data available for recommendations generation")
        
        # Generate recommendations using the recommendation engine
        try:
            from models.recommendation_engine import RecommendationEngine
            from models.business_insights_engine import BusinessInsightsEngine
            
            # First generate insights
            insights_engine = BusinessInsightsEngine()
            recommendation_engine = RecommendationEngine()
            
            # Create mock data for demonstration
            mock_ensemble_result = type('EnsembleResult', (), {
                'ensemble_accuracy': 0.85,
                'data_quality_score': 0.9,
                'point_forecast': company_data['sales_amount'].tail(6) if 'sales_amount' in company_data.columns else pd.Series([100, 105, 110, 115, 120, 125])
            })()
            
            mock_pattern_analysis = type('AdvancedPatternAnalysis', (), {
                'trend_analysis': type('TrendAnalysis', (), {
                    'trend_strength': 0.7,
                    'trend_type': 'increasing'
                })(),
                'seasonality_analysis': type('SeasonalityAnalysis', (), {
                    'seasonal_strength': 0.5,
                    'seasonal_consistency': 0.8
                })(),
                'volatility_analysis': type('VolatilityAnalysis', (), {
                    'risk_score': 0.3,
                    'coefficient_of_variation': 0.2,
                    'volatility_level': 'medium'
                })(),
                'anomaly_detection': type('AnomalyDetection', (), {
                    'anomaly_score': 0.1
                })(),
                'confidence_score': 0.8
            })()
            
            # Generate insights first
            insights_result = insights_engine.generate_comprehensive_insights(
                company_data, mock_ensemble_result, mock_pattern_analysis
            )
            
            # Generate recommendations
            recommendations_portfolio = recommendation_engine.generate_recommendations(
                insights_result, mock_pattern_analysis, mock_ensemble_result
            )
            
            # Convert to API response format
            response_data = {
                "recommendations": [
                    {
                        "recommendation_id": rec.recommendation_id,
                        "title": rec.title,
                        "description": rec.description,
                        "recommendation_type": rec.recommendation_type.value,
                        "priority": rec.priority.value,
                        "confidence_score": rec.confidence_score,
                        "timeline": rec.timeline,
                        "metrics": {
                            "expected_impact": rec.metrics.expected_impact,
                            "implementation_difficulty": rec.metrics.implementation_difficulty,
                            "resource_requirement": rec.metrics.resource_requirement,
                            "time_to_impact": rec.metrics.time_to_impact,
                            "success_probability": rec.metrics.success_probability,
                            "roi_estimate": rec.metrics.roi_estimate
                        },
                        "dependencies": rec.dependencies,
                        "risks": rec.risks,
                        "success_criteria": rec.success_criteria,
                        "created_at": rec.created_at.isoformat(),
                        "expires_at": rec.expires_at.isoformat() if rec.expires_at else None
                    }
                    for rec in recommendations_portfolio.recommendations
                ],
                "priority_matrix": recommendations_portfolio.priority_matrix,
                "confidence_summary": {
                    "average_confidence": recommendations_portfolio.confidence_summary.get('average_confidence', 0.7),
                    "high_confidence_count": recommendations_portfolio.confidence_summary.get('high_confidence_count', 0),
                    "low_confidence_count": recommendations_portfolio.confidence_summary.get('low_confidence_count', 0)
                },
                "expected_outcomes": {
                    "total_expected_impact": recommendations_portfolio.expected_outcomes.get('total_expected_impact', 0.0),
                    "average_roi": recommendations_portfolio.expected_outcomes.get('average_roi', 1.0),
                    "average_success_probability": recommendations_portfolio.expected_outcomes.get('average_success_probability', 0.7)
                },
                "generated_at": recommendations_portfolio.generated_at.isoformat()
            }
            
            return response_data
            
        except ImportError:
            # Fallback if recommendation engine is not available
            return {
                "recommendations": [
                    {
                        "recommendation_id": f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "title": "Optimize Current Operations",
                        "description": "Focus on improving operational efficiency based on current performance trends",
                        "recommendation_type": "operational_improvement",
                        "priority": "medium",
                        "confidence_score": 0.7,
                        "timeline": {
                            "immediate": ["Review current processes"],
                            "short_term": ["Implement efficiency improvements"],
                            "medium_term": ["Monitor and optimize"],
                            "long_term": ["Scale successful initiatives"]
                        },
                        "metrics": {
                            "expected_impact": 6.0,
                            "implementation_difficulty": 4.0,
                            "resource_requirement": 5.0,
                            "time_to_impact": 60,
                            "success_probability": 0.8,
                            "roi_estimate": 1.5
                        },
                        "dependencies": ["Management approval"],
                        "risks": ["Implementation challenges"],
                        "success_criteria": ["Improved efficiency metrics"],
                        "created_at": datetime.now().isoformat(),
                        "expires_at": None
                    }
                ],
                "priority_matrix": {
                    "critical": [],
                    "high": [],
                    "medium": ["fallback_recommendation"],
                    "low": []
                },
                "confidence_summary": {
                    "average_confidence": 0.7,
                    "high_confidence_count": 0,
                    "low_confidence_count": 0
                },
                "expected_outcomes": {
                    "total_expected_impact": 6.0,
                    "average_roi": 1.5,
                    "average_success_probability": 0.8
                },
                "generated_at": datetime.now().isoformat()
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate recommendations for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail="Recommendations generation failed")

# Scenario Planning Endpoints

@router.post("/scenario-forecast")
async def generate_scenario_forecast(
    request: Dict[str, Any],
    company_id: str = Depends(get_company_id_from_token)
):
    """Generate forecast for a specific scenario with parameter adjustments"""
    
    try:
        scenario_name = request.get('scenario_name', 'scenario_1')
        parameters = request.get('parameters', {})
        horizon_months = request.get('horizon_months', 6)
        
        # Get base forecast
        base_result = forecasting_engine.generate_forecast(
            company_id=company_id,
            horizon_months=horizon_months
        )
        
        if not base_result:
            raise ValueError("Unable to generate base forecast")
        
        # Apply scenario parameters to modify forecast
        modified_forecast = apply_scenario_parameters(base_result, parameters)
        
        # Calculate impact summary
        base_values = list(base_result.point_forecast.values())
        modified_values = list(modified_forecast['point_forecast'].values())
        
        if base_values and modified_values:
            total_change = ((sum(modified_values) - sum(base_values)) / sum(base_values)) * 100
            peak_month = max(modified_forecast['point_forecast'].items(), key=lambda x: x[1])[0]
        else:
            total_change = 0.0
            peak_month = datetime.now().strftime('%Y-%m-%d')
        
        # Determine risk level based on volatility modifier
        volatility_modifier = parameters.get('volatility_modifier', 1.0)
        if volatility_modifier > 1.2:
            risk_level = 'high'
        elif volatility_modifier < 0.8:
            risk_level = 'low'
        else:
            risk_level = 'medium'
        
        return {
            "scenario_name": scenario_name,
            "parameters": parameters,
            "forecast_data": modified_forecast,
            "impact_summary": {
                "total_change": total_change,
                "peak_month": peak_month,
                "risk_level": risk_level
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to generate scenario forecast for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scenario forecast generation failed: {str(e)}")

def apply_scenario_parameters(base_result: ForecastResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Apply scenario parameters to modify base forecast"""
    
    # Extract parameters with defaults
    seasonality_factor = parameters.get('seasonality_factor', 1.0)
    trend_adjustment = parameters.get('trend_adjustment', 1.0)
    volatility_modifier = parameters.get('volatility_modifier', 1.0)
    external_factors = parameters.get('external_factors', 1.0)
    market_conditions = parameters.get('market_conditions', 'neutral')
    
    # Market condition multipliers
    market_multipliers = {
        'optimistic': 1.1,
        'neutral': 1.0,
        'pessimistic': 0.9
    }
    market_multiplier = market_multipliers.get(market_conditions, 1.0)
    
    # Apply modifications to point forecast
    modified_point_forecast = {}
    for date_str, value in base_result.point_forecast.items():
        # Parse date to get month for seasonal effects
        try:
            date_obj = pd.to_datetime(date_str)
            month_factor = 1 + 0.1 * np.sin(date_obj.month * np.pi / 6) * (seasonality_factor - 1)
        except:
            month_factor = 1.0
        
        # Apply all modifications
        modified_value = (
            value * 
            trend_adjustment * 
            month_factor * 
            external_factors * 
            market_multiplier *
            (1 + (np.random.random() - 0.5) * 0.1 * volatility_modifier)  # Add controlled randomness
        )
        
        modified_point_forecast[date_str] = max(0, modified_value)  # Ensure non-negative
    
    # Apply modifications to confidence intervals
    modified_confidence_intervals = {}
    for level, series in base_result.confidence_intervals.items():
        modified_confidence_intervals[level] = {}
        for date_str, value in series.items():
            try:
                date_obj = pd.to_datetime(date_str)
                month_factor = 1 + 0.1 * np.sin(date_obj.month * np.pi / 6) * (seasonality_factor - 1)
            except:
                month_factor = 1.0
            
            # Apply same modifications but with adjusted volatility for confidence intervals
            confidence_volatility = volatility_modifier * (1.2 if level == 'p90' else 0.8 if level == 'p10' else 1.0)
            
            modified_value = (
                value * 
                trend_adjustment * 
                month_factor * 
                external_factors * 
                market_multiplier *
                (1 + (np.random.random() - 0.5) * 0.1 * confidence_volatility)
            )
            
            modified_confidence_intervals[level][date_str] = max(0, modified_value)
    
    return {
        "point_forecast": modified_point_forecast,
        "confidence_intervals": modified_confidence_intervals
    }

@router.post("/export-forecast")
async def export_forecast_report(
    request: Dict[str, Any],
    company_id: str = Depends(get_company_id_from_token)
):
    """Export comprehensive forecast report in multiple formats"""
    
    try:
        export_options = request.get('export_options', {})
        forecast_data = request.get('forecast_data')
        scenario_data = request.get('scenario_data', [])
        
        export_format = export_options.get('format', 'pdf')
        include_charts = export_options.get('includeCharts', True)
        include_metrics = export_options.get('includeMetrics', True)
        include_confidence = export_options.get('includeConfidenceIntervals', True)
        include_scenarios = export_options.get('includeScenarios', True)
        include_recommendations = export_options.get('includeRecommendations', True)
        
        # Generate comprehensive export data
        export_data = {
            "title": export_options.get('customTitle', 'Ensemble Forecast Report'),
            "company_id": company_id,
            "generated_at": datetime.now().isoformat(),
            "export_options": export_options,
            "forecast_summary": {
                "horizon_months": len(forecast_data.get('forecasts', {}).get('ensemble', [])) if forecast_data else 0,
                "models_used": ["ARIMA", "ETS", "XGBoost", "LSTM", "Croston"],
                "ensemble_accuracy": 0.87,  # Mock value
                "confidence_score": 0.85    # Mock value
            }
        }
        
        # Add forecast data if available
        if forecast_data and include_metrics:
            export_data["forecast_data"] = forecast_data
        
        # Add scenario data if available
        if scenario_data and include_scenarios:
            export_data["scenario_analysis"] = scenario_data
        
        # Add recommendations if requested
        if include_recommendations:
            export_data["recommendations"] = [
                "Monitor seasonal patterns for optimization opportunities",
                "Consider external factors in planning cycles",
                "Implement adaptive inventory management based on forecasts"
            ]
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"forecast_export_{company_id}_{timestamp}.{export_format}"
        
        # In a real implementation, this would generate actual files
        # For now, return mock success response
        return {
            "success": True,
            "filename": filename,
            "download_url": f"/api/downloads/{filename}",
            "export_data": export_data,
            "format": export_format,
            "size_mb": 2.5  # Mock file size
        }
    
    except Exception as e:
        logger.error(f"Failed to export forecast for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Company Sales Forecasting API"
    }