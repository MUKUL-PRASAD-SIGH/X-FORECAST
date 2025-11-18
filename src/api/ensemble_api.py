"""
Enhanced API Endpoints for Ensemble Integration
Provides comprehensive ensemble forecasting endpoints with real-time updates and WebSocket support
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime, timedelta
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager

# Import ensemble components
try:
    from ..models.ensemble_forecasting_engine import EnsembleForecastingEngine, EnsembleResult, ModelStatus
    from ..models.pattern_detection import PatternCharacteristics
    ENSEMBLE_ENGINE_AVAILABLE = True
except ImportError:
    ENSEMBLE_ENGINE_AVAILABLE = False

# Import model performance tracker
try:
    from ..models.model_performance_tracker import ModelPerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Router
router = APIRouter(prefix="/api/v1/ensemble", tags=["Ensemble Integration"])

# Global instances
ensemble_engine: Optional[EnsembleForecastingEngine] = None
websocket_manager = None

# WebSocket connection manager for real-time updates
class EnsembleWebSocketManager:
    """Manage WebSocket connections for real-time ensemble updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # connection_id -> [subscription_types]
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.subscriptions[connection_id] = []
        logger.info(f"Ensemble WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]
        logger.info(f"Ensemble WebSocket disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: dict):
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_subscribers(self, subscription_type: str, message: dict):
        """Broadcast message to all connections subscribed to a specific type"""
        disconnected = []
        
        for connection_id, subscriptions in self.subscriptions.items():
            if subscription_type in subscriptions:
                try:
                    await self.send_to_connection(connection_id, message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {e}")
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def subscribe(self, connection_id: str, subscription_type: str):
        if connection_id in self.subscriptions:
            if subscription_type not in self.subscriptions[connection_id]:
                self.subscriptions[connection_id].append(subscription_type)

# Initialize WebSocket manager
websocket_manager = EnsembleWebSocketManager()

# Pydantic models
class EnhancedUploadRequest(BaseModel):
    """Request model for enhanced data upload"""
    auto_detect_parameters: bool = Field(True, description="Automatically detect CSV parameters")
    initialize_ensemble: bool = Field(True, description="Initialize ensemble models after upload")
    run_pattern_detection: bool = Field(True, description="Run pattern detection on data")
    
class ColumnMapping(BaseModel):
    """Column mapping for CSV data"""
    date_column: str = Field(..., description="Name of date column")
    sales_amount_column: str = Field(..., description="Name of sales amount column")
    product_category_column: Optional[str] = Field(None, description="Name of product category column")
    region_column: Optional[str] = Field(None, description="Name of region column")

class EnhancedUploadResponse(BaseModel):
    """Response model for enhanced upload"""
    success: bool
    upload_id: str
    message: str
    detected_parameters: Optional[Dict[str, Any]] = None
    data_preview: Optional[Dict[str, Any]] = None
    models_initialized: Optional[List[str]] = None
    pattern_detected: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[float] = None
    ensemble_status: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None

class EnsemblePerformanceResponse(BaseModel):
    """Response model for ensemble performance metrics"""
    timestamp: datetime
    overall_accuracy: float
    model_performances: List[Dict[str, Any]]
    weight_distribution: Dict[str, float]
    confidence_score: float
    prediction_reliability: float
    total_predictions: int
    system_health_score: float
    drift_alerts: List[Dict[str, Any]]

class EnsembleForecastRequest(BaseModel):
    """Request model for ensemble forecasting"""
    horizon_months: int = Field(6, ge=1, le=24, description="Forecast horizon in months")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")
    include_individual_forecasts: bool = Field(True, description="Include individual model forecasts")
    confidence_levels: List[float] = Field([0.1, 0.5, 0.9], description="Confidence levels for intervals")
    
    @validator('confidence_levels')
    def validate_confidence_levels(cls, v):
        if not all(0 < level < 1 for level in v):
            raise ValueError('Confidence levels must be between 0 and 1')
        return sorted(v)

class EnsembleForecastResponse(BaseModel):
    """Response model for ensemble forecasting"""
    forecast_id: str
    generation_timestamp: datetime
    horizon_months: int
    point_forecast: Dict[str, float]  # date -> value
    confidence_intervals: Dict[str, Dict[str, float]]  # level -> {date -> value}
    individual_forecasts: Optional[Dict[str, Dict[str, float]]] = None  # model -> {date -> value}
    model_weights: Dict[str, float]
    ensemble_accuracy: float
    pattern_analysis: Dict[str, Any]
    data_quality_score: float
    forecast_metadata: Dict[str, Any]

# Helper functions
def get_user_id_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from authorization token"""
    # Simplified token handling for demo
    return credentials.credentials or "anonymous"

def initialize_ensemble_engine():
    """Initialize ensemble engine if not already initialized"""
    global ensemble_engine
    if not ENSEMBLE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ensemble engine not available")
    
    if ensemble_engine is None:
        ensemble_engine = EnsembleForecastingEngine()
        logger.info("Ensemble engine initialized")
    
    return ensemble_engine

async def validate_and_parse_file(file: UploadFile) -> pd.DataFrame:
    """Validate and parse uploaded file"""
    # Check file size (100MB limit)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
    
    # Check file format
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['csv', 'xlsx', 'json']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, Excel, or JSON")
    
    try:
        content = await file.read()
        
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_extension == 'xlsx':
            df = pd.read_excel(io.BytesIO(content))
        elif file_extension == 'json':
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File contains no data")
        
        return df
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

def detect_csv_parameters(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect CSV parameters and column mappings"""
    detected = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "column_types": {},
        "suggested_mappings": {},
        "data_quality_issues": []
    }
    
    # Analyze each column
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            detected["data_quality_issues"].append(f"Column '{col}' is entirely empty")
            continue
        
        # Determine column type and suggest mapping
        col_lower = col.lower()
        
        # Date column detection
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'period']):
            try:
                pd.to_datetime(col_data.head(10))
                detected["column_types"][col] = "date"
                detected["suggested_mappings"]["date_column"] = col
            except:
                detected["column_types"][col] = "text"
        
        # Numeric columns
        elif pd.api.types.is_numeric_dtype(col_data):
            if any(keyword in col_lower for keyword in ['sales', 'amount', 'revenue', 'value', 'price']):
                detected["column_types"][col] = "sales_amount"
                detected["suggested_mappings"]["sales_amount_column"] = col
            elif any(keyword in col_lower for keyword in ['quantity', 'units', 'count']):
                detected["column_types"][col] = "quantity"
            else:
                detected["column_types"][col] = "numeric"
        
        # Categorical columns
        else:
            unique_ratio = col_data.nunique() / len(col_data)
            if any(keyword in col_lower for keyword in ['category', 'product', 'sku', 'item']):
                detected["column_types"][col] = "product_category"
                detected["suggested_mappings"]["product_category_column"] = col
            elif any(keyword in col_lower for keyword in ['region', 'location', 'area', 'territory']):
                detected["column_types"][col] = "region"
                detected["suggested_mappings"]["region_column"] = col
            elif unique_ratio < 0.1:
                detected["column_types"][col] = "categorical"
            else:
                detected["column_types"][col] = "text"
    
    # Data quality checks
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage > 10:
        detected["data_quality_issues"].append(f"High missing data: {missing_percentage:.1f}%")
    
    # Check for required mappings
    if "date_column" not in detected["suggested_mappings"]:
        detected["data_quality_issues"].append("No date column detected")
    
    if "sales_amount_column" not in detected["suggested_mappings"]:
        detected["data_quality_issues"].append("No sales amount column detected")
    
    return detected

# API Endpoints

@router.post("/upload-enhanced", response_model=EnhancedUploadResponse)
async def upload_enhanced_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Sales data file for ensemble initialization"),
    auto_detect_parameters: bool = True,
    initialize_ensemble: bool = True,
    run_pattern_detection: bool = True,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Enhanced data upload with ensemble initialization and parameter detection
    
    This endpoint provides comprehensive data upload functionality including:
    - Automatic parameter detection
    - Data quality assessment
    - Ensemble model initialization
    - Pattern detection
    - Real-time progress updates via WebSocket
    """
    start_time = datetime.now()
    upload_id = str(uuid.uuid4())
    
    try:
        # Initialize ensemble engine
        engine = initialize_ensemble_engine()
        
        # Broadcast upload start
        await websocket_manager.broadcast_to_subscribers("upload_progress", {
            "type": "upload_started",
            "upload_id": upload_id,
            "timestamp": start_time.isoformat(),
            "user_id": user_id
        })
        
        # Parse and validate file
        df = await validate_and_parse_file(file)
        
        # Detect parameters if requested
        detected_parameters = None
        if auto_detect_parameters:
            detected_parameters = detect_csv_parameters(df)
            
            # Broadcast parameter detection complete
            await websocket_manager.broadcast_to_subscribers("upload_progress", {
                "type": "parameters_detected",
                "upload_id": upload_id,
                "detected_parameters": detected_parameters,
                "timestamp": datetime.now().isoformat()
            })
        
        # Create data preview
        data_preview = {
            "sample_rows": df.head(5).to_dict('records'),
            "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            "missing_values": df.isnull().sum().to_dict()
        }
        
        # Initialize ensemble models if requested
        models_initialized = None
        ensemble_status = None
        
        if initialize_ensemble:
            try:
                # Broadcast model initialization start
                await websocket_manager.broadcast_to_subscribers("upload_progress", {
                    "type": "model_initialization_started",
                    "upload_id": upload_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process data through ensemble engine
                result = await engine.process_new_data(df, target_column='sales_amount')
                
                models_initialized = list(engine.models.keys())
                ensemble_status = {
                    "initialized": True,
                    "total_models": len(models_initialized),
                    "model_weights": engine.model_weights,
                    "ensemble_accuracy": result.ensemble_accuracy,
                    "forecast_horizon": result.forecast_horizon
                }
                
                # Broadcast model initialization complete
                await websocket_manager.broadcast_to_subscribers("upload_progress", {
                    "type": "model_initialization_complete",
                    "upload_id": upload_id,
                    "models_initialized": models_initialized,
                    "ensemble_status": ensemble_status,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Ensemble initialization failed: {e}")
                ensemble_status = {
                    "initialized": False,
                    "error": str(e)
                }
        
        # Run pattern detection if requested
        pattern_detected = None
        if run_pattern_detection and 'sales_amount' in df.columns:
            try:
                # Broadcast pattern detection start
                await websocket_manager.broadcast_to_subscribers("upload_progress", {
                    "type": "pattern_detection_started",
                    "upload_id": upload_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                from ..models.pattern_detection import PatternDetector
                pattern_detector = PatternDetector()
                
                # Ensure we have a proper time series
                if detected_parameters and "date_column" in detected_parameters.get("suggested_mappings", {}):
                    date_col = detected_parameters["suggested_mappings"]["date_column"]
                    df_sorted = df.sort_values(date_col)
                    sales_series = df_sorted.set_index(pd.to_datetime(df_sorted[date_col]))['sales_amount']
                else:
                    sales_series = df['sales_amount']
                
                pattern_analysis = pattern_detector.detect_pattern(sales_series)
                pattern_detected = {
                    "pattern_type": pattern_analysis.pattern_type,
                    "seasonality_strength": pattern_analysis.seasonality_strength,
                    "trend_strength": pattern_analysis.trend_strength,
                    "volatility": pattern_analysis.volatility,
                    "confidence": pattern_analysis.confidence
                }
                
                # Broadcast pattern detection complete
                await websocket_manager.broadcast_to_subscribers("upload_progress", {
                    "type": "pattern_detection_complete",
                    "upload_id": upload_id,
                    "pattern_detected": pattern_detected,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Pattern detection failed: {e}")
                pattern_detected = {"error": str(e)}
        
        # Calculate data quality score
        data_quality_score = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Calculate processing time
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Broadcast upload complete
        await websocket_manager.broadcast_to_subscribers("upload_progress", {
            "type": "upload_complete",
            "upload_id": upload_id,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now().isoformat()
        })
        
        return EnhancedUploadResponse(
            success=True,
            upload_id=upload_id,
            message="Enhanced upload completed successfully",
            detected_parameters=detected_parameters,
            data_preview=data_preview,
            models_initialized=models_initialized,
            pattern_detected=pattern_detected,
            data_quality_score=data_quality_score,
            ensemble_status=ensemble_status,
            processing_time_ms=processing_time_ms
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced upload failed: {e}")
        
        # Broadcast upload error
        await websocket_manager.broadcast_to_subscribers("upload_progress", {
            "type": "upload_error",
            "upload_id": upload_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        
        raise HTTPException(status_code=500, detail=f"Enhanced upload failed: {str(e)}")

@router.get("/performance", response_model=EnsemblePerformanceResponse)
async def get_ensemble_performance(user_id: str = Depends(get_user_id_from_token)):
    """
    Get real-time ensemble performance metrics
    
    Returns comprehensive performance metrics including:
    - Overall ensemble accuracy
    - Individual model performances
    - Weight distribution
    - System health indicators
    - Drift detection alerts
    """
    try:
        engine = initialize_ensemble_engine()
        
        # Get performance metrics from ensemble engine
        performance_data = engine.get_performance_metrics()
        
        # Get model status summary
        status_summary = engine.get_model_status_summary()
        
        # Calculate system health score
        system_health_score = 1.0
        if status_summary['initialized_models'] < status_summary['total_models']:
            system_health_score -= 0.2
        
        if status_summary['models_with_performance'] < status_summary['initialized_models']:
            system_health_score -= 0.1
        
        # Get drift alerts if comprehensive tracker is available
        drift_alerts = []
        if hasattr(engine, 'comprehensive_tracker') and engine.comprehensive_tracker:
            try:
                # Get recent drift alerts for all models
                for model_name in engine.models.keys():
                    drift_result = await engine.comprehensive_tracker.detect_model_drift(model_name)
                    if drift_result.drift_score > 0.3:  # Threshold for alerting
                        drift_alerts.append({
                            "model_name": model_name,
                            "drift_type": drift_result.drift_type.value,
                            "drift_score": drift_result.drift_score,
                            "detection_timestamp": drift_result.detection_timestamp.isoformat(),
                            "severity": "high" if drift_result.drift_score > 0.7 else "medium"
                        })
            except Exception as e:
                logger.warning(f"Drift detection failed: {e}")
        
        return EnsemblePerformanceResponse(
            timestamp=datetime.now(),
            overall_accuracy=performance_data.get('overall_accuracy', 0.0),
            model_performances=performance_data.get('model_performances', []),
            weight_distribution=performance_data.get('weight_distribution', {}),
            confidence_score=min(performance_data.get('overall_accuracy', 0.0) + 0.1, 1.0),
            prediction_reliability=performance_data.get('overall_accuracy', 0.0),
            total_predictions=sum(perf.get('data_points', 0) for perf in performance_data.get('model_performances', [])),
            system_health_score=max(system_health_score, 0.0),
            drift_alerts=drift_alerts
        )
    
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/forecast", response_model=EnsembleForecastResponse)
async def generate_ensemble_forecast(
    request: EnsembleForecastRequest,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Generate ensemble forecast with all model outputs
    
    Provides comprehensive forecasting including:
    - Weighted ensemble forecast
    - Individual model forecasts
    - Confidence intervals
    - Pattern analysis
    - Model performance metrics
    """
    try:
        engine = initialize_ensemble_engine()
        
        # Generate ensemble forecast
        result = await engine.generate_forecast(horizon=request.horizon_months)
        
        forecast_id = str(uuid.uuid4())
        
        # Convert forecast data to response format
        point_forecast = {}
        if not result.point_forecast.empty:
            point_forecast = {
                date.strftime('%Y-%m-%d'): float(value)
                for date, value in result.point_forecast.items()
            }
        
        # Convert confidence intervals
        confidence_intervals = {}
        for level_name, level_series in result.confidence_intervals.items():
            if not level_series.empty:
                confidence_intervals[level_name] = {
                    date.strftime('%Y-%m-%d'): float(value)
                    for date, value in level_series.items()
                }
        
        # Convert individual forecasts if requested
        individual_forecasts = None
        if request.include_individual_forecasts:
            individual_forecasts = {}
            for model_name, forecast_series in result.individual_forecasts.items():
                if not forecast_series.empty:
                    individual_forecasts[model_name] = {
                        date.strftime('%Y-%m-%d'): float(value)
                        for date, value in forecast_series.items()
                    }
        
        # Convert pattern analysis
        pattern_analysis = {
            "pattern_type": result.pattern_analysis.pattern_type,
            "seasonality_strength": result.pattern_analysis.seasonality_strength,
            "trend_strength": result.pattern_analysis.trend_strength,
            "volatility": result.pattern_analysis.volatility,
            "confidence": result.pattern_analysis.confidence
        }
        
        # Create forecast metadata
        forecast_metadata = {
            "models_used": list(result.model_weights.keys()),
            "total_models": len(result.model_weights),
            "forecast_method": "adaptive_ensemble",
            "confidence_levels": request.confidence_levels,
            "generation_time_ms": 0  # TODO: Track actual generation time
        }
        
        # Broadcast forecast generation event
        await websocket_manager.broadcast_to_subscribers("forecast_updates", {
            "type": "forecast_generated",
            "forecast_id": forecast_id,
            "user_id": user_id,
            "horizon_months": request.horizon_months,
            "ensemble_accuracy": result.ensemble_accuracy,
            "timestamp": datetime.now().isoformat()
        })
        
        return EnsembleForecastResponse(
            forecast_id=forecast_id,
            generation_timestamp=result.forecast_date,
            horizon_months=request.horizon_months,
            point_forecast=point_forecast,
            confidence_intervals=confidence_intervals,
            individual_forecasts=individual_forecasts,
            model_weights=result.model_weights,
            ensemble_accuracy=result.ensemble_accuracy,
            pattern_analysis=pattern_analysis,
            data_quality_score=result.data_quality_score,
            forecast_metadata=forecast_metadata
        )
    
    except Exception as e:
        logger.error(f"Ensemble forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# WebSocket endpoint for real-time updates
@router.websocket("/ws/{connection_id}")
async def ensemble_websocket_endpoint(websocket: WebSocket, connection_id: str):
    """
    WebSocket endpoint for real-time ensemble updates
    
    Supports subscriptions to:
    - upload_progress: Data upload and processing progress
    - performance_updates: Real-time performance metrics
    - forecast_updates: Forecast generation events
    - model_updates: Model training and weight updates
    """
    await websocket_manager.connect(websocket, connection_id)
    
    try:
        # Send connection confirmation
        await websocket_manager.send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "available_subscriptions": [
                "upload_progress",
                "performance_updates", 
                "forecast_updates",
                "model_updates"
            ]
        })
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    subscription_type = message.get("subscription_type")
                    if subscription_type:
                        websocket_manager.subscribe(connection_id, subscription_type)
                        await websocket_manager.send_to_connection(connection_id, {
                            "type": "subscription_confirmed",
                            "subscription_type": subscription_type,
                            "timestamp": datetime.now().isoformat()
                        })
                
                elif message_type == "ping":
                    await websocket_manager.send_to_connection(connection_id, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message_type == "get_status":
                    # Send current ensemble status
                    if ensemble_engine:
                        status = ensemble_engine.get_model_status_summary()
                        await websocket_manager.send_to_connection(connection_id, {
                            "type": "status_update",
                            "status": status,
                            "timestamp": datetime.now().isoformat()
                        })
            
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(connection_id)

# Background task for periodic performance updates
async def broadcast_performance_updates():
    """Background task to broadcast periodic performance updates"""
    while True:
        try:
            if ensemble_engine:
                performance_data = ensemble_engine.get_performance_metrics()
                
                await websocket_manager.broadcast_to_subscribers("performance_updates", {
                    "type": "performance_update",
                    "data": performance_data,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"Performance broadcast failed: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# Initialize background tasks
@asynccontextmanager
async def ensemble_api_lifespan():
    """Manage ensemble API lifecycle"""
    # Start background tasks
    performance_task = asyncio.create_task(broadcast_performance_updates())
    
    try:
        yield
    finally:
        # Cleanup
        performance_task.cancel()
        try:
            await performance_task
        except asyncio.CancelledError:
            pass

# Additional utility endpoints
@router.get("/status")
async def get_ensemble_status():
    """Get current ensemble system status"""
    try:
        if not ensemble_engine:
            return {
                "initialized": False,
                "message": "Ensemble engine not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        status = ensemble_engine.get_model_status_summary()
        status["initialized"] = True
        status["api_version"] = "1.0.0"
        
        return status
    
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize_ensemble():
    """Manually initialize ensemble engine"""
    try:
        global ensemble_engine
        ensemble_engine = EnsembleForecastingEngine()
        
        return {
            "success": True,
            "message": "Ensemble engine initialized successfully",
            "models_available": list(ensemble_engine.models.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Ensemble initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")