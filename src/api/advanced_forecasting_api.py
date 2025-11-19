"""
Unified API Layer for Advanced Demand Forecasting System
Exposes all advanced forecasting capabilities through REST and GraphQL APIs
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio
from contextlib import asynccontextmanager
import jwt
import os
from passlib.context import CryptContext

# Import all system components
from ..models.hierarchical.hierarchical_forecaster import HierarchicalForecaster, ReconciliationMethod
from ..models.governance.fva_tracker import FVATracker
from ..models.governance.fqi_calculator import FQICalculator
from ..models.governance.workflow_engine import WorkflowEngine
from ..models.advanced.cross_category_effects import CrossCategoryEngine
from ..models.advanced.long_tail_optimizer import LongTailOptimizer
from ..models.advanced.multi_echelon_optimizer import MultiEchelonOptimizer
from ..models.advanced.otif_service_manager import OTIFServiceManager

logger = logging.getLogger(__name__)

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Load secret from environment (or use placeholder for local dev)
SECRET_KEY = os.environ.get("JWT_SECRET", "your-secret-key-here")
ALGORITHM = "HS256"
# Toggle to disable auth checks when running locally: set DISABLE_AUTH=1 or DISABLE_AUTH=true
AUTH_DISABLED = str(os.environ.get("DISABLE_AUTH", "0"))

# Pydantic models for API requests/responses
class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    horizon: int = 12
    reconciliation_method: Optional[str] = "mint"
    include_confidence_intervals: bool = True

class ForecastResponse(BaseModel):
    forecast_id: str
    base_forecasts: Dict[str, List[float]]
    reconciled_forecasts: Dict[str, List[float]]
    reconciliation_method: str
    coherence_score: float
    confidence_intervals: Optional[Dict[str, List[List[float]]]] = None
    metadata: Dict[str, Any]

class FVAAnalysisRequest(BaseModel):
    user_id: Optional[str] = None
    product_id: Optional[str] = None
    start_date: datetime
    end_date: datetime
    include_trends: bool = True

class FVAAnalysisResponse(BaseModel):
    analysis_id: str
    fva_score: float
    accuracy_metrics: Dict[str, float]
    trend_analysis: Optional[Dict[str, Any]] = None
    recommendations: List[str]

class FQIMonitoringRequest(BaseModel):
    forecast_data: List[Dict[str, Any]]
    actual_data: List[Dict[str, Any]]
    include_benchmarks: bool = True

class FQIMonitoringResponse(BaseModel):
    fqi_score: float
    component_scores: Dict[str, float]
    grade: str
    benchmark_comparison: Optional[Dict[str, Any]] = None
    alerts: List[str]

class OTIFAnalysisRequest(BaseModel):
    orders_data: List[Dict[str, Any]]
    include_root_cause: bool = True
    optimization_target: Optional[float] = None

class OTIFAnalysisResponse(BaseModel):
    otif_rate: float
    on_time_rate: float
    in_full_rate: float
    failure_breakdown: Dict[str, int]
    root_cause_analysis: Optional[Dict[str, Any]] = None
    optimization_recommendations: Optional[Dict[str, Any]] = None

class UserRole(BaseModel):
    user_id: str
    roles: List[str]
    permissions: List[str]

# Authentication and authorization
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token. When DISABLE_AUTH is set (local dev), bypass verification and return a dev user id."""
    if AUTH_DISABLED.lower() in ("1", "true", "yes"):
        # Bypass auth in development/test environments
        return "dev_admin"

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def check_permissions(user_id: str, required_permission: str):
    """Check if user has required permission"""
    # In production, this would query a user permissions database
    # For now, we'll use a simple mock
    user_permissions = {
        "admin": ["forecast:read", "forecast:write", "fva:read", "fva:write", "fqi:read", "fqi:write", "otif:read", "otif:write"],
        "analyst": ["forecast:read", "fva:read", "fqi:read", "otif:read"],
        "planner": ["forecast:read", "forecast:write", "fva:read", "otif:read"]
    }
    
    # Mock user role lookup
    user_role = "analyst"  # In production, query from database
    
    if required_permission not in user_permissions.get(user_role, []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

# System components initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system components on startup"""
    logger.info("Initializing Advanced Forecasting System components...")
    
    # Initialize all system components
    app.state.hierarchical_forecaster = HierarchicalForecaster()
    app.state.fva_tracker = FVATracker()
    app.state.fqi_calculator = FQICalculator()
    app.state.workflow_engine = WorkflowEngine()
    app.state.cross_category_engine = CrossCategoryEngine()
    app.state.long_tail_optimizer = LongTailOptimizer()
    app.state.multi_echelon_optimizer = MultiEchelonOptimizer()
    app.state.otif_service_manager = OTIFServiceManager()
    
    logger.info("All system components initialized successfully")
    yield
    
    logger.info("Shutting down Advanced Forecasting System...")

# FastAPI app initialization
app = FastAPI(
    title="Advanced Demand Forecasting API",
    description="Unified API for advanced demand forecasting capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",  # Frontend dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",  # Frontend dev server (127.0.0.1)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRF-Token",
        "Cache-Control"
    ],
    expose_headers=["*"],
    max_age=3600,
)

# REST API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/v1/forecast/hierarchical", response_model=ForecastResponse)
async def generate_hierarchical_forecast(
    request: ForecastRequest,
    user_id: str = Depends(verify_token)
):
    """Generate hierarchical forecast with reconciliation"""
    await check_permissions(user_id, "forecast:write")
    
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Get reconciliation method
        reconciliation_method = ReconciliationMethod(request.reconciliation_method.upper())
        
        # Generate forecast
        forecast_result = app.state.hierarchical_forecaster.forecast_hierarchical(
            data=df,
            horizon=request.horizon,
            reconciliation_method=reconciliation_method
        )
        
        # Convert to response format
        base_forecasts = {
            key: series.tolist() 
            for key, series in forecast_result.base_forecasts.items()
        }
        
        reconciled_forecasts = {
            key: series.tolist() 
            for key, series in forecast_result.reconciled_forecasts.items()
        }
        
        # Generate confidence intervals if requested
        confidence_intervals = None
        if request.include_confidence_intervals:
            confidence_intervals = {}
            for key, series in reconciled_forecasts.items():
                # Simple confidence interval calculation (±10%)
                lower = [v * 0.9 for v in series]
                upper = [v * 1.1 for v in series]
                confidence_intervals[key] = [lower, upper]
        
        response = ForecastResponse(
            forecast_id=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            base_forecasts=base_forecasts,
            reconciled_forecasts=reconciled_forecasts,
            reconciliation_method=forecast_result.reconciliation_method.value,
            coherence_score=forecast_result.coherence_score,
            confidence_intervals=confidence_intervals,
            metadata={
                "total_nodes": len(forecast_result.hierarchy_structure),
                "validation_result": forecast_result.validation_result.__dict__ if forecast_result.validation_result else None
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Hierarchical forecasting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/fva/analyze", response_model=FVAAnalysisResponse)
async def analyze_fva(
    request: FVAAnalysisRequest,
    user_id: str = Depends(verify_token)
):
    """Analyze Forecast Value Added (FVA)"""
    await check_permissions(user_id, "fva:read")
    
    try:
        # Perform FVA analysis
        if request.user_id:
            analysis_result = app.state.fva_tracker.analyze_user_fva(
                user_id=request.user_id,
                start_date=request.start_date,
                end_date=request.end_date
            )
        elif request.product_id:
            analysis_result = app.state.fva_tracker.analyze_product_fva(
                product_id=request.product_id,
                start_date=request.start_date,
                end_date=request.end_date
            )
        else:
            analysis_result = app.state.fva_tracker.analyze_overall_fva(
                start_date=request.start_date,
                end_date=request.end_date
            )
        
        # Generate trend analysis if requested
        trend_analysis = None
        if request.include_trends:
            trend_analysis = app.state.fva_tracker.analyze_fva_trends(
                analysis_result.analysis_id
            )
        
        response = FVAAnalysisResponse(
            analysis_id=analysis_result.analysis_id,
            fva_score=analysis_result.fva_score,
            accuracy_metrics=analysis_result.accuracy_metrics,
            trend_analysis=trend_analysis.__dict__ if trend_analysis else None,
            recommendations=analysis_result.recommendations
        )
        
        return response
        
    except Exception as e:
        logger.error(f"FVA analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/fqi/monitor", response_model=FQIMonitoringResponse)
async def monitor_fqi(
    request: FQIMonitoringRequest,
    user_id: str = Depends(verify_token)
):
    """Monitor Forecast Quality Index (FQI)"""
    await check_permissions(user_id, "fqi:read")
    
    try:
        # Convert data to DataFrames
        forecast_df = pd.DataFrame(request.forecast_data)
        actual_df = pd.DataFrame(request.actual_data)
        
        # Calculate FQI
        fqi_result = app.state.fqi_calculator.calculate_fqi(
            forecast_data=forecast_df,
            actual_data=actual_df
        )
        
        # Get benchmark comparison if requested
        benchmark_comparison = None
        if request.include_benchmarks:
            benchmark_comparison = app.state.fqi_calculator.compare_with_benchmarks(
                fqi_result.fqi_score
            )
        
        # Check for alerts
        alerts = app.state.fqi_calculator.check_fqi_alerts(fqi_result.fqi_score)
        
        response = FQIMonitoringResponse(
            fqi_score=fqi_result.fqi_score,
            component_scores=fqi_result.component_scores,
            grade=fqi_result.grade,
            benchmark_comparison=benchmark_comparison.__dict__ if benchmark_comparison else None,
            alerts=[alert.message for alert in alerts]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"FQI monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/otif/analyze", response_model=OTIFAnalysisResponse)
async def analyze_otif(
    request: OTIFAnalysisRequest,
    user_id: str = Depends(verify_token)
):
    """Analyze OTIF (On-Time In-Full) performance"""
    await check_permissions(user_id, "otif:read")
    
    try:
        # Convert data to DataFrame
        orders_df = pd.DataFrame(request.orders_data)
        
        # Track order performance
        order_records = app.state.otif_service_manager.track_order_performance(orders_df)
        
        # Calculate OTIF metrics
        start_date = min(pd.to_datetime(orders_df['order_date']))
        end_date = max(pd.to_datetime(orders_df['order_date']))
        
        otif_metrics = app.state.otif_service_manager.calculate_otif_metrics(
            start_date, end_date
        )
        
        # Perform root cause analysis if requested
        root_cause_analysis = None
        if request.include_root_cause:
            root_cause_analysis = app.state.otif_service_manager.root_cause_analyzer.analyze_failures(
                order_records
            )
        
        # Perform optimization if target specified
        optimization_recommendations = None
        if request.optimization_target:
            # Mock inventory data for optimization
            inventory_data = pd.DataFrame({
                'sku': orders_df['sku'].unique(),
                'location': 'default',
                'quantity': 100,
                'unit_cost': 10,
                'safety_stock': 20
            })
            
            cost_parameters = {
                'holding_cost_rate': 0.25,
                'revenue_per_order': 100,
                'average_unit_cost': 10
            }
            
            optimization_result = app.state.otif_service_manager.service_optimizer.optimize_service_inventory_tradeoff(
                current_metrics=otif_metrics,
                target_service_level=request.optimization_target,
                inventory_data=inventory_data,
                cost_parameters=cost_parameters
            )
            
            optimization_recommendations = optimization_result.__dict__
        
        response = OTIFAnalysisResponse(
            otif_rate=otif_metrics.otif_rate,
            on_time_rate=otif_metrics.on_time_rate,
            in_full_rate=otif_metrics.in_full_rate,
            failure_breakdown=dict(otif_metrics.failure_breakdown),
            root_cause_analysis=root_cause_analysis.__dict__ if root_cause_analysis else None,
            optimization_recommendations=optimization_recommendations
        )
        
        return response
        
    except Exception as e:
        logger.error(f"OTIF analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflow/templates")
async def get_workflow_templates(user_id: str = Depends(verify_token)):
    """Get available workflow templates"""
    await check_permissions(user_id, "workflow:read")
    
    try:
        templates = app.state.workflow_engine.get_workflow_templates()
        return {"templates": [template.__dict__ for template in templates]}
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflow/start")
async def start_workflow(
    template_id: str,
    context_data: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Start a new workflow instance"""
    await check_permissions(user_id, "workflow:write")
    
    try:
        workflow_instance = app.state.workflow_engine.start_workflow(
            template_id=template_id,
            context_data=context_data,
            initiated_by=user_id
        )
        return {"workflow_instance": workflow_instance.__dict__}
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GraphQL Schema
@strawberry.type
class ForecastNode:
    node_id: str
    level: str
    forecast_values: List[float]
    confidence_interval: Optional[List[List[float]]] = None

@strawberry.type
class HierarchicalForecastResult:
    forecast_id: str
    nodes: List[ForecastNode]
    coherence_score: float
    reconciliation_method: str

@strawberry.type
class Query:
    @strawberry.field
    async def hierarchical_forecast(
        self, 
        data: str,  # JSON string of data
        horizon: int = 12,
        reconciliation_method: str = "mint"
    ) -> HierarchicalForecastResult:
        """GraphQL query for hierarchical forecast"""
        
        # In a real implementation, you'd parse the JSON data
        # and call the hierarchical forecaster
        
        return HierarchicalForecastResult(
            forecast_id="gql_forecast_123",
            nodes=[
                ForecastNode(
                    node_id="total",
                    level="total",
                    forecast_values=[100.0, 105.0, 110.0],
                    confidence_interval=[[90.0, 95.0, 99.0], [110.0, 115.0, 121.0]]
                )
            ],
            coherence_score=0.95,
            reconciliation_method=reconciliation_method
        )
    
    @strawberry.field
    async def fva_analysis(
        self,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        start_date: str = "",
        end_date: str = ""
    ) -> Dict[str, Any]:
        """GraphQL query for FVA analysis"""
        
        return {
            "fva_score": 0.85,
            "accuracy_metrics": {"mape": 0.15, "mae": 10.5},
            "recommendations": ["Improve forecast accuracy for SKU-123"]
        }

# GraphQL router
schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

# Additional utility endpoints
@app.get("/api/v1/system/status")
async def get_system_status(user_id: str = Depends(verify_token)):
    """Get system status and component health"""
    await check_permissions(user_id, "system:read")
    
    status_info = {
        "system_status": "healthy",
        "components": {
            "hierarchical_forecaster": "active",
            "fva_tracker": "active",
            "fqi_calculator": "active",
            "workflow_engine": "active",
            "cross_category_engine": "active",
            "long_tail_optimizer": "active",
            "multi_echelon_optimizer": "active",
            "otif_service_manager": "active"
        },
        "last_updated": datetime.utcnow().isoformat()
    }
    
    return status_info

@app.get("/api/v1/system/metrics")
async def get_system_metrics(user_id: str = Depends(verify_token)):
    """Get system performance metrics"""
    await check_permissions(user_id, "system:read")
    
    # In production, these would be real metrics
    metrics = {
        "api_requests_per_minute": 150,
        "average_response_time_ms": 250,
        "active_workflows": 12,
        "forecast_accuracy": 0.92,
        "system_uptime_hours": 720
    }
    
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)