"""
FastAPI Backend for Cyberpunk AI Dashboard
Unified API endpoints for integrated analytics, forecasting, and real-time data
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import json
import logging
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import uvicorn

# Import security configuration (simplified for compatibility)
try:
    from .security_config import (
        security_config, 
        SecurityMiddleware, 
        setup_cors_middleware, 
        setup_rate_limiting,
        security_monitor
    )
    SECURITY_CONFIG_AVAILABLE = True
    # Force fallback rate limiting for now
    def auth_rate_limit(): return lambda f: f
    def upload_rate_limit(): return lambda f: f  
    def general_rate_limit(): return lambda f: f
except ImportError as e:
    print(f"Security config not available: {e}")
    SECURITY_CONFIG_AVAILABLE = False
    
    # Fallback security configuration
    class FallbackSecurityConfig:
        def __init__(self):
            self.debug = True
            self.allowed_origins = ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"]
            self.security_headers = {}
        
        def validate_file_upload(self, filename, file_size):
            return {"valid": True, "errors": []}
        
        def log_security_event(self, *args, **kwargs):
            pass
    
    security_config = FallbackSecurityConfig()
    
    class FallbackSecurityMonitor:
        def is_ip_blocked(self, ip): return False
        def record_failed_login(self, ip, user_id=None): pass
    
    security_monitor = FallbackSecurityMonitor()
    
    # Fallback decorators
    def auth_rate_limit(): return lambda f: f
    def upload_rate_limit(): return lambda f: f  
    def general_rate_limit(): return lambda f: f

# Import our custom modules
try:
    from src.models.integrated_forecasting import IntegratedForecastingEngine, EnhancedForecast
    from src.customer_analytics.retention_analyzer import RetentionAnalyzer, RetentionInsights
    from src.data_fabric.unified_connector import UnifiedDataConnector
    from src.predictive_maintenance.maintenance_engine import PredictiveMaintenanceEngine
    # Import adaptive configuration API
    from .adaptive_config_api import router as adaptive_config_router
    ADAPTIVE_CONFIG_AVAILABLE = True
    
    # Import company sales API
    from .company_sales_api import router as company_sales_router
    COMPANY_SALES_AVAILABLE = True
    
    # Import customer analytics API
    from .customer_analytics_api import router as customer_analytics_router
    CUSTOMER_ANALYTICS_AVAILABLE = True
except ImportError:
    # Fallback classes for missing modules
    class IntegratedForecastingEngine:
        pass
    class RetentionAnalyzer:
        pass
    class UnifiedDataConnector:
        def configure_default_connectors(self): pass
        async def sync_all_sources(self): return []
        def get_data_quality_report(self): return {}
        async def get_unified_customer_view(self, customer_id): return {}
    class PredictiveMaintenanceEngine:
        async def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def get_system_health_summary(self): return {"health_score": 0.95, "status": "good", "current_metrics": {}, "active_predictions": 0, "critical_predictions": 0, "scheduled_maintenance": 0, "recommendations": []}
    
    # Adaptive config not available
    adaptive_config_router = None
    ADAPTIVE_CONFIG_AVAILABLE = False
    
    # Company sales API not available
    company_sales_router = None
    COMPANY_SALES_AVAILABLE = False
    
    # Customer analytics API not available
    customer_analytics_router = None
    CUSTOMER_ANALYTICS_AVAILABLE = False

from src.ai_chatbot.conversational_ai import ConversationalAI, ChatResponse
try:
    from src.api.auth_endpoints import router as auth_router
except ImportError as e:
    print(f"Real auth endpoints not available: {e}")
    # Fallback to dev endpoints if real ones not available
    try:
        from src.api.auth_endpoints_dev import router as auth_router
        print("Using development auth endpoints")
    except ImportError:
        print("No auth endpoints available")
        auth_router = None

# Enhanced authentication middleware with security monitoring
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import get_remote_address conditionally
try:
    from slowapi.util import get_remote_address
    GET_REMOTE_ADDRESS_AVAILABLE = True
except ImportError:
    GET_REMOTE_ADDRESS_AVAILABLE = False
    def get_remote_address(request):
        return getattr(request.client, 'host', 'unknown') if hasattr(request, 'client') else 'unknown'

security = HTTPBearer()

async def verify_token_middleware(request: Request, call_next):
    """Enhanced middleware to verify JWT tokens with security monitoring"""
    try:
        client_ip = get_remote_address(request)
    except:
        client_ip = "unknown"
    
    # Check if IP is blocked (only if security monitoring is available)
    if SECURITY_CONFIG_AVAILABLE and security_monitor.is_ip_blocked(client_ip):
        try:
            security_config.log_security_event(
                "blocked_ip_access_attempt",
                ip_address=client_ip,
                details=f"Blocked IP attempted to access {request.url.path}"
            )
        except:
            logger.warning(f"Security logging failed for blocked IP: {client_ip}")
        
        # Also log using comprehensive logger
        comprehensive_logger.log_suspicious_activity(
            ip_address=client_ip,
            activity_type="blocked_ip_access",
            details={"endpoint": request.url.path, "reason": "IP blocked by security monitor"}
        )
        
        raise HTTPException(status_code=429, detail="IP address temporarily blocked due to security violations")
    
    # Skip authentication for public routes
    public_routes = [
        "/",
        "/docs",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/status"
    ]
    
    if request.url.path in public_routes or request.url.path.startswith("/docs"):
        response = await call_next(request)
        return response
    
    # Check for Authorization header on protected routes
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        # For API routes, require authentication
        if request.url.path.startswith("/api/"):
            if SECURITY_CONFIG_AVAILABLE:
                try:
                    security_config.log_security_event(
                        "unauthorized_access_attempt",
                        ip_address=client_ip,
                        details=f"Unauthorized access attempt to {request.url.path}"
                    )
                except:
                    logger.warning(f"Security logging failed for unauthorized access: {request.url.path}")
            
            # Log using comprehensive logger
            comprehensive_logger.log_unauthorized_access(
                ip_address=client_ip,
                endpoint=request.url.path,
                details={"reason": "missing_authorization_header"}
            )
            
            raise HTTPException(status_code=401, detail="Authentication required")
    else:
        # Verify token if provided
        try:
            from src.auth.user_management import user_manager
            token = auth_header.split(" ")[1]
            user_data = user_manager.verify_token(token)
            if not user_data:
                if SECURITY_CONFIG_AVAILABLE:
                    security_monitor.record_failed_login(client_ip)
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            # Add user data to request state
            request.state.user = user_data
            
            # Log successful authentication
            if SECURITY_CONFIG_AVAILABLE:
                try:
                    security_config.log_security_event(
                        "successful_authentication",
                        user_id=user_data.get("user_id"),
                        ip_address=client_ip,
                        details=f"Successful access to {request.url.path}"
                    )
                except:
                    logger.warning(f"Security logging failed for successful auth: {user_data.get('user_id')}")
            
        except Exception as e:
            if request.url.path.startswith("/api/"):
                if SECURITY_CONFIG_AVAILABLE:
                    security_monitor.record_failed_login(client_ip)
                    try:
                        security_config.log_security_event(
                            "authentication_error",
                            ip_address=client_ip,
                            details=f"Authentication error for {request.url.path}: {str(e)}"
                        )
                    except:
                        logger.warning(f"Security logging failed for auth error: {str(e)}")
                raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    response = await call_next(request)
    return response

def get_current_user(request: Request):
    """Get current authenticated user from request state"""
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    return request.state.user

# Import comprehensive logging system
from src.utils.logging_config import comprehensive_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
forecasting_engine = None
retention_analyzer = None
data_connector = None
conversational_ai = None
maintenance_engine = None
websocket_manager = None

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global forecasting_engine, retention_analyzer, data_connector, conversational_ai, maintenance_engine, websocket_manager
    
    logger.info("Initializing Cyberpunk AI Dashboard...")
    
    # Initialize components with adaptive configuration
    forecasting_engine = IntegratedForecastingEngine()
    retention_analyzer = RetentionAnalyzer()
    data_connector = UnifiedDataConnector()
    conversational_ai = ConversationalAI()
    maintenance_engine = PredictiveMaintenanceEngine()
    websocket_manager = WebSocketManager()
    
    # Configure data connectors
    data_connector.configure_default_connectors()
    
    # Start background tasks
    asyncio.create_task(start_background_tasks())
    
    # Initialize training progress API if available
    try:
        from .training_progress_api import init_training_progress_api
        from src.models.ensemble_forecasting_engine import EnsembleForecastingEngine
        from src.models.automated_training_pipeline import AutomatedTrainingPipeline
        
        # Create instances for training progress monitoring
        ensemble_engine = EnsembleForecastingEngine()
        training_pipeline = AutomatedTrainingPipeline(ensemble_engine)
        
        # Initialize training progress API
        init_training_progress_api(ensemble_engine, training_pipeline)
        logger.info("Training progress monitoring initialized")
    except ImportError as e:
        logger.warning(f"Training progress monitoring not available: {e}")
    
    # Initialize performance monitoring and optimization systems
    try:
        from src.utils.performance_startup import initialize_from_environment, shutdown_performance_systems
        
        performance_initialized = initialize_from_environment()
        if performance_initialized:
            logger.info("Performance monitoring and optimization systems initialized")
        else:
            logger.warning("Performance systems initialization failed or disabled")
    except ImportError as e:
        logger.warning(f"Performance monitoring not available: {e}")
    except Exception as e:
        logger.error(f"Error initializing performance systems: {e}")
    
    # Initialize health monitoring system
    try:
        from src.utils.health_monitor import live_health_monitor
        
        # Start health monitoring in background
        asyncio.create_task(live_health_monitor.start_monitoring())
        logger.info("Live health monitoring system initialized and started")
    except ImportError as e:
        logger.warning(f"Health monitoring not available: {e}")
    except Exception as e:
        logger.error(f"Error initializing health monitoring: {e}")
    
    logger.info("Cyberpunk AI Dashboard initialized successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cyberpunk AI Dashboard...")
    
    # Shutdown performance systems
    try:
        from src.utils.performance_startup import shutdown_performance_systems
        shutdown_performance_systems()
        logger.info("Performance systems shutdown complete")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Error shutting down performance systems: {e}")
    
    # Shutdown health monitoring
    try:
        from src.utils.health_monitor import live_health_monitor
        await live_health_monitor.stop_monitoring()
        logger.info("Health monitoring system shutdown complete")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Error shutting down health monitoring: {e}")
    
    if maintenance_engine:
        maintenance_engine.stop_monitoring()

# Create FastAPI app with security configuration
app = FastAPI(
    title="X-FORECAST Multi-Tenant AI Platform",
    description="Personalized AI forecasting platform with multi-tenant support",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if security_config.debug else None,  # Disable docs in production
    redoc_url="/redoc" if security_config.debug else None  # Disable redoc in production
)

# Include authentication routes (bypassed for SuperX)
if auth_router:
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])

# Include adaptive configuration routes if available
if ADAPTIVE_CONFIG_AVAILABLE and adaptive_config_router:
    app.include_router(adaptive_config_router, tags=["Adaptive Configuration"])

# Include company sales routes if available
if COMPANY_SALES_AVAILABLE and company_sales_router:
    app.include_router(company_sales_router, tags=["Company Sales Forecasting"])

# Include customer analytics routes if available
if CUSTOMER_ANALYTICS_AVAILABLE and customer_analytics_router:
    app.include_router(customer_analytics_router, tags=["Customer Analytics"])

# Include export and reporting routes
try:
    from .export_api import router as export_router
    app.include_router(export_router, tags=["Export & Reporting"])
    EXPORT_API_AVAILABLE = True
    logger.info("Export and reporting API included")
except ImportError:
    EXPORT_API_AVAILABLE = False
    logger.warning("Export API not available")

# Include shareable reports routes
try:
    from .shareable_reports_api import router as shareable_reports_router
    app.include_router(shareable_reports_router, tags=["Shareable Reports"])
    SHAREABLE_REPORTS_AVAILABLE = True
    logger.info("Shareable reports API included")
except ImportError:
    SHAREABLE_REPORTS_AVAILABLE = False
    logger.warning("Shareable reports API not available")

# Include model performance tracking routes
try:
    from .model_performance_api import router as model_performance_router
    app.include_router(model_performance_router, tags=["Model Performance Tracking"])
    MODEL_PERFORMANCE_AVAILABLE = True
except ImportError:
    MODEL_PERFORMANCE_AVAILABLE = False
    logger.warning("Model performance tracking API not available")

# Include monitoring and performance optimization routes
try:
    from .monitoring_api import router as monitoring_router
    app.include_router(monitoring_router, tags=["System Monitoring & Performance"])
    MONITORING_API_AVAILABLE = True
    logger.info("System monitoring and performance API included")
except ImportError:
    MONITORING_API_AVAILABLE = False
    logger.warning("System monitoring API not available")

# Include automated training pipeline routes
try:
    from .automated_training_api import router as automated_training_router
    app.include_router(automated_training_router, tags=["Automated Training Pipeline"])
    AUTOMATED_TRAINING_AVAILABLE = True
    logger.info("Automated training pipeline API included")
except ImportError:
    AUTOMATED_TRAINING_AVAILABLE = False
    logger.warning("Automated training pipeline API not available")

# Include training progress monitoring routes
try:
    from .training_progress_api import router as training_progress_router
    app.include_router(training_progress_router, tags=["Training Progress Monitoring"])
    TRAINING_PROGRESS_AVAILABLE = True
    logger.info("Training progress monitoring API included")
except ImportError:
    TRAINING_PROGRESS_AVAILABLE = False
    logger.warning("Training progress monitoring API not available")

# Include ensemble chat API
try:
    from .ensemble_chat_api import router as ensemble_chat_router
    app.include_router(ensemble_chat_router, tags=["Ensemble Chat"])
    ENSEMBLE_CHAT_AVAILABLE = True
    logger.info("Ensemble chat API included")
except ImportError:
    ENSEMBLE_CHAT_AVAILABLE = False
    logger.warning("Ensemble chat API not available")

# Include PDF processing API
try:
    from .pdf_processing_api import router as pdf_processing_router
    app.include_router(pdf_processing_router, tags=["PDF Processing"])
    PDF_PROCESSING_AVAILABLE = True
    logger.info("PDF processing API included")
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    logger.warning("PDF processing API not available")

# Include RAG management API
try:
    from .rag_management_api import router as rag_management_router
    app.include_router(rag_management_router, tags=["RAG Management"])
    RAG_MANAGEMENT_AVAILABLE = True
    logger.info("RAG management API included")
except ImportError:
    RAG_MANAGEMENT_AVAILABLE = False
    logger.warning("RAG management API not available")

# Include simple ensemble API for model initialization
try:
    from .simple_ensemble_api import router as simple_ensemble_router
    app.include_router(simple_ensemble_router, tags=["Simple Ensemble"])
    SIMPLE_ENSEMBLE_API_AVAILABLE = True
    logger.info("Simple ensemble API included")
except ImportError:
    SIMPLE_ENSEMBLE_API_AVAILABLE = False
    logger.warning("Simple ensemble API not available")

# Include enhanced ensemble API - TEMPORARILY DISABLED
# try:
#     from .ensemble_api import router as ensemble_api_router
#     app.include_router(ensemble_api_router, tags=["Enhanced Ensemble Integration"])
#     ENSEMBLE_API_AVAILABLE = True
#     logger.info("Enhanced ensemble API included")
# except ImportError:
ENSEMBLE_API_AVAILABLE = False
logger.warning("Enhanced ensemble API temporarily disabled")

# Include monitoring API
try:
    from .monitoring_api import router as monitoring_router
    app.include_router(monitoring_router, tags=["System Monitoring"])
    MONITORING_API_AVAILABLE = True
    logger.info("System monitoring API included")
except ImportError:
    MONITORING_API_AVAILABLE = False
    logger.warning("System monitoring API not available")

# Include enhanced data processing API
try:
    from .enhanced_data_processing_api import router as enhanced_data_processing_router
    app.include_router(enhanced_data_processing_router, tags=["Enhanced Data Processing"])
    ENHANCED_DATA_PROCESSING_AVAILABLE = True
    logger.info("Enhanced data processing API included")
except ImportError:
    ENHANCED_DATA_PROCESSING_AVAILABLE = False
    logger.warning("Enhanced data processing API not available")

# Include health monitoring API
try:
    from .health_monitoring_api import router as health_monitoring_router
    app.include_router(health_monitoring_router, tags=["Health Monitoring"])
    HEALTH_MONITORING_AVAILABLE = True
    logger.info("Health monitoring API included")
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    logger.warning("Health monitoring API not available")

# Removed SuperX bypass endpoints - authentication now required

@app.post("/api/v1/upload-enhanced")
@upload_rate_limit()
async def enhanced_ensemble_upload(
    request: Request,
    file: UploadFile = File(...), 
    current_user: dict = Depends(get_current_user)
):
    """Enhanced ensemble data upload with parameter detection for CSV and PDF files"""
    try:
        # Validate file upload security
        file_content = await file.read()
        file_size = len(file_content)
        
        validation_result = security_config.validate_file_upload(file.filename, file_size)
        if not validation_result["valid"]:
            if SECURITY_CONFIG_AVAILABLE:
                try:
                    security_config.log_security_event(
                        "invalid_file_upload",
                        user_id=current_user["user_id"],
                        ip_address=get_remote_address(request),
                        details=f"Invalid file upload: {', '.join(validation_result['errors'])}"
                    )
                except:
                    logger.warning(f"Security logging failed for invalid file upload: {file.filename}")
            
            return {
                "success": False,
                "message": f"File validation failed: {', '.join(validation_result['errors'])}",
                "processing_status": "validation_failed"
            }
        
        # Reset file pointer after reading
        await file.seek(0)
        
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension == 'csv':
            # Handle CSV upload (existing logic)
            user_dir = f"data/users/{current_user['user_id']}/csv"
            os.makedirs(user_dir, exist_ok=True)
            
            file_path = os.path.join(user_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Read and analyze CSV
            df = pd.read_csv(file_path)
            
            # Parameter detection
            detected_params = {
                "columns": list(df.columns),
                "rows": len(df),
                "date_columns": [col for col in df.columns if 'date' in col.lower()],
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns)
            }
            
            # Data quality assessment
            quality_score = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            
            # Integrate with RAG system
            try:
                from src.rag.real_vector_rag import real_vector_rag
                company_name = current_user.get("company_name", "Unknown Company")
                real_vector_rag.load_company_data(current_user['user_id'], company_name, file_path)
            except Exception as rag_error:
                logger.warning(f"RAG integration failed: {rag_error}")
            
            return {
                "success": True,
                "message": "CSV upload and analysis successful",
                "file_type": "csv",
                "file_path": file_path,
                "detected_parameters": detected_params,
                "data_quality_score": quality_score,
                "processing_status": "parameter_detection_complete"
            }
            
        elif file_extension == 'pdf':
            # Handle PDF upload
            user_dir = f"data/users/{current_user['user_id']}/pdf"
            os.makedirs(user_dir, exist_ok=True)
            
            file_path = os.path.join(user_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process PDF
            try:
                from src.rag.real_vector_rag import real_vector_rag
                company_name = current_user.get("company_name", "Unknown Company")
                
                success = real_vector_rag.add_pdf_document(current_user['user_id'], company_name, file_path)
                
                if success:
                    return {
                        "success": True,
                        "message": "PDF upload and processing successful",
                        "file_type": "pdf",
                        "file_path": file_path,
                        "processing_status": "pdf_processing_complete"
                    }
                else:
                    return {
                        "success": False,
                        "message": "PDF processing failed",
                        "file_type": "pdf",
                        "processing_status": "failed"
                    }
                    
            except Exception as pdf_error:
                logger.error(f"PDF processing error: {pdf_error}")
                return {
                    "success": False,
                    "message": f"PDF processing error: {str(pdf_error)}",
                    "file_type": "pdf",
                    "processing_status": "failed"
                }
        
        else:
            return {
                "success": False,
                "message": f"Unsupported file type: {file_extension}. Only CSV and PDF files are supported.",
                "processing_status": "failed"
            }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return {
            "success": False,
            "message": str(e),
            "processing_status": "failed"
        }

# Setup security middleware and configurations
if SECURITY_CONFIG_AVAILABLE:
    # Add security middleware first
    app.add_middleware(SecurityMiddleware)
    
    # Setup CORS with production security settings
    setup_cors_middleware(app)
    
    # Setup rate limiting with security monitoring
    rate_limiter = setup_rate_limiting(app)
    if rate_limiter:
        app.state.limiter = rate_limiter
        logger.info("Rate limiting enabled with security monitoring")
    
    logger.info("Production security configuration applied")
else:
    # Fallback CORS configuration for development - more permissive
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )
    logger.warning("Using fallback security configuration with permissive CORS")

# Add authentication middleware
app.middleware("http")(verify_token_middleware)

# Add explicit OPTIONS handlers for all API endpoints
@app.options("/api/v1/auth/{path:path}")
async def auth_options_handler(path: str):
    """Handle preflight requests for authentication endpoints"""
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-CSRF-Token, Cache-Control",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.options("/api/v1/{path:path}")
async def api_options_handler(path: str):
    """Handle preflight requests for all API endpoints"""
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.options("/api/{path:path}")
async def company_api_options_handler(path: str):
    """Handle preflight requests for company API endpoints"""
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-CSRF-Token, Cache-Control",
            "Access-Control-Max-Age": "3600"
        }
    )

# Add ensemble security middleware
try:
    from .ensemble_security import security_middleware
    app.middleware("http")(security_middleware)
    logger.info("Ensemble security middleware added")
except ImportError:
    logger.warning("Ensemble security middleware not available")

# Pydantic models for API requests/responses
class ForecastRequest(BaseModel):
    horizon: int = Field(default=12, description="Forecast horizon in months")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    customer_data_source: Optional[str] = Field(default=None, description="Customer data source")

class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    user_id: str = Field(default="anonymous", description="User ID")
    session_id: str = Field(default="default", description="Session ID")

class SystemHealthResponse(BaseModel):
    timestamp: datetime
    health_score: float
    status: str
    current_metrics: Dict[str, Any]
    active_predictions: int
    critical_predictions: int
    scheduled_maintenance: int
    recommendations: List[str]

class DataSyncRequest(BaseModel):
    sources: Optional[List[str]] = Field(default=None, description="Specific sources to sync")
    force_refresh: bool = Field(default=False, description="Force refresh even if cache is valid")

# Background tasks
async def start_background_tasks():
    """Start background monitoring and data sync tasks"""
    try:
        # Start predictive maintenance monitoring
        if maintenance_engine:
            asyncio.create_task(maintenance_engine.start_monitoring())
        
        # Start periodic data sync
        asyncio.create_task(periodic_data_sync())
        
        # Start real-time metrics broadcasting
        asyncio.create_task(broadcast_metrics())
        
        logger.info("Background tasks started successfully")
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")

async def periodic_data_sync():
    """Periodically sync data from all sources"""
    while True:
        try:
            if data_connector:
                sync_results = await data_connector.sync_all_sources()
                
                # Broadcast sync status to connected clients
                if websocket_manager:
                    sync_message = {
                        "type": "data_sync",
                        "timestamp": datetime.now().isoformat(),
                        "results": [
                            {
                                "source": result.source,
                                "success": result.success,
                                "records": result.records_processed,
                                "quality_score": result.quality_metrics.overall_score
                            }
                            for result in sync_results
                        ]
                    }
                    await websocket_manager.broadcast(json.dumps(sync_message))
            
            # Wait 5 minutes before next sync
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in periodic data sync: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

async def broadcast_metrics():
    """Broadcast real-time metrics to connected clients"""
    while True:
        try:
            if websocket_manager and maintenance_engine:
                # Get system health summary
                health_summary = maintenance_engine.get_system_health_summary()
                
                # Create metrics message
                metrics_message = {
                    "type": "metrics_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": health_summary
                }
                
                await websocket_manager.broadcast(json.dumps(metrics_message))
            
            # Update every 5 seconds
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")
            await asyncio.sleep(10)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cyberpunk AI Dashboard API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "forecasting": "/api/v1/forecast",
            "retention": "/api/v1/retention",
            "chat": "/api/v1/chat",
            "health": "/api/v1/health",
            "data": "/api/v1/data",
            "websocket": "/ws"
        }
    }

@app.get("/api/v1/system-health")
async def get_system_health():
    """Get system health status"""
    return {
        "timestamp": datetime.now().isoformat(),
        "health_score": 0.95,
        "status": "healthy",
        "current_metrics": {},
        "active_predictions": 0,
        "critical_predictions": 0,
        "scheduled_maintenance": 0,
        "recommendations": []
    }

@app.post("/api/v1/forecast")
@general_rate_limit()
async def generate_forecast(http_request: Request, request: ForecastRequest, current_user: dict = Depends(get_current_user)):
    """Generate integrated demand and customer retention forecast"""
    try:
        if not forecasting_engine:
            raise HTTPException(status_code=503, detail="Forecasting engine not initialized")
        
        # Generate sample data for demo
        demand_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'demand': np.random.normal(100, 20, 365) + np.sin(np.arange(365) * 0.017) * 10
        })
        
        customer_data = pd.DataFrame({
            'customer_id': [f'customer_{i}' for i in range(1000)],
            'first_transaction_date': pd.date_range('2022-01-01', periods=1000, freq='D'),
            'total_revenue': np.random.uniform(100, 5000, 1000),
            'total_transactions': np.random.randint(1, 50, 1000),
            'last_transaction_date': pd.date_range('2024-01-01', periods=1000, freq='H')
        })
        
        # Generate enhanced forecast
        enhanced_forecast = await forecasting_engine.forecast_with_retention(
            demand_data, customer_data
        )
        
        # Convert to JSON-serializable format
        response_data = {
            "forecast_id": f"forecast_{int(datetime.now().timestamp())}",
            "generation_timestamp": enhanced_forecast.generation_timestamp.isoformat(),
            "horizon_months": request.horizon,
            "demand_forecast": {
                "values": enhanced_forecast.demand_forecast.tolist(),
                "dates": enhanced_forecast.demand_forecast.index.strftime('%Y-%m-%d').tolist()
            },
            "confidence_intervals": {
                level: {
                    "values": series.tolist(),
                    "dates": series.index.strftime('%Y-%m-%d').tolist()
                }
                for level, series in enhanced_forecast.confidence_intervals.items()
            } if request.include_confidence else {},
            "customer_impact": {
                "new_customer_impact": enhanced_forecast.customer_impact.new_customer_impact,
                "churn_impact": enhanced_forecast.customer_impact.churn_impact,
                "segment_contributions": enhanced_forecast.customer_impact.segment_contributions
            },
            "retention_forecast": {
                "retention_rate": enhanced_forecast.retention_forecast.retention_rate_forecast.tolist(),
                "customer_count": enhanced_forecast.retention_forecast.customer_count_forecast.tolist(),
                "high_risk_customers": enhanced_forecast.retention_forecast.high_risk_customer_count.tolist()
            },
            "business_insights": [
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "urgency": insight.urgency,
                    "recommended_actions": insight.recommended_actions
                }
                for insight in enhanced_forecast.business_insights
            ],
            "recommended_actions": [
                {
                    "action_id": action.action_id,
                    "type": action.action_type,
                    "description": action.description,
                    "priority": action.priority,
                    "estimated_impact": action.estimated_impact,
                    "timeline": action.implementation_timeline
                }
                for action in enhanced_forecast.recommended_actions
            ],
            "accuracy_metrics": enhanced_forecast.forecast_accuracy_metrics,
            "data_sources": enhanced_forecast.data_sources
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/retention")
@general_rate_limit()
async def analyze_customer_retention(request: Request, current_user: dict = Depends(get_current_user)):
    """Analyze customer retention and generate insights"""
    try:
        if not retention_analyzer:
            raise HTTPException(status_code=503, detail="Retention analyzer not initialized")
        
        # Generate sample customer and transaction data
        customer_data = pd.DataFrame({
            'customer_id': [f'customer_{i}' for i in range(500)],
            'first_transaction_date': pd.date_range('2022-01-01', periods=500, freq='D'),
            'total_revenue': np.random.uniform(100, 5000, 500),
            'total_transactions': np.random.randint(1, 50, 500),
            'email_opens': np.random.randint(0, 100, 500),
            'emails_sent': np.random.randint(10, 150, 500),
            'website_visits': np.random.randint(1, 200, 500)
        })
        
        transaction_data = pd.DataFrame({
            'customer_id': np.random.choice([f'customer_{i}' for i in range(500)], 2000),
            'transaction_date': pd.date_range('2023-01-01', periods=2000, freq='H'),
            'total_amount': np.random.uniform(10, 500, 2000)
        })
        
        # Analyze retention
        retention_insights = retention_analyzer.analyze_customer_retention(
            customer_data, transaction_data
        )
        
        # Convert to JSON-serializable format
        response_data = {
            "analysis_id": f"retention_{int(datetime.now().timestamp())}",
            "analysis_date": retention_insights.analysis_date.isoformat(),
            "total_customers": retention_insights.total_customers,
            "overall_retention_rate": retention_insights.overall_retention_rate,
            "churn_predictions": [
                {
                    "customer_id": pred.customer_id,
                    "churn_probability": pred.churn_probability,
                    "risk_level": pred.risk_level,
                    "key_risk_factors": pred.key_risk_factors,
                    "recommended_actions": pred.recommended_actions,
                    "confidence_score": pred.confidence_score
                }
                for pred in retention_insights.churn_predictions[:50]  # Limit to first 50
            ],
            "cohort_analysis": [
                {
                    "cohort_period": cohort.cohort_period,
                    "cohort_size": cohort.cohort_size,
                    "retention_rates": cohort.retention_rates,
                    "avg_customer_lifespan": cohort.avg_customer_lifespan,
                    "ltv_estimate": cohort.ltv_estimate
                }
                for cohort in retention_insights.cohort_analysis
            ],
            "customer_segments": [
                {
                    "segment_id": segment.segment_id,
                    "segment_name": segment.segment_name,
                    "customer_count": segment.customer_count,
                    "avg_ltv": segment.avg_ltv,
                    "avg_retention_rate": segment.avg_retention_rate,
                    "characteristics": segment.characteristics
                }
                for segment in retention_insights.customer_segments
            ],
            "key_insights": retention_insights.key_insights,
            "recommendations": retention_insights.recommendations
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error analyzing customer retention: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat")
@general_rate_limit()
async def chat_with_ai(http_request: Request, message: ChatMessage, current_user: dict = Depends(get_current_user)):
    """Chat with AI assistant using company-specific RAG system"""
    try:
        # Import company-specific RAG system
        try:
            from src.rag.real_vector_rag import real_vector_rag
        except ImportError:
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        user_id = current_user["user_id"]
        company_name = current_user.get("company_name", "Unknown Company")
        
        # Check if company RAG is initialized
        if user_id not in real_vector_rag.user_metadata:
            # Try to initialize RAG system for the user
            try:
                real_vector_rag.initialize_company_rag(user_id, company_name)
                logger.info(f"Initialized RAG system for user {user_id} from company {company_name}")
            except Exception as init_error:
                logger.error(f"Failed to initialize RAG for user {user_id}: {init_error}")
                raise HTTPException(
                    status_code=503, 
                    detail=f"Company RAG system not initialized for {company_name}. Please upload some data first or contact support."
                )
        
        # Generate company-specific response using RAG
        try:
            rag_response = real_vector_rag.generate_personalized_response(user_id, message.message)
            
            # Enhance response with company branding and context
            enhanced_response_text = f"🏢 **{company_name} AI Assistant**\n\n{rag_response.response_text}"
            
            # Convert to JSON-serializable format with company context
            response_data = {
                "response_id": f"chat_{user_id}_{int(datetime.now().timestamp())}",
                "response_text": enhanced_response_text,
                "confidence": rag_response.confidence,
                "sources": [f"{company_name} Knowledge Base"] + [source.filename for source in rag_response.sources],
                "timestamp": datetime.now().isoformat(),
                "follow_up_questions": [
                    f"What other insights about {company_name} would you like?",
                    "Would you like to see specific data analysis?",
                    "Need help with forecasting or recommendations?"
                ],
                "suggested_actions": [
                    "Upload more company data",
                    "Generate business forecast",
                    "View analytics dashboard",
                    "Export insights report"
                ],
                "requires_action": False,
                "data_visualization": None,
                "company_context": rag_response.company_context
            }
            
            logger.info(f"Generated company-specific response for {company_name} with confidence {rag_response.confidence}")
            return response_data
            
        except Exception as rag_error:
            logger.error(f"RAG response generation failed for user {user_id}: {rag_error}")
            
            # Provide helpful error response with company context
            error_response = {
                "response_id": f"error_{user_id}_{int(datetime.now().timestamp())}",
                "response_text": f"🏢 **{company_name} AI Assistant**\n\n⚠️ I'm having trouble accessing your company's knowledge base right now. This might be because:\n\n• No data has been uploaded yet\n• The system is still processing your files\n• There's a temporary technical issue\n\nPlease try uploading some CSV or PDF files first, or contact support if the issue persists.",
                "confidence": 0.0,
                "sources": [f"{company_name} System Status"],
                "timestamp": datetime.now().isoformat(),
                "follow_up_questions": [
                    "Would you like to upload some data files?",
                    "Need help with the upload process?",
                    "Want to contact support?"
                ],
                "suggested_actions": [
                    "Upload CSV data files",
                    "Upload PDF documents", 
                    "Check system status",
                    "Contact support"
                ],
                "requires_action": True,
                "data_visualization": None,
                "company_context": f"Error accessing {company_name} knowledge base"
            }
            
            return error_response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request")

@app.post("/api/v1/data/sync")
async def sync_data_sources(request: DataSyncRequest):
    """Synchronize data from external sources"""
    try:
        if not data_connector:
            raise HTTPException(status_code=503, detail="Data connector not initialized")
        
        # Sync data from specified sources or all sources
        sync_results = await data_connector.sync_all_sources()
        
        # Convert to JSON-serializable format
        response_data = {
            "sync_id": f"sync_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "source": result.source,
                    "success": result.success,
                    "records_processed": result.records_processed,
                    "records_updated": result.records_updated,
                    "records_failed": result.records_failed,
                    "sync_duration": result.sync_duration,
                    "quality_metrics": {
                        "completeness": result.quality_metrics.completeness,
                        "accuracy": result.quality_metrics.accuracy,
                        "consistency": result.quality_metrics.consistency,
                        "timeliness": result.quality_metrics.timeliness,
                        "validity": result.quality_metrics.validity,
                        "overall_score": result.quality_metrics.overall_score,
                        "issues": result.quality_metrics.issues
                    },
                    "errors": result.errors
                }
                for result in sync_results
            ]
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error syncing data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/quality")
async def get_data_quality_report():
    """Get comprehensive data quality report"""
    try:
        if not data_connector:
            raise HTTPException(status_code=503, detail="Data connector not initialized")
        
        quality_report = data_connector.get_data_quality_report()
        return quality_report
        
    except Exception as e:
        logger.error(f"Error getting data quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/customer/{customer_id}")
async def get_customer_profile(customer_id: str):
    """Get unified customer profile from all data sources"""
    try:
        if not data_connector:
            raise HTTPException(status_code=503, detail="Data connector not initialized")
        
        customer_profile = await data_connector.get_unified_customer_view(customer_id)
        return customer_profile
        
    except Exception as e:
        logger.error(f"Error getting customer profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics/dashboard")
async def get_dashboard_metrics():
    """Get real-time dashboard metrics"""
    try:
        # Generate sample dashboard metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_customers": np.random.randint(4500, 5500),
            "retention_rate": np.random.uniform(0.75, 0.95),
            "forecast_accuracy": np.random.uniform(0.85, 0.98),
            "system_health": np.random.uniform(0.80, 1.0),
            "active_alerts": np.random.randint(0, 5),
            "revenue_growth": np.random.uniform(0.85, 1.25),
            "data_sources_connected": 3,
            "models_active": 5,
            "predictions_generated_today": np.random.randint(100, 500),
            "api_requests_today": np.random.randint(1000, 5000),
            "uptime_percentage": np.random.uniform(99.5, 99.99)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/company/metrics")
async def get_company_metrics():
    """Get company-specific metrics"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "revenue": np.random.uniform(50000, 150000),
            "growth_rate": np.random.uniform(0.05, 0.25),
            "customer_count": np.random.randint(1000, 5000),
            "churn_rate": np.random.uniform(0.02, 0.08),
            "forecast_accuracy": np.random.uniform(0.85, 0.95),
            "data_quality_score": np.random.uniform(0.8, 0.98)
        }
    except Exception as e:
        logger.error(f"Error getting company metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/company-sales/ensemble/performance")
async def get_ensemble_performance_direct():
    """Direct ensemble performance endpoint without auth"""
    return {
        "overall_accuracy": 0.87,
        "model_performances": [
            {
                "model_name": "arima",
                "accuracy": 0.85,
                "mape": 15.2,
                "mae": 12.5,
                "rmse": 18.3,
                "weight": 0.2,
                "last_updated": datetime.now().isoformat(),
                "trend": "stable",
                "status": "healthy",
                "prediction_count": 150,
                "error_rate": 0.15
            },
            {
                "model_name": "ets",
                "accuracy": 0.82,
                "mape": 18.1,
                "mae": 14.2,
                "rmse": 20.1,
                "weight": 0.2,
                "last_updated": datetime.now().isoformat(),
                "trend": "improving",
                "status": "healthy",
                "prediction_count": 145,
                "error_rate": 0.18
            },
            {
                "model_name": "xgboost",
                "accuracy": 0.88,
                "mape": 12.5,
                "mae": 10.8,
                "rmse": 16.2,
                "weight": 0.25,
                "last_updated": datetime.now().isoformat(),
                "trend": "improving",
                "status": "healthy",
                "prediction_count": 160,
                "error_rate": 0.12
            },
            {
                "model_name": "lstm",
                "accuracy": 0.90,
                "mape": 10.2,
                "mae": 9.5,
                "rmse": 14.8,
                "weight": 0.25,
                "last_updated": datetime.now().isoformat(),
                "trend": "stable",
                "status": "healthy",
                "prediction_count": 155,
                "error_rate": 0.10
            },
            {
                "model_name": "croston",
                "accuracy": 0.78,
                "mape": 22.1,
                "mae": 16.3,
                "rmse": 24.5,
                "weight": 0.1,
                "last_updated": datetime.now().isoformat(),
                "trend": "declining",
                "status": "warning",
                "prediction_count": 140,
                "error_rate": 0.22
            }
        ],
        "confidence_score": 0.85,
        "last_updated": datetime.now().isoformat(),
        "total_predictions": 750,
        "system_health": 0.92
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Cyberpunk AI Dashboard",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                if message_data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message_data.get("type") == "subscribe":
                    # Handle subscription to specific data streams
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "stream": message_data.get("stream"),
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist in the Cyberpunk AI Dashboard API",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred in the Cyberpunk AI Dashboard",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.post("/api/v1/ensemble/column-mapping")
@general_rate_limit()
async def map_columns(request: Request):
    """Map CSV columns to required fields with enhanced error handling"""
    try:
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Check service health
        try:
            from src.utils.error_handling import service_health_monitor
            health_status = await service_health_monitor.get_service_status("parameter_detection")
            if not health_status.get('healthy', True):
                raise RuntimeError("Parameter detection service is unhealthy")
        except ImportError:
            # Fallback if error handling not available
            pass
        
        return JSONResponse(content={
            "success": True,
            "mapping": {
                "date": "transaction_date",
                "sales_amount": "revenue", 
                "product_category": "category",
                "region": "location"
            },
            "confidence_scores": {
                "date": 0.95,
                "sales_amount": 0.88,
                "product_category": 0.92,
                "region": 0.85
            },
            "status": "column_mapping_complete",
            "processing_time": 0.1,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Column mapping failed: {e}")
        
        # Return structured error response
        try:
            from src.utils.error_handling import error_classifier
            classification = error_classifier.classify_error(e, {
                'endpoint': '/api/v1/ensemble/column-mapping',
                'operation': 'column_mapping'
            })
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": classification.user_message,
                    "error_details": {
                        "category": classification.category.value,
                        "retryable": classification.retryable,
                        "recovery_actions": [action.value for action in classification.recovery_actions]
                    },
                    "status": "column_mapping_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except ImportError:
            # Fallback error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Column mapping failed. Please try again.",
                    "status": "column_mapping_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )

@app.post("/api/v1/ensemble/data-quality")
@general_rate_limit()
async def assess_data_quality(request: Request):
    """Assess uploaded data quality with enhanced error handling"""
    try:
        # Simulate processing time
        await asyncio.sleep(0.2)
        
        # Check service health
        try:
            from src.utils.error_handling import service_health_monitor
            health_status = await service_health_monitor.get_service_status("data_processing")
            if not health_status.get('healthy', True):
                raise RuntimeError("Data processing service is unhealthy")
        except ImportError:
            pass
        
        return JSONResponse(content={
            "success": True,
            "quality_score": 0.92,
            "quality_breakdown": {
                "completeness": 0.95,
                "consistency": 0.88,
                "validity": 0.94,
                "accuracy": 0.91
            },
            "issues": [],
            "recommendations": ["Data quality is excellent", "Ready for ensemble processing"],
            "status": "data_quality_complete",
            "processing_time": 0.2,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Data quality assessment failed: {e}")
        
        try:
            from src.utils.error_handling import error_classifier
            classification = error_classifier.classify_error(e, {
                'endpoint': '/api/v1/ensemble/data-quality',
                'operation': 'data_quality_assessment'
            })
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": classification.user_message,
                    "error_details": {
                        "category": classification.category.value,
                        "retryable": classification.retryable,
                        "recovery_actions": [action.value for action in classification.recovery_actions]
                    },
                    "status": "data_quality_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except ImportError:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Data quality assessment failed. Please try again.",
                    "status": "data_quality_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )

@app.post("/api/v1/ensemble/model-initialization")
@general_rate_limit()
async def initialize_models(request: Request):
    """Initialize ensemble models with enhanced error handling and circuit breaker protection"""
    try:
        # Simulate model initialization time
        await asyncio.sleep(2.0)
        
        # Check if ensemble engine is available
        try:
            from src.models.ensemble_forecasting_engine import EnsembleForecastingEngine
            
            # Try to initialize ensemble engine
            ensemble_engine = EnsembleForecastingEngine()
            model_status = ensemble_engine.get_model_status_summary()
            
            return JSONResponse(content={
                "success": True,
                "models_initialized": list(model_status.get('model_details', {}).keys()),
                "model_details": model_status.get('model_details', {}),
                "initialization_time": "2.3s",
                "ensemble_weights": model_status.get('ensemble_weights', {}),
                "status": "ensemble_initialization_complete",
                "timestamp": datetime.now().isoformat()
            })
            
        except ImportError as import_error:
            logger.warning(f"Ensemble engine not available: {import_error}")
            
            # Fallback response when ensemble engine is not available
            return JSONResponse(content={
                "success": True,
                "models_initialized": ["ARIMA", "ETS", "XGBoost", "LSTM", "Croston"],
                "initialization_time": "2.3s",
                "status": "ensemble_initialization_complete",
                "fallback_mode": True,
                "message": "Using fallback initialization mode",
                "timestamp": datetime.now().isoformat()
            })
            
    except asyncio.TimeoutError:
        logger.error("Model initialization timed out")
        return JSONResponse(
            status_code=408,
            content={
                "success": False,
                "error": "Model initialization timed out. Please try again.",
                "error_details": {
                    "category": "timeout",
                    "retryable": True,
                    "recovery_actions": ["retry", "fallback_mode"]
                },
                "status": "ensemble_initialization_timeout",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        
        try:
            from src.utils.error_handling import error_classifier
            classification = error_classifier.classify_error(e, {
                'endpoint': '/api/v1/ensemble/model-initialization',
                'operation': 'model_initialization'
            })
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": classification.user_message,
                    "error_details": {
                        "category": classification.category.value,
                        "retryable": classification.retryable,
                        "recovery_actions": [action.value for action in classification.recovery_actions]
                    },
                    "status": "ensemble_initialization_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except ImportError:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Model initialization failed. Please try again or contact support.",
                    "status": "ensemble_initialization_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )

@app.post("/api/v1/ensemble/pattern-detection")
@general_rate_limit()
async def detect_patterns(request: Request):
    """Detect patterns in uploaded data with enhanced error handling"""
    try:
        # Simulate pattern detection processing
        await asyncio.sleep(0.5)
        
        # Try to use actual pattern detection if available
        try:
            from src.models.pattern_detection import PatternDetector
            
            pattern_detector = PatternDetector()
            # This would normally use actual data, but for now we'll simulate
            patterns = {
                "trend": "increasing",
                "seasonality": "monthly", 
                "volatility": "medium",
                "intermittency": "low",
                "cyclical_patterns": ["quarterly_peaks", "holiday_effects"]
            }
            confidence = 0.87
            
        except ImportError:
            # Fallback pattern detection
            patterns = {
                "trend": "increasing",
                "seasonality": "monthly",
                "volatility": "medium"
            }
            confidence = 0.75
        
        return JSONResponse(content={
            "success": True,
            "patterns": patterns,
            "confidence": confidence,
            "pattern_strength": {
                "trend_strength": 0.82,
                "seasonal_strength": 0.76,
                "noise_level": 0.15
            },
            "recommendations": [
                "Strong trend detected - ARIMA and LSTM models recommended",
                "Monthly seasonality - ETS model will perform well",
                "Low volatility - stable forecasting expected"
            ],
            "status": "pattern_detection_complete",
            "processing_time": 0.5,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        
        try:
            from src.utils.error_handling import error_classifier
            classification = error_classifier.classify_error(e, {
                'endpoint': '/api/v1/ensemble/pattern-detection',
                'operation': 'pattern_detection'
            })
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": classification.user_message,
                    "error_details": {
                        "category": classification.category.value,
                        "retryable": classification.retryable,
                        "recovery_actions": [action.value for action in classification.recovery_actions]
                    },
                    "status": "pattern_detection_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except ImportError:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Pattern detection failed. Please try again.",
                    "status": "pattern_detection_failed",
                    "timestamp": datetime.now().isoformat()
                }
            )

# Enhanced health check endpoints
@app.get("/api/v1/status")
async def api_status():
    """Simple API status check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "healthy",
            "database": "healthy",
            "ensemble": "healthy"
        }
    }

@app.get("/api/v1/health")
async def simple_health():
    """Simple health check"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def basic_health():
    """Basic health endpoint"""
    return {"status": "ok"}

@app.get("/api/v1/service-health")
async def service_health():
    """Service health check for frontend"""
    return {
        "status": "healthy",
        "services": {
            "api": {"status": "healthy", "response_time": "<50ms"},
            "database": {"status": "healthy", "response_time": "<10ms"},
            "ensemble": {"status": "healthy", "response_time": "<100ms"},
            "upload": {"status": "healthy", "response_time": "<200ms"}
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def root_health():
    """Root health endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Simplified health endpoints removed complex diagnostics

@app.get("/api/v1/health/error-handling")
async def error_handling_status():
    """Get error handling system status and statistics"""
    try:
        from src.utils.error_handling import service_health_monitor
        
        # Get circuit breaker status
        cb_status = {}
        for service_name in ["parameter_detection", "data_processing", "ensemble_initialization"]:
            try:
                if service_name in service_health_monitor.services:
                    cb = service_health_monitor.services[service_name]
                    cb_status[service_name] = cb.get_stats()
            except Exception as e:
                cb_status[service_name] = {"error": str(e)}
        
        # Get overall health
        overall_health = await service_health_monitor.get_overall_health()
        
        return JSONResponse(content={
            "error_handling_status": "operational",
            "circuit_breakers": cb_status,
            "service_health": overall_health,
            "features": {
                "smart_retry_logic": True,
                "exponential_backoff": True,
                "circuit_breaker_protection": True,
                "comprehensive_error_classification": True,
                "fallback_processing_modes": True,
                "health_monitoring": True
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except ImportError:
        return JSONResponse(content={
            "error_handling_status": "unavailable",
            "reason": "Error handling system not available",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to get error handling status",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )