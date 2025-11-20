"""
Ensemble-Aware Chat API
Provides endpoints for ensemble forecasting chat functionality with natural language processing
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

# Authentication dependency
def get_current_user(request: Request):
    """Get current authenticated user from request state"""
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    return request.state.user

# Import ensemble chat components
from src.ai_chatbot.ensemble_chat_processor import EnsembleChatProcessor, EnsembleChatResponse
from src.ai_chatbot.intelligent_query_processor import IntelligentQueryProcessor
from src.models.ensemble_forecasting_engine import EnsembleForecastingEngine
from src.models.business_insights_engine import BusinessInsightsEngine
from src.models.model_performance_tracker import ModelPerformanceTracker

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/ensemble-chat", tags=["ensemble-chat"])

# Request/Response models
class EnsembleChatRequest(BaseModel):
    """Request model for ensemble chat"""
    message: str = Field(..., description="User message/query")
    user_id: str = Field(default="anonymous", description="User ID")
    session_id: str = Field(default="default", description="Chat session ID")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class EnsembleChatResponseModel(BaseModel):
    """Response model for ensemble chat"""
    response_id: str
    response_text: str
    confidence: float
    sources: List[str]
    timestamp: str
    follow_up_questions: List[str]
    suggested_actions: List[str]
    ensemble_data: Optional[Dict[str, Any]] = None
    model_performance_data: Optional[Dict[str, Any]] = None
    forecast_data: Optional[Dict[str, Any]] = None
    insights_data: Optional[Dict[str, Any]] = None
    technical_explanation: Optional[str] = None
    plain_language_summary: Optional[str] = None

# Global instances (will be initialized when needed)
ensemble_chat_processor: Optional[EnsembleChatProcessor] = None
intelligent_query_processor: Optional[IntelligentQueryProcessor] = None

def get_ensemble_chat_processor() -> EnsembleChatProcessor:
    """Get or create ensemble chat processor instance"""
    global ensemble_chat_processor
    
    if ensemble_chat_processor is None:
        try:
            # Initialize ensemble components
            ensemble_engine = EnsembleForecastingEngine()
            insights_engine = BusinessInsightsEngine()
            performance_tracker = ModelPerformanceTracker()
            
            ensemble_chat_processor = EnsembleChatProcessor(
                ensemble_engine=ensemble_engine,
                insights_engine=insights_engine,
                performance_tracker=performance_tracker
            )
            
            logger.info("Ensemble chat processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble chat processor: {e}")
            # Create basic processor without engines for graceful degradation
            ensemble_chat_processor = EnsembleChatProcessor()
    
    return ensemble_chat_processor

def get_intelligent_query_processor() -> IntelligentQueryProcessor:
    """Get or create intelligent query processor instance"""
    global intelligent_query_processor
    
    if intelligent_query_processor is None:
        try:
            ensemble_processor = get_ensemble_chat_processor()
            
            # Try to get insights engine
            insights_engine = None
            try:
                insights_engine = BusinessInsightsEngine()
            except Exception as e:
                logger.warning(f"Could not initialize insights engine: {e}")
            
            intelligent_query_processor = IntelligentQueryProcessor(
                ensemble_chat_processor=ensemble_processor,
                insights_engine=insights_engine
            )
            
            logger.info("Intelligent query processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligent query processor: {e}")
            # Fallback to basic ensemble processor
            ensemble_processor = get_ensemble_chat_processor()
            intelligent_query_processor = IntelligentQueryProcessor(
                ensemble_chat_processor=ensemble_processor
            )
    
    return intelligent_query_processor

@router.post("/query", response_model=EnsembleChatResponseModel)
async def process_ensemble_query(request: EnsembleChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Process ensemble-aware chat query with intelligent understanding using company-specific RAG
    
    This endpoint provides natural language processing for ensemble forecasting queries,
    model performance questions, and business insights with plain-language explanations.
    """
    try:
        logger.info(f"Processing ensemble chat query from user {request.user_id}: {request.message[:100]}...")
        
        # Validate user authentication
        if request.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Add company context to request
        company_name = current_user.get("company_name", "Unknown Company")
        enhanced_context = request.context or {}
        enhanced_context.update({
            "company_name": company_name,
            "user_id": current_user["user_id"],
            "authenticated": True
        })
        
        # Check if company RAG is available
        try:
            from src.rag.real_vector_rag import real_vector_rag
            
            if current_user["user_id"] not in real_vector_rag.user_metadata:
                # Company RAG not initialized
                return EnsembleChatResponseModel(
                    response_id=f"ensemble_error_{int(datetime.now().timestamp())}",
                    response_text=f"🏢 **{company_name} Ensemble AI**\n\n⚠️ Your company's AI system isn't fully set up yet. Please upload some data files (CSV or PDF) to enable personalized forecasting and insights.\n\nOnce you upload data, I'll be able to provide:\n• Company-specific forecasts\n• Personalized business insights\n• Data-driven recommendations",
                    confidence=0.0,
                    sources=[f"{company_name} System Status"],
                    timestamp=datetime.now().isoformat(),
                    follow_up_questions=[
                        "Would you like help uploading data?",
                        "What type of forecasting do you need?",
                        "Need guidance on data requirements?"
                    ],
                    suggested_actions=[
                        "Upload CSV data files",
                        "Upload PDF documents",
                        "View data upload guide",
                        "Contact support"
                    ]
                )
        except ImportError:
            raise HTTPException(status_code=503, detail="Company RAG system not available")
        
        # Get intelligent query processor
        query_processor = get_intelligent_query_processor()
        
        # Process query with intelligent understanding and company context
        response = await query_processor.process_intelligent_query(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            user_context=enhanced_context
        )
        
        # Enhance response with company branding
        enhanced_response_text = f"🏢 **{company_name} Ensemble AI**\n\n{response.response_text}"
        
        # Add company-specific sources
        company_sources = [f"{company_name} Knowledge Base", f"{company_name} Ensemble Models"]
        enhanced_sources = company_sources + (response.sources or [])
        
        # Convert to API response format with company context
        api_response = EnsembleChatResponseModel(
            response_id=f"ensemble_{current_user['user_id']}_{int(datetime.now().timestamp())}",
            response_text=enhanced_response_text,
            confidence=response.confidence,
            sources=enhanced_sources,
            timestamp=datetime.now().isoformat(),
            follow_up_questions=response.follow_up_questions or [],
            suggested_actions=response.suggested_actions or [],
            ensemble_data=response.ensemble_data,
            model_performance_data=response.model_performance_data,
            forecast_data=response.forecast_data,
            insights_data=response.insights_data,
            technical_explanation=response.technical_explanation,
            plain_language_summary=response.plain_language_summary
        )
        
        logger.info(f"Successfully processed ensemble query for {company_name} with confidence {response.confidence:.2f}")
        return api_response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing ensemble chat query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process ensemble query: {str(e)}"
        )

@router.post("/basic", response_model=EnsembleChatResponseModel)
async def process_basic_ensemble_query(request: EnsembleChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Process basic ensemble query without intelligent processing using company-specific context
    
    Fallback endpoint for direct ensemble chat processing without advanced
    context management and conversation intelligence, but with company-specific RAG.
    """
    try:
        logger.info(f"Processing basic ensemble query from user {request.user_id}: {request.message[:100]}...")
        
        # Validate user authentication
        if request.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        company_name = current_user.get("company_name", "Unknown Company")
        
        # Add company context to request
        enhanced_context = request.context or {}
        enhanced_context.update({
            "company_name": company_name,
            "user_id": current_user["user_id"],
            "authenticated": True
        })
        
        # Get ensemble chat processor
        chat_processor = get_ensemble_chat_processor()
        
        # Process query directly with company context
        response = await chat_processor.process_ensemble_query(
            query=request.message,
            user_context=enhanced_context
        )
        
        # Enhance response with company branding
        enhanced_response_text = f"🏢 **{company_name} Ensemble AI**\n\n{response.response_text}"
        
        # Convert to API response format with company context
        api_response = EnsembleChatResponseModel(
            response_id=f"ensemble_basic_{current_user['user_id']}_{int(datetime.now().timestamp())}",
            response_text=enhanced_response_text,
            confidence=response.confidence,
            sources=[f"{company_name} Ensemble System"] + (response.sources or []),
            timestamp=datetime.now().isoformat(),
            follow_up_questions=response.follow_up_questions or [],
            suggested_actions=response.suggested_actions or [],
            ensemble_data=response.ensemble_data,
            model_performance_data=response.model_performance_data,
            forecast_data=response.forecast_data,
            insights_data=response.insights_data,
            technical_explanation=response.technical_explanation,
            plain_language_summary=response.plain_language_summary
        )
        
        logger.info(f"Successfully processed basic ensemble query for {company_name} with confidence {response.confidence:.2f}")
        return api_response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing basic ensemble query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process basic ensemble query: {str(e)}"
        )

@router.get("/capabilities")
async def get_ensemble_chat_capabilities():
    """
    Get information about ensemble chat capabilities
    
    Returns information about what types of queries the ensemble chat system can handle.
    """
    try:
        capabilities = {
            "query_types": [
                {
                    "type": "forecast",
                    "description": "Generate ensemble forecasts with confidence intervals",
                    "examples": [
                        "What will sales be next month?",
                        "Forecast demand for the next 6 months",
                        "Show me the ensemble prediction"
                    ]
                },
                {
                    "type": "performance",
                    "description": "Analyze model performance and accuracy metrics",
                    "examples": [
                        "How accurate are the models?",
                        "Which model is performing best?",
                        "Show me the forecast accuracy"
                    ]
                },
                {
                    "type": "model_comparison",
                    "description": "Compare different forecasting models",
                    "examples": [
                        "Compare ARIMA vs LSTM performance",
                        "Which model should I trust more?",
                        "Show me model rankings"
                    ]
                },
                {
                    "type": "insights",
                    "description": "Get business insights and recommendations",
                    "examples": [
                        "What insights do you have?",
                        "Give me business recommendations",
                        "What opportunities should I focus on?"
                    ]
                },
                {
                    "type": "weights",
                    "description": "Understand model weight distribution",
                    "examples": [
                        "Why is ARIMA weighted higher?",
                        "Show me model contributions",
                        "How are weights calculated?"
                    ]
                },
                {
                    "type": "confidence",
                    "description": "Understand forecast confidence and uncertainty",
                    "examples": [
                        "How confident is this forecast?",
                        "What are the confidence intervals?",
                        "How reliable is this prediction?"
                    ]
                }
            ],
            "supported_models": [
                "ARIMA - Time series analysis with trend and seasonality",
                "ETS - Exponential smoothing for seasonal patterns", 
                "XGBoost - Machine learning for complex relationships",
                "LSTM - Neural networks for sequence learning",
                "Croston - Specialized for intermittent demand"
            ],
            "features": [
                "Natural language query processing",
                "Plain language explanations of technical concepts",
                "Context-aware conversation management",
                "Intelligent follow-up question generation",
                "Multi-format response data (text, technical, structured)",
                "Real-time model performance monitoring",
                "Adaptive weight calculation explanations"
            ]
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting ensemble chat capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get capabilities: {str(e)}"
        )

@router.get("/health")
async def ensemble_chat_health_check():
    """
    Health check for ensemble chat system
    
    Returns the status of ensemble chat components and their availability.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check ensemble chat processor
        try:
            processor = get_ensemble_chat_processor()
            health_status["components"]["ensemble_chat_processor"] = {
                "status": "available",
                "has_ensemble_engine": processor.ensemble_engine is not None,
                "has_insights_engine": processor.insights_engine is not None,
                "has_performance_tracker": processor.performance_tracker is not None
            }
        except Exception as e:
            health_status["components"]["ensemble_chat_processor"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check intelligent query processor
        try:
            intelligent_processor = get_intelligent_query_processor()
            health_status["components"]["intelligent_query_processor"] = {
                "status": "available",
                "max_context_history": intelligent_processor.max_context_history
            }
        except Exception as e:
            health_status["components"]["intelligent_query_processor"] = {
                "status": "error", 
                "error": str(e)
            }
        
        # Determine overall status
        component_statuses = [comp.get("status") for comp in health_status["components"].values()]
        if "error" in component_statuses:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in ensemble chat health check: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }