"""
Progressive Enhancement API
API endpoints for progressive data enhancement system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio

# Import progressive enhancement components
from ..models.progressive_enhancement_integration import (
    ProgressiveEnhancementIntegration, IntegratedEnhancementResult
)
from ..models.ensemble_forecasting_engine import EnsembleForecastingEngine

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/progressive-enhancement", tags=["Progressive Enhancement"])

# Global progressive enhancement system (will be initialized by main app)
progressive_enhancement_system: Optional[ProgressiveEnhancementIntegration] = None

# Pydantic models for API
class DataUploadRequest(BaseModel):
    company_id: str = Field(..., description="Company identifier")
    data: List[Dict[str, Any]] = Field(..., description="Data records")
    target_column: str = Field(default="sales_amount", description="Target column for forecasting")
    is_incremental: bool = Field(default=True, description="Whether this is incremental data")

class EnhancementHistoryResponse(BaseModel):
    company_id: str
    progressive_enhancements: List[Dict[str, Any]]
    quality_enhancements: List[Dict[str, Any]]
    validation_suggestions: List[Dict[str, Any]]
    maturity_assessment: Optional[Dict[str, Any]]
    confidence_evolution: List[Dict[str, Any]]

class MaturityAssessmentResponse(BaseModel):
    company_id: str
    maturity_level: str
    data_months: int
    pattern_detection_capability: str
    seasonality_detection_enabled: bool
    advanced_trend_analysis_enabled: bool
    confidence_interval_precision: float
    recommended_forecast_horizon: int
    data_quality_score: float

class IntegratedProcessingResponse(BaseModel):
    company_id: str
    processing_timestamp: str
    quality_improvement: float
    progressive_enhancements_applied: int
    quality_enhancements_applied: int
    validation_suggestions_count: int
    forecast_continuity_maintained: bool
    overall_improvement_score: float
    ensemble_accuracy: float
    forecast_data: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    integration_overview: Dict[str, Any]
    progressive_learning_status: Dict[str, Any]
    data_quality_status: Dict[str, Any]
    recent_processing_sessions: List[Dict[str, Any]]
    system_health: Dict[str, bool]

def get_progressive_enhancement_system() -> ProgressiveEnhancementIntegration:
    """Dependency to get progressive enhancement system"""
    if progressive_enhancement_system is None:
        raise HTTPException(status_code=500, detail="Progressive enhancement system not initialized")
    return progressive_enhancement_system

def initialize_progressive_enhancement_system(ensemble_engine: EnsembleForecastingEngine):
    """Initialize the progressive enhancement system"""
    global progressive_enhancement_system
    progressive_enhancement_system = ProgressiveEnhancementIntegration(ensemble_engine)
    logger.info("Progressive enhancement system initialized")

@router.post("/process-data", response_model=IntegratedProcessingResponse)
async def process_company_data(
    request: DataUploadRequest,
    background_tasks: BackgroundTasks,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Process company data with integrated progressive and quality enhancements
    """
    try:
        logger.info(f"Processing data for company {request.company_id}")
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Validate required columns
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' not found in data"
            )
        
        # Process data through integrated enhancement system
        if request.is_incremental:
            # For incremental processing, we need historical data
            # In a real implementation, this would be retrieved from database
            historical_data = df.copy()  # Simplified for this example
            result = await system.process_company_data(
                request.company_id, df, historical_data, request.target_column
            )
        else:
            # First-time processing
            result = await system.process_company_data(
                request.company_id, df, None, request.target_column
            )
        
        # Convert ensemble result to API response format
        forecast_data = {}
        if result.ensemble_result:
            forecast_data = {
                "point_forecast": result.ensemble_result.point_forecast.to_dict() if hasattr(result.ensemble_result.point_forecast, 'to_dict') else {},
                "confidence_intervals": {
                    level: interval.to_dict() if hasattr(interval, 'to_dict') else {}
                    for level, interval in result.ensemble_result.confidence_intervals.items()
                },
                "model_weights": result.ensemble_result.model_weights,
                "pattern_analysis": {
                    "pattern_type": result.ensemble_result.pattern_analysis.pattern_type,
                    "seasonality_strength": result.ensemble_result.pattern_analysis.seasonality_strength,
                    "trend_strength": result.ensemble_result.pattern_analysis.trend_strength,
                    "confidence": result.ensemble_result.pattern_analysis.confidence
                } if result.ensemble_result.pattern_analysis else {},
                "forecast_horizon": result.ensemble_result.forecast_horizon,
                "data_quality_score": result.ensemble_result.data_quality_score
            }
        
        response = IntegratedProcessingResponse(
            company_id=result.company_id,
            processing_timestamp=result.processing_timestamp.isoformat(),
            quality_improvement=result.quality_score_after - result.quality_score_before,
            progressive_enhancements_applied=len(result.progressive_enhancements),
            quality_enhancements_applied=len(result.quality_enhancements),
            validation_suggestions_count=len(result.validation_suggestions),
            forecast_continuity_maintained=result.forecast_continuity_maintained,
            overall_improvement_score=result.overall_improvement_score,
            ensemble_accuracy=result.ensemble_result.ensemble_accuracy if result.ensemble_result else 0.0,
            forecast_data=forecast_data
        )
        
        logger.info(f"Data processing complete for company {request.company_id}")
        return response
        
    except Exception as e:
        logger.error(f"Data processing failed for company {request.company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")

@router.get("/maturity-assessment/{company_id}", response_model=MaturityAssessmentResponse)
async def get_maturity_assessment(
    company_id: str,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get data maturity assessment for a company
    """
    try:
        assessment = system.progressive_learning.get_maturity_assessment(company_id)
        
        if not assessment:
            raise HTTPException(status_code=404, detail=f"No maturity assessment found for company {company_id}")
        
        return MaturityAssessmentResponse(
            company_id=company_id,
            maturity_level=assessment['maturity_level'],
            data_months=assessment['data_months'],
            pattern_detection_capability=assessment['pattern_detection_capability'],
            seasonality_detection_enabled=assessment['seasonality_detection_enabled'],
            advanced_trend_analysis_enabled=assessment['advanced_trend_analysis_enabled'],
            confidence_interval_precision=assessment['confidence_interval_precision'],
            recommended_forecast_horizon=assessment['recommended_forecast_horizon'],
            data_quality_score=assessment['data_quality_score']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get maturity assessment for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get maturity assessment: {str(e)}")

@router.get("/enhancement-history/{company_id}", response_model=EnhancementHistoryResponse)
async def get_enhancement_history(
    company_id: str,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get comprehensive enhancement history for a company
    """
    try:
        # Get enhancement history from subsystems
        progressive_history = system.progressive_learning.get_enhancement_history(company_id)
        quality_history = system.quality_enhancement.get_quality_enhancement_history(company_id)
        validation_suggestions = system.quality_enhancement.get_validation_suggestions(company_id)
        maturity_assessment = system.progressive_learning.get_maturity_assessment(company_id)
        confidence_evolution = system.progressive_learning.get_confidence_evolution(company_id)
        
        return EnhancementHistoryResponse(
            company_id=company_id,
            progressive_enhancements=progressive_history,
            quality_enhancements=quality_history,
            validation_suggestions=validation_suggestions,
            maturity_assessment=maturity_assessment,
            confidence_evolution=confidence_evolution
        )
        
    except Exception as e:
        logger.error(f"Failed to get enhancement history for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement history: {str(e)}")

@router.get("/confidence-evolution/{company_id}")
async def get_confidence_evolution(
    company_id: str,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get confidence interval evolution for a company
    """
    try:
        evolution = system.progressive_learning.get_confidence_evolution(company_id)
        
        if not evolution:
            raise HTTPException(status_code=404, detail=f"No confidence evolution data found for company {company_id}")
        
        return {
            "company_id": company_id,
            "confidence_evolution": evolution,
            "evolution_summary": {
                "total_points": len(evolution),
                "latest_interval_width": evolution[-1]['interval_width'] if evolution else 0,
                "average_interval_width": np.mean([point['interval_width'] for point in evolution]) if evolution else 0,
                "trend": "narrowing" if len(evolution) > 1 and evolution[-1]['interval_width'] < evolution[0]['interval_width'] else "stable"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get confidence evolution for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get confidence evolution: {str(e)}")

@router.get("/validation-suggestions/{company_id}")
async def get_validation_suggestions(
    company_id: str,
    priority: Optional[str] = None,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get validation improvement suggestions for a company
    """
    try:
        suggestions = system.quality_enhancement.get_validation_suggestions(company_id)
        
        # Filter by priority if specified
        if priority:
            suggestions = [s for s in suggestions if s.get('priority') == priority.lower()]
        
        return {
            "company_id": company_id,
            "suggestions": suggestions,
            "summary": {
                "total_suggestions": len(suggestions),
                "high_priority": len([s for s in suggestions if s.get('priority') == 'high']),
                "medium_priority": len([s for s in suggestions if s.get('priority') == 'medium']),
                "low_priority": len([s for s in suggestions if s.get('priority') == 'low'])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get validation suggestions for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation suggestions: {str(e)}")

@router.get("/integration-history/{company_id}")
async def get_integration_history(
    company_id: str,
    limit: int = 10,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get integration processing history for a company
    """
    try:
        history = system.get_integration_history(company_id)
        
        # Limit results
        if limit > 0:
            history = history[-limit:]
        
        return {
            "company_id": company_id,
            "integration_history": history,
            "summary": {
                "total_sessions": len(history),
                "average_quality_improvement": np.mean([h['quality_improvement'] for h in history]) if history else 0,
                "average_overall_improvement": np.mean([h['overall_improvement_score'] for h in history]) if history else 0,
                "latest_maturity_level": history[-1]['data_maturity_level'] if history else 'unknown'
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get integration history for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration history: {str(e)}")

@router.get("/company-summary/{company_id}")
async def get_company_enhancement_summary(
    company_id: str,
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get comprehensive enhancement summary for a company
    """
    try:
        summary = await system.get_company_enhancement_summary(company_id)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get company summary for {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get company summary: {str(e)}")

@router.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status(
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Get overall progressive enhancement system status
    """
    try:
        status = system.get_integration_status()
        
        return SystemStatusResponse(
            integration_overview=status['integration_overview'],
            progressive_learning_status=status['progressive_learning_status'],
            data_quality_status=status['data_quality_status'],
            recent_processing_sessions=status['recent_processing_sessions'],
            system_health=status['system_health']
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.post("/simulate-data-growth/{company_id}")
async def simulate_data_growth(
    company_id: str,
    months_to_simulate: int = Field(default=6, ge=1, le=24, description="Number of months to simulate"),
    system: ProgressiveEnhancementIntegration = Depends(get_progressive_enhancement_system)
):
    """
    Simulate progressive enhancement over time as more data becomes available
    """
    try:
        logger.info(f"Simulating data growth for company {company_id} over {months_to_simulate} months")
        
        # Get current maturity assessment
        current_assessment = system.progressive_learning.get_maturity_assessment(company_id)
        
        if not current_assessment:
            raise HTTPException(status_code=404, detail=f"No existing data found for company {company_id}")
        
        current_months = current_assessment['data_months']
        
        # Simulate progression through maturity levels
        simulation_results = []
        
        for month in range(1, months_to_simulate + 1):
            simulated_months = current_months + month
            
            # Determine maturity level for simulated months
            if simulated_months <= 3:
                maturity_level = "initial"
                capabilities = ["basic_pattern_detection"]
                confidence_precision = 0.3
            elif simulated_months <= 8:
                maturity_level = "developing"
                capabilities = ["intermediate_pattern_detection", "basic_seasonality"]
                confidence_precision = 0.5
            elif simulated_months <= 18:
                maturity_level = "mature"
                capabilities = ["advanced_pattern_detection", "full_seasonality", "trend_analysis"]
                confidence_precision = 0.7
            else:
                maturity_level = "advanced"
                capabilities = ["full_advanced_capabilities", "extended_forecasting", "dynamic_rebalancing"]
                confidence_precision = 0.9
            
            # Simulate quality improvements
            base_quality = current_assessment['data_quality_score']
            quality_improvement = min(0.1 * month / months_to_simulate, 0.1)  # Up to 10% improvement
            simulated_quality = min(base_quality + quality_improvement, 1.0)
            
            # Simulate forecast accuracy improvements
            base_accuracy = 0.7  # Assume baseline accuracy
            accuracy_improvement = min(0.2 * (simulated_months / 24), 0.2)  # Up to 20% improvement over 2 years
            simulated_accuracy = min(base_accuracy + accuracy_improvement, 0.95)
            
            simulation_results.append({
                "month": month,
                "total_months": simulated_months,
                "maturity_level": maturity_level,
                "capabilities_enabled": capabilities,
                "confidence_interval_precision": confidence_precision,
                "estimated_data_quality": simulated_quality,
                "estimated_forecast_accuracy": simulated_accuracy,
                "recommended_forecast_horizon": min(18, 3 + simulated_months // 2)
            })
        
        return {
            "company_id": company_id,
            "simulation_parameters": {
                "months_simulated": months_to_simulate,
                "starting_months": current_months,
                "starting_maturity": current_assessment['maturity_level']
            },
            "simulation_results": simulation_results,
            "summary": {
                "final_maturity_level": simulation_results[-1]["maturity_level"],
                "quality_improvement": simulation_results[-1]["estimated_data_quality"] - current_assessment['data_quality_score'],
                "accuracy_improvement": simulation_results[-1]["estimated_forecast_accuracy"] - 0.7,
                "capabilities_gained": len(simulation_results[-1]["capabilities_enabled"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data growth simulation failed for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for progressive enhancement system
    """
    try:
        if progressive_enhancement_system is None:
            return {"status": "unhealthy", "message": "Progressive enhancement system not initialized"}
        
        return {
            "status": "healthy",
            "message": "Progressive enhancement system is operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "progressive_learning": True,
                "data_quality_enhancement": True,
                "integration_system": True
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "message": f"Health check failed: {str(e)}"}

# WebSocket endpoint for real-time enhancement monitoring
@router.websocket("/ws/enhancement-monitor/{company_id}")
async def enhancement_monitor_websocket(websocket, company_id: str):
    """
    WebSocket endpoint for real-time enhancement monitoring
    """
    await websocket.accept()
    
    try:
        while True:
            # Send current enhancement status
            if progressive_enhancement_system:
                summary = await progressive_enhancement_system.get_company_enhancement_summary(company_id)
                await websocket.send_json({
                    "type": "enhancement_status",
                    "data": summary,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except Exception as e:
        logger.error(f"WebSocket connection error for company {company_id}: {e}")
    finally:
        await websocket.close()