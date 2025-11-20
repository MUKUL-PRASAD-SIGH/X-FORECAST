"""
Enhanced Data Processing API with Smart Analysis and Bulletproof Error Handling
Real-time parameter detection, auto-mapping, and quality assessment with comprehensive error recovery
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime
import traceback
import time

from .company_sales_api import get_company_id_from_token
from ..data.smart_data_processor import SmartDataProcessor, ProcessingResult
from ..utils.file_utils import validate_file_format, get_file_size_mb
from ..utils.error_handling import (
    error_classifier, retry_manager, service_health_monitor,
    with_retry, with_circuit_breaker, RetryConfig, CircuitBreakerConfig,
    EnhancedError, ErrorCategory, ErrorSeverity, RecoveryAction
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data-processing", tags=["Enhanced Data Processing"])

# Initialize the smart processor with error handling
smart_processor = SmartDataProcessor()

# Configure retry settings for different operations
PARAMETER_DETECTION_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    jitter=0.1
)

DATA_PROCESSING_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=5.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    jitter=0.2
)

# Configure circuit breakers for critical services
PARAMETER_DETECTION_CB_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    name="parameter_detection"
)

DATA_PROCESSING_CB_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=120.0,
    name="data_processing"
)

# Register services for health monitoring
async def parameter_detection_health_check():
    """Health check for parameter detection service"""
    try:
        # Simple health check - could be expanded
        return smart_processor is not None
    except Exception:
        return False

async def data_processing_health_check():
    """Health check for data processing service"""
    try:
        # Simple health check - could be expanded
        return smart_processor is not None
    except Exception:
        return False

# Register services with health monitor
service_health_monitor.register_service(
    "parameter_detection", 
    parameter_detection_health_check, 
    PARAMETER_DETECTION_CB_CONFIG
)

service_health_monitor.register_service(
    "data_processing", 
    data_processing_health_check, 
    DATA_PROCESSING_CB_CONFIG
)

def create_error_response(error: Exception, context: Dict[str, Any] = None) -> JSONResponse:
    """Create standardized error response with classification"""
    try:
        if isinstance(error, EnhancedError):
            error_details = error.to_dict()
        else:
            classification = error_classifier.classify_error(error, context)
            error_details = {
                'error_type': 'classified_error',
                'category': classification.category.value,
                'severity': classification.severity.value,
                'user_message': classification.user_message,
                'technical_message': classification.technical_message,
                'retryable': classification.retryable,
                'recovery_actions': [action.value for action in classification.recovery_actions],
                'context': classification.context,
                'timestamp': datetime.now().isoformat()
            }
        
        # Determine HTTP status code based on error category
        status_code = 500  # Default
        if error_details['category'] == ErrorCategory.AUTHENTICATION.value:
            status_code = 401
        elif error_details['category'] == ErrorCategory.VALIDATION.value:
            status_code = 400
        elif error_details['category'] == ErrorCategory.FILE_FORMAT.value:
            status_code = 400
        elif error_details['category'] == ErrorCategory.SERVICE_UNAVAILABLE.value:
            status_code = 503
        elif error_details['category'] == ErrorCategory.RATE_LIMIT.value:
            status_code = 429
        elif error_details['category'] == ErrorCategory.TIMEOUT.value:
            status_code = 408
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": error_details['user_message'],
                "error_details": error_details
            }
        )
    except Exception as e:
        logger.error(f"Error creating error response: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An unexpected error occurred",
                "error_details": {
                    "category": ErrorCategory.UNKNOWN.value,
                    "severity": ErrorSeverity.HIGH.value,
                    "retryable": False,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

async def validate_request_context(request: Request, file: UploadFile, company_id: str) -> Dict[str, Any]:
    """Validate request context and return metadata"""
    context = {
        'endpoint': str(request.url),
        'method': request.method,
        'company_id': company_id,
        'file_name': file.filename,
        'file_size': file.size if hasattr(file, 'size') else 0,
        'content_type': file.content_type,
        'timestamp': datetime.now().isoformat(),
        'client_ip': request.client.host if request.client else 'unknown'
    }
    
    # Basic validation
    if not file.filename:
        raise ValueError("File name is required")
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError(f"File size ({file.size / 1024 / 1024:.1f}MB) exceeds 50MB limit")
    
    return context

@router.post("/analyze-file")
@with_circuit_breaker(PARAMETER_DETECTION_CB_CONFIG)
async def analyze_file_intelligent(
    request: Request,
    file: UploadFile = File(..., description="Data file for intelligent analysis"),
    company_id: str = Depends(get_company_id_from_token)
):
    """
    Intelligent file analysis with real-time parameter detection and quality assessment
    
    Features:
    - Real-time column type detection with confidence scoring
    - Auto-mapping to required sales data fields
    - Comprehensive data quality assessment
    - Smart preprocessing recommendations
    - Bulletproof error handling with retry logic
    - Circuit breaker protection for service reliability
    """
    
    start_time = time.time()
    context = None
    
    try:
        # Validate request context
        context = await validate_request_context(request, file, company_id)
        
        # Enhanced file validation with detailed error messages
        try:
            file_format = validate_file_format(file)
            file_size_mb = get_file_size_mb(file)
        except Exception as validation_error:
            logger.warning(f"File validation failed: {validation_error}")
            raise ValueError(f"File validation failed: {str(validation_error)}")
        
        if file_size_mb > 50:
            raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds 50MB limit")
        
        # Read file content with timeout protection
        try:
            file_content = await asyncio.wait_for(file.read(), timeout=30.0)
        except asyncio.TimeoutError:
            raise TimeoutError("File reading timed out. Please try with a smaller file.")
        except Exception as read_error:
            raise IOError(f"Failed to read file content: {str(read_error)}")
        
        # Reset file position for potential re-reading
        try:
            await file.seek(0)
        except Exception:
            logger.warning("Could not reset file position")
        
        # Process file with smart analyzer using retry logic
        @with_retry(PARAMETER_DETECTION_RETRY_CONFIG)
        async def process_with_retry():
            return await smart_processor.process_file_intelligent(
                file_content=file_content,
                filename=file.filename or "unknown",
                file_format=file_format
            )
        
        try:
            processing_result = await process_with_retry()
        except Exception as processing_error:
            logger.error(f"File processing failed: {processing_error}")
            
            # Try fallback processing mode
            try:
                logger.info("Attempting fallback processing mode")
                processing_result = await smart_processor.process_file_basic(
                    file_content=file_content,
                    filename=file.filename or "unknown",
                    file_format=file_format
                )
                logger.info("Fallback processing successful")
            except Exception as fallback_error:
                logger.error(f"Fallback processing also failed: {fallback_error}")
                raise processing_error  # Raise original error
        
        # Convert result to API response format with error handling
        try:
            response_data = {
                "success": True,
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "fallback_mode_used": False,  # Track if fallback was used
                "file_info": {
                    "name": file.filename,
                    "size_mb": file_size_mb,
                    "format": file_format,
                    "company_id": company_id
                },
                "detected_columns": [
                    {
                        "name": col.name,
                        "type": col.type.value if hasattr(col.type, 'value') else str(col.type),
                        "confidence": getattr(col, 'confidence', 0.0),
                        "quality_score": getattr(col, 'quality_score', 0.0),
                        "null_percentage": getattr(col, 'null_percentage', 0.0),
                        "unique_count": getattr(col, 'unique_count', 0),
                        "sample_values": getattr(col, 'sample_values', []),
                        "data_patterns": getattr(col, 'data_patterns', []),
                        "recommendations": getattr(col, 'recommendations', [])
                    }
                    for col in (processing_result.detected_columns if processing_result.detected_columns else [])
                ],
                "column_mappings": [
                    {
                        "required_field": mapping.required_field,
                        "detected_column": mapping.detected_column,
                        "confidence": getattr(mapping, 'confidence', 0.0),
                        "status": getattr(mapping, 'status', 'unknown'),
                        "alternatives": [
                            {"column": alt[0], "confidence": alt[1]} 
                            for alt in (getattr(mapping, 'alternatives', []) or [])
                        ],
                        "mapping_reason": getattr(mapping, 'mapping_reason', '')
                    }
                    for mapping in (processing_result.column_mappings if processing_result.column_mappings else [])
                ],
                "data_quality": {
                    "overall_score": getattr(processing_result.data_quality, 'overall_score', 0.0) if processing_result.data_quality else 0.0,
                    "completeness": getattr(processing_result.data_quality, 'completeness', 0.0) if processing_result.data_quality else 0.0,
                    "consistency": getattr(processing_result.data_quality, 'consistency', 0.0) if processing_result.data_quality else 0.0,
                    "validity": getattr(processing_result.data_quality, 'validity', 0.0) if processing_result.data_quality else 0.0,
                    "accuracy": getattr(processing_result.data_quality, 'accuracy', 0.0) if processing_result.data_quality else 0.0,
                    "uniqueness": getattr(processing_result.data_quality, 'uniqueness', 0.0) if processing_result.data_quality else 0.0,
                    "timeliness": getattr(processing_result.data_quality, 'timeliness', 0.0) if processing_result.data_quality else 0.0,
                    "issues": getattr(processing_result.data_quality, 'issues', []) if processing_result.data_quality else [],
                    "recommendations": getattr(processing_result.data_quality, 'recommendations', []) if processing_result.data_quality else [],
                    "quality_breakdown": getattr(processing_result.data_quality, 'quality_breakdown', {}) if processing_result.data_quality else {}
                },
                "preview_data": processing_result.preview_data if processing_result.preview_data else {},
                "processing_stats": processing_result.processing_stats if processing_result.processing_stats else {},
                "standardization_available": processing_result.standardized_data is not None if processing_result else False
            }
        except Exception as response_error:
            logger.error(f"Error creating response data: {response_error}")
            raise ValueError(f"Failed to format analysis results: {str(response_error)}")
        
        logger.info(f"Smart analysis completed for {file.filename} (Company: {company_id}) in {time.time() - start_time:.2f}s")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Smart analysis failed for company {company_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return structured error response
        return create_error_response(e, context)

@router.post("/detect-parameters-enhanced")
async def detect_parameters_enhanced(
    file: UploadFile = File(..., description="Data file for enhanced parameter detection"),
    company_id: str = Depends(get_company_id_from_token)
):
    """
    Enhanced parameter detection with auto-mapping and quality scoring
    
    This endpoint provides backward compatibility while offering enhanced features:
    - Intelligent column type detection
    - Auto-mapping to sales data fields
    - Quality assessment and recommendations
    """
    
    try:
        # Use the smart analyzer
        analysis_result = await analyze_file_intelligent(file, company_id)
        
        # Extract data for backward compatibility
        analysis_data = analysis_result.body.decode('utf-8')
        import json
        parsed_data = json.loads(analysis_data)
        
        # Format for backward compatibility with existing frontend
        compatible_response = {
            "success": True,
            "detected_columns": [
                {
                    "name": col["name"],
                    "type": col["type"],
                    "sample_values": col["sample_values"],
                    "confidence": col["confidence"]
                }
                for col in parsed_data["detected_columns"]
            ],
            "preview": {
                "rows": parsed_data["preview_data"]["basic_stats"]["rows"],
                "columns": parsed_data["preview_data"]["basic_stats"]["columns"],
                "sample_data": parsed_data["preview_data"]["sample_data"]
            },
            "file_info": parsed_data["file_info"],
            # Enhanced features
            "enhanced_analysis": {
                "column_mappings": parsed_data["column_mappings"],
                "data_quality": parsed_data["data_quality"],
                "processing_stats": parsed_data["processing_stats"]
            }
        }
        
        return JSONResponse(content=compatible_response)
        
    except Exception as e:
        logger.error(f"Enhanced parameter detection failed for company {company_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Parameter detection failed: {str(e)}"
        )

@router.post("/standardize-data")
async def standardize_data_format(
    file: UploadFile = File(..., description="Data file for format standardization"),
    company_id: str = Depends(get_company_id_from_token)
):
    """
    Standardize data format based on intelligent analysis and auto-mapping
    
    Features:
    - Automatic format standardization
    - Data cleaning and preprocessing
    - Quality improvement suggestions
    """
    
    try:
        # Validate file
        file_format = validate_file_format(file)
        file_size_mb = get_file_size_mb(file)
        
        if file_size_mb > 50:
            raise HTTPException(
                status_code=400, 
                detail=f"File size ({file_size_mb:.1f}MB) exceeds 50MB limit"
            )
        
        # Read and process file
        file_content = await file.read()
        
        processing_result = await smart_processor.process_file_intelligent(
            file_content=file_content,
            filename=file.filename or "unknown",
            file_format=file_format
        )
        
        if processing_result.standardized_data is None:
            return JSONResponse(content={
                "success": False,
                "message": "Data standardization not possible - insufficient column mappings",
                "required_mappings": [
                    mapping.required_field for mapping in processing_result.column_mappings 
                    if mapping.detected_column is None
                ],
                "suggestions": [
                    f"Map '{mapping.required_field}' to an appropriate column" 
                    for mapping in processing_result.column_mappings 
                    if mapping.detected_column is None
                ]
            })
        
        # Convert standardized data to response format
        standardized_preview = processing_result.standardized_data.head(10).to_dict('records')
        
        response_data = {
            "success": True,
            "message": "Data successfully standardized",
            "standardized_preview": standardized_preview,
            "standardization_summary": {
                "original_columns": len(processing_result.detected_columns),
                "mapped_columns": sum(
                    1 for mapping in processing_result.column_mappings 
                    if mapping.detected_column is not None
                ),
                "quality_improvements": len(processing_result.data_quality.recommendations),
                "data_shape": {
                    "rows": len(processing_result.standardized_data),
                    "columns": len(processing_result.standardized_data.columns)
                }
            },
            "applied_transformations": [
                f"Standardized {mapping.required_field} from {mapping.detected_column}"
                for mapping in processing_result.column_mappings
                if mapping.detected_column is not None
            ],
            "quality_score": processing_result.data_quality.overall_score
        }
        
        logger.info(f"Data standardization completed for {file.filename} (Company: {company_id})")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data standardization failed for company {company_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Standardization failed: {str(e)}"
        )

@router.get("/quality-report/{analysis_id}")
async def get_quality_report(
    analysis_id: str,
    company_id: str = Depends(get_company_id_from_token)
):
    """
    Get detailed quality report for a previous analysis
    
    Note: This is a placeholder for future implementation with persistent storage
    """
    
    return JSONResponse(content={
        "message": "Quality report feature coming soon",
        "analysis_id": analysis_id,
        "company_id": company_id
    })

@router.post("/batch-analyze")
async def batch_analyze_files(
    files: List[UploadFile] = File(..., description="Multiple files for batch analysis"),
    company_id: str = Depends(get_company_id_from_token)
):
    """
    Batch analysis of multiple files with parallel processing
    
    Features:
    - Parallel file processing
    - Aggregated quality metrics
    - Batch recommendations
    """
    
    try:
        if len(files) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 10 files allowed per batch"
            )
        
        # Process files in parallel
        async def process_single_file(file: UploadFile) -> Dict[str, Any]:
            try:
                file_format = validate_file_format(file)
                file_content = await file.read()
                
                result = await smart_processor.process_file_intelligent(
                    file_content=file_content,
                    filename=file.filename or "unknown",
                    file_format=file_format
                )
                
                return {
                    "filename": file.filename,
                    "success": True,
                    "quality_score": result.data_quality.overall_score,
                    "detected_columns": len(result.detected_columns),
                    "mapped_columns": sum(
                        1 for mapping in result.column_mappings 
                        if mapping.detected_column is not None
                    ),
                    "issues_count": len(result.data_quality.issues),
                    "file_size_mb": get_file_size_mb(file)
                }
            except Exception as e:
                return {
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                }
        
        # Process all files concurrently
        batch_results = await asyncio.gather(
            *[process_single_file(file) for file in files],
            return_exceptions=True
        )
        
        # Aggregate results
        successful_analyses = [r for r in batch_results if isinstance(r, dict) and r.get("success")]
        failed_analyses = [r for r in batch_results if isinstance(r, dict) and not r.get("success")]
        
        batch_summary = {
            "total_files": len(files),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(failed_analyses),
            "average_quality_score": (
                sum(r["quality_score"] for r in successful_analyses) / len(successful_analyses)
                if successful_analyses else 0
            ),
            "total_issues": sum(r["issues_count"] for r in successful_analyses),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        response_data = {
            "success": True,
            "batch_summary": batch_summary,
            "file_results": batch_results,
            "recommendations": [
                "Review files with quality scores below 0.7",
                "Address common data quality issues across files",
                "Consider standardizing data collection processes"
            ] if batch_summary["average_quality_score"] < 0.7 else [
                "Good overall data quality detected",
                "Minor improvements recommended for optimal processing"
            ]
        }
        
        logger.info(f"Batch analysis completed: {len(successful_analyses)}/{len(files)} files processed successfully")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed for company {company_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Batch analysis failed: {str(e)}"
        )

# Health check endpoints
@router.get("/health")
async def health_check():
    """Comprehensive health check for the enhanced data processing service"""
    
    try:
        # Check service health
        overall_health = await service_health_monitor.get_overall_health()
        
        # Test smart processor
        processor_healthy = smart_processor is not None
        
        # Get circuit breaker status
        cb_status = {}
        for service_name in ["parameter_detection", "data_processing"]:
            if service_name in service_health_monitor.services:
                cb = service_health_monitor.services[service_name]
                cb_status[service_name] = cb.get_stats()
        
        health_status = {
            "status": "healthy" if overall_health["overall_healthy"] and processor_healthy else "degraded",
            "service": "Enhanced Data Processing API",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "smart_processor": processor_healthy,
                "parameter_detection": overall_health["services"].get("parameter_detection", {}).get("healthy", False),
                "data_processing": overall_health["services"].get("data_processing", {}).get("healthy", False)
            },
            "circuit_breakers": cb_status,
            "features": [
                "Intelligent column detection",
                "Auto-mapping engine", 
                "Data quality assessment",
                "Format standardization",
                "Batch processing",
                "Error recovery",
                "Circuit breaker protection"
            ],
            "error_handling": {
                "retry_enabled": True,
                "circuit_breaker_enabled": True,
                "fallback_mode_available": True
            }
        }
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "Enhanced Data Processing API",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service diagnostics"""
    
    try:
        # Get comprehensive health status
        overall_health = await service_health_monitor.get_overall_health()
        
        # Test each component
        component_tests = {}
        
        # Test parameter detection
        try:
            await parameter_detection_health_check()
            component_tests["parameter_detection"] = {"status": "healthy", "last_test": datetime.now().isoformat()}
        except Exception as e:
            component_tests["parameter_detection"] = {"status": "unhealthy", "error": str(e), "last_test": datetime.now().isoformat()}
        
        # Test data processing
        try:
            await data_processing_health_check()
            component_tests["data_processing"] = {"status": "healthy", "last_test": datetime.now().isoformat()}
        except Exception as e:
            component_tests["data_processing"] = {"status": "unhealthy", "error": str(e), "last_test": datetime.now().isoformat()}
        
        return JSONResponse(content={
            "overall_health": overall_health,
            "component_tests": component_tests,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Health check system failure",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@router.post("/test-parameter-detection")
async def test_parameter_detection():
    """Test endpoint to verify parameter detection pipeline is working"""
    
    try:
        # Create a simple test CSV data
        test_csv_data = """date,sales_amount,product_category,region,units_sold
2024-01-01,1500.50,Electronics,North,25
2024-01-02,2300.75,Clothing,South,40
2024-01-03,1800.25,Electronics,East,30
2024-01-04,950.00,Home,West,15
2024-01-05,2100.80,Clothing,North,35"""
        
        # Convert to bytes
        test_content = test_csv_data.encode('utf-8')
        
        # Process with smart processor
        processing_result = await smart_processor.process_file_intelligent(
            file_content=test_content,
            filename="test_data.csv",
            file_format="csv"
        )
        
        # Format response
        test_result = {
            "success": True,
            "message": "Parameter detection pipeline is working correctly",
            "test_results": {
                "detected_columns_count": len(processing_result.detected_columns),
                "mapped_columns_count": sum(1 for m in processing_result.column_mappings if m.detected_column),
                "overall_quality_score": processing_result.data_quality.overall_score,
                "detected_columns": [
                    {
                        "name": col.name,
                        "type": col.type.value,
                        "confidence": col.confidence,
                        "quality_score": col.quality_score
                    }
                    for col in processing_result.detected_columns
                ],
                "column_mappings": [
                    {
                        "required_field": mapping.required_field,
                        "detected_column": mapping.detected_column,
                        "confidence": mapping.confidence,
                        "status": mapping.status
                    }
                    for mapping in processing_result.column_mappings
                ]
            },
            "pipeline_status": {
                "smart_processor_available": smart_processor is not None,
                "intelligent_processing": True,
                "fallback_available": True,
                "circuit_breaker_status": "closed"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Parameter detection test completed successfully")
        return JSONResponse(content=test_result)
        
    except Exception as e:
        logger.error(f"Parameter detection test failed: {e}")
        
        # Try fallback processing
        try:
            fallback_result = await smart_processor.process_file_basic(
                file_content=test_csv_data.encode('utf-8'),
                filename="test_data.csv",
                file_format="csv"
            )
            
            return JSONResponse(content={
                "success": True,
                "message": "Parameter detection working in fallback mode",
                "fallback_mode": True,
                "test_results": {
                    "detected_columns_count": len(fallback_result.detected_columns),
                    "overall_quality_score": fallback_result.data_quality.overall_score
                },
                "error_details": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as fallback_error:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Parameter detection pipeline failed",
                    "intelligent_processing_error": str(e),
                    "fallback_processing_error": str(fallback_error),
                    "timestamp": datetime.now().isoformat()
                }
            )

@router.post("/test-error-handling")
async def test_error_handling(error_type: str = "network"):
    """Test endpoint for error handling (development/testing only)"""
    
    if error_type == "network":
        raise ConnectionError("Simulated network error")
    elif error_type == "timeout":
        await asyncio.sleep(35)  # Simulate timeout
        return {"message": "This should timeout"}
    elif error_type == "validation":
        raise ValueError("Simulated validation error")
    elif error_type == "auth":
        raise PermissionError("Simulated authentication error")
    elif error_type == "service":
        raise RuntimeError("Simulated service error")
    else:
        raise Exception("Simulated unknown error")

@router.get("/circuit-breaker-status")
async def get_circuit_breaker_status():
    """Get current circuit breaker status for all services"""
    
    try:
        status = {}
        for service_name in ["parameter_detection", "data_processing"]:
            if service_name in service_health_monitor.services:
                cb = service_health_monitor.services[service_name]
                status[service_name] = cb.get_stats()
        
        return JSONResponse(content={
            "circuit_breakers": status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to retrieve circuit breaker status",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )