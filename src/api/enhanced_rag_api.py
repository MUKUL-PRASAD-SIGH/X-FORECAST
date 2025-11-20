"""
Enhanced RAG API with Integrated Reliability Features
Provides API endpoints for the enhanced RAG manager with startup validation,
health monitoring, diagnostics, and comprehensive error handling.
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime
from typing import Dict, Any

# Import enhanced RAG manager
try:
    from src.rag.enhanced_rag_manager import enhanced_rag_manager
except ImportError:
    from ..rag.enhanced_rag_manager import enhanced_rag_manager

logger = logging.getLogger(__name__)

def create_enhanced_rag_routes(app: Flask):
    """Create enhanced RAG API routes"""
    
    @app.route('/api/rag/system/validate', methods=['POST'])
    def validate_system():
        """Run system startup validation"""
        try:
            data = request.get_json() or {}
            force_validation = data.get('force_validation', False)
            
            logger.info("API: Running system startup validation")
            
            # Run startup validation
            validation_result = enhanced_rag_manager.startup_system_validation(force_validation)
            
            return jsonify({
                "success": True,
                "validation_result": {
                    "overall_status": validation_result.overall_status.value,
                    "total_duration": validation_result.total_duration,
                    "system_health_score": validation_result.system_health_score,
                    "ready_for_operation": validation_result.ready_for_operation,
                    "critical_issues": validation_result.critical_issues,
                    "warnings": validation_result.warnings,
                    "recommendations": validation_result.recommendations,
                    "phase_count": len(validation_result.phase_results),
                    "timestamp": validation_result.timestamp.isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error in system validation API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "System validation failed"
            }), 500
    
    @app.route('/api/rag/user/<user_id>/initialize', methods=['POST'])
    def initialize_user_rag(user_id: str):
        """Initialize RAG system for user with enhanced reliability"""
        try:
            data = request.get_json() or {}
            company_name = data.get('company_name', 'Unknown Company')
            force_reinit = data.get('force_reinit', False)
            validate_system = data.get('validate_system', True)
            
            logger.info(f"API: Initializing RAG for user {user_id}")
            
            # Initialize RAG with enhanced features
            result = enhanced_rag_manager.initialize_rag_for_user(
                user_id=user_id,
                company_name=company_name,
                force_reinit=force_reinit,
                validate_system=validate_system
            )
            
            return jsonify({
                "success": result.success,
                "operation": result.operation,
                "user_id": result.user_id,
                "message": result.message,
                "status": result.status.value,
                "duration": result.duration,
                "recovery_attempted": result.recovery_attempted,
                "recovery_successful": result.recovery_successful,
                "recommendations": result.recommendations,
                "error_details": result.error_details,
                "metadata": result.metadata
            })
            
        except Exception as e:
            logger.error(f"Error in user RAG initialization API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": f"RAG initialization failed for user {user_id}"
            }), 500
    
    @app.route('/api/rag/user/<user_id>/status', methods=['GET'])
    def get_user_rag_status(user_id: str):
        """Get enhanced RAG status for user"""
        try:
            logger.info(f"API: Getting enhanced RAG status for user {user_id}")
            
            # Get enhanced status
            status = enhanced_rag_manager.get_enhanced_rag_status(user_id)
            
            return jsonify({
                "success": True,
                "user_id": status.user_id,
                "company_name": status.company_name,
                "system_status": status.system_status.value,
                "health_score": status.health_score,
                "is_initialized": status.is_initialized,
                "startup_validated": status.startup_validated,
                "dependencies_healthy": status.dependencies_healthy,
                "monitoring_active": status.monitoring_active,
                "last_health_check": status.last_health_check.isoformat() if status.last_health_check else None,
                "active_alerts": [
                    {
                        "id": alert.id,
                        "level": alert.level.value,
                        "component": alert.component,
                        "title": alert.title,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in status.active_alerts
                ],
                "recommendations": status.recommendations,
                "error_message": status.error_message,
                "performance_metrics": status.performance_metrics
            })
            
        except Exception as e:
            logger.error(f"Error in user RAG status API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": f"Failed to get RAG status for user {user_id}"
            }), 500
    
    @app.route('/api/rag/system/diagnostics', methods=['GET'])
    def run_system_diagnostics():
        """Run comprehensive system diagnostics"""
        try:
            user_id = request.args.get('user_id')
            
            logger.info("API: Running comprehensive system diagnostics")
            
            # Run diagnostics
            diagnostics = enhanced_rag_manager.run_comprehensive_diagnostics(user_id)
            
            return jsonify({
                "success": True,
                "diagnostics": diagnostics
            })
            
        except Exception as e:
            logger.error(f"Error in system diagnostics API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "System diagnostics failed"
            }), 500
    
    @app.route('/api/rag/system/recovery', methods=['POST'])
    def handle_system_recovery():
        """Handle system recovery operations"""
        try:
            data = request.get_json() or {}
            recovery_type = data.get('recovery_type', 'auto')
            target_user = data.get('target_user')
            
            logger.info(f"API: Handling system recovery - Type: {recovery_type}")
            
            # Handle recovery
            recovery_result = enhanced_rag_manager.handle_system_recovery(
                recovery_type=recovery_type,
                target_user=target_user
            )
            
            return jsonify({
                "success": recovery_result["success"],
                "recovery_result": recovery_result
            })
            
        except Exception as e:
            logger.error(f"Error in system recovery API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "System recovery failed"
            }), 500
    
    @app.route('/api/rag/system/health', methods=['GET'])
    def get_system_health():
        """Get system health overview"""
        try:
            logger.info("API: Getting system health overview")
            
            # Get health monitoring status
            from src.rag.health_monitor import health_monitor
            health_status = health_monitor.get_health_status_report()
            
            # Get startup validation status
            from src.rag.startup_validator import startup_validator
            validation_report = startup_validator.get_validation_report()
            
            # Get dependency status
            from src.rag.dependency_validator import dependency_validator
            dependency_status = dependency_validator.get_system_status_report()
            
            return jsonify({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "health_monitoring": {
                    "active": health_status["monitoring_active"],
                    "current_score": health_status["current_health"]["overall_score"],
                    "active_alerts": health_status["alerts"]["active_count"],
                    "critical_alerts": health_status["alerts"]["critical_count"]
                },
                "startup_validation": {
                    "last_status": validation_report.get("overall_status", "not_run"),
                    "health_score": validation_report.get("system_health_score", 0),
                    "ready_for_operation": validation_report.get("ready_for_operation", False)
                },
                "dependencies": {
                    "overall_status": dependency_status["overall_status"],
                    "available": dependency_status["summary"]["available"],
                    "critical_missing": dependency_status["summary"]["critical_missing"],
                    "optional_missing": dependency_status["summary"]["optional_missing"]
                }
            })
            
        except Exception as e:
            logger.error(f"Error in system health API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "Failed to get system health"
            }), 500
    
    @app.route('/api/rag/system/info', methods=['GET'])
    def get_system_info():
        """Get system information and capabilities"""
        try:
            logger.info("API: Getting system information")
            
            return jsonify({
                "success": True,
                "system_info": {
                    "enhanced_rag_manager": {
                        "available": True,
                        "monitoring_enabled": enhanced_rag_manager.enable_monitoring,
                        "auto_recovery_enabled": enhanced_rag_manager.auto_recovery,
                        "system_validated": enhanced_rag_manager.system_validated,
                        "last_validation": enhanced_rag_manager.last_validation_time.isoformat() if enhanced_rag_manager.last_validation_time else None
                    },
                    "reliability_features": {
                        "startup_validation": True,
                        "dependency_validation": True,
                        "health_monitoring": True,
                        "system_diagnostics": True,
                        "automatic_recovery": True,
                        "schema_migration": True
                    },
                    "operation_history": len(enhanced_rag_manager.operation_history),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error in system info API: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "Failed to get system information"
            }), 500

# Example usage function
def test_enhanced_rag_api():
    """Test function to demonstrate enhanced RAG API usage"""
    try:
        print("🧪 Testing Enhanced RAG API Integration")
        
        # Test 1: System validation
        print("\n1. Testing system startup validation...")
        validation_result = enhanced_rag_manager.startup_system_validation()
        print(f"   Validation Status: {validation_result.overall_status.value}")
        print(f"   Health Score: {validation_result.system_health_score}/100")
        print(f"   Ready for Operation: {validation_result.ready_for_operation}")
        
        # Test 2: System diagnostics
        print("\n2. Testing comprehensive diagnostics...")
        diagnostics = enhanced_rag_manager.run_comprehensive_diagnostics()
        print(f"   Overall Assessment: {diagnostics['overall_assessment']['status']}")
        print(f"   System Score: {diagnostics['overall_assessment']['overall_score']}/100")
        
        # Test 3: Enhanced status (using a test user)
        test_user_id = "test_user_123"
        print(f"\n3. Testing enhanced status for user {test_user_id}...")
        status = enhanced_rag_manager.get_enhanced_rag_status(test_user_id)
        print(f"   System Status: {status.system_status.value}")
        print(f"   Health Score: {status.health_score}/100")
        print(f"   Dependencies Healthy: {status.dependencies_healthy}")
        print(f"   Monitoring Active: {status.monitoring_active}")
        
        print("\n✅ Enhanced RAG API integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced RAG API integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    test_enhanced_rag_api()