"""
Performance Monitoring Startup Script
Initializes performance monitoring and optimization systems
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def initialize_performance_systems(
    enable_monitoring: bool = True,
    enable_pdf_optimization: bool = True,
    enable_rag_caching: bool = True,
    monitoring_interval_seconds: int = 60
) -> bool:
    """
    Initialize all performance optimization and monitoring systems
    
    Args:
        enable_monitoring: Enable performance monitoring
        enable_pdf_optimization: Enable PDF processing optimization
        enable_rag_caching: Enable RAG query caching
        monitoring_interval_seconds: Monitoring check interval
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing performance optimization systems...")
        
        # Initialize PDF processing optimizer
        if enable_pdf_optimization:
            try:
                from src.utils.performance_optimizer import pdf_optimizer
                logger.info("PDF processing optimizer initialized")
            except ImportError as e:
                logger.warning(f"Could not initialize PDF optimizer: {e}")
                enable_pdf_optimization = False
        
        # Initialize RAG query cache
        if enable_rag_caching:
            try:
                from src.utils.performance_optimizer import rag_cache
                logger.info("RAG query cache initialized")
            except ImportError as e:
                logger.warning(f"Could not initialize RAG cache: {e}")
                enable_rag_caching = False
        
        # Initialize performance monitoring
        if enable_monitoring:
            try:
                from src.utils.performance_monitor import performance_monitor
                
                # Configure monitoring interval
                for threshold in performance_monitor.thresholds.values():
                    threshold.check_interval_seconds = monitoring_interval_seconds
                
                # Start monitoring
                performance_monitor.start_monitoring()
                logger.info(f"Performance monitoring started (interval: {monitoring_interval_seconds}s)")
                
            except ImportError as e:
                logger.warning(f"Could not initialize performance monitoring: {e}")
                enable_monitoring = False
            except Exception as e:
                logger.error(f"Error starting performance monitoring: {e}")
                enable_monitoring = False
        
        # Log initialization status
        enabled_systems = []
        if enable_pdf_optimization:
            enabled_systems.append("PDF Optimization")
        if enable_rag_caching:
            enabled_systems.append("RAG Caching")
        if enable_monitoring:
            enabled_systems.append("Performance Monitoring")
        
        if enabled_systems:
            logger.info(f"Performance systems initialized: {', '.join(enabled_systems)}")
            return True
        else:
            logger.warning("No performance systems could be initialized")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing performance systems: {e}")
        return False

def shutdown_performance_systems():
    """Shutdown performance monitoring and optimization systems"""
    try:
        logger.info("Shutting down performance systems...")
        
        # Stop performance monitoring
        try:
            from src.utils.performance_monitor import performance_monitor
            performance_monitor.stop_monitoring()
            logger.info("Performance monitoring stopped")
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
        
        # Clean up caches
        try:
            from src.utils.performance_optimizer import pdf_optimizer, rag_cache
            
            # Clean PDF processing cache
            if hasattr(pdf_optimizer, 'processing_cache'):
                pdf_optimizer.processing_cache.clear()
            
            # Clean RAG cache
            if hasattr(rag_cache, 'memory_cache'):
                rag_cache.memory_cache.clear()
            
            logger.info("Performance caches cleaned up")
            
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error cleaning up caches: {e}")
        
        logger.info("Performance systems shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down performance systems: {e}")

def get_performance_status() -> dict:
    """Get status of all performance systems"""
    try:
        status = {
            "timestamp": "datetime.now().isoformat()",
            "pdf_optimization": False,
            "rag_caching": False,
            "performance_monitoring": False,
            "details": {}
        }
        
        # Check PDF optimizer
        try:
            from src.utils.performance_optimizer import pdf_optimizer
            status["pdf_optimization"] = True
            status["details"]["pdf_cache_size"] = len(getattr(pdf_optimizer, 'processing_cache', {}))
            status["details"]["pdf_metrics_count"] = len(getattr(pdf_optimizer, 'performance_metrics', []))
        except ImportError:
            status["details"]["pdf_optimization_error"] = "Module not available"
        except Exception as e:
            status["details"]["pdf_optimization_error"] = str(e)
        
        # Check RAG cache
        try:
            from src.utils.performance_optimizer import rag_cache
            status["rag_caching"] = True
            cache_stats = rag_cache.get_cache_statistics()
            status["details"]["rag_cache_stats"] = cache_stats
        except ImportError:
            status["details"]["rag_caching_error"] = "Module not available"
        except Exception as e:
            status["details"]["rag_caching_error"] = str(e)
        
        # Check performance monitoring
        try:
            from src.utils.performance_monitor import performance_monitor
            status["performance_monitoring"] = performance_monitor.monitoring_active
            status["details"]["monitoring_active"] = performance_monitor.monitoring_active
            status["details"]["active_alerts"] = len([a for a in performance_monitor.active_alerts if not a.resolved])
            status["details"]["metrics_history_size"] = len(performance_monitor.metrics_history)
        except ImportError:
            status["details"]["performance_monitoring_error"] = "Module not available"
        except Exception as e:
            status["details"]["performance_monitoring_error"] = str(e)
        
        return status
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": "datetime.now().isoformat()"
        }

def configure_performance_thresholds(
    cpu_warning: Optional[float] = None,
    cpu_critical: Optional[float] = None,
    memory_warning: Optional[float] = None,
    memory_critical: Optional[float] = None,
    disk_warning: Optional[float] = None,
    disk_critical: Optional[float] = None,
    pdf_processing_warning: Optional[float] = None,
    pdf_processing_critical: Optional[float] = None,
    rag_query_warning: Optional[float] = None,
    rag_query_critical: Optional[float] = None
):
    """Configure performance monitoring thresholds"""
    try:
        from src.utils.performance_monitor import performance_monitor
        
        # Update CPU thresholds
        if cpu_warning is not None:
            performance_monitor.thresholds["cpu_usage"].warning_threshold = cpu_warning
        if cpu_critical is not None:
            performance_monitor.thresholds["cpu_usage"].critical_threshold = cpu_critical
        
        # Update memory thresholds
        if memory_warning is not None:
            performance_monitor.thresholds["memory_usage"].warning_threshold = memory_warning
        if memory_critical is not None:
            performance_monitor.thresholds["memory_usage"].critical_threshold = memory_critical
        
        # Update disk thresholds
        if disk_warning is not None:
            performance_monitor.thresholds["disk_usage"].warning_threshold = disk_warning
        if disk_critical is not None:
            performance_monitor.thresholds["disk_usage"].critical_threshold = disk_critical
        
        # Update PDF processing thresholds
        if pdf_processing_warning is not None:
            performance_monitor.thresholds["pdf_processing_time"].warning_threshold = pdf_processing_warning
        if pdf_processing_critical is not None:
            performance_monitor.thresholds["pdf_processing_time"].critical_threshold = pdf_processing_critical
        
        # Update RAG query thresholds
        if rag_query_warning is not None:
            performance_monitor.thresholds["rag_query_time"].warning_threshold = rag_query_warning
        if rag_query_critical is not None:
            performance_monitor.thresholds["rag_query_time"].critical_threshold = rag_query_critical
        
        logger.info("Performance thresholds updated")
        return True
        
    except ImportError:
        logger.error("Performance monitor not available")
        return False
    except Exception as e:
        logger.error(f"Error configuring thresholds: {e}")
        return False

# Environment-based configuration
def load_performance_config_from_env():
    """Load performance configuration from environment variables"""
    config = {
        "enable_monitoring": os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true",
        "enable_pdf_optimization": os.getenv("ENABLE_PDF_OPTIMIZATION", "true").lower() == "true",
        "enable_rag_caching": os.getenv("ENABLE_RAG_CACHING", "true").lower() == "true",
        "monitoring_interval_seconds": int(os.getenv("PERFORMANCE_MONITORING_INTERVAL", "60")),
        
        # Threshold configuration
        "cpu_warning": float(os.getenv("CPU_WARNING_THRESHOLD", "80")),
        "cpu_critical": float(os.getenv("CPU_CRITICAL_THRESHOLD", "95")),
        "memory_warning": float(os.getenv("MEMORY_WARNING_THRESHOLD", "85")),
        "memory_critical": float(os.getenv("MEMORY_CRITICAL_THRESHOLD", "95")),
        "disk_warning": float(os.getenv("DISK_WARNING_THRESHOLD", "90")),
        "disk_critical": float(os.getenv("DISK_CRITICAL_THRESHOLD", "98")),
    }
    
    return config

def initialize_from_environment():
    """Initialize performance systems using environment configuration"""
    try:
        config = load_performance_config_from_env()
        
        # Initialize systems
        success = initialize_performance_systems(
            enable_monitoring=config["enable_monitoring"],
            enable_pdf_optimization=config["enable_pdf_optimization"],
            enable_rag_caching=config["enable_rag_caching"],
            monitoring_interval_seconds=config["monitoring_interval_seconds"]
        )
        
        if success and config["enable_monitoring"]:
            # Configure thresholds
            configure_performance_thresholds(
                cpu_warning=config["cpu_warning"],
                cpu_critical=config["cpu_critical"],
                memory_warning=config["memory_warning"],
                memory_critical=config["memory_critical"],
                disk_warning=config["disk_warning"],
                disk_critical=config["disk_critical"]
            )
        
        return success
        
    except Exception as e:
        logger.error(f"Error initializing from environment: {e}")
        return False

if __name__ == "__main__":
    # Initialize performance systems when run directly
    initialize_from_environment()