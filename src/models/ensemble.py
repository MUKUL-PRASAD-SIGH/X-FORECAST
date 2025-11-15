import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.model_selection import TimeSeriesSplit
from .classical.arima import ARIMAForecaster
from .classical.ets import ETSForecaster
from .intermittent.croston import CrostonForecaster
from .pattern_detection import PatternDetector, PatternCharacteristics

try:
    from .ml_deep.xgboost_model import XGBoostForecaster
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from .ml_deep.lstm_model import LSTMForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightUpdateMethod(Enum):
    """Enumeration of available weight update algorithms"""
    INVERSE_ERROR = "inverse_error"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    RANK_BASED = "rank_based"

@dataclass
class ConfidenceInterval:
    """Container for confidence interval forecasts"""
    p10: pd.Series  # 10th percentile (lower bound)
    p50: pd.Series  # 50th percentile (median/point forecast)
    p90: pd.Series  # 90th percentile (upper bound)
    forecast_date: datetime
    model_name: str = "ensemble"

@dataclass
class WeightUpdateConfig:
    """Configuration for weight update algorithms"""
    method: WeightUpdateMethod = WeightUpdateMethod.INVERSE_ERROR
    min_weight: float = 0.05
    max_weight: float = 0.7
    smoothing_factor: float = 0.1
    max_weight_change: float = 0.2
    performance_window: int = 30
    degradation_threshold: float = 0.15
    enable_alerts: bool = True

@dataclass
class AdaptiveConfig:
    """
    Configuration for all tunable parameters in adaptive ensemble forecasting
    
    This dataclass centralizes all configuration parameters for the adaptive ensemble
    forecasting system, including learning windows, update frequencies, thresholds,
    and monitoring settings.
    """
    
    # Learning and adaptation settings
    adaptive_learning_enabled: bool = True
    learning_window_days: int = 30
    weight_update_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    weight_update_method: str = 'inverse_error'  # 'inverse_error', 'exponential_smoothing', 'rank_based'
    
    # Weight management thresholds
    min_model_weight: float = 0.05
    max_model_weight: float = 0.7
    max_weight_change_per_update: float = 0.2
    weight_smoothing_factor: float = 0.1
    weight_normalization_enabled: bool = True
    
    # Performance monitoring configuration
    performance_tracking_enabled: bool = True
    performance_history_limit: int = 1000
    performance_window_days: int = 30
    degradation_alert_threshold: float = 0.15
    significant_change_threshold: float = 0.2
    performance_metrics: List[str] = None  # ['mae', 'mape', 'rmse', 'r_squared']
    
    # Pattern detection settings
    pattern_detection_enabled: bool = True
    pattern_change_threshold: float = 0.3
    pattern_confidence_threshold: float = 0.6
    pattern_redetection_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    
    # Backtesting configuration
    backtesting_enabled: bool = True
    backtesting_validation_split: float = 0.3
    backtesting_n_splits: int = 5
    backtesting_min_data_points: int = 50
    backtesting_metrics: List[str] = None  # ['mae', 'mape', 'rmse']
    
    # Confidence intervals and uncertainty quantification
    bootstrap_samples: int = 1000
    confidence_levels: List[float] = None  # [0.1, 0.5, 0.9] for P10, P50, P90
    uncertainty_estimation_enabled: bool = True
    
    # Alerting and monitoring thresholds
    enable_performance_alerts: bool = True
    enable_weight_change_alerts: bool = True
    enable_pattern_change_alerts: bool = True
    enable_model_failure_alerts: bool = True
    alert_cooldown_hours: int = 24
    critical_performance_threshold: float = 0.5  # Threshold for critical alerts
    
    # Data quality and validation
    data_quality_checks_enabled: bool = True
    min_data_points_for_update: int = 10
    outlier_detection_enabled: bool = True
    outlier_threshold_std: float = 3.0
    
    # Computational and resource limits
    max_models_in_ensemble: int = 10
    parallel_processing_enabled: bool = True
    max_concurrent_models: int = 4
    memory_limit_mb: Optional[int] = None
    
    # Logging and debugging
    detailed_logging_enabled: bool = False
    log_weight_changes: bool = True
    log_performance_updates: bool = False
    export_performance_data: bool = False
    
    def __post_init__(self):
        """Validate and set default values for configuration parameters"""
        # Set default confidence levels
        if self.confidence_levels is None:
            self.confidence_levels = [0.1, 0.5, 0.9]  # P10, P50, P90
        
        # Set default performance metrics
        if self.performance_metrics is None:
            self.performance_metrics = ['mae', 'mape', 'rmse', 'r_squared']
        
        # Set default backtesting metrics
        if self.backtesting_metrics is None:
            self.backtesting_metrics = ['mae', 'mape', 'rmse']
        
        # Validate configuration parameters
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters and raise errors for invalid values"""
        # Validate weight parameters
        if not 0 < self.min_model_weight <= 1.0:
            raise ValueError(f"min_model_weight must be between 0 and 1, got {self.min_model_weight}")
        
        if not 0 < self.max_model_weight <= 1.0:
            raise ValueError(f"max_model_weight must be between 0 and 1, got {self.max_model_weight}")
        
        if self.min_model_weight >= self.max_model_weight:
            raise ValueError("min_model_weight must be less than max_model_weight")
        
        if not 0 < self.max_weight_change_per_update <= 1.0:
            raise ValueError(f"max_weight_change_per_update must be between 0 and 1, got {self.max_weight_change_per_update}")
        
        if not 0 <= self.weight_smoothing_factor <= 1.0:
            raise ValueError(f"weight_smoothing_factor must be between 0 and 1, got {self.weight_smoothing_factor}")
        
        # Validate time windows
        if self.learning_window_days <= 0:
            raise ValueError(f"learning_window_days must be positive, got {self.learning_window_days}")
        
        if self.performance_window_days <= 0:
            raise ValueError(f"performance_window_days must be positive, got {self.performance_window_days}")
        
        # Validate frequency parameters
        valid_frequencies = ['daily', 'weekly', 'monthly']
        if self.weight_update_frequency not in valid_frequencies:
            raise ValueError(f"weight_update_frequency must be one of {valid_frequencies}, got {self.weight_update_frequency}")
        
        if self.pattern_redetection_frequency not in valid_frequencies:
            raise ValueError(f"pattern_redetection_frequency must be one of {valid_frequencies}, got {self.pattern_redetection_frequency}")
        
        # Validate weight update method
        valid_methods = ['inverse_error', 'exponential_smoothing', 'rank_based']
        if self.weight_update_method not in valid_methods:
            raise ValueError(f"weight_update_method must be one of {valid_methods}, got {self.weight_update_method}")
        
        # Validate thresholds
        if not 0 < self.degradation_alert_threshold <= 1.0:
            raise ValueError(f"degradation_alert_threshold must be between 0 and 1, got {self.degradation_alert_threshold}")
        
        if not 0 < self.significant_change_threshold <= 1.0:
            raise ValueError(f"significant_change_threshold must be between 0 and 1, got {self.significant_change_threshold}")
        
        if not 0 < self.pattern_change_threshold <= 1.0:
            raise ValueError(f"pattern_change_threshold must be between 0 and 1, got {self.pattern_change_threshold}")
        
        if not 0 < self.pattern_confidence_threshold <= 1.0:
            raise ValueError(f"pattern_confidence_threshold must be between 0 and 1, got {self.pattern_confidence_threshold}")
        
        # Validate backtesting parameters
        if not 0 < self.backtesting_validation_split < 1.0:
            raise ValueError(f"backtesting_validation_split must be between 0 and 1, got {self.backtesting_validation_split}")
        
        if self.backtesting_n_splits <= 0:
            raise ValueError(f"backtesting_n_splits must be positive, got {self.backtesting_n_splits}")
        
        if self.backtesting_min_data_points <= 0:
            raise ValueError(f"backtesting_min_data_points must be positive, got {self.backtesting_min_data_points}")
        
        # Validate confidence levels
        if not all(0 < level < 1 for level in self.confidence_levels):
            raise ValueError("All confidence_levels must be between 0 and 1")
        
        # Validate bootstrap samples
        if self.bootstrap_samples <= 0:
            raise ValueError(f"bootstrap_samples must be positive, got {self.bootstrap_samples}")
        
        # Validate resource limits
        if self.max_models_in_ensemble <= 0:
            raise ValueError(f"max_models_in_ensemble must be positive, got {self.max_models_in_ensemble}")
        
        if self.max_concurrent_models <= 0:
            raise ValueError(f"max_concurrent_models must be positive, got {self.max_concurrent_models}")
        
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be positive or None, got {self.memory_limit_mb}")
        
        # Validate data quality parameters
        if self.min_data_points_for_update <= 0:
            raise ValueError(f"min_data_points_for_update must be positive, got {self.min_data_points_for_update}")
        
        if self.outlier_threshold_std <= 0:
            raise ValueError(f"outlier_threshold_std must be positive, got {self.outlier_threshold_std}")
        
        # Validate alert parameters
        if self.alert_cooldown_hours < 0:
            raise ValueError(f"alert_cooldown_hours must be non-negative, got {self.alert_cooldown_hours}")
        
        if not 0 < self.critical_performance_threshold <= 1.0:
            raise ValueError(f"critical_performance_threshold must be between 0 and 1, got {self.critical_performance_threshold}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdaptiveConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'AdaptiveConfig':
        """Create a new configuration with updated parameters"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def get_weight_update_config(self) -> Dict[str, Any]:
        """Extract weight update specific configuration"""
        return {
            'method': self.weight_update_method,
            'min_weight': self.min_model_weight,
            'max_weight': self.max_model_weight,
            'smoothing_factor': self.weight_smoothing_factor,
            'max_weight_change': self.max_weight_change_per_update,
            'performance_window': self.performance_window_days,
            'degradation_threshold': self.degradation_alert_threshold,
            'enable_alerts': self.enable_weight_change_alerts
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Extract performance monitoring specific configuration"""
        return {
            'tracking_enabled': self.performance_tracking_enabled,
            'history_limit': self.performance_history_limit,
            'window_days': self.performance_window_days,
            'metrics': self.performance_metrics,
            'degradation_threshold': self.degradation_alert_threshold,
            'enable_alerts': self.enable_performance_alerts
        }
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Extract backtesting specific configuration"""
        return {
            'enabled': self.backtesting_enabled,
            'validation_split': self.backtesting_validation_split,
            'n_splits': self.backtesting_n_splits,
            'min_data_points': self.backtesting_min_data_points,
            'metrics': self.backtesting_metrics
        }

@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    r_squared: float  # R-squared
    model_name: str
    evaluation_date: datetime
    data_points: int

@dataclass
class PerformanceRecord:
    """Individual performance record for tracking model performance over time"""
    model_name: str
    timestamp: datetime
    predicted_value: float
    actual_value: float
    error_metrics: PerformanceMetrics
    data_pattern: Optional[str] = None
    forecast_horizon: int = 1
    weight_at_time: float = 0.0

@dataclass
class WeightChangeRecord:
    """Record of weight changes for audit trail"""
    timestamp: datetime
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    trigger_reason: str  # 'scheduled', 'performance_alert', 'pattern_change', 'manual'
    performance_metrics: Dict[str, PerformanceMetrics]
    pattern_detected: Optional[str] = None
    weight_update_method: Optional[str] = None

@dataclass
class PerformanceAlert:
    """Alert for significant performance changes or issues"""
    alert_id: str
    timestamp: datetime
    alert_type: str  # 'performance_degradation', 'weight_change', 'pattern_change', 'model_failure'
    severity: str  # 'low', 'medium', 'high', 'critical'
    model_name: Optional[str]
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None

@dataclass
class BacktestResult:
    """Container for backtesting results"""
    model_performance: Dict[str, PerformanceMetrics]
    initial_weights: Dict[str, float]
    validation_periods: int
    total_data_points: int
    backtest_date: datetime
    detected_pattern: Optional[PatternCharacteristics] = None

class PerformanceTracker:
    """
    Comprehensive performance tracking system for detailed monitoring with time windows
    and degradation detection
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.performance_records: List[PerformanceRecord] = []
        self.model_performance_history: Dict[str, List[PerformanceRecord]] = {}
        self.alerts: List[PerformanceAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Performance aggregation cache
        self._performance_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_validity_minutes = 5
        
        logger.info("PerformanceTracker initialized")
    
    def update_performance(self, model_name: str, predicted: Union[float, pd.Series], 
                         actual: Union[float, pd.Series], forecast_date: datetime,
                         weight_at_time: float = 0.0, data_pattern: Optional[str] = None,
                         forecast_horizon: int = 1):
        """
        Update performance tracking with new prediction vs actual data
        
        Args:
            model_name: Name of the model
            predicted: Predicted value(s)
            actual: Actual value(s)
            forecast_date: Date of the forecast
            weight_at_time: Model weight when forecast was made
            data_pattern: Detected data pattern at time of forecast
            forecast_horizon: Number of periods forecasted
        """
        try:
            # Convert to pandas Series if needed
            if not isinstance(predicted, pd.Series):
                predicted = pd.Series([predicted] if np.isscalar(predicted) else predicted)
            if not isinstance(actual, pd.Series):
                actual = pd.Series([actual] if np.isscalar(actual) else actual)
            
            # Calculate error metrics
            error_metrics = self._calculate_error_metrics(actual, predicted, model_name)
            
            # Create performance record
            record = PerformanceRecord(
                model_name=model_name,
                timestamp=forecast_date,
                predicted_value=float(predicted.iloc[0]) if len(predicted) > 0 else 0.0,
                actual_value=float(actual.iloc[0]) if len(actual) > 0 else 0.0,
                error_metrics=error_metrics,
                data_pattern=data_pattern,
                forecast_horizon=forecast_horizon,
                weight_at_time=weight_at_time
            )
            
            # Add to records
            self.performance_records.append(record)
            
            # Add to model-specific history
            if model_name not in self.model_performance_history:
                self.model_performance_history[model_name] = []
            self.model_performance_history[model_name].append(record)
            
            # Maintain history limits
            self._maintain_history_limits()
            
            # Clear performance cache
            self._clear_performance_cache()
            
            # Check for performance degradation
            if self.config.enable_performance_alerts:
                self._check_performance_degradation(model_name, record)
            
            logger.debug(f"Performance updated for {model_name}: MAE={error_metrics.mae:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update performance for {model_name}: {e}")
    
    def get_recent_performance(self, model_name: str, days: int = None) -> Dict[str, float]:
        """
        Get recent performance metrics for a specific model
        
        Args:
            model_name: Name of the model
            days: Number of days to look back (uses config default if None)
            
        Returns:
            Dictionary of aggregated performance metrics
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get recent records for the model
            recent_records = [
                record for record in self.model_performance_history.get(model_name, [])
                if record.timestamp >= cutoff_date
            ]
            
            if not recent_records:
                return {
                    'mae': float('inf'),
                    'mape': float('inf'),
                    'rmse': float('inf'),
                    'r_squared': 0.0,
                    'record_count': 0,
                    'avg_weight': 0.0
                }
            
            # Aggregate metrics
            mae_values = [r.error_metrics.mae for r in recent_records if np.isfinite(r.error_metrics.mae)]
            mape_values = [r.error_metrics.mape for r in recent_records if np.isfinite(r.error_metrics.mape)]
            rmse_values = [r.error_metrics.rmse for r in recent_records if np.isfinite(r.error_metrics.rmse)]
            r2_values = [r.error_metrics.r_squared for r in recent_records if np.isfinite(r.error_metrics.r_squared)]
            weights = [r.weight_at_time for r in recent_records]
            
            return {
                'mae': np.mean(mae_values) if mae_values else float('inf'),
                'mape': np.mean(mape_values) if mape_values else float('inf'),
                'rmse': np.mean(rmse_values) if rmse_values else float('inf'),
                'r_squared': np.mean(r2_values) if r2_values else 0.0,
                'record_count': len(recent_records),
                'avg_weight': np.mean(weights) if weights else 0.0,
                'latest_timestamp': max(r.timestamp for r in recent_records),
                'data_patterns': list(set(r.data_pattern for r in recent_records if r.data_pattern))
            }
            
        except Exception as e:
            logger.error(f"Failed to get recent performance for {model_name}: {e}")
            return {'mae': float('inf'), 'mape': float('inf'), 'rmse': float('inf'), 
                   'r_squared': 0.0, 'record_count': 0, 'avg_weight': 0.0}
    
    def get_performance_history(self, model_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get performance history as a DataFrame for analysis and visualization
        
        Args:
            model_name: Name of the model
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with performance history
        """
        try:
            records = self.model_performance_history.get(model_name, [])
            
            if limit:
                records = records[-limit:]
            
            if not records:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    'timestamp': record.timestamp,
                    'predicted_value': record.predicted_value,
                    'actual_value': record.actual_value,
                    'mae': record.error_metrics.mae,
                    'mape': record.error_metrics.mape,
                    'rmse': record.error_metrics.rmse,
                    'r_squared': record.error_metrics.r_squared,
                    'weight_at_time': record.weight_at_time,
                    'data_pattern': record.data_pattern,
                    'forecast_horizon': record.forecast_horizon
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Failed to get performance history for {model_name}: {e}")
            return pd.DataFrame()
    
    def detect_performance_degradation(self, model_name: str, threshold: Optional[float] = None) -> bool:
        """
        Detect if a model's performance has degraded significantly
        
        Args:
            model_name: Name of the model to check
            threshold: Degradation threshold (uses config default if None)
            
        Returns:
            True if significant degradation detected
        """
        try:
            if threshold is None:
                threshold = self.config.degradation_alert_threshold
            
            # Get recent and historical performance
            recent_perf = self.get_recent_performance(model_name, days=7)  # Last week
            historical_perf = self.get_recent_performance(model_name, days=30)  # Last month
            
            if recent_perf['record_count'] < 3 or historical_perf['record_count'] < 10:
                return False  # Insufficient data
            
            # Compare recent vs historical MAPE
            if (np.isfinite(recent_perf['mape']) and np.isfinite(historical_perf['mape']) and
                historical_perf['mape'] > 0):
                
                degradation_ratio = (recent_perf['mape'] - historical_perf['mape']) / historical_perf['mape']
                
                if degradation_ratio > threshold:
                    logger.warning(f"Performance degradation detected for {model_name}: "
                                 f"{degradation_ratio:.2%} increase in MAPE")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect performance degradation for {model_name}: {e}")
            return False
    
    def get_model_rankings(self, days: int = None, metric: str = 'mape') -> List[Tuple[str, float]]:
        """
        Get model rankings based on recent performance
        
        Args:
            days: Number of days to consider (uses config default if None)
            metric: Metric to rank by ('mae', 'mape', 'rmse', 'r_squared')
            
        Returns:
            List of (model_name, metric_value) tuples sorted by performance
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            rankings = []
            
            for model_name in self.model_performance_history.keys():
                perf = self.get_recent_performance(model_name, days)
                
                if perf['record_count'] > 0:
                    metric_value = perf.get(metric, float('inf'))
                    rankings.append((model_name, metric_value))
            
            # Sort by metric (ascending for error metrics, descending for r_squared)
            reverse_sort = metric == 'r_squared'
            rankings.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Failed to get model rankings: {e}")
            return []
    
    def generate_performance_report(self, days: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            days: Number of days to include in report
            
        Returns:
            Dictionary containing performance report
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            report = {
                'report_timestamp': datetime.now(),
                'analysis_period_days': days,
                'model_performance': {},
                'rankings': {},
                'alerts_summary': {},
                'overall_stats': {}
            }
            
            # Model performance details
            for model_name in self.model_performance_history.keys():
                perf = self.get_recent_performance(model_name, days)
                report['model_performance'][model_name] = perf
            
            # Rankings by different metrics
            for metric in ['mae', 'mape', 'rmse', 'r_squared']:
                report['rankings'][metric] = self.get_model_rankings(days, metric)
            
            # Alert summary
            recent_alerts = [
                alert for alert in self.alerts
                if alert.timestamp >= datetime.now() - timedelta(days=days)
            ]
            
            report['alerts_summary'] = {
                'total_alerts': len(recent_alerts),
                'by_type': {},
                'by_severity': {},
                'unresolved_count': sum(1 for a in recent_alerts if not a.resolved)
            }
            
            # Count alerts by type and severity
            for alert in recent_alerts:
                report['alerts_summary']['by_type'][alert.alert_type] = \
                    report['alerts_summary']['by_type'].get(alert.alert_type, 0) + 1
                report['alerts_summary']['by_severity'][alert.severity] = \
                    report['alerts_summary']['by_severity'].get(alert.severity, 0) + 1
            
            # Overall statistics
            all_models = list(self.model_performance_history.keys())
            if all_models:
                # Calculate ensemble-level statistics
                all_recent_performance = [
                    self.get_recent_performance(model, days) for model in all_models
                ]
                
                valid_performances = [p for p in all_recent_performance if p['record_count'] > 0]
                
                if valid_performances:
                    report['overall_stats'] = {
                        'best_model_mae': min(p['mae'] for p in valid_performances if np.isfinite(p['mae'])),
                        'worst_model_mae': max(p['mae'] for p in valid_performances if np.isfinite(p['mae'])),
                        'avg_mae': np.mean([p['mae'] for p in valid_performances if np.isfinite(p['mae'])]),
                        'best_model_mape': min(p['mape'] for p in valid_performances if np.isfinite(p['mape'])),
                        'worst_model_mape': max(p['mape'] for p in valid_performances if np.isfinite(p['mape'])),
                        'avg_mape': np.mean([p['mape'] for p in valid_performances if np.isfinite(p['mape'])]),
                        'total_predictions': sum(p['record_count'] for p in valid_performances),
                        'active_models': len(valid_performances)
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {
                'report_timestamp': datetime.now(),
                'analysis_period_days': days,
                'error': str(e)
            }
    
    def create_alert(self, alert_type: str, severity: str, model_name: Optional[str],
                    message: str, details: Dict[str, Any] = None) -> PerformanceAlert:
        """
        Create and store a performance alert
        
        Args:
            alert_type: Type of alert ('performance_degradation', 'weight_change', etc.)
            severity: Alert severity ('low', 'medium', 'high', 'critical')
            model_name: Associated model name (if applicable)
            message: Alert message
            details: Additional alert details
            
        Returns:
            Created PerformanceAlert object
        """
        try:
            # Check alert cooldown to prevent spam
            alert_key = f"{alert_type}_{model_name or 'system'}"
            now = datetime.now()
            
            if alert_key in self.last_alert_times:
                time_since_last = now - self.last_alert_times[alert_key]
                cooldown_hours = self.config.alert_cooldown_hours
                
                if time_since_last.total_seconds() < cooldown_hours * 3600:
                    logger.debug(f"Alert {alert_key} suppressed due to cooldown")
                    return None
            
            # Create alert
            alert = PerformanceAlert(
                alert_id=f"{alert_type}_{now.strftime('%Y%m%d_%H%M%S')}_{model_name or 'system'}",
                timestamp=now,
                alert_type=alert_type,
                severity=severity,
                model_name=model_name,
                message=message,
                details=details or {}
            )
            
            # Store alert
            self.alerts.append(alert)
            self.last_alert_times[alert_key] = now
            
            # Log alert
            logger.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return None
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was found and resolved
        """
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_timestamp = datetime.now()
                    logger.info(f"Alert {alert_id} resolved")
                    return True
            
            logger.warning(f"Alert {alert_id} not found or already resolved")
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[PerformanceAlert]:
        """
        Get list of active (unresolved) alerts
        
        Args:
            severity_filter: Optional severity filter ('low', 'medium', 'high', 'critical')
            
        Returns:
            List of active alerts
        """
        try:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if severity_filter:
                active_alerts = [alert for alert in active_alerts if alert.severity == severity_filter]
            
            # Sort by timestamp (newest first)
            active_alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return active_alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    def get_weight_evolution_data(self, days: int = None) -> pd.DataFrame:
        """
        Get weight evolution data for visualization
        
        Args:
            days: Number of days to include (uses config default if None)
            
        Returns:
            DataFrame with weight evolution over time
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get weight changes from performance records
            weight_data = []
            
            for record in self.performance_records:
                if record.timestamp >= cutoff_date and record.weight_at_time > 0:
                    weight_data.append({
                        'timestamp': record.timestamp,
                        'model_name': record.model_name,
                        'weight': record.weight_at_time,
                        'mae': record.error_metrics.mae,
                        'mape': record.error_metrics.mape
                    })
            
            if not weight_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(weight_data)
            return df.pivot_table(
                index='timestamp', 
                columns='model_name', 
                values='weight', 
                fill_value=0
            )
            
        except Exception as e:
            logger.error(f"Failed to get weight evolution data: {e}")
            return pd.DataFrame()
    
    def export_performance_data(self, filepath: str, days: int = None) -> bool:
        """
        Export performance data to CSV file
        
        Args:
            filepath: Path to save the CSV file
            days: Number of days to include (uses config default if None)
            
        Returns:
            True if export successful
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Prepare export data
            export_data = []
            
            for record in self.performance_records:
                if record.timestamp >= cutoff_date:
                    export_data.append({
                        'timestamp': record.timestamp,
                        'model_name': record.model_name,
                        'predicted_value': record.predicted_value,
                        'actual_value': record.actual_value,
                        'mae': record.error_metrics.mae,
                        'mape': record.error_metrics.mape,
                        'rmse': record.error_metrics.rmse,
                        'r_squared': record.error_metrics.r_squared,
                        'weight_at_time': record.weight_at_time,
                        'data_pattern': record.data_pattern,
                        'forecast_horizon': record.forecast_horizon
                    })
            
            if not export_data:
                logger.warning("No performance data to export")
                return False
            
            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Performance data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False
    
    def _calculate_error_metrics(self, actual: pd.Series, predicted: pd.Series, 
                               model_name: str) -> PerformanceMetrics:
        """Calculate comprehensive error metrics for model evaluation"""
        try:
            # Ensure both series have the same length and no NaN values
            actual = actual.dropna()
            predicted = predicted.dropna()
            
            # Align series by index
            common_index = actual.index.intersection(predicted.index)
            if len(common_index) == 0:
                raise ValueError("No common indices between actual and predicted values")
            
            actual_aligned = actual.loc[common_index]
            predicted_aligned = predicted.loc[common_index]
            
            # Calculate MAE (Mean Absolute Error)
            mae = np.mean(np.abs(actual_aligned - predicted_aligned))
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            non_zero_mask = actual_aligned != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((actual_aligned[non_zero_mask] - predicted_aligned[non_zero_mask]) / 
                                    actual_aligned[non_zero_mask])) * 100
            else:
                mape = float('inf')
            
            # Calculate RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((actual_aligned - predicted_aligned) ** 2))
            
            # Calculate R-squared
            ss_res = np.sum((actual_aligned - predicted_aligned) ** 2)
            ss_tot = np.sum((actual_aligned - np.mean(actual_aligned)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return PerformanceMetrics(
                mae=mae,
                mape=mape,
                rmse=rmse,
                r_squared=r_squared,
                model_name=model_name,
                evaluation_date=datetime.now(),
                data_points=len(actual_aligned)
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            # Return default metrics in case of error
            return PerformanceMetrics(
                mae=float('inf'),
                mape=float('inf'),
                rmse=float('inf'),
                r_squared=0.0,
                model_name=model_name,
                evaluation_date=datetime.now(),
                data_points=0
            )
    
    def _maintain_history_limits(self):
        """Maintain performance history within configured limits"""
        try:
            max_records = self.config.performance_history_limit
            
            # Trim overall performance records
            if len(self.performance_records) > max_records:
                self.performance_records = self.performance_records[-max_records:]
            
            # Trim model-specific history
            for model_name in self.model_performance_history:
                if len(self.model_performance_history[model_name]) > max_records:
                    self.model_performance_history[model_name] = \
                        self.model_performance_history[model_name][-max_records:]
            
            # Trim alerts (keep last 1000)
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to maintain history limits: {e}")
    
    def _clear_performance_cache(self):
        """Clear performance aggregation cache"""
        self._performance_cache.clear()
        self._cache_timestamp = None
    
    def _check_performance_degradation(self, model_name: str, record: PerformanceRecord):
        """Check for performance degradation and create alerts if needed"""
        try:
            if not self.config.enable_performance_alerts:
                return
            
            # Check if model performance has degraded significantly
            if self.detect_performance_degradation(model_name):
                recent_perf = self.get_recent_performance(model_name, days=7)
                historical_perf = self.get_recent_performance(model_name, days=30)
                
                degradation_pct = ((recent_perf['mape'] - historical_perf['mape']) / 
                                 historical_perf['mape'] * 100) if historical_perf['mape'] > 0 else 0
                
                self.create_alert(
                    alert_type='performance_degradation',
                    severity='high' if degradation_pct > 30 else 'medium',
                    model_name=model_name,
                    message=f"Model {model_name} performance degraded by {degradation_pct:.1f}%",
                    details={
                        'recent_mape': recent_perf['mape'],
                        'historical_mape': historical_perf['mape'],
                        'degradation_percentage': degradation_pct,
                        'recent_record_count': recent_perf['record_count'],
                        'historical_record_count': historical_perf['record_count']
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to check performance degradation for {model_name}: {e}")


class WeightEvolutionTracker:
    """
    Tracks weight changes over time for audit trails and visualization
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.weight_changes: List[WeightChangeRecord] = []
        self.significant_changes: List[WeightChangeRecord] = []
        self.weight_snapshots: Dict[datetime, Dict[str, float]] = {}
        self.volatility_metrics: Dict[str, List[float]] = {}
        
        logger.info("WeightEvolutionTracker initialized")
        
    def record_weight_change(self, old_weights: Dict[str, float], new_weights: Dict[str, float],
                           trigger_reason: str, performance_metrics: Dict[str, PerformanceMetrics],
                           pattern_detected: Optional[str] = None,
                           weight_update_method: Optional[str] = None):
        """
        Record a weight change for audit trail
        
        Args:
            old_weights: Previous model weights
            new_weights: New model weights
            trigger_reason: Reason for weight change
            performance_metrics: Performance metrics that triggered the change
            pattern_detected: Detected data pattern (if applicable)
            weight_update_method: Method used for weight update
        """
        try:
            record = WeightChangeRecord(
                timestamp=datetime.now(),
                old_weights=old_weights.copy(),
                new_weights=new_weights.copy(),
                trigger_reason=trigger_reason,
                performance_metrics=performance_metrics.copy(),
                pattern_detected=pattern_detected,
                weight_update_method=weight_update_method
            )
            
            self.weight_changes.append(record)
            
            # Check if this is a significant change
            if self._is_significant_change(old_weights, new_weights):
                self.significant_changes.append(record)
                logger.info(f"Significant weight change recorded: {trigger_reason}")
            
            # Maintain history limits
            self._maintain_history_limits()
            
        except Exception as e:
            logger.error(f"Failed to record weight change: {e}")
    
    def _is_significant_change(self, old_weights: Dict[str, float], 
                             new_weights: Dict[str, float]) -> bool:
        """Check if weight change is significant based on threshold"""
        try:
            threshold = self.config.significant_change_threshold
            
            for model_name in old_weights:
                old_weight = old_weights.get(model_name, 0.0)
                new_weight = new_weights.get(model_name, 0.0)
                
                if abs(new_weight - old_weight) > threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check significant change: {e}")
            return False
    
    def get_weight_evolution_history(self, days: int = None) -> List[WeightChangeRecord]:
        """Get weight change history for specified period"""
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            return [
                record for record in self.weight_changes
                if record.timestamp >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Failed to get weight evolution history: {e}")
            return []
    
    def get_significant_changes(self, days: int = None) -> List[WeightChangeRecord]:
        """Get significant weight changes for specified period"""
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            return [
                record for record in self.significant_changes
                if record.timestamp >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Failed to get significant changes: {e}")
            return []
    
    def _maintain_history_limits(self):
        """Maintain weight change history within limits"""
        try:
            max_records = 1000  # Keep last 1000 weight changes
            
            if len(self.weight_changes) > max_records:
                self.weight_changes = self.weight_changes[-max_records:]
            
            if len(self.significant_changes) > max_records // 2:
                self.significant_changes = self.significant_changes[-max_records // 2:]
            
            # Maintain weight snapshots (keep last 500)
            if len(self.weight_snapshots) > 500:
                sorted_timestamps = sorted(self.weight_snapshots.keys())
                for ts in sorted_timestamps[:-500]:
                    del self.weight_snapshots[ts]
                
        except Exception as e:
            logger.error(f"Failed to maintain weight history limits: {e}")
    
    def create_weight_snapshot(self, weights: Dict[str, float], timestamp: Optional[datetime] = None):
        """Create a snapshot of current weights for visualization"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            self.weight_snapshots[timestamp] = weights.copy()
            
            # Update volatility metrics
            for model_name, weight in weights.items():
                if model_name not in self.volatility_metrics:
                    self.volatility_metrics[model_name] = []
                self.volatility_metrics[model_name].append(weight)
                
                # Keep only recent volatility data (last 100 points)
                if len(self.volatility_metrics[model_name]) > 100:
                    self.volatility_metrics[model_name] = self.volatility_metrics[model_name][-100:]
            
        except Exception as e:
            logger.error(f"Failed to create weight snapshot: {e}")
    
    def get_weight_evolution_dataframe(self, days: int = None) -> pd.DataFrame:
        """
        Get weight evolution as a DataFrame for visualization
        
        Args:
            days: Number of days to include (uses config default if None)
            
        Returns:
            DataFrame with timestamps as index and model weights as columns
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter snapshots by date
            filtered_snapshots = {
                ts: weights for ts, weights in self.weight_snapshots.items()
                if ts >= cutoff_date
            }
            
            if not filtered_snapshots:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(filtered_snapshots, orient='index')
            df.index.name = 'timestamp'
            df = df.sort_index()
            df = df.fillna(0.0)  # Fill missing weights with 0
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get weight evolution DataFrame: {e}")
            return pd.DataFrame()
    
    def calculate_weight_volatility(self, model_name: str, window_size: int = 30) -> float:
        """
        Calculate weight volatility for a specific model
        
        Args:
            model_name: Name of the model
            window_size: Number of recent observations to consider
            
        Returns:
            Standard deviation of recent weight changes
        """
        try:
            if model_name not in self.volatility_metrics:
                return 0.0
            
            recent_weights = self.volatility_metrics[model_name][-window_size:]
            
            if len(recent_weights) < 2:
                return 0.0
            
            return float(np.std(recent_weights))
            
        except Exception as e:
            logger.error(f"Failed to calculate weight volatility for {model_name}: {e}")
            return 0.0
    
    def get_weight_change_audit_trail(self, model_name: Optional[str] = None, 
                                    days: int = None) -> List[Dict[str, Any]]:
        """
        Get detailed audit trail of weight changes
        
        Args:
            model_name: Optional model name to filter by
            days: Number of days to include
            
        Returns:
            List of audit trail entries
        """
        try:
            if days is None:
                days = self.config.performance_window_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            audit_trail = []
            
            for record in self.weight_changes:
                if record.timestamp < cutoff_date:
                    continue
                
                # Filter by model if specified
                if model_name and model_name not in record.old_weights:
                    continue
                
                entry = {
                    'timestamp': record.timestamp,
                    'trigger_reason': record.trigger_reason,
                    'pattern_detected': record.pattern_detected,
                    'weight_update_method': record.weight_update_method,
                    'weight_changes': {}
                }
                
                # Calculate weight changes for each model
                for model in record.old_weights:
                    if model_name and model != model_name:
                        continue
                    
                    old_weight = record.old_weights.get(model, 0.0)
                    new_weight = record.new_weights.get(model, 0.0)
                    change = new_weight - old_weight
                    change_pct = (change / old_weight * 100) if old_weight > 0 else 0.0
                    
                    entry['weight_changes'][model] = {
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'absolute_change': change,
                        'percentage_change': change_pct,
                        'performance_metrics': record.performance_metrics.get(model, {})
                    }
                
                audit_trail.append(entry)
            
            return audit_trail
            
        except Exception as e:
            logger.error(f"Failed to get weight change audit trail: {e}")
            return []
    
    def export_weight_history(self, filepath: str, format: str = 'csv') -> bool:
        """
        Export weight history to file
        
        Args:
            filepath: Path to save the file
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            True if export successful
        """
        try:
            if format.lower() == 'csv':
                df = self.get_weight_evolution_dataframe()
                if not df.empty:
                    df.to_csv(filepath)
                    logger.info(f"Weight history exported to {filepath}")
                    return True
                else:
                    logger.warning("No weight history data to export")
                    return False
            
            elif format.lower() == 'json':
                audit_trail = self.get_weight_change_audit_trail()
                import json
                with open(filepath, 'w') as f:
                    json.dump(audit_trail, f, indent=2, default=str)
                logger.info(f"Weight audit trail exported to {filepath}")
                return True
            
            elif format.lower() == 'excel':
                df = self.get_weight_evolution_dataframe()
                if not df.empty:
                    df.to_excel(filepath)
                    logger.info(f"Weight history exported to {filepath}")
                    return True
                else:
                    logger.warning("No weight history data to export")
                    return False
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export weight history: {e}")
            return False


class PerformanceDashboard:
    """
    Performance dashboard integration for monitoring and visualization
    """
    
    def __init__(self, performance_tracker: PerformanceTracker, 
                 weight_tracker: WeightEvolutionTracker):
        self.performance_tracker = performance_tracker
        self.weight_tracker = weight_tracker
    
    def generate_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data
        
        Args:
            days: Number of days to include in dashboard
            
        Returns:
            Dictionary containing all dashboard data
        """
        try:
            dashboard_data = {
                'timestamp': datetime.now(),
                'period_days': days,
                'performance_summary': self.performance_tracker.generate_performance_report(days),
                'weight_evolution': self._get_weight_evolution_summary(days),
                'active_alerts': self._get_alerts_summary(),
                'model_rankings': self._get_model_rankings_summary(days),
                'trend_analysis': self._get_trend_analysis(days)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def _get_weight_evolution_summary(self, days: int) -> Dict[str, Any]:
        """Get weight evolution summary for dashboard"""
        try:
            weight_history = self.weight_tracker.get_weight_evolution_history(days)
            significant_changes = self.weight_tracker.get_significant_changes(days)
            
            return {
                'total_changes': len(weight_history),
                'significant_changes': len(significant_changes),
                'recent_changes': weight_history[-10:] if weight_history else [],
                'change_frequency': len(weight_history) / max(days, 1),
                'most_volatile_model': self._get_most_volatile_model(weight_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get weight evolution summary: {e}")
            return {}
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary for dashboard"""
        try:
            active_alerts = self.performance_tracker.get_active_alerts()
            
            return {
                'total_active': len(active_alerts),
                'by_severity': {
                    'critical': len([a for a in active_alerts if a.severity == 'critical']),
                    'high': len([a for a in active_alerts if a.severity == 'high']),
                    'medium': len([a for a in active_alerts if a.severity == 'medium']),
                    'low': len([a for a in active_alerts if a.severity == 'low'])
                },
                'recent_alerts': active_alerts[:5]  # Last 5 alerts
            }
            
        except Exception as e:
            logger.error(f"Failed to get alerts summary: {e}")
            return {}
    
    def _get_model_rankings_summary(self, days: int) -> Dict[str, Any]:
        """Get model rankings summary for dashboard"""
        try:
            rankings = {}
            
            for metric in ['mae', 'mape', 'rmse', 'r_squared']:
                rankings[metric] = self.performance_tracker.get_model_rankings(days, metric)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Failed to get model rankings summary: {e}")
            return {}
    
    def _get_trend_analysis(self, days: int) -> Dict[str, Any]:
        """Get trend analysis for dashboard"""
        try:
            # This would implement trend analysis of performance over time
            # For now, return basic statistics
            return {
                'performance_trend': 'stable',  # Would be calculated from historical data
                'weight_stability': 'moderate',  # Would be calculated from weight changes
                'prediction_accuracy': 'improving'  # Would be calculated from recent performance
            }
            
        except Exception as e:
            logger.error(f"Failed to get trend analysis: {e}")
            return {}
    
    def _get_most_volatile_model(self, weight_history: List[WeightChangeRecord]) -> Optional[str]:
        """Identify the model with most weight volatility"""
        try:
            if not weight_history:
                return None
            
            model_volatility = {}
            
            for record in weight_history:
                for model_name in record.old_weights:
                    old_weight = record.old_weights.get(model_name, 0.0)
                    new_weight = record.new_weights.get(model_name, 0.0)
                    change = abs(new_weight - old_weight)
                    
                    if model_name not in model_volatility:
                        model_volatility[model_name] = []
                    model_volatility[model_name].append(change)
            
            # Calculate average volatility for each model
            avg_volatility = {
                model: np.mean(changes) 
                for model, changes in model_volatility.items()
            }
            
            if avg_volatility:
                return max(avg_volatility, key=avg_volatility.get)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get most volatile model: {e}")
            return None
    
    def generate_weight_evolution_chart_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate data for weight evolution visualization charts
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary containing chart data and configuration
        """
        try:
            weight_df = self.weight_tracker.get_weight_evolution_dataframe(days)
            
            if weight_df.empty:
                return {'error': 'No weight evolution data available'}
            
            # Prepare time series data
            chart_data = {
                'timestamps': weight_df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'models': {},
                'summary_stats': {},
                'annotations': []
            }
            
            # Add weight series for each model
            for model_name in weight_df.columns:
                chart_data['models'][model_name] = {
                    'weights': weight_df[model_name].tolist(),
                    'volatility': self.weight_tracker.calculate_weight_volatility(model_name),
                    'current_weight': float(weight_df[model_name].iloc[-1]) if len(weight_df) > 0 else 0.0,
                    'avg_weight': float(weight_df[model_name].mean()),
                    'min_weight': float(weight_df[model_name].min()),
                    'max_weight': float(weight_df[model_name].max())
                }
            
            # Add significant change annotations
            significant_changes = self.weight_tracker.get_significant_changes(days)
            for change in significant_changes[-10:]:  # Last 10 significant changes
                chart_data['annotations'].append({
                    'timestamp': change.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'reason': change.trigger_reason,
                    'pattern': change.pattern_detected,
                    'method': change.weight_update_method
                })
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Failed to generate weight evolution chart data: {e}")
            return {'error': str(e)}
    
    def generate_performance_heatmap_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate data for performance heatmap visualization
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary containing heatmap data
        """
        try:
            # Get performance data for all models
            models = list(self.performance_tracker.model_performance_history.keys())
            
            heatmap_data = {
                'models': models,
                'metrics': ['mae', 'mape', 'rmse', 'r_squared'],
                'data': [],
                'color_scale': {
                    'mae': {'min': 0, 'max': 1, 'reverse': True},
                    'mape': {'min': 0, 'max': 100, 'reverse': True},
                    'rmse': {'min': 0, 'max': 1, 'reverse': True},
                    'r_squared': {'min': 0, 'max': 1, 'reverse': False}
                }
            }
            
            # Calculate performance matrix
            for model in models:
                model_perf = self.performance_tracker.get_recent_performance(model, days)
                row_data = []
                
                for metric in heatmap_data['metrics']:
                    value = model_perf.get(metric, 0.0)
                    if np.isfinite(value):
                        row_data.append(float(value))
                    else:
                        row_data.append(None)
                
                heatmap_data['data'].append(row_data)
            
            return heatmap_data
            
        except Exception as e:
            logger.error(f"Failed to generate performance heatmap data: {e}")
            return {'error': str(e)}
    
    def create_alert_for_significant_weight_change(self, change_record: WeightChangeRecord) -> Optional[PerformanceAlert]:
        """
        Create an alert for significant weight changes
        
        Args:
            change_record: Weight change record that triggered the alert
            
        Returns:
            Created alert or None if alert creation failed
        """
        try:
            # Calculate the magnitude of change
            max_change = 0.0
            changed_models = []
            
            for model_name in change_record.old_weights:
                old_weight = change_record.old_weights.get(model_name, 0.0)
                new_weight = change_record.new_weights.get(model_name, 0.0)
                change = abs(new_weight - old_weight)
                
                if change > max_change:
                    max_change = change
                
                if change > 0.1:  # 10% threshold for individual model changes
                    changed_models.append(f"{model_name}: {old_weight:.3f} → {new_weight:.3f}")
            
            # Determine alert severity
            if max_change > 0.3:
                severity = 'critical'
            elif max_change > 0.2:
                severity = 'high'
            elif max_change > 0.1:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Create alert message
            message = f"Significant weight change detected (max change: {max_change:.1%})"
            
            # Create alert details
            details = {
                'trigger_reason': change_record.trigger_reason,
                'pattern_detected': change_record.pattern_detected,
                'weight_update_method': change_record.weight_update_method,
                'max_change': max_change,
                'changed_models': changed_models,
                'old_weights': change_record.old_weights,
                'new_weights': change_record.new_weights
            }
            
            # Create the alert
            alert = self.performance_tracker.create_alert(
                alert_type='weight_change',
                severity=severity,
                model_name=None,  # System-wide alert
                message=message,
                details=details
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create weight change alert: {e}")
            return None
    
    def generate_model_comparison_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate data for model comparison visualization
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary containing model comparison data
        """
        try:
            models = list(self.performance_tracker.model_performance_history.keys())
            
            comparison_data = {
                'models': models,
                'performance_comparison': {},
                'weight_comparison': {},
                'trend_analysis': {}
            }
            
            # Performance comparison
            for model in models:
                perf = self.performance_tracker.get_recent_performance(model, days)
                comparison_data['performance_comparison'][model] = {
                    'mae': perf.get('mae', float('inf')),
                    'mape': perf.get('mape', float('inf')),
                    'rmse': perf.get('rmse', float('inf')),
                    'r_squared': perf.get('r_squared', 0.0),
                    'record_count': perf.get('record_count', 0),
                    'avg_weight': perf.get('avg_weight', 0.0)
                }
            
            # Weight comparison
            weight_df = self.weight_tracker.get_weight_evolution_dataframe(days)
            if not weight_df.empty:
                for model in models:
                    if model in weight_df.columns:
                        comparison_data['weight_comparison'][model] = {
                            'current_weight': float(weight_df[model].iloc[-1]) if len(weight_df) > 0 else 0.0,
                            'avg_weight': float(weight_df[model].mean()),
                            'volatility': self.weight_tracker.calculate_weight_volatility(model),
                            'trend': self._calculate_weight_trend(weight_df[model])
                        }
            
            # Rankings
            for metric in ['mae', 'mape', 'rmse', 'r_squared']:
                rankings = self.performance_tracker.get_model_rankings(days, metric)
                comparison_data[f'{metric}_rankings'] = rankings
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to generate model comparison data: {e}")
            return {'error': str(e)}
    
    def _calculate_weight_trend(self, weight_series: pd.Series) -> str:
        """Calculate weight trend direction"""
        try:
            if len(weight_series) < 2:
                return 'stable'
            
            # Calculate linear trend
            x = np.arange(len(weight_series))
            slope = np.polyfit(x, weight_series.values, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Failed to calculate weight trend: {e}")
            return 'unknown'
    
    def export_dashboard_data(self, filepath: str, days: int = 30) -> bool:
        """
        Export comprehensive dashboard data to file
        
        Args:
            filepath: Path to save the file
            days: Number of days to include
            
        Returns:
            True if export successful
        """
        try:
            dashboard_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_days': days,
                'dashboard_summary': self.generate_dashboard_data(days),
                'weight_evolution': self.generate_weight_evolution_chart_data(days),
                'performance_heatmap': self.generate_performance_heatmap_data(days),
                'model_comparison': self.generate_model_comparison_data(days),
                'audit_trail': self.weight_tracker.get_weight_change_audit_trail(days=days)
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            return False


class PerformanceVisualization:
    """
    Advanced performance visualization methods for ensemble forecasting
    """
    
    def __init__(self, performance_tracker: PerformanceTracker, 
                 weight_tracker: WeightEvolutionTracker):
        self.performance_tracker = performance_tracker
        self.weight_tracker = weight_tracker
    
    def create_performance_timeline_data(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Create timeline data for model performance visualization
        
        Args:
            model_name: Name of the model
            days: Number of days to include
            
        Returns:
            Dictionary containing timeline data
        """
        try:
            performance_df = self.performance_tracker.get_performance_history(model_name, limit=None)
            
            if performance_df.empty:
                return {'error': f'No performance data for model {model_name}'}
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            performance_df = performance_df[performance_df.index >= cutoff_date]
            
            timeline_data = {
                'model_name': model_name,
                'timestamps': performance_df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'metrics': {
                    'mae': performance_df['mae'].tolist(),
                    'mape': performance_df['mape'].tolist(),
                    'rmse': performance_df['rmse'].tolist(),
                    'r_squared': performance_df['r_squared'].tolist()
                },
                'predictions': {
                    'predicted_values': performance_df['predicted_value'].tolist(),
                    'actual_values': performance_df['actual_value'].tolist(),
                    'errors': (performance_df['actual_value'] - performance_df['predicted_value']).tolist()
                },
                'weights': performance_df['weight_at_time'].tolist(),
                'patterns': performance_df['data_pattern'].tolist(),
                'summary_stats': {
                    'avg_mae': float(performance_df['mae'].mean()),
                    'avg_mape': float(performance_df['mape'].mean()),
                    'avg_rmse': float(performance_df['rmse'].mean()),
                    'avg_r_squared': float(performance_df['r_squared'].mean()),
                    'prediction_count': len(performance_df),
                    'avg_weight': float(performance_df['weight_at_time'].mean())
                }
            }
            
            return timeline_data
            
        except Exception as e:
            logger.error(f"Failed to create performance timeline data for {model_name}: {e}")
            return {'error': str(e)}
    
    def create_ensemble_performance_comparison(self, days: int = 30) -> Dict[str, Any]:
        """
        Create comprehensive ensemble performance comparison data
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary containing comparison data
        """
        try:
            models = list(self.performance_tracker.model_performance_history.keys())
            
            comparison_data = {
                'period_days': days,
                'models': {},
                'ensemble_metrics': {},
                'performance_rankings': {},
                'correlation_matrix': {},
                'efficiency_metrics': {}
            }
            
            # Individual model performance
            for model in models:
                timeline_data = self.create_performance_timeline_data(model, days)
                if 'error' not in timeline_data:
                    comparison_data['models'][model] = timeline_data
            
            # Calculate ensemble-level metrics
            all_predictions = []
            all_actuals = []
            all_weights = {}
            
            for model in models:
                perf_df = self.performance_tracker.get_performance_history(model)
                if not perf_df.empty:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    recent_df = perf_df[perf_df.index >= cutoff_date]
                    
                    if not recent_df.empty:
                        all_predictions.extend(recent_df['predicted_value'].tolist())
                        all_actuals.extend(recent_df['actual_value'].tolist())
                        all_weights[model] = recent_df['weight_at_time'].mean()
            
            if all_predictions and all_actuals:
                # Calculate ensemble performance metrics
                mae = np.mean(np.abs(np.array(all_actuals) - np.array(all_predictions)))
                mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100
                rmse = np.sqrt(np.mean((np.array(all_actuals) - np.array(all_predictions)) ** 2))
                
                comparison_data['ensemble_metrics'] = {
                    'mae': float(mae),
                    'mape': float(mape),
                    'rmse': float(rmse),
                    'prediction_count': len(all_predictions),
                    'average_weights': all_weights
                }
            
            # Performance rankings
            for metric in ['mae', 'mape', 'rmse', 'r_squared']:
                rankings = self.performance_tracker.get_model_rankings(days, metric)
                comparison_data['performance_rankings'][metric] = rankings
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to create ensemble performance comparison: {e}")
            return {'error': str(e)}
    
    def create_weight_volatility_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Create weight volatility analysis data
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary containing volatility analysis
        """
        try:
            weight_df = self.weight_tracker.get_weight_evolution_dataframe(days)
            
            if weight_df.empty:
                return {'error': 'No weight evolution data available'}
            
            volatility_data = {
                'period_days': days,
                'models': {},
                'overall_stability': {},
                'change_frequency': {},
                'volatility_rankings': []
            }
            
            # Calculate volatility metrics for each model
            for model in weight_df.columns:
                model_weights = weight_df[model]
                
                volatility_data['models'][model] = {
                    'volatility': float(model_weights.std()),
                    'mean_weight': float(model_weights.mean()),
                    'min_weight': float(model_weights.min()),
                    'max_weight': float(model_weights.max()),
                    'weight_range': float(model_weights.max() - model_weights.min()),
                    'coefficient_of_variation': float(model_weights.std() / model_weights.mean()) if model_weights.mean() > 0 else 0.0,
                    'trend': self._calculate_weight_trend_direction(model_weights),
                    'stability_score': self._calculate_stability_score(model_weights)
                }
            
            # Overall ensemble stability metrics
            total_volatility = sum(data['volatility'] for data in volatility_data['models'].values())
            volatility_data['overall_stability'] = {
                'total_volatility': float(total_volatility),
                'avg_volatility': float(total_volatility / len(volatility_data['models'])),
                'most_stable_model': min(volatility_data['models'].items(), key=lambda x: x[1]['volatility'])[0],
                'most_volatile_model': max(volatility_data['models'].items(), key=lambda x: x[1]['volatility'])[0],
                'stability_rating': self._get_stability_rating(total_volatility)
            }
            
            # Volatility rankings
            volatility_data['volatility_rankings'] = sorted(
                [(model, data['volatility']) for model, data in volatility_data['models'].items()],
                key=lambda x: x[1]
            )
            
            return volatility_data
            
        except Exception as e:
            logger.error(f"Failed to create weight volatility analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_weight_trend_direction(self, weight_series: pd.Series) -> Dict[str, Any]:
        """Calculate detailed weight trend information"""
        try:
            if len(weight_series) < 3:
                return {'direction': 'insufficient_data', 'slope': 0.0, 'r_squared': 0.0}
            
            x = np.arange(len(weight_series))
            coeffs = np.polyfit(x, weight_series.values, 1)
            slope = coeffs[0]
            
            # Calculate R-squared for trend line
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((weight_series.values - y_pred) ** 2)
            ss_tot = np.sum((weight_series.values - np.mean(weight_series.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            if abs(slope) < 0.001:
                direction = 'stable'
            elif slope > 0.001:
                direction = 'increasing'
            else:
                direction = 'decreasing'
            
            return {
                'direction': direction,
                'slope': float(slope),
                'r_squared': float(r_squared),
                'trend_strength': 'strong' if abs(slope) > 0.01 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate weight trend direction: {e}")
            return {'direction': 'error', 'slope': 0.0, 'r_squared': 0.0}
    
    def _calculate_stability_score(self, weight_series: pd.Series) -> float:
        """Calculate a stability score (0-1, higher is more stable)"""
        try:
            if len(weight_series) < 2:
                return 1.0
            
            # Calculate coefficient of variation (lower is more stable)
            cv = weight_series.std() / weight_series.mean() if weight_series.mean() > 0 else float('inf')
            
            # Convert to stability score (0-1 scale)
            stability_score = 1.0 / (1.0 + cv)
            
            return float(stability_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate stability score: {e}")
            return 0.0
    
    def _get_stability_rating(self, total_volatility: float) -> str:
        """Get qualitative stability rating"""
        if total_volatility < 0.1:
            return 'very_stable'
        elif total_volatility < 0.2:
            return 'stable'
        elif total_volatility < 0.4:
            return 'moderate'
        elif total_volatility < 0.6:
            return 'volatile'
        else:
            return 'very_volatile'
    
    def create_alert_dashboard_data(self) -> Dict[str, Any]:
        """
        Create comprehensive alert dashboard data
        
        Returns:
            Dictionary containing alert dashboard information
        """
        try:
            active_alerts = self.performance_tracker.get_active_alerts()
            
            alert_data = {
                'summary': {
                    'total_active': len(active_alerts),
                    'by_severity': {
                        'critical': len([a for a in active_alerts if a.severity == 'critical']),
                        'high': len([a for a in active_alerts if a.severity == 'high']),
                        'medium': len([a for a in active_alerts if a.severity == 'medium']),
                        'low': len([a for a in active_alerts if a.severity == 'low'])
                    },
                    'by_type': {}
                },
                'recent_alerts': [],
                'alert_trends': {},
                'resolution_stats': {}
            }
            
            # Count alerts by type
            for alert in active_alerts:
                alert_type = alert.alert_type
                if alert_type not in alert_data['summary']['by_type']:
                    alert_data['summary']['by_type'][alert_type] = 0
                alert_data['summary']['by_type'][alert_type] += 1
            
            # Recent alerts (last 10)
            for alert in active_alerts[:10]:
                alert_data['recent_alerts'].append({
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'model_name': alert.model_name,
                    'message': alert.message,
                    'details': alert.details
                })
            
            # Calculate alert trends (last 7 days)
            seven_days_ago = datetime.now() - timedelta(days=7)
            recent_alerts = [a for a in self.performance_tracker.alerts if a.timestamp >= seven_days_ago]
            
            alert_data['alert_trends'] = {
                'total_last_7_days': len(recent_alerts),
                'daily_counts': self._calculate_daily_alert_counts(recent_alerts),
                'trending_up': len(recent_alerts) > len(active_alerts) * 0.7  # Simple trend indicator
            }
            
            # Resolution statistics
            resolved_alerts = [a for a in self.performance_tracker.alerts if a.resolved]
            if resolved_alerts:
                resolution_times = []
                for alert in resolved_alerts:
                    if alert.resolved_timestamp:
                        resolution_time = (alert.resolved_timestamp - alert.timestamp).total_seconds() / 3600  # hours
                        resolution_times.append(resolution_time)
                
                if resolution_times:
                    alert_data['resolution_stats'] = {
                        'avg_resolution_time_hours': float(np.mean(resolution_times)),
                        'median_resolution_time_hours': float(np.median(resolution_times)),
                        'total_resolved': len(resolved_alerts),
                        'resolution_rate': len(resolved_alerts) / len(self.performance_tracker.alerts) if self.performance_tracker.alerts else 0.0
                    }
            
            return alert_data
            
        except Exception as e:
            logger.error(f"Failed to create alert dashboard data: {e}")
            return {'error': str(e)}
    
    def _calculate_daily_alert_counts(self, alerts: List) -> Dict[str, int]:
        """Calculate daily alert counts for the last 7 days"""
        try:
            daily_counts = {}
            
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_counts[date] = 0
            
            for alert in alerts:
                date = alert.timestamp.strftime('%Y-%m-%d')
                if date in daily_counts:
                    daily_counts[date] += 1
            
            return daily_counts
            
        except Exception as e:
            logger.error(f"Failed to calculate daily alert counts: {e}")
            return {}
    def __init__(self, use_ml=False, adaptive_learning=True, pattern_detection=True):
        self.models = {
            'arima': ARIMAForecaster(),
            'ets': ETSForecaster(),
            'croston': CrostonForecaster()
        }
        
        if use_ml and XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBoostForecaster()
            
        if use_ml and LSTM_AVAILABLE:
            self.models['lstm'] = LSTMForecaster()
        
        # Pattern detection configuration
        self.adaptive_learning = adaptive_learning
        self.pattern_detection_enabled = pattern_detection
        self.pattern_detector = PatternDetector() if pattern_detection else None
        self.current_pattern: Optional[PatternCharacteristics] = None
        
        # Initialize weights - will be updated by pattern detection if enabled
        num_models = len(self.models)
        base_weight = 1.0 / num_models
        self.weights = {name: base_weight for name in self.models.keys()}
        
        # Performance tracking
        self.performance_log: List[PerformanceMetrics] = []
        self.backtest_results: Optional[BacktestResult] = None
        self.historical_performance: Dict[str, List[PerformanceMetrics]] = {
            name: [] for name in self.models.keys()
        }
        
        # Pattern change tracking
        self.pattern_change_threshold = 0.3
        self.last_pattern_check: Optional[datetime] = None
        
        # Weight update configuration
        self.weight_update_config = WeightUpdateConfig()
        self.weight_history: List[Dict[str, Any]] = []
        self.performance_alerts: List[Dict[str, Any]] = []
        
        # Bootstrap configuration for confidence intervals
        self.bootstrap_samples = 1000
        self.confidence_levels = [0.1, 0.5, 0.9]  # P10, P50, P90
    
    def fit(self, data, features_df=None):
        """Fit all models. Use features_df for ML models if available"""
        # Detect pattern and set intelligent initial weights if enabled
        if self.pattern_detection_enabled and self.pattern_detector:
            self.current_pattern = self.pattern_detector.detect_pattern(data)
            pattern_weights = self.pattern_detector.get_pattern_specific_weights(
                self.current_pattern.pattern_type, 
                list(self.models.keys())
            )
            self.weights = pattern_weights
            logger.info(f"Detected pattern: {self.current_pattern.pattern_type} "
                       f"(confidence: {self.current_pattern.confidence:.2f})")
            logger.info(f"Applied pattern-specific weights: {self.weights}")
        
        # Fit all models
        for name, model in self.models.items():
            try:
                if name in ['xgboost'] and features_df is not None:
                    model.fit(features_df)
                else:
                    if hasattr(data, 'values'):
                        model.fit(data)
                    else:
                        model.fit(pd.Series(data))
            except Exception as e:
                print(f"Warning: {name} model failed to fit: {e}")
        return self
    
    def forecast(self, steps: int, features_df=None) -> pd.Series:
        """Generate ensemble forecast"""
        forecasts = {}
        successful_models = []
        
        for name, model in self.models.items():
            try:
                if name in ['xgboost'] and features_df is not None:
                    forecast = model.forecast(features_df, steps)
                else:
                    forecast = model.forecast(steps)
                
                # Ensure forecast is a pandas Series with consistent index
                if not isinstance(forecast, pd.Series):
                    forecast = pd.Series(forecast)
                
                # Reset index to ensure consistent indexing (0, 1, 2, ...)
                forecast = pd.Series(forecast.values, index=range(len(forecast)))
                
                if len(forecast) == steps:
                    forecasts[name] = forecast
                    successful_models.append(name)
                else:
                    print(f"Warning: {name} model returned {len(forecast)} steps instead of {steps}")
                    
            except Exception as e:
                print(f"Warning: {name} model failed to forecast: {e}")
        
        if not successful_models:
            # If all models failed, return zeros
            return pd.Series([0.0] * steps, index=range(steps))
        
        # Calculate ensemble forecast using only successful models
        # Renormalize weights for successful models only
        successful_weights = {name: self.weights[name] for name in successful_models}
        total_weight = sum(successful_weights.values())
        
        if total_weight > 0:
            normalized_weights = {name: weight / total_weight for name, weight in successful_weights.items()}
        else:
            # Equal weights if all weights are zero
            normalized_weights = {name: 1.0 / len(successful_models) for name in successful_models}
        
        # Create ensemble forecast
        ensemble_forecast = pd.Series([0.0] * steps, index=range(steps))
        
        for name in successful_models:
            ensemble_forecast += forecasts[name] * normalized_weights[name]
        
        return ensemble_forecast
    
    def forecast_with_confidence(self, steps: int, features_df=None, 
                               bootstrap_samples: Optional[int] = None) -> ConfidenceInterval:
        """
        Generate ensemble forecast with confidence intervals using bootstrap methods
        
        Args:
            steps: Number of forecast steps
            features_df: Optional features for ML models
            bootstrap_samples: Number of bootstrap samples (uses default if None)
            
        Returns:
            ConfidenceInterval object with P10/P50/P90 forecasts
        """
        try:
            if bootstrap_samples is None:
                bootstrap_samples = self.bootstrap_samples
            
            logger.info(f"Generating forecast with confidence intervals using {bootstrap_samples} bootstrap samples")
            
            # Generate individual model forecasts
            model_forecasts = {}
            successful_models = []
            
            for name, model in self.models.items():
                try:
                    if name in ['xgboost'] and features_df is not None:
                        forecast = model.forecast(features_df, steps)
                    else:
                        forecast = model.forecast(steps)
                    
                    if not isinstance(forecast, pd.Series):
                        forecast = pd.Series(forecast)
                    
                    forecast = pd.Series(forecast.values, index=range(len(forecast)))
                    
                    if len(forecast) == steps:
                        model_forecasts[name] = forecast
                        successful_models.append(name)
                        
                except Exception as e:
                    logger.warning(f"Model {name} failed to generate forecast: {e}")
            
            if not successful_models:
                # Return zero forecasts if all models failed
                zero_forecast = pd.Series([0.0] * steps, index=range(steps))
                return ConfidenceInterval(
                    p10=zero_forecast,
                    p50=zero_forecast,
                    p90=zero_forecast,
                    forecast_date=datetime.now(),
                    model_name="ensemble"
                )
            
            # Bootstrap ensemble forecasts
            bootstrap_forecasts = []
            
            for _ in range(bootstrap_samples):
                # Generate bootstrap weights by adding noise to current weights
                bootstrap_weights = self._generate_bootstrap_weights(successful_models)
                
                # Calculate bootstrap ensemble forecast
                bootstrap_forecast = pd.Series([0.0] * steps, index=range(steps))
                for model_name in successful_models:
                    bootstrap_forecast += model_forecasts[model_name] * bootstrap_weights[model_name]
                
                bootstrap_forecasts.append(bootstrap_forecast)
            
            # Calculate confidence intervals from bootstrap samples
            bootstrap_array = np.array([forecast.values for forecast in bootstrap_forecasts])
            
            p10_forecast = pd.Series(np.percentile(bootstrap_array, 10, axis=0), index=range(steps))
            p50_forecast = pd.Series(np.percentile(bootstrap_array, 50, axis=0), index=range(steps))
            p90_forecast = pd.Series(np.percentile(bootstrap_array, 90, axis=0), index=range(steps))
            
            confidence_interval = ConfidenceInterval(
                p10=p10_forecast,
                p50=p50_forecast,
                p90=p90_forecast,
                forecast_date=datetime.now(),
                model_name="ensemble"
            )
            
            logger.info("Confidence interval forecast generated successfully")
            return confidence_interval
            
        except Exception as e:
            logger.error(f"Confidence interval forecast failed: {e}")
            # Return point forecast as fallback
            point_forecast = self.forecast(steps, features_df)
            return ConfidenceInterval(
                p10=point_forecast,
                p50=point_forecast,
                p90=point_forecast,
                forecast_date=datetime.now(),
                model_name="ensemble"
            )
    
    def _generate_bootstrap_weights(self, model_names: List[str]) -> Dict[str, float]:
        """
        Generate bootstrap weights by adding controlled noise to current weights
        
        Args:
            model_names: List of successful model names
            
        Returns:
            Dictionary of bootstrap weights
        """
        try:
            # Get current weights for successful models
            current_weights = {name: self.weights[name] for name in model_names}
            
            # Add Gaussian noise to weights (scaled by current weight)
            noise_scale = 0.1  # 10% noise
            bootstrap_weights = {}
            
            for name in model_names:
                current_weight = current_weights[name]
                # Add proportional noise
                noise = np.random.normal(0, current_weight * noise_scale)
                bootstrap_weight = max(0.01, current_weight + noise)  # Ensure positive
                bootstrap_weights[name] = bootstrap_weight
            
            # Normalize to sum to 1.0
            total_weight = sum(bootstrap_weights.values())
            bootstrap_weights = {name: weight / total_weight 
                               for name, weight in bootstrap_weights.items()}
            
            return bootstrap_weights
            
        except Exception as e:
            logger.error(f"Bootstrap weight generation failed: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(model_names)
            return {name: equal_weight for name in model_names}
    
    def export_results(self, forecast: pd.Series, filename: str):
        forecast.to_csv(filename, header=['forecast'])
    
    def get_model_weights(self) -> dict:
        """Return current model weights"""
        return self.weights.copy()
    
    def set_model_weights(self, weights: dict):
        """Set custom model weights"""
        for name in weights:
            if name in self.models:
                self.weights[name] = weights[name]
    
    def calculate_error_metrics(self, actual: pd.Series, predicted: pd.Series, 
                              model_name: str = "ensemble") -> PerformanceMetrics:
        """Calculate comprehensive error metrics for model evaluation"""
        try:
            # Ensure both series have the same length and no NaN values
            actual = actual.dropna()
            predicted = predicted.dropna()
            
            # Align series by index
            common_index = actual.index.intersection(predicted.index)
            if len(common_index) == 0:
                raise ValueError("No common indices between actual and predicted values")
            
            actual_aligned = actual.loc[common_index]
            predicted_aligned = predicted.loc[common_index]
            
            # Calculate MAE (Mean Absolute Error)
            mae = np.mean(np.abs(actual_aligned - predicted_aligned))
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            non_zero_mask = actual_aligned != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((actual_aligned[non_zero_mask] - predicted_aligned[non_zero_mask]) / 
                                    actual_aligned[non_zero_mask])) * 100
            else:
                mape = float('inf')
            
            # Calculate RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((actual_aligned - predicted_aligned) ** 2))
            
            # Calculate R-squared
            ss_res = np.sum((actual_aligned - predicted_aligned) ** 2)
            ss_tot = np.sum((actual_aligned - np.mean(actual_aligned)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return PerformanceMetrics(
                mae=mae,
                mape=mape,
                rmse=rmse,
                r_squared=r_squared,
                model_name=model_name,
                evaluation_date=datetime.now(),
                data_points=len(actual_aligned)
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            # Return default metrics in case of error
            return PerformanceMetrics(
                mae=float('inf'),
                mape=float('inf'),
                rmse=float('inf'),
                r_squared=0.0,
                model_name=model_name,
                evaluation_date=datetime.now(),
                data_points=0
            )
    
    def evaluate_models(self, actual_data: pd.Series, forecast_horizon: int = 12,
                       features_df: Optional[pd.DataFrame] = None) -> Dict[str, PerformanceMetrics]:
        """
        Evaluate all models using out-of-sample testing
        
        Args:
            actual_data: Historical time series data
            forecast_horizon: Number of periods to forecast for evaluation
            features_df: Optional features for ML models
            
        Returns:
            Dictionary of performance metrics for each model
        """
        model_metrics = {}
        
        if len(actual_data) < forecast_horizon + 10:
            logger.warning("Insufficient data for proper model evaluation")
            return model_metrics
        
        try:
            # Split data into train and test
            train_size = len(actual_data) - forecast_horizon
            train_data = actual_data.iloc[:train_size]
            test_data = actual_data.iloc[train_size:]
            
            # Evaluate each model
            for model_name, model in self.models.items():
                try:
                    # Fit model on training data
                    if model_name in ['xgboost'] and features_df is not None:
                        train_features = features_df.iloc[:train_size] if len(features_df) >= train_size else features_df
                        model.fit(train_features)
                        
                        # Generate forecast
                        test_features = features_df.iloc[train_size:] if len(features_df) > train_size else features_df.tail(forecast_horizon)
                        forecast = model.forecast(test_features, forecast_horizon)
                    else:
                        model.fit(train_data)
                        forecast = model.forecast(forecast_horizon)
                    
                    # Calculate metrics
                    metrics = self.calculate_error_metrics(test_data, forecast, model_name)
                    model_metrics[model_name] = metrics
                    
                    # Store in historical performance
                    self.historical_performance[model_name].append(metrics)
                    
                    logger.info(f"Evaluated {model_name}: MAE={metrics.mae:.2f}, MAPE={metrics.mape:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {e}")
                    # Create default metrics for failed model
                    model_metrics[model_name] = PerformanceMetrics(
                        mae=float('inf'),
                        mape=float('inf'),
                        rmse=float('inf'),
                        r_squared=0.0,
                        model_name=model_name,
                        evaluation_date=datetime.now(),
                        data_points=0
                    )
            
            # Store evaluation results
            self.performance_log.extend(model_metrics.values())
            
            return model_metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return model_metrics
    
    def run_backtesting(self, data: pd.Series, validation_split: float = 0.3, 
                       n_splits: int = 5, features_df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run time series cross-validation backtesting
        
        Args:
            data: Historical time series data
            validation_split: Proportion of data to use for validation
            n_splits: Number of cross-validation splits
            features_df: Optional features for ML models
            
        Returns:
            BacktestResult containing performance metrics and initial weights
        """
        try:
            logger.info(f"Starting backtesting with {n_splits} splits...")
            
            # Initialize time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            model_performance_history = {name: [] for name in self.models.keys()}
            
            # Perform cross-validation
            for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
                logger.info(f"Processing fold {fold + 1}/{n_splits}")
                
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Evaluate each model on this fold
                for model_name, model in self.models.items():
                    try:
                        # Fit model
                        if model_name in ['xgboost'] and features_df is not None:
                            train_features = features_df.iloc[train_idx] if len(features_df) > max(train_idx) else features_df
                            model.fit(train_features)
                            
                            # Generate forecast
                            test_features = features_df.iloc[test_idx] if len(features_df) > max(test_idx) else features_df.tail(len(test_idx))
                            forecast = model.forecast(test_features, len(test_idx))
                        else:
                            model.fit(train_data)
                            forecast = model.forecast(len(test_idx))
                        
                        # Calculate metrics for this fold
                        metrics = self.calculate_error_metrics(test_data, forecast, model_name)
                        model_performance_history[model_name].append(metrics)
                        
                    except Exception as e:
                        logger.error(f"Fold {fold + 1} failed for {model_name}: {e}")
                        # Add default metrics for failed fold
                        model_performance_history[model_name].append(
                            PerformanceMetrics(
                                mae=float('inf'),
                                mape=float('inf'),
                                rmse=float('inf'),
                                r_squared=0.0,
                                model_name=model_name,
                                evaluation_date=datetime.now(),
                                data_points=0
                            )
                        )
            
            # Calculate average performance across folds
            avg_performance = {}
            for model_name, metrics_list in model_performance_history.items():
                if metrics_list:
                    # Filter out infinite values for averaging
                    valid_metrics = [m for m in metrics_list if np.isfinite(m.mae)]
                    
                    if valid_metrics:
                        avg_mae = np.mean([m.mae for m in valid_metrics])
                        avg_mape = np.mean([m.mape for m in valid_metrics if np.isfinite(m.mape)])
                        avg_rmse = np.mean([m.rmse for m in valid_metrics])
                        avg_r_squared = np.mean([m.r_squared for m in valid_metrics])
                        total_points = sum([m.data_points for m in valid_metrics])
                    else:
                        avg_mae = avg_mape = avg_rmse = float('inf')
                        avg_r_squared = 0.0
                        total_points = 0
                    
                    avg_performance[model_name] = PerformanceMetrics(
                        mae=avg_mae,
                        mape=avg_mape,
                        rmse=avg_rmse,
                        r_squared=avg_r_squared,
                        model_name=model_name,
                        evaluation_date=datetime.now(),
                        data_points=total_points
                    )
            
            # Detect pattern in the data if pattern detection is enabled
            detected_pattern = None
            if self.pattern_detection_enabled and self.pattern_detector:
                detected_pattern = self.pattern_detector.detect_pattern(data)
                self.current_pattern = detected_pattern
                logger.info(f"Backtesting detected pattern: {detected_pattern.pattern_type} "
                           f"(confidence: {detected_pattern.confidence:.2f})")
            
            # Calculate initial weights - combine performance and pattern-based weights
            if detected_pattern and self.pattern_detection_enabled:
                pattern_weights = self.pattern_detector.get_pattern_specific_weights(
                    detected_pattern.pattern_type, 
                    list(self.models.keys())
                )
                performance_weights = self._calculate_performance_based_weights(avg_performance)
                
                # Blend pattern-based and performance-based weights
                # Higher pattern confidence = more weight to pattern-based weights
                pattern_confidence = detected_pattern.confidence
                initial_weights = {}
                for model_name in self.models.keys():
                    pattern_weight = pattern_weights.get(model_name, 1.0 / len(self.models))
                    perf_weight = performance_weights.get(model_name, 1.0 / len(self.models))
                    
                    # Weighted combination based on pattern confidence
                    blended_weight = (pattern_confidence * pattern_weight + 
                                    (1 - pattern_confidence) * perf_weight)
                    initial_weights[model_name] = blended_weight
                
                # Normalize weights
                total_weight = sum(initial_weights.values())
                initial_weights = {name: weight / total_weight for name, weight in initial_weights.items()}
                
                logger.info(f"Blended initial weights (pattern confidence {pattern_confidence:.2f}): {initial_weights}")
            else:
                initial_weights = self._calculate_performance_based_weights(avg_performance)
            
            # Create backtest result
            backtest_result = BacktestResult(
                model_performance=avg_performance,
                initial_weights=initial_weights,
                validation_periods=n_splits,
                total_data_points=len(data),
                backtest_date=datetime.now(),
                detected_pattern=detected_pattern
            )
            
            self.backtest_results = backtest_result
            logger.info("Backtesting completed successfully")
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            # Return default backtest result
            return BacktestResult(
                model_performance={},
                initial_weights=self.weights.copy(),
                validation_periods=0,
                total_data_points=len(data),
                backtest_date=datetime.now()
            )
    
    def _calculate_performance_based_weights(self, performance_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
        """Calculate model weights based on performance metrics"""
        try:
            # Use inverse MAPE for weight calculation (lower MAPE = higher weight)
            inverse_errors = {}
            
            for model_name, metrics in performance_metrics.items():
                if np.isfinite(metrics.mape) and metrics.mape > 0:
                    # Use inverse of MAPE, with minimum threshold to avoid extreme weights
                    inverse_errors[model_name] = 1.0 / max(metrics.mape, 1.0)
                else:
                    # Assign very low weight to models with infinite or zero MAPE
                    inverse_errors[model_name] = 0.01
            
            # Normalize to sum to 1.0
            total_inverse_error = sum(inverse_errors.values())
            
            if total_inverse_error > 0:
                weights = {name: error / total_inverse_error for name, error in inverse_errors.items()}
            else:
                # Fallback to equal weights
                num_models = len(self.models)
                weights = {name: 1.0 / num_models for name in self.models.keys()}
            
            # Ensure minimum weight threshold (5% minimum)
            min_weight = 0.05
            for name in weights:
                if weights[name] < min_weight:
                    weights[name] = min_weight
            
            # Renormalize after applying minimum weights
            total_weight = sum(weights.values())
            weights = {name: weight / total_weight for name, weight in weights.items()}
            
            logger.info(f"Calculated performance-based weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")
            # Return equal weights as fallback
            num_models = len(self.models)
            return {name: 1.0 / num_models for name in self.models.keys()}
    
    def fit_with_backtesting(self, data: pd.Series, validation_split: float = 0.3,
                           n_splits: int = 5, features_df: Optional[pd.DataFrame] = None):
        """
        Fit ensemble with backtesting to determine optimal initial weights
        
        Args:
            data: Historical time series data
            validation_split: Proportion of data for validation
            n_splits: Number of cross-validation splits
            features_df: Optional features for ML models
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info("Starting fit with backtesting...")
            
            # Run backtesting to get performance-based weights
            backtest_result = self.run_backtesting(data, validation_split, n_splits, features_df)
            
            # Update model weights based on backtesting results
            if backtest_result.initial_weights:
                self.weights = backtest_result.initial_weights
                logger.info(f"Updated weights based on backtesting: {self.weights}")
            
            # Fit all models on the full dataset
            self.fit(data, features_df)
            
            logger.info("Fit with backtesting completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Fit with backtesting failed: {e}")
            # Fallback to regular fit
            return self.fit(data, features_df)
    
    def get_performance_history(self, model_name: Optional[str] = None) -> Dict[str, List[PerformanceMetrics]]:
        """
        Get historical performance metrics
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Dictionary of performance history
        """
        if model_name:
            return {model_name: self.historical_performance.get(model_name, [])}
        return self.historical_performance.copy()
    
    def get_latest_performance(self) -> Dict[str, PerformanceMetrics]:
        """Get the most recent performance metrics for each model"""
        latest_performance = {}
        
        for model_name, metrics_list in self.historical_performance.items():
            if metrics_list:
                latest_performance[model_name] = metrics_list[-1]
        
        return latest_performance
    
    def get_backtest_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of backtesting results"""
        if not self.backtest_results:
            return None
        
        summary = {
            'backtest_date': self.backtest_results.backtest_date,
            'validation_periods': self.backtest_results.validation_periods,
            'total_data_points': self.backtest_results.total_data_points,
            'model_performance': {},
            'optimal_weights': self.backtest_results.initial_weights
        }
        
        # Add pattern information if available
        if self.backtest_results.detected_pattern:
            pattern = self.backtest_results.detected_pattern
            summary['detected_pattern'] = {
                'pattern_type': pattern.pattern_type,
                'seasonality_strength': pattern.seasonality_strength,
                'trend_strength': pattern.trend_strength,
                'intermittency_ratio': pattern.intermittency_ratio,
                'confidence': pattern.confidence,
                'detected_period': pattern.detected_period
            }
        
        # Add performance summary for each model
        for model_name, metrics in self.backtest_results.model_performance.items():
            summary['model_performance'][model_name] = {
                'mae': metrics.mae,
                'mape': metrics.mape,
                'rmse': metrics.rmse,
                'r_squared': metrics.r_squared,
                'data_points': metrics.data_points
            }
        
        return summary
    
    def check_pattern_change(self, new_data: pd.Series, 
                           rebalance_weights: bool = True) -> bool:
        """
        Check if data pattern has changed and optionally rebalance weights
        
        Args:
            new_data: New data to analyze for pattern changes
            rebalance_weights: Whether to automatically rebalance weights if change detected
            
        Returns:
            True if significant pattern change detected
        """
        if not self.pattern_detection_enabled or not self.pattern_detector or not self.current_pattern:
            return False
        
        try:
            # Detect pattern change
            pattern_changed = self.pattern_detector.detect_pattern_change(
                self.current_pattern, new_data, self.pattern_change_threshold
            )
            
            if pattern_changed:
                logger.info("Significant pattern change detected")
                
                # Update current pattern
                new_pattern = self.pattern_detector.detect_pattern(new_data)
                old_pattern_type = self.current_pattern.pattern_type
                self.current_pattern = new_pattern
                
                logger.info(f"Pattern changed from {old_pattern_type} to {new_pattern.pattern_type}")
                
                if rebalance_weights:
                    # Get new pattern-specific weights
                    new_weights = self.pattern_detector.get_pattern_specific_weights(
                        new_pattern.pattern_type, 
                        list(self.models.keys())
                    )
                    
                    # Apply smooth transition to avoid abrupt changes
                    self.weights = self._smooth_weight_transition(self.weights, new_weights)
                    
                    logger.info(f"Rebalanced weights for new pattern: {self.weights}")
                
                # Update last pattern check timestamp
                self.last_pattern_check = datetime.now()
                
                return True
            
            # Update last check timestamp even if no change
            self.last_pattern_check = datetime.now()
            return False
            
        except Exception as e:
            logger.error(f"Pattern change detection failed: {e}")
            return False
    
    def _smooth_weight_transition(self, old_weights: Dict[str, float], 
                                new_weights: Dict[str, float], 
                                smoothing_factor: float = 0.3) -> Dict[str, float]:
        """
        Apply smooth transition between old and new weights to avoid abrupt changes
        
        Args:
            old_weights: Current model weights
            new_weights: Target model weights
            smoothing_factor: How much to move toward new weights (0-1)
            
        Returns:
            Smoothed weights
        """
        try:
            smoothed_weights = {}
            
            for model_name in old_weights.keys():
                old_weight = old_weights.get(model_name, 0.0)
                new_weight = new_weights.get(model_name, 0.0)
                
                # Linear interpolation between old and new weights
                smoothed_weight = old_weight + smoothing_factor * (new_weight - old_weight)
                smoothed_weights[model_name] = smoothed_weight
            
            # Normalize to ensure weights sum to 1.0
            total_weight = sum(smoothed_weights.values())
            if total_weight > 0:
                smoothed_weights = {name: weight / total_weight 
                                  for name, weight in smoothed_weights.items()}
            else:
                # Fallback to equal weights
                num_models = len(old_weights)
                smoothed_weights = {name: 1.0 / num_models for name in old_weights.keys()}
            
            return smoothed_weights
            
        except Exception as e:
            logger.error(f"Weight smoothing failed: {e}")
            return old_weights.copy()
    
    def get_current_pattern(self) -> Optional[PatternCharacteristics]:
        """Get the currently detected pattern characteristics"""
        return self.current_pattern
    
    def set_pattern_change_threshold(self, threshold: float):
        """Set the threshold for detecting significant pattern changes"""
        if 0.0 <= threshold <= 1.0:
            self.pattern_change_threshold = threshold
            logger.info(f"Pattern change threshold set to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}. Must be between 0.0 and 1.0")
    
    def enable_pattern_detection(self, enabled: bool = True):
        """Enable or disable pattern detection"""
        self.pattern_detection_enabled = enabled
        if enabled and not self.pattern_detector:
            self.pattern_detector = PatternDetector()
        logger.info(f"Pattern detection {'enabled' if enabled else 'disabled'}")
    
    def get_pattern_weights_history(self) -> List[Dict[str, Any]]:
        """Get history of pattern-based weight changes"""
        # This would be implemented with proper weight change tracking
        # For now, return current state
        if self.current_pattern:
            return [{
                'timestamp': datetime.now(),
                'pattern_type': self.current_pattern.pattern_type,
                'weights': self.weights.copy(),
                'pattern_confidence': self.current_pattern.confidence
            }]
        return []
    
    def update_weights(self, performance_metrics: Dict[str, PerformanceMetrics], 
                      method: Optional[WeightUpdateMethod] = None) -> Dict[str, float]:
        """
        Update model weights using advanced algorithms based on recent performance
        
        Args:
            performance_metrics: Recent performance metrics for each model
            method: Weight update algorithm to use (uses config default if None)
            
        Returns:
            Updated weights dictionary
        """
        try:
            if method is None:
                method = self.weight_update_config.method
            
            logger.info(f"Updating weights using {method.value} method")
            
            # Store old weights for comparison and alerts
            old_weights = self.weights.copy()
            
            # Calculate new weights based on selected method
            if method == WeightUpdateMethod.INVERSE_ERROR:
                new_weights = self._calculate_inverse_error_weights(performance_metrics)
            elif method == WeightUpdateMethod.EXPONENTIAL_SMOOTHING:
                new_weights = self._calculate_exponential_smoothing_weights(performance_metrics)
            elif method == WeightUpdateMethod.RANK_BASED:
                new_weights = self._calculate_rank_based_weights(performance_metrics)
            else:
                logger.warning(f"Unknown weight update method: {method}")
                return self.weights.copy()
            
            # Apply weight update safeguards
            safeguarded_weights = self._apply_weight_safeguards(old_weights, new_weights)
            
            # Check for performance degradation and generate alerts
            self._check_performance_degradation(performance_metrics, old_weights, safeguarded_weights)
            
            # Update weights and log the change
            self.weights = safeguarded_weights
            self._log_weight_update(old_weights, safeguarded_weights, method, performance_metrics)
            
            logger.info(f"Weights updated successfully: {self.weights}")
            return self.weights.copy()
            
        except Exception as e:
            logger.error(f"Weight update failed: {e}")
            return self.weights.copy()
    
    def _calculate_inverse_error_weights(self, performance_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
        """Calculate weights using inverse error method (lower error = higher weight)"""
        try:
            inverse_errors = {}
            
            for model_name, metrics in performance_metrics.items():
                if model_name in self.models:
                    # Use MAPE as primary error metric, with fallback to MAE
                    error = metrics.mape if np.isfinite(metrics.mape) and metrics.mape > 0 else metrics.mae
                    
                    if np.isfinite(error) and error > 0:
                        # Use inverse of error with minimum threshold to avoid extreme weights
                        inverse_errors[model_name] = 1.0 / max(error, 0.1)
                    else:
                        # Assign very low weight to models with infinite or zero error
                        inverse_errors[model_name] = 0.01
            
            # Normalize to sum to 1.0
            total_inverse_error = sum(inverse_errors.values())
            
            if total_inverse_error > 0:
                weights = {name: error / total_inverse_error for name, error in inverse_errors.items()}
            else:
                # Fallback to equal weights
                weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Inverse error weight calculation failed: {e}")
            return {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def _calculate_exponential_smoothing_weights(self, performance_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
        """Calculate weights using exponential smoothing method"""
        try:
            alpha = 2.0  # Exponential smoothing parameter
            exp_weights = {}
            
            for model_name, metrics in performance_metrics.items():
                if model_name in self.models:
                    # Use MAPE as primary error metric
                    error = metrics.mape if np.isfinite(metrics.mape) and metrics.mape > 0 else metrics.mae
                    
                    if np.isfinite(error):
                        # Exponential decay based on error (lower error = higher weight)
                        exp_weights[model_name] = np.exp(-alpha * max(error / 100.0, 0.01))
                    else:
                        exp_weights[model_name] = 0.01
            
            # Normalize to sum to 1.0
            total_exp_weight = sum(exp_weights.values())
            
            if total_exp_weight > 0:
                weights = {name: weight / total_exp_weight for name, weight in exp_weights.items()}
            else:
                weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Exponential smoothing weight calculation failed: {e}")
            return {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def _calculate_rank_based_weights(self, performance_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, float]:
        """Calculate weights using rank-based method (best performing model gets highest weight)"""
        try:
            # Create list of (model_name, error) tuples for ranking
            model_errors = []
            
            for model_name, metrics in performance_metrics.items():
                if model_name in self.models:
                    error = metrics.mape if np.isfinite(metrics.mape) and metrics.mape > 0 else metrics.mae
                    
                    if np.isfinite(error):
                        model_errors.append((model_name, error))
                    else:
                        model_errors.append((model_name, float('inf')))
            
            # Sort by error (ascending - lower error is better)
            model_errors.sort(key=lambda x: x[1])
            
            # Assign weights based on rank (best model gets highest weight)
            n_models = len(model_errors)
            rank_weights = {}
            
            for rank, (model_name, _) in enumerate(model_errors):
                # Linear weight assignment: best model gets n, worst gets 1
                rank_weights[model_name] = n_models - rank
            
            # Normalize to sum to 1.0
            total_rank_weight = sum(rank_weights.values())
            
            if total_rank_weight > 0:
                weights = {name: weight / total_rank_weight for name, weight in rank_weights.items()}
            else:
                weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Rank-based weight calculation failed: {e}")
            return {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def _apply_weight_safeguards(self, old_weights: Dict[str, float], 
                               new_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply weight update safeguards to prevent extreme changes and ensure stability
        
        Args:
            old_weights: Current model weights
            new_weights: Proposed new weights
            
        Returns:
            Safeguarded weights
        """
        try:
            config = self.weight_update_config
            safeguarded_weights = {}
            
            for model_name in self.models.keys():
                old_weight = old_weights.get(model_name, 1.0 / len(self.models))
                new_weight = new_weights.get(model_name, 1.0 / len(self.models))
                
                # Apply maximum weight change limit
                max_change = config.max_weight_change
                weight_change = new_weight - old_weight
                
                if abs(weight_change) > max_change:
                    # Limit the change to maximum allowed
                    limited_change = max_change if weight_change > 0 else -max_change
                    safeguarded_weight = old_weight + limited_change
                else:
                    safeguarded_weight = new_weight
                
                # Apply min/max weight limits
                safeguarded_weight = max(config.min_weight, 
                                       min(config.max_weight, safeguarded_weight))
                
                safeguarded_weights[model_name] = safeguarded_weight
            
            # Apply smooth transitions using exponential smoothing
            smoothed_weights = {}
            smoothing_factor = config.smoothing_factor
            
            for model_name in self.models.keys():
                old_weight = old_weights.get(model_name, 1.0 / len(self.models))
                safeguarded_weight = safeguarded_weights[model_name]
                
                # Exponential smoothing: new = α * target + (1-α) * old
                smoothed_weight = (smoothing_factor * safeguarded_weight + 
                                 (1 - smoothing_factor) * old_weight)
                smoothed_weights[model_name] = smoothed_weight
            
            # Final normalization to ensure weights sum to 1.0
            total_weight = sum(smoothed_weights.values())
            if total_weight > 0:
                normalized_weights = {name: weight / total_weight 
                                    for name, weight in smoothed_weights.items()}
            else:
                # Fallback to equal weights
                normalized_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Weight safeguarding failed: {e}")
            return old_weights.copy()
    
    def _check_performance_degradation(self, performance_metrics: Dict[str, PerformanceMetrics],
                                     old_weights: Dict[str, float], 
                                     new_weights: Dict[str, float]):
        """
        Check for performance degradation and generate alerts if needed
        
        Args:
            performance_metrics: Recent performance metrics
            old_weights: Previous weights
            new_weights: New weights after update
        """
        try:
            if not self.weight_update_config.enable_alerts:
                return
            
            config = self.weight_update_config
            current_time = datetime.now()
            
            # Check for significant weight changes
            for model_name in self.models.keys():
                old_weight = old_weights.get(model_name, 0.0)
                new_weight = new_weights.get(model_name, 0.0)
                weight_change = abs(new_weight - old_weight)
                
                if weight_change > config.degradation_threshold:
                    alert = {
                        'timestamp': current_time,
                        'alert_type': 'significant_weight_change',
                        'model_name': model_name,
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'weight_change': weight_change,
                        'threshold': config.degradation_threshold,
                        'performance_metrics': performance_metrics.get(model_name)
                    }
                    
                    self.performance_alerts.append(alert)
                    logger.warning(f"Significant weight change alert for {model_name}: "
                                 f"{old_weight:.3f} -> {new_weight:.3f} "
                                 f"(change: {weight_change:.3f})")
            
            # Check for overall performance degradation
            if len(self.historical_performance) > 0:
                for model_name, metrics in performance_metrics.items():
                    if model_name in self.historical_performance:
                        historical_metrics = self.historical_performance[model_name]
                        
                        if len(historical_metrics) >= 2:
                            # Compare with recent historical performance
                            recent_avg_error = np.mean([m.mape for m in historical_metrics[-3:] 
                                                      if np.isfinite(m.mape)])
                            current_error = metrics.mape if np.isfinite(metrics.mape) else metrics.mae
                            
                            if (np.isfinite(recent_avg_error) and np.isfinite(current_error) and 
                                current_error > recent_avg_error * (1 + config.degradation_threshold)):
                                
                                alert = {
                                    'timestamp': current_time,
                                    'alert_type': 'performance_degradation',
                                    'model_name': model_name,
                                    'current_error': current_error,
                                    'recent_avg_error': recent_avg_error,
                                    'degradation_ratio': current_error / recent_avg_error,
                                    'threshold': config.degradation_threshold
                                }
                                
                                self.performance_alerts.append(alert)
                                logger.warning(f"Performance degradation alert for {model_name}: "
                                             f"Error increased from {recent_avg_error:.2f} to {current_error:.2f}")
            
        except Exception as e:
            logger.error(f"Performance degradation check failed: {e}")
    
    def _log_weight_update(self, old_weights: Dict[str, float], new_weights: Dict[str, float],
                          method: WeightUpdateMethod, performance_metrics: Dict[str, PerformanceMetrics]):
        """Log weight update details for audit trail"""
        try:
            weight_update_log = {
                'timestamp': datetime.now(),
                'update_method': method.value,
                'old_weights': old_weights.copy(),
                'new_weights': new_weights.copy(),
                'weight_changes': {name: new_weights[name] - old_weights.get(name, 0.0) 
                                 for name in new_weights.keys()},
                'performance_metrics': {name: {
                    'mae': metrics.mae,
                    'mape': metrics.mape,
                    'rmse': metrics.rmse,
                    'r_squared': metrics.r_squared
                } for name, metrics in performance_metrics.items()},
                'pattern_type': self.current_pattern.pattern_type if self.current_pattern else None
            }
            
            self.weight_history.append(weight_update_log)
            
            # Keep only recent history (last 100 updates)
            if len(self.weight_history) > 100:
                self.weight_history = self.weight_history[-100:]
                
        except Exception as e:
            logger.error(f"Weight update logging failed: {e}")
    
    def get_weight_update_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of weight updates
        
        Args:
            limit: Maximum number of recent updates to return
            
        Returns:
            List of weight update records
        """
        if limit:
            return self.weight_history[-limit:]
        return self.weight_history.copy()
    
    def get_performance_alerts(self, alert_type: Optional[str] = None, 
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get performance alerts
        
        Args:
            alert_type: Filter by alert type ('significant_weight_change', 'performance_degradation')
            limit: Maximum number of recent alerts to return
            
        Returns:
            List of performance alerts
        """
        alerts = self.performance_alerts
        
        if alert_type:
            alerts = [alert for alert in alerts if alert.get('alert_type') == alert_type]
        
        if limit:
            alerts = alerts[-limit:]
            
        return alerts
    
    def configure_weight_updates(self, config: WeightUpdateConfig):
        """Update weight update configuration"""
        self.weight_update_config = config
        logger.info(f"Weight update configuration updated: {config}")
    
    def update_with_actuals(self, actual_values: pd.Series, predicted_values: Dict[str, pd.Series],
                          forecast_dates: Optional[pd.DatetimeIndex] = None,
                          auto_update_weights: bool = True):
        """
        Update performance tracking with actual vs predicted values and optionally update weights
        
        Args:
            actual_values: Actual observed values
            predicted_values: Dictionary of model predictions
            forecast_dates: Optional dates for the forecasts
            auto_update_weights: Whether to automatically update weights based on performance
        """
        try:
            logger.info("Updating performance tracking with actual values")
            
            # Calculate performance metrics for each model
            current_performance = {}
            
            for model_name, predictions in predicted_values.items():
                if model_name in self.models:
                    metrics = self.calculate_error_metrics(actual_values, predictions, model_name)
                    current_performance[model_name] = metrics
                    
                    # Add to historical performance
                    self.historical_performance[model_name].append(metrics)
                    
                    # Keep only recent history (last 50 evaluations per model)
                    if len(self.historical_performance[model_name]) > 50:
                        self.historical_performance[model_name] = self.historical_performance[model_name][-50:]
            
            # Update weights if auto-update is enabled
            if auto_update_weights and current_performance:
                self.update_weights(current_performance)
            
            # Check for pattern changes if pattern detection is enabled
            if self.pattern_detection_enabled and len(actual_values) >= 12:
                self.check_pattern_change(actual_values, rebalance_weights=auto_update_weights)
            
            logger.info("Performance tracking update completed")
            
        except Exception as e:
            logger.error(f"Performance tracking update failed: {e}")