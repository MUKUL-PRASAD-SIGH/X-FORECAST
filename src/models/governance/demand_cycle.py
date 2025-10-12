"""
Weekly Demand Cycle Governance Workflow
Implements automated baseline → exception list → demand review → minimal human overrides
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CyclePhase(Enum):
    BASELINE_GENERATION = "baseline_generation"
    EXCEPTION_DETECTION = "exception_detection"
    DEMAND_REVIEW = "demand_review"
    OVERRIDE_APPROVAL = "override_approval"
    FORECAST_FINALIZATION = "forecast_finalization"
    PERFORMANCE_MONITORING = "performance_monitoring"

class ExceptionType(Enum):
    HIGH_FORECAST_ERROR = "high_forecast_error"
    SIGNIFICANT_BIAS = "significant_bias"
    UNUSUAL_PATTERN = "unusual_pattern"
    NEW_PRODUCT_LAUNCH = "new_product_launch"
    PROMOTIONAL_EVENT = "promotional_event"
    SUPPLY_CONSTRAINT = "supply_constraint"
    MARKET_DISRUPTION = "market_disruption"
    SEASONAL_ANOMALY = "seasonal_anomaly"

class ExceptionSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ForecastException:
    """Represents an exception requiring human review"""
    exception_id: str
    sku: str
    location: str
    channel: str
    exception_type: ExceptionType
    severity: ExceptionSeverity
    detected_at: datetime
    description: str
    current_forecast: float
    suggested_adjustment: Optional[float]
    confidence: float
    supporting_data: Dict[str, Any]
    requires_approval: bool
    assigned_reviewer: Optional[str] = None
    review_deadline: Optional[datetime] = None
    status: str = "pending"

@dataclass
class CycleMetrics:
    """Metrics for demand cycle performance"""
    total_forecasts_generated: int
    exceptions_detected: int
    exception_rate: float
    auto_approved_rate: float
    human_review_rate: float
    cycle_completion_time: float  # hours
    forecast_accuracy_improvement: float
    sla_compliance_rate: float

@dataclass
class DemandCycleResult:
    """Result of a complete demand cycle"""
    cycle_id: str
    start_time: datetime
    end_time: datetime
    phase: CyclePhase
    baseline_forecasts: Dict[str, pd.Series]
    exceptions: List[ForecastException]
    final_forecasts: Dict[str, pd.Series]
    metrics: CycleMetrics
    approvals_pending: int
    next_cycle_date: datetime

class DemandCycleEngine:
    """
    Automated Weekly Demand Cycle Engine
    Orchestrates the complete forecast governance workflow
    """
    
    def __init__(self):
        self.exception_thresholds = {
            ExceptionType.HIGH_FORECAST_ERROR: 0.15,  # 15% error threshold
            ExceptionType.SIGNIFICANT_BIAS: 0.10,     # 10% bias threshold
            ExceptionType.UNUSUAL_PATTERN: 2.0,       # 2 std dev threshold
        }
        
        self.auto_approval_rules = {
            'max_adjustment_percentage': 0.05,  # Auto-approve up to 5% adjustments
            'min_confidence_threshold': 0.8,    # Require 80% confidence
            'max_forecast_value': 10000,        # Auto-approve only below this value
        }
        
        self.sla_targets = {
            'cycle_completion_hours': 48,       # Complete cycle in 48 hours
            'exception_review_hours': 24,       # Review exceptions in 24 hours
            'approval_response_hours': 8,       # Approve/reject in 8 hours
        }
        
        self.current_cycle = None
        self.cycle_history = []
        
    async def run_weekly_cycle(self, data: pd.DataFrame) -> DemandCycleResult:
        """Execute complete weekly demand cycle"""
        
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting demand cycle {cycle_id}")
        
        try:
            # Phase 1: Generate baseline forecasts
            baseline_forecasts = await self._generate_baseline_forecasts(data)
            
            # Phase 2: Detect exceptions
            exceptions = await self._detect_exceptions(baseline_forecasts, data)
            
            # Phase 3: Conduct demand review
            reviewed_exceptions = await self._conduct_demand_review(exceptions)
            
            # Phase 4: Process overrides and approvals
            final_forecasts = await self._process_overrides(baseline_forecasts, reviewed_exceptions)
            
            # Phase 5: Finalize forecasts
            await self._finalize_forecasts(final_forecasts)
            
            # Calculate metrics
            metrics = self._calculate_cycle_metrics(baseline_forecasts, exceptions, final_forecasts, start_time)
            
            end_time = datetime.now()
            
            result = DemandCycleResult(
                cycle_id=cycle_id,
                start_time=start_time,
                end_time=end_time,
                phase=CyclePhase.FORECAST_FINALIZATION,
                baseline_forecasts=baseline_forecasts,
                exceptions=reviewed_exceptions,
                final_forecasts=final_forecasts,
                metrics=metrics,
                approvals_pending=sum(1 for e in reviewed_exceptions if e.status == "pending"),
                next_cycle_date=start_time + timedelta(days=7)
            )
            
            self.current_cycle = result
            self.cycle_history.append(result)
            
            logger.info(f"Completed demand cycle {cycle_id} in {(end_time - start_time).total_seconds() / 3600:.2f} hours")
            
            return result
            
        except Exception as e:
            logger.error(f"Demand cycle {cycle_id} failed: {e}")
            raise
    
    async def _generate_baseline_forecasts(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Phase 1: Generate baseline forecasts using ensemble models"""
        
        logger.info("Generating baseline forecasts...")
        
        from ..ensemble import EnsembleForecaster
        from ..hierarchical.hierarchical_forecaster import HierarchicalForecaster
        
        baseline_forecasts = {}
        
        # Use hierarchical forecasting for coherent forecasts
        hierarchical_forecaster = HierarchicalForecaster()
        hierarchical_result = hierarchical_forecaster.forecast_hierarchical(data, horizon=12)
        
        # Use reconciled forecasts as baseline
        baseline_forecasts = hierarchical_result.reconciled_forecasts
        
        logger.info(f"Generated {len(baseline_forecasts)} baseline forecasts")
        
        return baseline_forecasts
    
    async def _detect_exceptions(self, forecasts: Dict[str, pd.Series], data: pd.DataFrame) -> List[ForecastException]:
        """Phase 2: Detect exceptions requiring human review"""
        
        logger.info("Detecting forecast exceptions...")
        
        exceptions = []
        
        for forecast_key, forecast in forecasts.items():
            try:
                # Parse forecast key (assuming format: sku_location_channel)
                parts = forecast_key.split('_')
                if len(parts) >= 3:
                    sku, location, channel = parts[0], parts[1], parts[2]
                else:
                    sku, location, channel = forecast_key, "all", "all"
                
                # Get historical data for this combination
                historical_data = self._get_historical_data(data, sku, location, channel)
                
                if len(historical_data) == 0:
                    continue
                
                # Check for various exception types
                detected_exceptions = []
                
                # 1. High forecast error (based on recent performance)
                if len(historical_data) >= 12:
                    recent_error = self._calculate_recent_forecast_error(historical_data, forecast)
                    if recent_error > self.exception_thresholds[ExceptionType.HIGH_FORECAST_ERROR]:
                        detected_exceptions.append((ExceptionType.HIGH_FORECAST_ERROR, ExceptionSeverity.HIGH))
                
                # 2. Significant bias
                bias = self._calculate_forecast_bias(historical_data, forecast)
                if abs(bias) > self.exception_thresholds[ExceptionType.SIGNIFICANT_BIAS]:
                    severity = ExceptionSeverity.HIGH if abs(bias) > 0.2 else ExceptionSeverity.MEDIUM
                    detected_exceptions.append((ExceptionType.SIGNIFICANT_BIAS, severity))
                
                # 3. Unusual pattern detection
                if self._detect_unusual_pattern(historical_data, forecast):
                    detected_exceptions.append((ExceptionType.UNUSUAL_PATTERN, ExceptionSeverity.MEDIUM))
                
                # 4. New product launch detection
                if self._is_new_product_launch(historical_data, sku):
                    detected_exceptions.append((ExceptionType.NEW_PRODUCT_LAUNCH, ExceptionSeverity.HIGH))
                
                # 5. Promotional event detection
                if self._detect_promotional_event(historical_data, forecast):
                    detected_exceptions.append((ExceptionType.PROMOTIONAL_EVENT, ExceptionSeverity.MEDIUM))
                
                # Create exception objects
                for exception_type, severity in detected_exceptions:
                    exception = ForecastException(
                        exception_id=f"exc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(exceptions)}",
                        sku=sku,
                        location=location,
                        channel=channel,
                        exception_type=exception_type,
                        severity=severity,
                        detected_at=datetime.now(),
                        description=self._generate_exception_description(exception_type, severity),
                        current_forecast=forecast.iloc[0] if len(forecast) > 0 else 0,
                        suggested_adjustment=self._calculate_suggested_adjustment(historical_data, forecast, exception_type),
                        confidence=self._calculate_exception_confidence(exception_type, historical_data),
                        supporting_data=self._gather_supporting_data(historical_data, forecast, exception_type),
                        requires_approval=severity in [ExceptionSeverity.HIGH, ExceptionSeverity.CRITICAL],
                        review_deadline=datetime.now() + timedelta(hours=self.sla_targets['exception_review_hours'])
                    )
                    exceptions.append(exception)
                    
            except Exception as e:
                logger.warning(f"Exception detection failed for {forecast_key}: {e}")
        
        logger.info(f"Detected {len(exceptions)} exceptions")
        
        return exceptions
    
    def _get_historical_data(self, data: pd.DataFrame, sku: str, location: str, channel: str) -> pd.DataFrame:
        """Get historical data for specific SKU/location/channel combination"""
        
        mask = pd.Series([True] * len(data))
        
        if sku != "all" and 'sku' in data.columns:
            mask &= (data['sku'] == sku)
        if location != "all" and 'location' in data.columns:
            mask &= (data['location'] == location)
        if channel != "all" and 'channel' in data.columns:
            mask &= (data['channel'] == channel)
        
        return data[mask].sort_values('date') if 'date' in data.columns else data[mask]
    
    def _calculate_recent_forecast_error(self, historical_data: pd.DataFrame, forecast: pd.Series) -> float:
        """Calculate recent forecast error for exception detection"""
        
        if len(historical_data) < 12 or 'demand' not in historical_data.columns:
            return 0.0
        
        # Use last 12 periods for error calculation
        recent_actual = historical_data['demand'].tail(12)
        
        # Simulate what the forecast would have been (mock implementation)
        # In real implementation, this would use stored historical forecasts
        simulated_forecast = recent_actual.mean() * np.ones(len(recent_actual))
        
        # Calculate MAPE
        mape = np.mean(np.abs((recent_actual - simulated_forecast) / recent_actual))
        
        return mape
    
    def _calculate_forecast_bias(self, historical_data: pd.DataFrame, forecast: pd.Series) -> float:
        """Calculate forecast bias"""
        
        if len(historical_data) < 6 or 'demand' not in historical_data.columns:
            return 0.0
        
        recent_actual = historical_data['demand'].tail(6)
        avg_actual = recent_actual.mean()
        avg_forecast = forecast.mean() if len(forecast) > 0 else avg_actual
        
        bias = (avg_forecast - avg_actual) / avg_actual if avg_actual != 0 else 0
        
        return bias
    
    def _detect_unusual_pattern(self, historical_data: pd.DataFrame, forecast: pd.Series) -> bool:
        """Detect unusual patterns in forecast vs historical data"""
        
        if len(historical_data) < 12 or 'demand' not in historical_data.columns:
            return False
        
        recent_demand = historical_data['demand'].tail(12)
        demand_std = recent_demand.std()
        demand_mean = recent_demand.mean()
        
        if demand_std == 0:
            return False
        
        # Check if forecast is more than 2 standard deviations from recent mean
        forecast_mean = forecast.mean() if len(forecast) > 0 else demand_mean
        z_score = abs(forecast_mean - demand_mean) / demand_std
        
        return z_score > 2.0
    
    def _is_new_product_launch(self, historical_data: pd.DataFrame, sku: str) -> bool:
        """Detect if this is a new product launch"""
        
        # Simple heuristic: if less than 3 months of data, consider it new
        return len(historical_data) < 12
    
    def _detect_promotional_event(self, historical_data: pd.DataFrame, forecast: pd.Series) -> bool:
        """Detect promotional events"""
        
        if len(historical_data) < 6 or 'demand' not in historical_data.columns:
            return False
        
        recent_demand = historical_data['demand'].tail(6)
        baseline_demand = recent_demand.median()
        forecast_peak = forecast.max() if len(forecast) > 0 else baseline_demand
        
        # If forecast shows >50% increase, might be promotional
        return forecast_peak > baseline_demand * 1.5
    
    def _generate_exception_description(self, exception_type: ExceptionType, severity: ExceptionSeverity) -> str:
        """Generate human-readable exception description"""
        
        descriptions = {
            ExceptionType.HIGH_FORECAST_ERROR: f"Forecast error exceeds threshold ({severity.value} severity)",
            ExceptionType.SIGNIFICANT_BIAS: f"Systematic forecast bias detected ({severity.value} severity)",
            ExceptionType.UNUSUAL_PATTERN: f"Unusual demand pattern identified ({severity.value} severity)",
            ExceptionType.NEW_PRODUCT_LAUNCH: f"New product launch detected ({severity.value} severity)",
            ExceptionType.PROMOTIONAL_EVENT: f"Promotional event identified ({severity.value} severity)",
            ExceptionType.SUPPLY_CONSTRAINT: f"Supply constraint impact ({severity.value} severity)",
            ExceptionType.MARKET_DISRUPTION: f"Market disruption detected ({severity.value} severity)",
            ExceptionType.SEASONAL_ANOMALY: f"Seasonal pattern anomaly ({severity.value} severity)"
        }
        
        return descriptions.get(exception_type, f"Exception detected ({severity.value} severity)")
    
    def _calculate_suggested_adjustment(self, historical_data: pd.DataFrame, 
                                      forecast: pd.Series, exception_type: ExceptionType) -> Optional[float]:
        """Calculate suggested forecast adjustment"""
        
        if len(forecast) == 0:
            return None
        
        current_forecast = forecast.iloc[0]
        
        if exception_type == ExceptionType.HIGH_FORECAST_ERROR:
            # Suggest adjustment towards recent average
            if len(historical_data) > 0 and 'demand' in historical_data.columns:
                recent_avg = historical_data['demand'].tail(6).mean()
                return (recent_avg + current_forecast) / 2
        
        elif exception_type == ExceptionType.SIGNIFICANT_BIAS:
            # Adjust for bias
            bias = self._calculate_forecast_bias(historical_data, forecast)
            return current_forecast * (1 - bias)
        
        elif exception_type == ExceptionType.PROMOTIONAL_EVENT:
            # Suggest promotional uplift
            return current_forecast * 1.3
        
        return None
    
    def _calculate_exception_confidence(self, exception_type: ExceptionType, historical_data: pd.DataFrame) -> float:
        """Calculate confidence in exception detection"""
        
        # Base confidence levels by exception type
        base_confidence = {
            ExceptionType.HIGH_FORECAST_ERROR: 0.8,
            ExceptionType.SIGNIFICANT_BIAS: 0.7,
            ExceptionType.UNUSUAL_PATTERN: 0.6,
            ExceptionType.NEW_PRODUCT_LAUNCH: 0.9,
            ExceptionType.PROMOTIONAL_EVENT: 0.7,
        }
        
        confidence = base_confidence.get(exception_type, 0.5)
        
        # Adjust based on data quality
        if len(historical_data) > 24:
            confidence += 0.1  # More data = higher confidence
        elif len(historical_data) < 6:
            confidence -= 0.2  # Less data = lower confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _gather_supporting_data(self, historical_data: pd.DataFrame, 
                              forecast: pd.Series, exception_type: ExceptionType) -> Dict[str, Any]:
        """Gather supporting data for exception"""
        
        supporting_data = {
            "historical_periods": len(historical_data),
            "forecast_horizon": len(forecast),
            "detection_timestamp": datetime.now().isoformat()
        }
        
        if len(historical_data) > 0 and 'demand' in historical_data.columns:
            supporting_data.update({
                "recent_demand_avg": historical_data['demand'].tail(6).mean(),
                "recent_demand_std": historical_data['demand'].tail(6).std(),
                "demand_trend": "increasing" if historical_data['demand'].tail(6).mean() > historical_data['demand'].head(6).mean() else "decreasing"
            })
        
        if len(forecast) > 0:
            supporting_data.update({
                "forecast_avg": forecast.mean(),
                "forecast_std": forecast.std(),
                "forecast_peak": forecast.max(),
                "forecast_min": forecast.min()
            })
        
        return supporting_data
    
    async def _conduct_demand_review(self, exceptions: List[ForecastException]) -> List[ForecastException]:
        """Phase 3: Conduct automated and human demand review"""
        
        logger.info(f"Conducting demand review for {len(exceptions)} exceptions")
        
        reviewed_exceptions = []
        
        for exception in exceptions:
            # Auto-approve low-severity exceptions with high confidence
            if (exception.severity == ExceptionSeverity.LOW and 
                exception.confidence > 0.8 and
                exception.suggested_adjustment is not None):
                
                adjustment_pct = abs(exception.suggested_adjustment - exception.current_forecast) / exception.current_forecast
                
                if adjustment_pct <= self.auto_approval_rules['max_adjustment_percentage']:
                    exception.status = "auto_approved"
                    exception.assigned_reviewer = "system"
                    logger.info(f"Auto-approved exception {exception.exception_id}")
                else:
                    exception.status = "pending_review"
                    exception.assigned_reviewer = self._assign_reviewer(exception)
            
            else:
                exception.status = "pending_review"
                exception.assigned_reviewer = self._assign_reviewer(exception)
            
            reviewed_exceptions.append(exception)
        
        return reviewed_exceptions
    
    def _assign_reviewer(self, exception: ForecastException) -> str:
        """Assign reviewer based on exception characteristics"""
        
        # Simple assignment logic (in real implementation, would use workload balancing)
        if exception.exception_type == ExceptionType.NEW_PRODUCT_LAUNCH:
            return "product_manager"
        elif exception.exception_type == ExceptionType.PROMOTIONAL_EVENT:
            return "marketing_analyst"
        elif exception.severity == ExceptionSeverity.CRITICAL:
            return "senior_analyst"
        else:
            return "demand_planner"
    
    async def _process_overrides(self, baseline_forecasts: Dict[str, pd.Series], 
                                exceptions: List[ForecastException]) -> Dict[str, pd.Series]:
        """Phase 4: Process overrides and create final forecasts"""
        
        logger.info("Processing overrides and finalizing forecasts")
        
        final_forecasts = baseline_forecasts.copy()
        
        # Apply approved adjustments
        for exception in exceptions:
            if exception.status == "auto_approved" and exception.suggested_adjustment is not None:
                forecast_key = f"{exception.sku}_{exception.location}_{exception.channel}"
                
                if forecast_key in final_forecasts:
                    # Apply adjustment to first period (can be extended to full horizon)
                    adjustment_factor = exception.suggested_adjustment / exception.current_forecast
                    final_forecasts[forecast_key] = final_forecasts[forecast_key] * adjustment_factor
                    
                    logger.info(f"Applied adjustment to {forecast_key}: {adjustment_factor:.3f}")
        
        return final_forecasts
    
    async def _finalize_forecasts(self, final_forecasts: Dict[str, pd.Series]):
        """Phase 5: Finalize and publish forecasts"""
        
        logger.info(f"Finalizing {len(final_forecasts)} forecasts")
        
        # In real implementation, this would:
        # 1. Store forecasts in database
        # 2. Trigger downstream systems
        # 3. Generate forecast packages
        # 4. Send notifications
        
        # For now, just log completion
        logger.info("Forecasts finalized and published")
    
    def _calculate_cycle_metrics(self, baseline_forecasts: Dict[str, pd.Series], 
                                exceptions: List[ForecastException],
                                final_forecasts: Dict[str, pd.Series],
                                start_time: datetime) -> CycleMetrics:
        """Calculate performance metrics for the cycle"""
        
        total_forecasts = len(baseline_forecasts)
        total_exceptions = len(exceptions)
        auto_approved = sum(1 for e in exceptions if e.status == "auto_approved")
        pending_review = sum(1 for e in exceptions if e.status == "pending_review")
        
        cycle_time = (datetime.now() - start_time).total_seconds() / 3600  # hours
        
        return CycleMetrics(
            total_forecasts_generated=total_forecasts,
            exceptions_detected=total_exceptions,
            exception_rate=total_exceptions / total_forecasts if total_forecasts > 0 else 0,
            auto_approved_rate=auto_approved / total_exceptions if total_exceptions > 0 else 0,
            human_review_rate=pending_review / total_exceptions if total_exceptions > 0 else 0,
            cycle_completion_time=cycle_time,
            forecast_accuracy_improvement=0.02,  # Mock 2% improvement
            sla_compliance_rate=0.95 if cycle_time <= self.sla_targets['cycle_completion_hours'] else 0.8
        )
    
    def get_cycle_status(self) -> Optional[Dict[str, Any]]:
        """Get current cycle status"""
        
        if not self.current_cycle:
            return None
        
        return {
            "cycle_id": self.current_cycle.cycle_id,
            "phase": self.current_cycle.phase.value,
            "start_time": self.current_cycle.start_time.isoformat(),
            "elapsed_hours": (datetime.now() - self.current_cycle.start_time).total_seconds() / 3600,
            "exceptions_pending": self.current_cycle.approvals_pending,
            "next_cycle": self.current_cycle.next_cycle_date.isoformat(),
            "sla_compliance": self.current_cycle.metrics.sla_compliance_rate
        }