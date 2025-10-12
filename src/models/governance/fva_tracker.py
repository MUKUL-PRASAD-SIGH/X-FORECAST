"""
Enhanced Forecast Value Add (FVA) Tracking System
Measures the impact of human overrides on forecast accuracy with advanced analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .fva_database import FVADataAccessLayer

logger = logging.getLogger(__name__)

class OverrideType(Enum):
    MANUAL_ADJUSTMENT = "manual_adjustment"
    PROMOTIONAL_LIFT = "promotional_lift"
    MARKET_INTELLIGENCE = "market_intelligence"
    SUPPLY_CONSTRAINT = "supply_constraint"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"
    NEW_PRODUCT_LAUNCH = "new_product_launch"
    COMPETITIVE_RESPONSE = "competitive_response"

@dataclass
class ForecastOverride:
    """Represents a human override to the forecast"""
    override_id: str
    timestamp: datetime
    user_id: str
    sku: str
    location: str
    channel: str
    period: str
    original_forecast: float
    adjusted_forecast: float
    override_type: OverrideType
    reason: str
    confidence: float  # 0-1
    business_justification: str
    expected_impact: float
    approval_status: str
    approver_id: Optional[str] = None

@dataclass
class FVAMetrics:
    """Enhanced FVA performance metrics"""
    total_overrides: int
    override_rate: float  # % of forecasts overridden
    avg_override_magnitude: float
    fva_accuracy_improvement: float  # % improvement vs baseline
    fva_bias_reduction: float
    positive_fva_rate: float  # % of overrides that improved accuracy
    override_type_performance: Dict[OverrideType, float]
    user_performance: Dict[str, float]
    time_to_decision: float  # avg hours from forecast to override
    override_persistence: float  # how long overrides remain active
    
    # Enhanced metrics
    seasonal_fva_performance: Dict[str, float]  # FVA by season/month
    horizon_fva_performance: Dict[int, float]  # FVA by forecast horizon
    confidence_calibration: float  # How well confidence matches actual performance
    fva_volatility: float  # Consistency of FVA performance
    cumulative_value_add: float  # Total business value added
    
@dataclass
class FVAConfidenceInterval:
    """Confidence intervals for FVA measurements"""
    metric_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    sample_size: int

@dataclass
class FVAReport:
    """Comprehensive FVA analysis report"""
    period: str
    metrics: FVAMetrics
    top_performers: List[str]
    improvement_opportunities: List[str]
    override_patterns: Dict[str, Any]
    recommendations: List[str]
    forecast_accuracy_baseline: float
    forecast_accuracy_with_fva: float

class FVATracker:
    """
    Enhanced Tracks and analyzes Forecast Value Add from human interventions
    """
    
    def __init__(self, database_url: str = "sqlite:///fva_tracking.db"):
        self.overrides_db = []
        self.baseline_forecasts = {}
        self.actual_demand = {}
        self.performance_cache = {}
        
        # Enhanced features
        self.data_access = FVADataAccessLayer(database_url)
        self.seasonal_patterns = {}
        self.horizon_adjustments = {}
        self.confidence_models = {}
        
    def record_override(self, override: ForecastOverride) -> str:
        """Record a forecast override"""
        self.overrides_db.append(override)
        logger.info(f"Recorded override {override.override_id} by {override.user_id}")
        return override.override_id
    
    def calculate_baseline_accuracy(self, forecasts: Dict[str, pd.Series], 
                                  actuals: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate baseline forecast accuracy without overrides"""
        accuracy_metrics = {}
        
        for key in forecasts.keys():
            if key in actuals:
                forecast = forecasts[key]
                actual = actuals[key]
                
                # Align series
                common_index = forecast.index.intersection(actual.index)
                if len(common_index) > 0:
                    f_aligned = forecast.loc[common_index]
                    a_aligned = actual.loc[common_index]
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((a_aligned - f_aligned) / a_aligned)) * 100
                    accuracy_metrics[key] = mape
        
        return accuracy_metrics
    
    def calculate_fva_impact(self, period_start: datetime, period_end: datetime) -> FVAMetrics:
        """Calculate FVA impact for a specific period"""
        
        # Filter overrides for the period
        period_overrides = [
            override for override in self.overrides_db
            if period_start <= override.timestamp <= period_end
        ]
        
        if not period_overrides:
            return FVAMetrics(
                total_overrides=0,
                override_rate=0.0,
                avg_override_magnitude=0.0,
                fva_accuracy_improvement=0.0,
                fva_bias_reduction=0.0,
                positive_fva_rate=0.0,
                override_type_performance={},
                user_performance={},
                time_to_decision=0.0,
                override_persistence=0.0
            )
        
        # Calculate basic metrics
        total_overrides = len(period_overrides)
        
        # Calculate override magnitude
        override_magnitudes = []
        for override in period_overrides:
            magnitude = abs(override.adjusted_forecast - override.original_forecast) / override.original_forecast
            override_magnitudes.append(magnitude)
        
        avg_override_magnitude = np.mean(override_magnitudes) if override_magnitudes else 0.0
        
        # Calculate accuracy improvement
        accuracy_improvements = []
        for override in period_overrides:
            # Get actual demand for this override
            actual_demand = self._get_actual_demand(override)
            
            if actual_demand is not None:
                # Calculate original error
                original_error = abs(actual_demand - override.original_forecast)
                # Calculate adjusted error
                adjusted_error = abs(actual_demand - override.adjusted_forecast)
                
                # Calculate improvement (negative means override made it worse)
                improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                accuracy_improvements.append(improvement)
        
        fva_accuracy_improvement = np.mean(accuracy_improvements) if accuracy_improvements else 0.0
        positive_fva_rate = sum(1 for imp in accuracy_improvements if imp > 0) / len(accuracy_improvements) if accuracy_improvements else 0.0
        
        # Calculate bias reduction
        original_bias = np.mean([override.original_forecast - self._get_actual_demand(override) 
                               for override in period_overrides 
                               if self._get_actual_demand(override) is not None])
        adjusted_bias = np.mean([override.adjusted_forecast - self._get_actual_demand(override) 
                               for override in period_overrides 
                               if self._get_actual_demand(override) is not None])
        
        fva_bias_reduction = abs(original_bias) - abs(adjusted_bias) if original_bias and adjusted_bias else 0.0
        
        # Performance by override type
        override_type_performance = {}
        for override_type in OverrideType:
            type_overrides = [o for o in period_overrides if o.override_type == override_type]
            if type_overrides:
                type_improvements = []
                for override in type_overrides:
                    actual = self._get_actual_demand(override)
                    if actual is not None:
                        original_error = abs(actual - override.original_forecast)
                        adjusted_error = abs(actual - override.adjusted_forecast)
                        improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                        type_improvements.append(improvement)
                
                override_type_performance[override_type] = np.mean(type_improvements) if type_improvements else 0.0
        
        # Performance by user
        user_performance = {}
        users = set(override.user_id for override in period_overrides)
        for user in users:
            user_overrides = [o for o in period_overrides if o.user_id == user]
            user_improvements = []
            for override in user_overrides:
                actual = self._get_actual_demand(override)
                if actual is not None:
                    original_error = abs(actual - override.original_forecast)
                    adjusted_error = abs(actual - override.adjusted_forecast)
                    improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                    user_improvements.append(improvement)
            
            user_performance[user] = np.mean(user_improvements) if user_improvements else 0.0
        
        # Enhanced metrics calculation
        seasonal_fva = self._calculate_seasonal_fva_performance(period_overrides)
        horizon_fva = self._calculate_horizon_fva_performance(period_overrides)
        confidence_calibration = self._calculate_confidence_calibration(period_overrides)
        fva_volatility = self._calculate_fva_volatility(period_overrides)
        cumulative_value_add = self._calculate_cumulative_value_add(period_overrides)
        
        # Calculate time to decision (mock data for now)
        time_to_decision = 2.5  # hours
        
        # Calculate override persistence (mock data for now)
        override_persistence = 0.75  # 75% of overrides remain for full forecast horizon
        
        # Calculate override rate (need total forecast count)
        total_forecasts = self._estimate_total_forecasts(period_start, period_end)
        override_rate = total_overrides / total_forecasts if total_forecasts > 0 else 0.0
        
        return FVAMetrics(
            total_overrides=total_overrides,
            override_rate=override_rate,
            avg_override_magnitude=avg_override_magnitude,
            fva_accuracy_improvement=fva_accuracy_improvement,
            fva_bias_reduction=fva_bias_reduction,
            positive_fva_rate=positive_fva_rate,
            override_type_performance=override_type_performance,
            user_performance=user_performance,
            time_to_decision=time_to_decision,
            override_persistence=override_persistence,
            seasonal_fva_performance=seasonal_fva,
            horizon_fva_performance=horizon_fva,
            confidence_calibration=confidence_calibration,
            fva_volatility=fva_volatility,
            cumulative_value_add=cumulative_value_add
        )
    
    def _get_actual_demand(self, override: ForecastOverride) -> Optional[float]:
        """Get actual demand for an override (mock implementation)"""
        # In real implementation, this would query actual demand data
        # For now, simulate with some realistic variance
        base_demand = override.original_forecast
        noise = np.random.normal(0, base_demand * 0.15)  # 15% noise
        return max(0, base_demand + noise)
    
    def _estimate_total_forecasts(self, period_start: datetime, period_end: datetime) -> int:
        """Estimate total number of forecasts generated in period"""
        # Mock implementation - in reality would query forecast generation logs
        days = (period_end - period_start).days
        return days * 1000  # Assume 1000 forecasts per day
    
    def generate_fva_report(self, period_start: datetime, period_end: datetime) -> FVAReport:
        """Generate comprehensive FVA report"""
        
        metrics = self.calculate_fva_impact(period_start, period_end)
        
        # Identify top performers
        top_performers = sorted(
            metrics.user_performance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        top_performers = [user for user, score in top_performers if score > 0]
        
        # Identify improvement opportunities
        improvement_opportunities = []
        
        if metrics.positive_fva_rate < 0.6:
            improvement_opportunities.append("Low positive FVA rate - review override criteria and training")
        
        if metrics.avg_override_magnitude > 0.3:
            improvement_opportunities.append("High override magnitude - consider model recalibration")
        
        if metrics.time_to_decision > 4.0:
            improvement_opportunities.append("Slow decision making - streamline approval process")
        
        # Analyze override patterns
        override_patterns = {
            "most_common_type": max(metrics.override_type_performance.items(), key=lambda x: x[1])[0].value if metrics.override_type_performance else "none",
            "peak_override_hours": "9-11 AM",  # Mock data
            "seasonal_patterns": "Higher overrides during promotional periods",
            "sku_concentration": "80% of overrides on top 20% of SKUs"
        }
        
        # Generate recommendations
        recommendations = []
        
        if metrics.fva_accuracy_improvement < 0.05:
            recommendations.append("Consider additional training for forecasting team")
            recommendations.append("Review and update override guidelines")
        
        if metrics.override_rate > 0.3:
            recommendations.append("High override rate suggests model issues - investigate root causes")
        
        if metrics.fva_bias_reduction < 0:
            recommendations.append("Overrides are increasing bias - review systematic patterns")
        
        recommendations.append("Implement automated override suggestions for high-confidence scenarios")
        recommendations.append("Create feedback loop to improve base forecasting models")
        
        # Calculate baseline vs FVA accuracy
        baseline_accuracy = 85.2  # Mock baseline MAPE
        fva_accuracy = baseline_accuracy * (1 + metrics.fva_accuracy_improvement)
        
        return FVAReport(
            period=f"{period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
            metrics=metrics,
            top_performers=top_performers,
            improvement_opportunities=improvement_opportunities,
            override_patterns=override_patterns,
            recommendations=recommendations,
            forecast_accuracy_baseline=baseline_accuracy,
            forecast_accuracy_with_fva=fva_accuracy
        )
    
    def track_override_effectiveness(self, override_id: str, actual_demand: float) -> Dict[str, float]:
        """Track the effectiveness of a specific override after actual demand is known"""
        
        override = next((o for o in self.overrides_db if o.override_id == override_id), None)
        if not override:
            return {"error": "Override not found"}
        
        # Calculate errors
        original_error = abs(actual_demand - override.original_forecast)
        adjusted_error = abs(actual_demand - override.adjusted_forecast)
        
        # Calculate improvement
        improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
        
        # Calculate accuracy metrics
        original_mape = abs(actual_demand - override.original_forecast) / actual_demand * 100
        adjusted_mape = abs(actual_demand - override.adjusted_forecast) / actual_demand * 100
        
        return {
            "override_id": override_id,
            "improvement_percentage": improvement * 100,
            "original_mape": original_mape,
            "adjusted_mape": adjusted_mape,
            "accuracy_gain": original_mape - adjusted_mape,
            "override_was_beneficial": improvement > 0
        }
    
    def get_user_fva_score(self, user_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Get FVA performance score for a specific user"""
        
        cutoff_date = datetime.now() - timedelta(days=period_days)
        user_overrides = [
            o for o in self.overrides_db 
            if o.user_id == user_id and o.timestamp >= cutoff_date
        ]
        
        if not user_overrides:
            return {"user_id": user_id, "score": 0, "overrides_count": 0}
        
        # Calculate user's FVA score
        improvements = []
        for override in user_overrides:
            actual = self._get_actual_demand(override)
            if actual is not None:
                original_error = abs(actual - override.original_forecast)
                adjusted_error = abs(actual - override.adjusted_forecast)
                improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                improvements.append(improvement)
        
        avg_improvement = np.mean(improvements) if improvements else 0
        positive_rate = sum(1 for imp in improvements if imp > 0) / len(improvements) if improvements else 0
        
        # Composite score (0-100)
        fva_score = (avg_improvement * 50 + positive_rate * 50)
        
        return {
            "user_id": user_id,
            "fva_score": max(0, min(100, fva_score)),
            "overrides_count": len(user_overrides),
            "positive_override_rate": positive_rate,
            "avg_improvement": avg_improvement,
            "period_days": period_days
        }
    
    def _calculate_seasonal_fva_performance(self, overrides: List[ForecastOverride]) -> Dict[str, float]:
        """Calculate FVA performance by season/month"""
        seasonal_performance = {}
        
        # Group overrides by month
        monthly_groups = {}
        for override in overrides:
            month = override.timestamp.strftime('%B')
            if month not in monthly_groups:
                monthly_groups[month] = []
            monthly_groups[month].append(override)
        
        # Calculate FVA for each month
        for month, month_overrides in monthly_groups.items():
            improvements = []
            for override in month_overrides:
                actual = self._get_actual_demand(override)
                if actual is not None:
                    original_error = abs(actual - override.original_forecast)
                    adjusted_error = abs(actual - override.adjusted_forecast)
                    improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                    improvements.append(improvement)
            
            seasonal_performance[month] = np.mean(improvements) if improvements else 0.0
        
        return seasonal_performance
    
    def _calculate_horizon_fva_performance(self, overrides: List[ForecastOverride]) -> Dict[int, float]:
        """Calculate FVA performance by forecast horizon"""
        horizon_performance = {}
        
        # Group overrides by forecast horizon (mock implementation)
        for horizon in [1, 3, 6, 12]:  # 1, 3, 6, 12 months ahead
            horizon_overrides = [o for o in overrides if self._get_forecast_horizon(o) == horizon]
            
            if horizon_overrides:
                improvements = []
                for override in horizon_overrides:
                    actual = self._get_actual_demand(override)
                    if actual is not None:
                        original_error = abs(actual - override.original_forecast)
                        adjusted_error = abs(actual - override.adjusted_forecast)
                        improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                        improvements.append(improvement)
                
                horizon_performance[horizon] = np.mean(improvements) if improvements else 0.0
        
        return horizon_performance
    
    def _calculate_confidence_calibration(self, overrides: List[ForecastOverride]) -> float:
        """Calculate how well confidence scores match actual performance"""
        if not overrides:
            return 0.0
        
        # Group overrides by confidence bins
        confidence_bins = np.arange(0, 1.1, 0.2)  # 0-0.2, 0.2-0.4, etc.
        calibration_errors = []
        
        for i in range(len(confidence_bins) - 1):
            bin_start, bin_end = confidence_bins[i], confidence_bins[i + 1]
            bin_overrides = [o for o in overrides if bin_start <= o.confidence < bin_end]
            
            if bin_overrides:
                # Calculate actual success rate for this confidence bin
                successes = []
                for override in bin_overrides:
                    actual = self._get_actual_demand(override)
                    if actual is not None:
                        original_error = abs(actual - override.original_forecast)
                        adjusted_error = abs(actual - override.adjusted_forecast)
                        success = adjusted_error < original_error
                        successes.append(success)
                
                if successes:
                    actual_success_rate = sum(successes) / len(successes)
                    expected_confidence = (bin_start + bin_end) / 2
                    calibration_error = abs(actual_success_rate - expected_confidence)
                    calibration_errors.append(calibration_error)
        
        # Return calibration score (1 - average calibration error)
        return 1 - np.mean(calibration_errors) if calibration_errors else 0.5
    
    def _calculate_fva_volatility(self, overrides: List[ForecastOverride]) -> float:
        """Calculate consistency of FVA performance over time"""
        if len(overrides) < 5:
            return 0.0
        
        # Calculate weekly FVA performance
        weekly_performance = {}
        for override in overrides:
            week = override.timestamp.strftime('%Y-W%U')
            if week not in weekly_performance:
                weekly_performance[week] = []
            
            actual = self._get_actual_demand(override)
            if actual is not None:
                original_error = abs(actual - override.original_forecast)
                adjusted_error = abs(actual - override.adjusted_forecast)
                improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                weekly_performance[week].append(improvement)
        
        # Calculate weekly averages
        weekly_averages = [np.mean(improvements) for improvements in weekly_performance.values() if improvements]
        
        if len(weekly_averages) < 2:
            return 0.0
        
        # Return coefficient of variation (lower = more consistent)
        mean_performance = np.mean(weekly_averages)
        std_performance = np.std(weekly_averages)
        
        return std_performance / abs(mean_performance) if mean_performance != 0 else float('inf')
    
    def _calculate_cumulative_value_add(self, overrides: List[ForecastOverride]) -> float:
        """Calculate total business value added by overrides"""
        total_value = 0.0
        
        for override in overrides:
            actual = self._get_actual_demand(override)
            if actual is not None:
                # Calculate error reduction
                original_error = abs(actual - override.original_forecast)
                adjusted_error = abs(actual - override.adjusted_forecast)
                error_reduction = original_error - adjusted_error
                
                # Convert to business value (assume $1 per unit of error reduction)
                unit_value = 1.0  # Could be parameterized based on product margin
                value_add = error_reduction * unit_value
                total_value += value_add
        
        return total_value
    
    def _get_forecast_horizon(self, override: ForecastOverride) -> int:
        """Get forecast horizon for an override (mock implementation)"""
        # In real implementation, this would be calculated from the forecast period
        # For now, return a random horizon
        return np.random.choice([1, 3, 6, 12])
    
    def calculate_fva_confidence_intervals(self, period_start: datetime, period_end: datetime,
                                         confidence_level: float = 0.95) -> List[FVAConfidenceInterval]:
        """Calculate confidence intervals for FVA metrics"""
        
        period_overrides = [
            override for override in self.overrides_db
            if period_start <= override.timestamp <= period_end
        ]
        
        if len(period_overrides) < 10:  # Need minimum sample size
            return []
        
        confidence_intervals = []
        
        # Calculate confidence interval for accuracy improvement
        improvements = []
        for override in period_overrides:
            actual = self._get_actual_demand(override)
            if actual is not None:
                original_error = abs(actual - override.original_forecast)
                adjusted_error = abs(actual - override.adjusted_forecast)
                improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                improvements.append(improvement)
        
        if len(improvements) >= 10:
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements, ddof=1)
            n = len(improvements)
            
            # Calculate t-statistic for confidence interval
            alpha = 1 - confidence_level
            t_stat = stats.t.ppf(1 - alpha/2, n - 1)
            margin_error = t_stat * (std_improvement / np.sqrt(n))
            
            confidence_intervals.append(FVAConfidenceInterval(
                metric_name='fva_accuracy_improvement',
                point_estimate=mean_improvement,
                lower_bound=mean_improvement - margin_error,
                upper_bound=mean_improvement + margin_error,
                confidence_level=confidence_level,
                sample_size=n
            ))
        
        # Calculate confidence interval for positive FVA rate
        positive_count = sum(1 for imp in improvements if imp > 0)
        total_count = len(improvements)
        
        if total_count >= 10:
            positive_rate = positive_count / total_count
            
            # Binomial confidence interval (Wilson score interval)
            z_score = stats.norm.ppf(1 - (1 - confidence_level)/2)
            denominator = 1 + z_score**2 / total_count
            center = (positive_rate + z_score**2 / (2 * total_count)) / denominator
            margin = z_score * np.sqrt((positive_rate * (1 - positive_rate) + z_score**2 / (4 * total_count)) / total_count) / denominator
            
            confidence_intervals.append(FVAConfidenceInterval(
                metric_name='positive_fva_rate',
                point_estimate=positive_rate,
                lower_bound=max(0, center - margin),
                upper_bound=min(1, center + margin),
                confidence_level=confidence_level,
                sample_size=total_count
            ))
        
        return confidence_intervals
    
    def detect_fva_anomalies(self, period_start: datetime, period_end: datetime) -> List[Dict[str, Any]]:
        """Detect anomalies in FVA performance"""
        
        period_overrides = [
            override for override in self.overrides_db
            if period_start <= override.timestamp <= period_end
        ]
        
        anomalies = []
        
        # Detect users with unusually poor FVA performance
        user_performance = {}
        for override in period_overrides:
            if override.user_id not in user_performance:
                user_performance[override.user_id] = []
            
            actual = self._get_actual_demand(override)
            if actual is not None:
                original_error = abs(actual - override.original_forecast)
                adjusted_error = abs(actual - override.adjusted_forecast)
                improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                user_performance[override.user_id].append(improvement)
        
        # Calculate z-scores for user performance
        all_improvements = [imp for improvements in user_performance.values() for imp in improvements]
        if len(all_improvements) > 10:
            overall_mean = np.mean(all_improvements)
            overall_std = np.std(all_improvements)
            
            for user_id, improvements in user_performance.items():
                if len(improvements) >= 3:  # Minimum sample size
                    user_mean = np.mean(improvements)
                    z_score = (user_mean - overall_mean) / overall_std if overall_std > 0 else 0
                    
                    if z_score < -2.0:  # Significantly below average
                        anomalies.append({
                            'type': 'poor_user_performance',
                            'user_id': user_id,
                            'performance_score': user_mean,
                            'z_score': z_score,
                            'sample_size': len(improvements),
                            'severity': 'high' if z_score < -3.0 else 'medium'
                        })
        
        # Detect SKUs with consistently negative FVA
        sku_performance = {}
        for override in period_overrides:
            if override.sku not in sku_performance:
                sku_performance[override.sku] = []
            
            actual = self._get_actual_demand(override)
            if actual is not None:
                original_error = abs(actual - override.original_forecast)
                adjusted_error = abs(actual - override.adjusted_forecast)
                improvement = (original_error - adjusted_error) / original_error if original_error > 0 else 0
                sku_performance[override.sku].append(improvement)
        
        for sku, improvements in sku_performance.items():
            if len(improvements) >= 3:
                negative_rate = sum(1 for imp in improvements if imp < 0) / len(improvements)
                if negative_rate > 0.7:  # More than 70% negative FVA
                    anomalies.append({
                        'type': 'consistently_negative_fva',
                        'sku': sku,
                        'negative_fva_rate': negative_rate,
                        'sample_size': len(improvements),
                        'avg_performance': np.mean(improvements),
                        'severity': 'high' if negative_rate > 0.8 else 'medium'
                    })
        
        return anomalies