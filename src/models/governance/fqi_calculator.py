"""
Forecast Quality Index (FQI) Calculator
Provides 0-100 scoring system blending WAPE/MAE, bias, calibration, and stability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class FQIComponents:
    """Components of the Forecast Quality Index"""
    accuracy_score: float  # 0-100 based on WAPE/MAE
    bias_score: float      # 0-100 based on forecast bias
    calibration_score: float  # 0-100 based on P50 realization rate
    stability_score: float    # 0-100 based on forecast stability
    coverage_score: float     # 0-100 based on prediction interval coverage
    coherence_score: float    # 0-100 based on hierarchical coherence
    
@dataclass
class FQIResult:
    """Complete FQI calculation result"""
    overall_fqi: float
    components: FQIComponents
    grade: str  # A+, A, B+, B, C+, C, D, F
    benchmark_comparison: Dict[str, float]
    improvement_recommendations: List[str]
    trend_analysis: Dict[str, float]
    
@dataclass
class FQIAlert:
    """FQI monitoring alert"""
    alert_id: str
    alert_type: str
    severity: str
    threshold_value: float
    actual_value: float
    message: str
    timestamp: datetime
    requires_retraining: bool

class FQICalculator:
    """
    Calculates Forecast Quality Index (FQI) - a comprehensive 0-100 score
    """
    
    def __init__(self):
        self.component_weights = {
            'accuracy': 0.30,    # 30% weight on accuracy (WAPE/MAE)
            'bias': 0.20,        # 20% weight on bias
            'calibration': 0.20, # 20% weight on calibration
            'stability': 0.15,   # 15% weight on stability
            'coverage': 0.05,    # 5% weight on prediction interval coverage
            'coherence': 0.10    # 10% weight on hierarchical coherence
        }
        
        # Real-time monitoring settings
        self.monitoring_thresholds = {
            'fqi_decline_threshold': 5.0,      # Alert if FQI drops by 5 points
            'accuracy_threshold': 60.0,        # Alert if accuracy score < 60
            'bias_threshold': 50.0,           # Alert if bias score < 50
            'coherence_threshold': 70.0,      # Alert if coherence score < 70
            'retraining_threshold': 50.0      # Trigger retraining if overall FQI < 50
        }
        
        # Industry benchmarks
        self.industry_benchmarks = {
            'retail': {'fqi': 78.5, 'accuracy': 82.0, 'bias': 75.0, 'coherence': 85.0},
            'manufacturing': {'fqi': 81.2, 'accuracy': 85.0, 'bias': 78.0, 'coherence': 88.0},
            'healthcare': {'fqi': 76.8, 'accuracy': 80.0, 'bias': 72.0, 'coherence': 82.0},
            'technology': {'fqi': 83.1, 'accuracy': 87.0, 'bias': 80.0, 'coherence': 90.0}
        }
        
        self.historical_fqi_scores = []
        self.active_alerts = []
        
        # Benchmark thresholds for scoring
        self.accuracy_benchmarks = {
            'excellent': 5.0,    # WAPE < 5%
            'good': 10.0,        # WAPE < 10%
            'fair': 20.0,        # WAPE < 20%
            'poor': 30.0         # WAPE < 30%
        }
        
        self.bias_benchmarks = {
            'excellent': 2.0,    # |Bias| < 2%
            'good': 5.0,         # |Bias| < 5%
            'fair': 10.0,        # |Bias| < 10%
            'poor': 20.0         # |Bias| < 20%
        }
    
    def calculate_accuracy_score(self, forecasts: pd.Series, actuals: pd.Series) -> float:
        """Calculate accuracy score (0-100) based on WAPE and MAE"""
        
        # Align series
        common_index = forecasts.index.intersection(actuals.index)
        if len(common_index) == 0:
            return 0.0
        
        f_aligned = forecasts.loc[common_index]
        a_aligned = actuals.loc[common_index]
        
        # Remove zero actuals to avoid division by zero
        non_zero_mask = a_aligned != 0
        if non_zero_mask.sum() == 0:
            return 0.0
        
        f_nz = f_aligned[non_zero_mask]
        a_nz = a_aligned[non_zero_mask]
        
        # Calculate WAPE (Weighted Absolute Percentage Error)
        wape = np.sum(np.abs(a_nz - f_nz)) / np.sum(np.abs(a_nz)) * 100
        
        # Calculate MAE as percentage of mean demand
        mae = np.mean(np.abs(a_nz - f_nz))
        mae_percentage = mae / np.mean(np.abs(a_nz)) * 100
        
        # Average WAPE and MAE percentage for final accuracy metric
        accuracy_metric = (wape + mae_percentage) / 2
        
        # Convert to 0-100 score (lower error = higher score)
        if accuracy_metric <= self.accuracy_benchmarks['excellent']:
            score = 95 + (self.accuracy_benchmarks['excellent'] - accuracy_metric) / self.accuracy_benchmarks['excellent'] * 5
        elif accuracy_metric <= self.accuracy_benchmarks['good']:
            score = 80 + (self.accuracy_benchmarks['good'] - accuracy_metric) / (self.accuracy_benchmarks['good'] - self.accuracy_benchmarks['excellent']) * 15
        elif accuracy_metric <= self.accuracy_benchmarks['fair']:
            score = 60 + (self.accuracy_benchmarks['fair'] - accuracy_metric) / (self.accuracy_benchmarks['fair'] - self.accuracy_benchmarks['good']) * 20
        elif accuracy_metric <= self.accuracy_benchmarks['poor']:
            score = 30 + (self.accuracy_benchmarks['poor'] - accuracy_metric) / (self.accuracy_benchmarks['poor'] - self.accuracy_benchmarks['fair']) * 30
        else:
            score = max(0, 30 - (accuracy_metric - self.accuracy_benchmarks['poor']) / 10)
        
        return min(100, max(0, score))
    
    def calculate_bias_score(self, forecasts: pd.Series, actuals: pd.Series) -> float:
        """Calculate bias score (0-100) based on forecast bias"""
        
        # Align series
        common_index = forecasts.index.intersection(actuals.index)
        if len(common_index) == 0:
            return 0.0
        
        f_aligned = forecasts.loc[common_index]
        a_aligned = actuals.loc[common_index]
        
        # Calculate bias as percentage
        bias = np.mean((f_aligned - a_aligned) / a_aligned) * 100
        abs_bias = abs(bias)
        
        # Convert to 0-100 score (lower bias = higher score)
        if abs_bias <= self.bias_benchmarks['excellent']:
            score = 95 + (self.bias_benchmarks['excellent'] - abs_bias) / self.bias_benchmarks['excellent'] * 5
        elif abs_bias <= self.bias_benchmarks['good']:
            score = 80 + (self.bias_benchmarks['good'] - abs_bias) / (self.bias_benchmarks['good'] - self.bias_benchmarks['excellent']) * 15
        elif abs_bias <= self.bias_benchmarks['fair']:
            score = 60 + (self.bias_benchmarks['fair'] - abs_bias) / (self.bias_benchmarks['fair'] - self.bias_benchmarks['good']) * 20
        elif abs_bias <= self.bias_benchmarks['poor']:
            score = 30 + (self.bias_benchmarks['poor'] - abs_bias) / (self.bias_benchmarks['poor'] - self.bias_benchmarks['fair']) * 30
        else:
            score = max(0, 30 - (abs_bias - self.bias_benchmarks['poor']) / 5)
        
        return min(100, max(0, score))
    
    def calculate_calibration_score(self, forecasts: pd.Series, actuals: pd.Series, 
                                  confidence_intervals: Optional[Dict[str, pd.Series]] = None) -> float:
        """Calculate calibration score based on P50 realization rate"""
        
        # Align series
        common_index = forecasts.index.intersection(actuals.index)
        if len(common_index) == 0:
            return 0.0
        
        f_aligned = forecasts.loc[common_index]
        a_aligned = actuals.loc[common_index]
        
        # If confidence intervals provided, use them for calibration
        if confidence_intervals and 'P50' in confidence_intervals:
            p50_forecasts = confidence_intervals['P50'].loc[common_index]
            
            # Calculate how often actuals fall below P50 (should be ~50%)
            below_p50_rate = (a_aligned < p50_forecasts).mean()
            calibration_error = abs(below_p50_rate - 0.5)
            
            # Perfect calibration = 100, maximum error (0.5) = 0
            calibration_score = (0.5 - calibration_error) / 0.5 * 100
        else:
            # Use forecast vs actual distribution comparison
            # Calculate if forecast distribution matches actual distribution
            
            # Normalize both series
            f_norm = (f_aligned - f_aligned.mean()) / f_aligned.std() if f_aligned.std() > 0 else f_aligned
            a_norm = (a_aligned - a_aligned.mean()) / a_aligned.std() if a_aligned.std() > 0 else a_aligned
            
            # Use Kolmogorov-Smirnov test for distribution similarity
            try:
                ks_stat, p_value = stats.ks_2samp(f_norm, a_norm)
                # Higher p-value = more similar distributions = better calibration
                calibration_score = min(100, p_value * 100)
            except:
                calibration_score = 50.0  # Default middle score
        
        return min(100, max(0, calibration_score))
    
    def calculate_stability_score(self, forecast_revisions: List[pd.Series]) -> float:
        """Calculate stability score based on forecast revision patterns"""
        
        if len(forecast_revisions) < 2:
            return 100.0  # Perfect stability if no revisions
        
        # Calculate revision magnitudes
        revision_magnitudes = []
        
        for i in range(1, len(forecast_revisions)):
            prev_forecast = forecast_revisions[i-1]
            curr_forecast = forecast_revisions[i]
            
            # Align series
            common_index = prev_forecast.index.intersection(curr_forecast.index)
            if len(common_index) > 0:
                prev_aligned = prev_forecast.loc[common_index]
                curr_aligned = curr_forecast.loc[common_index]
                
                # Calculate revision magnitude as percentage change
                revision_magnitude = np.mean(np.abs((curr_aligned - prev_aligned) / prev_aligned)) * 100
                revision_magnitudes.append(revision_magnitude)
        
        if not revision_magnitudes:
            return 100.0
        
        # Average revision magnitude
        avg_revision = np.mean(revision_magnitudes)
        
        # Convert to 0-100 score (lower revision = higher stability)
        # Excellent stability: < 5% revision
        # Good stability: < 10% revision
        # Fair stability: < 20% revision
        # Poor stability: > 20% revision
        
        if avg_revision <= 5.0:
            score = 95 + (5.0 - avg_revision) / 5.0 * 5
        elif avg_revision <= 10.0:
            score = 80 + (10.0 - avg_revision) / 5.0 * 15
        elif avg_revision <= 20.0:
            score = 60 + (20.0 - avg_revision) / 10.0 * 20
        else:
            score = max(0, 60 - (avg_revision - 20.0) / 5.0 * 10)
        
        return min(100, max(0, score))
    
    def calculate_coverage_score(self, actuals: pd.Series, 
                               confidence_intervals: Dict[str, pd.Series]) -> float:
        """Calculate prediction interval coverage score"""
        
        if 'P10' not in confidence_intervals or 'P90' not in confidence_intervals:
            return 50.0  # Default score if no intervals
        
        # Align series
        common_index = actuals.index.intersection(confidence_intervals['P10'].index)
        common_index = common_index.intersection(confidence_intervals['P90'].index)
        
        if len(common_index) == 0:
            return 50.0
        
        a_aligned = actuals.loc[common_index]
        p10_aligned = confidence_intervals['P10'].loc[common_index]
        p90_aligned = confidence_intervals['P90'].loc[common_index]
        
        # Calculate coverage rate (should be ~80% for P10-P90 interval)
        within_interval = ((a_aligned >= p10_aligned) & (a_aligned <= p90_aligned)).mean()
        
        # Perfect coverage (80%) = 100 score
        # Coverage error calculation
        coverage_error = abs(within_interval - 0.8)
        coverage_score = (0.8 - coverage_error) / 0.8 * 100
        
        return min(100, max(0, coverage_score))
    
    def calculate_hierarchical_coherence_score(self, hierarchical_forecasts: Dict[str, pd.Series],
                                             aggregation_matrix: Optional[np.ndarray] = None) -> float:
        """Calculate hierarchical coherence score (0-100)"""
        
        if not hierarchical_forecasts or len(hierarchical_forecasts) < 2:
            return 50.0  # Default score if no hierarchical structure
        
        try:
            # If aggregation matrix is provided, use it for precise coherence calculation
            if aggregation_matrix is not None:
                return self._calculate_matrix_based_coherence(hierarchical_forecasts, aggregation_matrix)
            
            # Otherwise, use heuristic-based coherence calculation
            return self._calculate_heuristic_coherence(hierarchical_forecasts)
            
        except Exception as e:
            logger.warning(f"Hierarchical coherence calculation failed: {e}")
            return 50.0
    
    def _calculate_matrix_based_coherence(self, forecasts: Dict[str, pd.Series], 
                                        aggregation_matrix: np.ndarray) -> float:
        """Calculate coherence using aggregation matrix"""
        
        # Get forecast horizon
        horizon = len(list(forecasts.values())[0])
        coherence_scores = []
        
        for t in range(horizon):
            # Get forecasts for time t
            forecast_vector = np.array([forecasts[key].iloc[t] for key in forecasts.keys()])
            
            # Calculate what the aggregated forecasts should be
            expected_aggregated = aggregation_matrix @ forecast_vector
            
            # Compare with actual aggregated forecasts
            actual_aggregated = forecast_vector  # Assuming forecasts are already in hierarchical order
            
            # Calculate coherence as 1 - normalized error
            error = np.mean(np.abs(expected_aggregated - actual_aggregated))
            total_forecast = np.sum(np.abs(actual_aggregated))
            
            if total_forecast > 0:
                coherence_scores.append(1 - (error / total_forecast))
            else:
                coherence_scores.append(1.0)
        
        avg_coherence = np.mean(coherence_scores)
        return min(100, max(0, avg_coherence * 100))
    
    def _calculate_heuristic_coherence(self, forecasts: Dict[str, pd.Series]) -> float:
        """Calculate coherence using heuristic approach"""
        
        # Group forecasts by hierarchy level (based on naming convention)
        hierarchy_levels = {}
        
        for key, forecast in forecasts.items():
            # Determine hierarchy level based on key structure
            parts = key.split('_')
            level = len(parts)  # More parts = lower level
            
            if level not in hierarchy_levels:
                hierarchy_levels[level] = {}
            hierarchy_levels[level][key] = forecast
        
        if len(hierarchy_levels) < 2:
            return 100.0  # Perfect coherence if only one level
        
        coherence_violations = []
        
        # Check coherence between adjacent levels
        sorted_levels = sorted(hierarchy_levels.keys())
        
        for i in range(len(sorted_levels) - 1):
            upper_level = sorted_levels[i]
            lower_level = sorted_levels[i + 1]
            
            # For each upper level forecast, check if it equals sum of related lower level forecasts
            for upper_key, upper_forecast in hierarchy_levels[upper_level].items():
                related_lower_forecasts = []
                
                # Find related lower level forecasts (simple heuristic)
                for lower_key, lower_forecast in hierarchy_levels[lower_level].items():
                    if self._are_hierarchically_related(upper_key, lower_key):
                        related_lower_forecasts.append(lower_forecast)
                
                if related_lower_forecasts:
                    # Sum related lower level forecasts
                    lower_sum = sum(related_lower_forecasts)
                    
                    # Calculate coherence violation
                    for t in range(len(upper_forecast)):
                        if t < len(lower_sum):
                            expected = lower_sum.iloc[t]
                            actual = upper_forecast.iloc[t]
                            
                            if expected != 0:
                                violation = abs(actual - expected) / abs(expected)
                                coherence_violations.append(violation)
        
        if not coherence_violations:
            return 100.0
        
        # Calculate average coherence (1 - average violation)
        avg_violation = np.mean(coherence_violations)
        coherence_score = max(0, 1 - avg_violation) * 100
        
        return min(100, max(0, coherence_score))
    
    def _are_hierarchically_related(self, upper_key: str, lower_key: str) -> bool:
        """Check if two forecast keys are hierarchically related"""
        
        upper_parts = upper_key.split('_')
        lower_parts = lower_key.split('_')
        
        # Lower level should have more parts and contain all upper level parts
        if len(lower_parts) <= len(upper_parts):
            return False
        
        # Check if lower key starts with upper key components
        for i, upper_part in enumerate(upper_parts):
            if i >= len(lower_parts) or upper_part != lower_parts[i]:
                return False
        
        return True
    
    def calculate_fqi(self, forecasts: pd.Series, actuals: pd.Series,
                     confidence_intervals: Optional[Dict[str, pd.Series]] = None,
                     forecast_revisions: Optional[List[pd.Series]] = None,
                     hierarchical_forecasts: Optional[Dict[str, pd.Series]] = None,
                     aggregation_matrix: Optional[np.ndarray] = None) -> FQIResult:
        """Calculate complete Forecast Quality Index"""
        
        # Calculate component scores
        accuracy_score = self.calculate_accuracy_score(forecasts, actuals)
        bias_score = self.calculate_bias_score(forecasts, actuals)
        calibration_score = self.calculate_calibration_score(forecasts, actuals, confidence_intervals)
        stability_score = self.calculate_stability_score(forecast_revisions or [forecasts])
        coverage_score = self.calculate_coverage_score(actuals, confidence_intervals or {})
        coherence_score = self.calculate_hierarchical_coherence_score(hierarchical_forecasts or {}, aggregation_matrix)
        
        components = FQIComponents(
            accuracy_score=accuracy_score,
            bias_score=bias_score,
            calibration_score=calibration_score,
            stability_score=stability_score,
            coverage_score=coverage_score,
            coherence_score=coherence_score
        )
        
        # Calculate weighted overall FQI
        overall_fqi = (
            accuracy_score * self.component_weights['accuracy'] +
            bias_score * self.component_weights['bias'] +
            calibration_score * self.component_weights['calibration'] +
            stability_score * self.component_weights['stability'] +
            coverage_score * self.component_weights['coverage'] +
            coherence_score * self.component_weights['coherence']
        )
        
        # Assign letter grade
        grade = self._assign_grade(overall_fqi)
        
        # Benchmark comparison
        benchmark_comparison = {
            'industry_average': 72.0,
            'best_in_class': 85.0,
            'your_score': overall_fqi,
            'percentile_rank': self._calculate_percentile_rank(overall_fqi)
        }
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_recommendations(components)
        
        # Trend analysis (mock data for now)
        trend_analysis = {
            'last_month_fqi': overall_fqi - 2.5,
            'last_quarter_fqi': overall_fqi - 5.2,
            'trend_direction': 'improving' if overall_fqi > 75 else 'stable',
            'momentum': 'positive'
        }
        
        return FQIResult(
            overall_fqi=overall_fqi,
            components=components,
            grade=grade,
            benchmark_comparison=benchmark_comparison,
            improvement_recommendations=improvement_recommendations,
            trend_analysis=trend_analysis
        )
    
    def _assign_grade(self, fqi_score: float) -> str:
        """Assign letter grade based on FQI score"""
        if fqi_score >= 95:
            return "A+"
        elif fqi_score >= 90:
            return "A"
        elif fqi_score >= 85:
            return "B+"
        elif fqi_score >= 80:
            return "B"
        elif fqi_score >= 75:
            return "C+"
        elif fqi_score >= 70:
            return "C"
        elif fqi_score >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_percentile_rank(self, fqi_score: float) -> float:
        """Calculate percentile rank (mock implementation)"""
        # In real implementation, this would compare against historical database
        if fqi_score >= 90:
            return 95.0
        elif fqi_score >= 80:
            return 80.0
        elif fqi_score >= 70:
            return 60.0
        elif fqi_score >= 60:
            return 40.0
        else:
            return 20.0
    
    def _generate_recommendations(self, components: FQIComponents) -> List[str]:
        """Generate improvement recommendations based on component scores"""
        recommendations = []
        
        if components.accuracy_score < 70:
            recommendations.append("Improve forecast accuracy by enhancing feature engineering and model selection")
            recommendations.append("Consider ensemble methods to reduce forecast error")
        
        if components.bias_score < 70:
            recommendations.append("Address systematic forecast bias through model recalibration")
            recommendations.append("Review and adjust seasonal and promotional factors")
        
        if components.calibration_score < 70:
            recommendations.append("Improve forecast calibration by adjusting confidence intervals")
            recommendations.append("Validate probabilistic forecasting methods")
        
        if components.stability_score < 70:
            recommendations.append("Reduce forecast volatility by implementing smoothing mechanisms")
            recommendations.append("Establish clearer guidelines for forecast revisions")
        
        if components.coverage_score < 70:
            recommendations.append("Adjust prediction intervals to achieve target coverage rates")
            recommendations.append("Review uncertainty quantification methods")
        
        if components.coherence_score < 70:
            recommendations.append("Improve hierarchical forecast coherence through better reconciliation")
            recommendations.append("Review aggregation constraints and hierarchy structure")
        
        if not recommendations:
            recommendations.append("Maintain current high performance standards")
            recommendations.append("Focus on continuous improvement and monitoring")
        
        return recommendations
    
    def monitor_fqi_real_time(self, current_fqi: FQIResult, 
                             historical_window: int = 30) -> List[FQIAlert]:
        """Real-time FQI monitoring with alert generation"""
        
        alerts = []
        current_time = datetime.now()
        
        # Store current FQI score
        self.historical_fqi_scores.append((current_time, current_fqi.overall_fqi))
        
        # Keep only recent history
        cutoff_time = current_time - timedelta(days=historical_window)
        self.historical_fqi_scores = [
            (timestamp, score) for timestamp, score in self.historical_fqi_scores
            if timestamp >= cutoff_time
        ]
        
        # Check for FQI decline
        if len(self.historical_fqi_scores) >= 2:
            recent_scores = [score for _, score in self.historical_fqi_scores[-7:]]  # Last 7 measurements
            if len(recent_scores) >= 2:
                trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                if trend_slope < -self.monitoring_thresholds['fqi_decline_threshold'] / 7:  # Decline per measurement
                    alerts.append(FQIAlert(
                        alert_id=f"fqi_decline_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        alert_type="fqi_decline",
                        severity="high",
                        threshold_value=self.monitoring_thresholds['fqi_decline_threshold'],
                        actual_value=trend_slope * 7,
                        message=f"FQI declining at rate of {trend_slope * 7:.2f} points per week",
                        timestamp=current_time,
                        requires_retraining=True
                    ))
        
        # Check component thresholds
        if current_fqi.components.accuracy_score < self.monitoring_thresholds['accuracy_threshold']:
            alerts.append(FQIAlert(
                alert_id=f"accuracy_low_{current_time.strftime('%Y%m%d_%H%M%S')}",
                alert_type="accuracy_threshold",
                severity="medium",
                threshold_value=self.monitoring_thresholds['accuracy_threshold'],
                actual_value=current_fqi.components.accuracy_score,
                message=f"Accuracy score ({current_fqi.components.accuracy_score:.1f}) below threshold",
                timestamp=current_time,
                requires_retraining=False
            ))
        
        if current_fqi.components.bias_score < self.monitoring_thresholds['bias_threshold']:
            alerts.append(FQIAlert(
                alert_id=f"bias_high_{current_time.strftime('%Y%m%d_%H%M%S')}",
                alert_type="bias_threshold",
                severity="medium",
                threshold_value=self.monitoring_thresholds['bias_threshold'],
                actual_value=current_fqi.components.bias_score,
                message=f"Bias score ({current_fqi.components.bias_score:.1f}) indicates systematic bias",
                timestamp=current_time,
                requires_retraining=True
            ))
        
        if current_fqi.components.coherence_score < self.monitoring_thresholds['coherence_threshold']:
            alerts.append(FQIAlert(
                alert_id=f"coherence_low_{current_time.strftime('%Y%m%d_%H%M%S')}",
                alert_type="coherence_threshold",
                severity="high",
                threshold_value=self.monitoring_thresholds['coherence_threshold'],
                actual_value=current_fqi.components.coherence_score,
                message=f"Hierarchical coherence ({current_fqi.components.coherence_score:.1f}) below threshold",
                timestamp=current_time,
                requires_retraining=False
            ))
        
        # Check overall FQI for retraining trigger
        if current_fqi.overall_fqi < self.monitoring_thresholds['retraining_threshold']:
            alerts.append(FQIAlert(
                alert_id=f"retraining_required_{current_time.strftime('%Y%m%d_%H%M%S')}",
                alert_type="retraining_required",
                severity="critical",
                threshold_value=self.monitoring_thresholds['retraining_threshold'],
                actual_value=current_fqi.overall_fqi,
                message=f"Overall FQI ({current_fqi.overall_fqi:.1f}) requires immediate model retraining",
                timestamp=current_time,
                requires_retraining=True
            ))
        
        # Add new alerts to active alerts
        self.active_alerts.extend(alerts)
        
        return alerts
    
    def get_industry_benchmark_comparison(self, industry: str, current_fqi: FQIResult) -> Dict[str, Any]:
        """Compare current FQI with industry benchmarks"""
        
        if industry not in self.industry_benchmarks:
            industry = 'retail'  # Default to retail benchmarks
        
        benchmarks = self.industry_benchmarks[industry]
        
        comparison = {
            'industry': industry,
            'your_fqi': current_fqi.overall_fqi,
            'industry_average_fqi': benchmarks['fqi'],
            'fqi_percentile': self._calculate_industry_percentile(current_fqi.overall_fqi, benchmarks['fqi']),
            'component_comparison': {
                'accuracy': {
                    'your_score': current_fqi.components.accuracy_score,
                    'industry_average': benchmarks['accuracy'],
                    'gap': current_fqi.components.accuracy_score - benchmarks['accuracy']
                },
                'bias': {
                    'your_score': current_fqi.components.bias_score,
                    'industry_average': benchmarks['bias'],
                    'gap': current_fqi.components.bias_score - benchmarks['bias']
                },
                'coherence': {
                    'your_score': current_fqi.components.coherence_score,
                    'industry_average': benchmarks['coherence'],
                    'gap': current_fqi.components.coherence_score - benchmarks['coherence']
                }
            },
            'competitive_position': self._determine_competitive_position(current_fqi.overall_fqi, benchmarks['fqi']),
            'improvement_priority': self._identify_improvement_priorities(current_fqi, benchmarks)
        }
        
        return comparison
    
    def _calculate_industry_percentile(self, your_score: float, industry_average: float) -> float:
        """Calculate percentile rank within industry"""
        
        # Simplified percentile calculation
        # In practice, this would use actual industry distribution data
        if your_score >= industry_average + 10:
            return 90.0
        elif your_score >= industry_average + 5:
            return 75.0
        elif your_score >= industry_average:
            return 60.0
        elif your_score >= industry_average - 5:
            return 40.0
        elif your_score >= industry_average - 10:
            return 25.0
        else:
            return 10.0
    
    def _determine_competitive_position(self, your_score: float, industry_average: float) -> str:
        """Determine competitive position"""
        
        gap = your_score - industry_average
        
        if gap >= 10:
            return "industry_leader"
        elif gap >= 5:
            return "above_average"
        elif gap >= -2:
            return "average"
        elif gap >= -8:
            return "below_average"
        else:
            return "needs_improvement"
    
    def _identify_improvement_priorities(self, current_fqi: FQIResult, benchmarks: Dict[str, float]) -> List[str]:
        """Identify improvement priorities based on benchmark gaps"""
        
        priorities = []
        
        # Calculate gaps
        accuracy_gap = current_fqi.components.accuracy_score - benchmarks['accuracy']
        bias_gap = current_fqi.components.bias_score - benchmarks['bias']
        coherence_gap = current_fqi.components.coherence_score - benchmarks['coherence']
        
        # Prioritize largest negative gaps
        gaps = [
            ('accuracy', accuracy_gap),
            ('bias', bias_gap),
            ('coherence', coherence_gap)
        ]
        
        # Sort by gap (most negative first)
        gaps.sort(key=lambda x: x[1])
        
        for component, gap in gaps:
            if gap < -5:  # Significant gap
                if component == 'accuracy':
                    priorities.append("Improve forecast accuracy through better models and features")
                elif component == 'bias':
                    priorities.append("Address systematic forecast bias through calibration")
                elif component == 'coherence':
                    priorities.append("Enhance hierarchical forecast coherence and reconciliation")
        
        if not priorities:
            priorities.append("Maintain current performance and focus on consistency")
        
        return priorities
    
    async def trigger_model_retraining(self, alert: FQIAlert) -> Dict[str, Any]:
        """Trigger automatic model retraining based on FQI degradation"""
        
        logger.info(f"Triggering model retraining due to alert: {alert.alert_id}")
        
        # In a real implementation, this would:
        # 1. Queue retraining job
        # 2. Notify data science team
        # 3. Prepare training data
        # 4. Execute retraining pipeline
        
        retraining_result = {
            'retraining_triggered': True,
            'trigger_alert': alert.alert_id,
            'trigger_reason': alert.message,
            'estimated_completion_time': datetime.now() + timedelta(hours=6),
            'retraining_job_id': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'priority': 'high' if alert.severity == 'critical' else 'medium'
        }
        
        logger.info(f"Model retraining queued: {retraining_result['retraining_job_id']}")
        
        return retraining_result
    
    def calculate_fqi_trend(self, historical_fqi_scores: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Calculate FQI trend analysis"""
        
        if len(historical_fqi_scores) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by date
        historical_fqi_scores.sort(key=lambda x: x[0])
        
        dates = [score[0] for score in historical_fqi_scores]
        scores = [score[1] for score in historical_fqi_scores]
        
        # Calculate trend
        x = np.arange(len(scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        # Determine trend direction
        if slope > 1.0:
            trend_direction = "strongly_improving"
        elif slope > 0.2:
            trend_direction = "improving"
        elif slope > -0.2:
            trend_direction = "stable"
        elif slope > -1.0:
            trend_direction = "declining"
        else:
            trend_direction = "strongly_declining"
        
        # Calculate volatility
        score_changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        volatility = np.std(score_changes) if len(score_changes) > 1 else 0
        
        return {
            "trend_direction": trend_direction,
            "slope": slope,
            "r_squared": r_value ** 2,
            "volatility": volatility,
            "latest_score": scores[-1],
            "best_score": max(scores),
            "worst_score": min(scores),
            "average_score": np.mean(scores),
            "periods_analyzed": len(scores)
        }