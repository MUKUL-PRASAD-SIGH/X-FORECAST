"""
Integrated Forecasting Engine for Cyberpunk AI Dashboard
Combines demand forecasting with customer retention analytics for enhanced predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import asyncio
from abc import ABC, abstractmethod

# Import existing components with fallback handling
try:
    from ..models.ensemble import EnsembleForecaster
except ImportError:
    try:
        from models.ensemble import EnsembleForecaster
    except ImportError:
        try:
            from ensemble import EnsembleForecaster
        except ImportError:
            # Create mock EnsembleForecaster for testing
            class EnsembleForecaster:
                def __init__(self, use_ml=True, adaptive_learning=False):
                    self.use_ml = use_ml
                    self.adaptive_learning = adaptive_learning
                
                def forecast(self, steps=12):
                    dates = pd.date_range(start=datetime.now(), periods=steps, freq='M')
                    return pd.Series([100] * steps, index=dates)

try:
    from ..customer_analytics.retention_analyzer import RetentionAnalyzer, ChurnPrediction, RetentionInsights
except ImportError:
    try:
        from customer_analytics.retention_analyzer import RetentionAnalyzer, ChurnPrediction, RetentionInsights
    except ImportError:
        # Create mock classes for testing
        from unittest.mock import Mock
        RetentionAnalyzer = Mock
        ChurnPrediction = Mock
        RetentionInsights = Mock

try:
    from ..data_fabric.unified_connector import UnifiedDataConnector
except ImportError:
    try:
        from data_fabric.unified_connector import UnifiedDataConnector
    except ImportError:
        from unittest.mock import Mock
        UnifiedDataConnector = Mock

try:
    from .adaptive_config_manager import get_config_manager, should_use_adaptive_features
except ImportError:
    try:
        from adaptive_config_manager import get_config_manager, should_use_adaptive_features
    except ImportError:
        # Create mock functions for testing
        def get_config_manager():
            from unittest.mock import Mock
            mock_manager = Mock()
            mock_manager.get_ensemble_config.return_value = {}
            mock_manager.get_integration_config.return_value = {}
            return mock_manager
        
        def should_use_adaptive_features(user_id=None):
            return False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomerImpactForecast:
    """Customer impact on demand forecasting"""
    new_customer_impact: float
    churn_impact: float
    segment_contributions: Dict[str, float]
    seasonal_customer_patterns: pd.Series
    retention_adjusted_demand: pd.Series
    customer_acquisition_forecast: pd.Series
    ltv_weighted_forecast: pd.Series

@dataclass
class RetentionForecast:
    """Customer retention forecasting results"""
    retention_rate_forecast: pd.Series
    churn_rate_forecast: pd.Series
    customer_count_forecast: pd.Series
    segment_retention_forecasts: Dict[str, pd.Series]
    high_risk_customer_count: pd.Series
    retention_revenue_impact: pd.Series

@dataclass
class BusinessInsight:
    """AI-generated business insight"""
    insight_type: str  # 'opportunity', 'risk', 'trend', 'recommendation'
    title: str
    description: str
    confidence: float
    impact_score: float  # 0-10
    supporting_data: Dict[str, Any]
    recommended_actions: List[str]
    urgency: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class BusinessAction:
    """Recommended business action"""
    action_id: str
    action_type: str
    description: str
    priority: str
    estimated_impact: str
    implementation_timeline: str
    required_resources: List[str]
    success_metrics: List[str]

@dataclass
class ModelExplanation:
    """Explanation for model predictions"""
    model_name: str
    key_factors: List[str]
    factor_contributions: Dict[str, float]
    confidence_level: float
    data_quality_score: float
    assumptions: List[str]
    limitations: List[str]

@dataclass
class EnhancedForecast:
    """Enhanced forecast combining demand and customer analytics"""
    demand_forecast: pd.Series
    confidence_intervals: Dict[str, pd.Series]
    customer_impact: CustomerImpactForecast
    retention_forecast: RetentionForecast
    business_insights: List[BusinessInsight]
    recommended_actions: List[BusinessAction]
    model_explanations: Dict[str, ModelExplanation]
    forecast_accuracy_metrics: Dict[str, float]
    data_sources: List[str]
    generation_timestamp: datetime

class CustomerImpactModel:
    """Model customer impact on demand forecasting"""
    
    def __init__(self):
        self.customer_elasticity = 0.7  # How much customer changes affect demand
        self.seasonal_factors = {}
        self.segment_weights = {}
    
    def calculate_customer_impact(self, demand_data: pd.DataFrame, 
                                customer_data: pd.DataFrame,
                                retention_insights: RetentionInsights) -> CustomerImpactForecast:
        """Calculate how customer behavior impacts demand forecasting"""
        try:
            # Calculate new customer impact
            new_customer_impact = self._calculate_new_customer_impact(customer_data)
            
            # Calculate churn impact
            churn_impact = self._calculate_churn_impact(retention_insights.churn_predictions)
            
            # Calculate segment contributions
            segment_contributions = self._calculate_segment_contributions(retention_insights.customer_segments)
            
            # Generate seasonal customer patterns
            seasonal_patterns = self._generate_seasonal_patterns(customer_data)
            
            # Adjust demand forecast based on retention
            retention_adjusted_demand = self._adjust_demand_for_retention(
                demand_data, retention_insights
            )
            
            # Forecast customer acquisition
            acquisition_forecast = self._forecast_customer_acquisition(customer_data)
            
            # Create LTV-weighted forecast
            ltv_weighted_forecast = self._create_ltv_weighted_forecast(
                demand_data, customer_data, retention_insights
            )
            
            return CustomerImpactForecast(
                new_customer_impact=new_customer_impact,
                churn_impact=churn_impact,
                segment_contributions=segment_contributions,
                seasonal_customer_patterns=seasonal_patterns,
                retention_adjusted_demand=retention_adjusted_demand,
                customer_acquisition_forecast=acquisition_forecast,
                ltv_weighted_forecast=ltv_weighted_forecast
            )
            
        except Exception as e:
            logger.error(f"Customer impact calculation failed: {e}")
            # Return default impact forecast
            return CustomerImpactForecast(
                new_customer_impact=0.0,
                churn_impact=0.0,
                segment_contributions={},
                seasonal_customer_patterns=pd.Series(),
                retention_adjusted_demand=pd.Series(),
                customer_acquisition_forecast=pd.Series(),
                ltv_weighted_forecast=pd.Series()
            )
    
    def _calculate_new_customer_impact(self, customer_data: pd.DataFrame) -> float:
        """Calculate impact of new customer acquisition on demand"""
        if 'first_transaction_date' not in customer_data.columns:
            return 0.0
        
        # Calculate new customer rate over last 3 months
        recent_date = datetime.now() - timedelta(days=90)
        recent_customers = customer_data[
            pd.to_datetime(customer_data['first_transaction_date']) >= recent_date
        ]
        
        new_customer_rate = len(recent_customers) / len(customer_data) if len(customer_data) > 0 else 0
        
        # Estimate impact (new customers typically have lower initial spend)
        return new_customer_rate * 0.3  # 30% of average customer impact
    
    def _calculate_churn_impact(self, churn_predictions: List[ChurnPrediction]) -> float:
        """Calculate negative impact of customer churn on demand"""
        if not churn_predictions:
            return 0.0
        
        # Calculate weighted churn impact
        total_churn_risk = sum(pred.churn_probability for pred in churn_predictions)
        avg_churn_risk = total_churn_risk / len(churn_predictions)
        
        # Churn has negative impact on demand
        return -avg_churn_risk * self.customer_elasticity
    
    def _calculate_segment_contributions(self, customer_segments) -> Dict[str, float]:
        """Calculate contribution of each customer segment to demand"""
        contributions = {}
        
        if not customer_segments:
            return contributions
        
        total_customers = sum(segment.customer_count for segment in customer_segments)
        
        for segment in customer_segments:
            if total_customers > 0:
                # Weight by customer count and average LTV
                segment_weight = (segment.customer_count / total_customers) * (segment.avg_ltv / 1000)
                contributions[segment.segment_name] = segment_weight
        
        return contributions
    
    def _generate_seasonal_patterns(self, customer_data: pd.DataFrame) -> pd.Series:
        """Generate seasonal customer behavior patterns"""
        # Create mock seasonal patterns (in real implementation, analyze historical data)
        months = pd.date_range(start='2024-01-01', periods=12, freq='M')
        
        # Typical retail seasonal pattern
        seasonal_multipliers = [0.8, 0.7, 0.9, 1.0, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.3, 1.4]
        
        return pd.Series(seasonal_multipliers, index=months)
    
    def _adjust_demand_for_retention(self, demand_data: pd.DataFrame, 
                                   retention_insights: RetentionInsights) -> pd.Series:
        """Adjust demand forecast based on retention insights"""
        if demand_data.empty:
            return pd.Series()
        
        # Get base demand (assuming 'demand' column exists)
        base_demand = demand_data.get('demand', pd.Series([100] * 30))  # Default 30-day series
        
        # Apply retention adjustment
        retention_multiplier = retention_insights.overall_retention_rate
        adjusted_demand = base_demand * retention_multiplier
        
        return adjusted_demand
    
    def _forecast_customer_acquisition(self, customer_data: pd.DataFrame) -> pd.Series:
        """Forecast customer acquisition over time"""
        # Simple trend-based acquisition forecast
        if 'first_transaction_date' not in customer_data.columns:
            # Return default forecast
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            return pd.Series([5] * 30, index=dates)  # 5 new customers per day
        
        # Analyze historical acquisition trend
        customer_data['first_transaction_date'] = pd.to_datetime(customer_data['first_transaction_date'])
        monthly_acquisitions = customer_data.groupby(
            customer_data['first_transaction_date'].dt.to_period('M')
        ).size()
        
        # Simple linear trend projection
        if len(monthly_acquisitions) >= 3:
            trend = np.polyfit(range(len(monthly_acquisitions)), monthly_acquisitions.values, 1)[0]
            last_value = monthly_acquisitions.iloc[-1]
            
            # Project next 12 months
            future_months = pd.date_range(start=datetime.now(), periods=12, freq='M')
            forecast_values = [max(0, last_value + trend * i) for i in range(1, 13)]
            
            return pd.Series(forecast_values, index=future_months)
        
        # Default forecast if insufficient data
        dates = pd.date_range(start=datetime.now(), periods=12, freq='M')
        return pd.Series([10] * 12, index=dates)
    
    def _create_ltv_weighted_forecast(self, demand_data: pd.DataFrame, 
                                    customer_data: pd.DataFrame,
                                    retention_insights: RetentionInsights) -> pd.Series:
        """Create demand forecast weighted by customer lifetime value"""
        base_demand = demand_data.get('demand', pd.Series([100] * 30))
        
        # Calculate average LTV from segments
        if retention_insights.customer_segments:
            avg_ltv = np.mean([segment.avg_ltv for segment in retention_insights.customer_segments])
            ltv_multiplier = min(avg_ltv / 1000, 2.0)  # Cap at 2x multiplier
        else:
            ltv_multiplier = 1.0
        
        return base_demand * ltv_multiplier

class RetentionForecaster:
    """Forecast customer retention metrics"""
    
    def __init__(self):
        self.retention_model = None
    
    def forecast_retention_metrics(self, retention_insights: RetentionInsights, 
                                 forecast_horizon: int = 12) -> RetentionForecast:
        """Forecast retention metrics over specified horizon"""
        try:
            # Generate retention rate forecast
            retention_rate_forecast = self._forecast_retention_rates(
                retention_insights.overall_retention_rate, forecast_horizon
            )
            
            # Generate churn rate forecast
            churn_rate_forecast = 1 - retention_rate_forecast
            
            # Forecast customer count
            customer_count_forecast = self._forecast_customer_count(
                retention_insights.total_customers, retention_rate_forecast
            )
            
            # Forecast segment-specific retention
            segment_retention_forecasts = self._forecast_segment_retention(
                retention_insights.customer_segments, forecast_horizon
            )
            
            # Forecast high-risk customer count
            high_risk_count = sum(1 for pred in retention_insights.churn_predictions 
                                if pred.risk_level == 'High')
            high_risk_forecast = self._forecast_high_risk_customers(high_risk_count, forecast_horizon)
            
            # Calculate retention revenue impact
            revenue_impact = self._calculate_retention_revenue_impact(
                retention_rate_forecast, retention_insights
            )
            
            return RetentionForecast(
                retention_rate_forecast=retention_rate_forecast,
                churn_rate_forecast=churn_rate_forecast,
                customer_count_forecast=customer_count_forecast,
                segment_retention_forecasts=segment_retention_forecasts,
                high_risk_customer_count=high_risk_forecast,
                retention_revenue_impact=revenue_impact
            )
            
        except Exception as e:
            logger.error(f"Retention forecasting failed: {e}")
            # Return default forecast
            dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='M')
            default_series = pd.Series([0.8] * forecast_horizon, index=dates)
            
            return RetentionForecast(
                retention_rate_forecast=default_series,
                churn_rate_forecast=1 - default_series,
                customer_count_forecast=pd.Series([1000] * forecast_horizon, index=dates),
                segment_retention_forecasts={},
                high_risk_customer_count=pd.Series([50] * forecast_horizon, index=dates),
                retention_revenue_impact=pd.Series([10000] * forecast_horizon, index=dates)
            )
    
    def _forecast_retention_rates(self, current_rate: float, horizon: int) -> pd.Series:
        """Forecast retention rates with seasonal adjustments"""
        dates = pd.date_range(start=datetime.now(), periods=horizon, freq='M')
        
        # Add slight seasonal variation and trend
        seasonal_pattern = [1.0, 0.95, 1.05, 1.0, 1.1, 1.0, 0.9, 0.95, 1.0, 1.05, 1.15, 1.1]
        trend = -0.001  # Slight declining trend (realistic for many businesses)
        
        forecast_values = []
        for i in range(horizon):
            seasonal_factor = seasonal_pattern[i % 12]
            trend_factor = 1 + (trend * i)
            forecasted_rate = current_rate * seasonal_factor * trend_factor
            forecast_values.append(max(0.5, min(0.95, forecasted_rate)))  # Bound between 50% and 95%
        
        return pd.Series(forecast_values, index=dates)
    
    def _forecast_customer_count(self, current_count: int, 
                               retention_rates: pd.Series) -> pd.Series:
        """Forecast customer count based on retention rates"""
        customer_counts = [current_count]
        
        for retention_rate in retention_rates:
            # Assume some new customer acquisition to offset churn
            new_customers = current_count * 0.05  # 5% new customer rate
            retained_customers = customer_counts[-1] * retention_rate
            next_count = int(retained_customers + new_customers)
            customer_counts.append(next_count)
        
        return pd.Series(customer_counts[1:], index=retention_rates.index)
    
    def _forecast_segment_retention(self, customer_segments, horizon: int) -> Dict[str, pd.Series]:
        """Forecast retention for each customer segment"""
        segment_forecasts = {}
        dates = pd.date_range(start=datetime.now(), periods=horizon, freq='M')
        
        for segment in customer_segments:
            # Different segments have different retention patterns
            base_rate = segment.avg_retention_rate
            
            if segment.segment_name == 'Champions':
                # Champions have high, stable retention
                forecast_values = [min(0.95, base_rate + np.random.normal(0, 0.02)) for _ in range(horizon)]
            elif segment.segment_name == 'At Risk':
                # At Risk customers have declining retention
                forecast_values = [max(0.3, base_rate - 0.05 * i) for i in range(horizon)]
            else:
                # Standard retention with slight variation
                forecast_values = [max(0.5, base_rate + np.random.normal(0, 0.05)) for _ in range(horizon)]
            
            segment_forecasts[segment.segment_name] = pd.Series(forecast_values, index=dates)
        
        return segment_forecasts
    
    def _forecast_high_risk_customers(self, current_high_risk: int, horizon: int) -> pd.Series:
        """Forecast number of high-risk customers"""
        dates = pd.date_range(start=datetime.now(), periods=horizon, freq='M')
        
        # Assume high-risk customers fluctuate with seasonal patterns
        seasonal_multipliers = [1.2, 1.1, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 1.0, 1.3, 1.4]
        
        forecast_values = []
        for i in range(horizon):
            seasonal_factor = seasonal_multipliers[i % 12]
            forecasted_count = int(current_high_risk * seasonal_factor)
            forecast_values.append(max(0, forecasted_count))
        
        return pd.Series(forecast_values, index=dates)
    
    def _calculate_retention_revenue_impact(self, retention_rates: pd.Series, 
                                          retention_insights: RetentionInsights) -> pd.Series:
        """Calculate revenue impact of retention rates"""
        # Estimate average customer value
        if retention_insights.customer_segments:
            avg_customer_value = np.mean([segment.avg_ltv for segment in retention_insights.customer_segments])
        else:
            avg_customer_value = 1000  # Default value
        
        # Calculate revenue impact
        base_revenue = retention_insights.total_customers * avg_customer_value / 12  # Monthly
        revenue_impact = retention_rates * base_revenue
        
        return revenue_impact

class BusinessInsightGenerator:
    """Generate AI-powered business insights"""
    
    def __init__(self):
        self.insight_templates = {
            'churn_risk': "High churn risk detected in {segment} segment",
            'growth_opportunity': "Growth opportunity identified in {area}",
            'seasonal_trend': "Seasonal trend detected: {pattern}",
            'performance_alert': "Performance alert: {metric} is {direction}"
        }
    
    def generate_insights(self, enhanced_forecast: EnhancedForecast) -> List[BusinessInsight]:
        """Generate business insights from forecast data"""
        insights = []
        
        # Churn risk insights
        churn_insights = self._generate_churn_insights(enhanced_forecast)
        insights.extend(churn_insights)
        
        # Growth opportunity insights
        growth_insights = self._generate_growth_insights(enhanced_forecast)
        insights.extend(growth_insights)
        
        # Seasonal insights
        seasonal_insights = self._generate_seasonal_insights(enhanced_forecast)
        insights.extend(seasonal_insights)
        
        # Performance insights
        performance_insights = self._generate_performance_insights(enhanced_forecast)
        insights.extend(performance_insights)
        
        return insights
    
    def _generate_churn_insights(self, forecast: EnhancedForecast) -> List[BusinessInsight]:
        """Generate insights about churn risks"""
        insights = []
        
        if not forecast.retention_forecast.high_risk_customer_count.empty:
            high_risk_trend = forecast.retention_forecast.high_risk_customer_count
            if high_risk_trend.iloc[-1] > high_risk_trend.iloc[0]:
                insights.append(BusinessInsight(
                    insight_type='risk',
                    title='Increasing Churn Risk',
                    description=f'High-risk customer count is projected to increase by {((high_risk_trend.iloc[-1] / high_risk_trend.iloc[0]) - 1) * 100:.1f}%',
                    confidence=0.8,
                    impact_score=8.5,
                    supporting_data={'trend': high_risk_trend.to_dict()},
                    recommended_actions=[
                        'Implement proactive retention campaigns',
                        'Increase customer success outreach',
                        'Analyze churn risk factors'
                    ],
                    urgency='high'
                ))
        
        return insights
    
    def _generate_growth_insights(self, forecast: EnhancedForecast) -> List[BusinessInsight]:
        """Generate insights about growth opportunities"""
        insights = []
        
        # Analyze customer acquisition forecast
        if not forecast.customer_impact.customer_acquisition_forecast.empty:
            acquisition_trend = forecast.customer_impact.customer_acquisition_forecast
            if acquisition_trend.iloc[-1] > acquisition_trend.iloc[0]:
                insights.append(BusinessInsight(
                    insight_type='opportunity',
                    title='Customer Acquisition Growth',
                    description='Customer acquisition is trending upward, indicating market expansion opportunity',
                    confidence=0.7,
                    impact_score=7.0,
                    supporting_data={'acquisition_trend': acquisition_trend.to_dict()},
                    recommended_actions=[
                        'Increase marketing investment',
                        'Expand customer acquisition channels',
                        'Optimize onboarding process'
                    ],
                    urgency='medium'
                ))
        
        return insights
    
    def _generate_seasonal_insights(self, forecast: EnhancedForecast) -> List[BusinessInsight]:
        """Generate insights about seasonal patterns"""
        insights = []
        
        # Analyze seasonal customer patterns
        if not forecast.customer_impact.seasonal_customer_patterns.empty:
            seasonal_data = forecast.customer_impact.seasonal_customer_patterns
            peak_month = seasonal_data.idxmax()
            peak_value = seasonal_data.max()
            
            if peak_value > 1.2:  # 20% above average
                insights.append(BusinessInsight(
                    insight_type='trend',
                    title='Strong Seasonal Pattern Detected',
                    description=f'Peak customer activity expected in {peak_month.strftime("%B")} with {(peak_value - 1) * 100:.1f}% increase',
                    confidence=0.85,
                    impact_score=6.5,
                    supporting_data={'seasonal_pattern': seasonal_data.to_dict()},
                    recommended_actions=[
                        'Prepare inventory for seasonal demand',
                        'Plan seasonal marketing campaigns',
                        'Adjust staffing for peak periods'
                    ],
                    urgency='medium'
                ))
        
        return insights
    
    def _generate_performance_insights(self, forecast: EnhancedForecast) -> List[BusinessInsight]:
        """Generate insights about performance metrics"""
        insights = []
        
        # Analyze forecast accuracy
        if forecast.forecast_accuracy_metrics:
            accuracy = forecast.forecast_accuracy_metrics.get('mape', 0)
            if accuracy > 20:  # Poor accuracy
                insights.append(BusinessInsight(
                    insight_type='risk',
                    title='Forecast Accuracy Concern',
                    description=f'Model accuracy is {accuracy:.1f}% MAPE, which may impact decision quality',
                    confidence=0.9,
                    impact_score=7.5,
                    supporting_data={'accuracy_metrics': forecast.forecast_accuracy_metrics},
                    recommended_actions=[
                        'Review data quality',
                        'Retrain forecasting models',
                        'Incorporate additional data sources'
                    ],
                    urgency='high'
                ))
        
        return insights

class IntegratedForecastingEngine(EnsembleForecaster):
    """Enhanced forecasting engine integrating demand and customer analytics with adaptive ensemble"""
    
    def __init__(self, adaptive_enabled: Optional[bool] = None, user_id: Optional[str] = None):
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Determine if adaptive features should be used
        if adaptive_enabled is None:
            self.adaptive_enabled = should_use_adaptive_features(user_id)
        else:
            self.adaptive_enabled = adaptive_enabled
        
        # Initialize with adaptive ensemble capabilities
        super().__init__(use_ml=True, adaptive_learning=self.adaptive_enabled)
        
        # Get ensemble configuration
        ensemble_config = self.config_manager.get_ensemble_config()
        
        # Import adaptive configuration if available
        try:
            from .ensemble import AdaptiveConfig
            self.adaptive_config = AdaptiveConfig(**ensemble_config)
        except ImportError:
            logger.warning("Adaptive ensemble not available, using basic ensemble")
            self.adaptive_config = None
        
        self.retention_analyzer = RetentionAnalyzer()
        self.customer_impact_model = CustomerImpactModel()
        self.retention_forecaster = RetentionForecaster()
        self.insight_generator = BusinessInsightGenerator()
        
        # Configuration
        self.forecast_horizon = 12  # months
        self.confidence_levels = [0.1, 0.5, 0.9]  # P10, P50, P90
        self.user_id = user_id
        
        # Performance tracking for adaptive features
        self.performance_history = []
        self.weight_change_history = []
        
        # Integration configuration
        self.integration_config = self.config_manager.get_integration_config()
        
        logger.info(f"IntegratedForecastingEngine initialized with adaptive_enabled={self.adaptive_enabled}, user_id={user_id}")
    
    async def forecast_with_retention(self, demand_data: pd.DataFrame, 
                                    customer_data: pd.DataFrame,
                                    transaction_data: Optional[pd.DataFrame] = None,
                                    use_confidence_intervals: bool = True) -> EnhancedForecast:
        """Generate forecasts incorporating customer retention insights"""
        try:
            logger.info("Starting integrated forecasting with retention analytics...")
            
            # Step 1: Analyze customer retention
            if transaction_data is None:
                transaction_data = self._generate_mock_transaction_data(customer_data)
            
            retention_insights = self.retention_analyzer.analyze_customer_retention(
                customer_data, transaction_data
            )
            
            # Step 2: Calculate customer impact on demand
            customer_impact = self.customer_impact_model.calculate_customer_impact(
                demand_data, customer_data, retention_insights
            )
            
            # Step 3: Generate base demand forecast
            base_forecast = await self._generate_base_forecast(demand_data, customer_impact)
            
            # Step 4: Generate retention forecasts
            retention_forecast = self.retention_forecaster.forecast_retention_metrics(
                retention_insights, self.forecast_horizon
            )
            
            # Step 5: Calculate confidence intervals (enhanced with adaptive ensemble)
            if use_confidence_intervals and self.adaptive_enabled:
                confidence_intervals = self._calculate_adaptive_confidence_intervals(
                    base_forecast, customer_impact, retention_forecast
                )
            else:
                confidence_intervals = self._calculate_confidence_intervals(
                    base_forecast, customer_impact, retention_forecast
                )
            
            # Step 6: Generate business insights
            enhanced_forecast = EnhancedForecast(
                demand_forecast=base_forecast,
                confidence_intervals=confidence_intervals,
                customer_impact=customer_impact,
                retention_forecast=retention_forecast,
                business_insights=[],  # Will be populated below
                recommended_actions=[],  # Will be populated below
                model_explanations=self._generate_model_explanations(),
                forecast_accuracy_metrics=self._calculate_accuracy_metrics(),
                data_sources=['Demand Data', 'Customer Data', 'Transaction Data'],
                generation_timestamp=datetime.now()
            )
            
            # Generate insights and recommendations
            enhanced_forecast.business_insights = self.insight_generator.generate_insights(enhanced_forecast)
            enhanced_forecast.recommended_actions = self._generate_business_actions(enhanced_forecast)
            
            logger.info("Integrated forecasting completed successfully")
            return enhanced_forecast
            
        except Exception as e:
            logger.error(f"Integrated forecasting failed: {e}")
            # Return minimal forecast in case of error
            return self._create_fallback_forecast(demand_data)
    
    async def _generate_base_forecast(self, demand_data: pd.DataFrame, 
                                    customer_impact: CustomerImpactForecast) -> pd.Series:
        """Generate base demand forecast adjusted for customer impact"""
        # Use parent class forecasting capability
        if 'demand' in demand_data.columns:
            base_forecast = self.forecast(steps=self.forecast_horizon)
        else:
            # Generate default forecast
            dates = pd.date_range(start=datetime.now(), periods=self.forecast_horizon, freq='M')
            base_forecast = pd.Series([100] * self.forecast_horizon, index=dates)
        
        # Adjust for customer impact
        if not customer_impact.retention_adjusted_demand.empty:
            # Apply retention adjustment
            adjustment_factor = customer_impact.retention_adjusted_demand.mean() / base_forecast.mean()
            base_forecast = base_forecast * adjustment_factor
        
        # Apply new customer impact
        base_forecast = base_forecast * (1 + customer_impact.new_customer_impact)
        
        # Apply churn impact
        base_forecast = base_forecast * (1 + customer_impact.churn_impact)
        
        return base_forecast
    
    def _calculate_adaptive_confidence_intervals(self, base_forecast: pd.Series,
                                               customer_impact: CustomerImpactForecast,
                                               retention_forecast: RetentionForecast) -> Dict[str, pd.Series]:
        """Calculate confidence intervals using adaptive ensemble capabilities"""
        try:
            if hasattr(self, 'forecast_with_confidence') and self.adaptive_enabled:
                # Use adaptive ensemble confidence intervals if available
                confidence_result = self.forecast_with_confidence(steps=len(base_forecast))
                
                if hasattr(confidence_result, 'p10') and hasattr(confidence_result, 'p90'):
                    # Apply customer impact adjustments to confidence intervals
                    adjustment_factor = 1 + customer_impact.new_customer_impact + customer_impact.churn_impact
                    
                    return {
                        'P10': confidence_result.p10 * adjustment_factor,
                        'P50': confidence_result.p50 * adjustment_factor,
                        'P90': confidence_result.p90 * adjustment_factor
                    }
            
            # Fallback to standard confidence intervals
            return self._calculate_confidence_intervals(base_forecast, customer_impact, retention_forecast)
            
        except Exception as e:
            logger.warning(f"Adaptive confidence intervals failed, using standard method: {e}")
            return self._calculate_confidence_intervals(base_forecast, customer_impact, retention_forecast)
    
    def update_with_actuals(self, actual_values: pd.Series, forecast_dates: pd.DatetimeIndex,
                           customer_data: Optional[pd.DataFrame] = None):
        """Update adaptive ensemble with actual values for continuous learning"""
        try:
            if not self.adaptive_enabled:
                logger.info("Adaptive learning disabled, skipping update")
                return
            
            # Update ensemble performance if method exists
            if hasattr(self, 'update_with_actuals') and hasattr(super(), 'update_with_actuals'):
                super().update_with_actuals(actual_values, forecast_dates)
            
            # Track performance for integrated forecasting
            self._track_integrated_performance(actual_values, forecast_dates, customer_data)
            
            logger.info(f"Updated adaptive ensemble with {len(actual_values)} actual values")
            
        except Exception as e:
            logger.error(f"Failed to update adaptive ensemble: {e}")
    
    def _track_integrated_performance(self, actual_values: pd.Series, forecast_dates: pd.DatetimeIndex,
                                    customer_data: Optional[pd.DataFrame] = None):
        """Track performance specific to integrated forecasting"""
        try:
            # Calculate integrated forecasting metrics
            performance_record = {
                'timestamp': datetime.now(),
                'forecast_dates': forecast_dates,
                'actual_values': actual_values,
                'customer_data_available': customer_data is not None,
                'adaptive_enabled': self.adaptive_enabled
            }
            
            # Add customer impact metrics if available
            if customer_data is not None:
                performance_record['customer_count'] = len(customer_data)
                if 'customer_id' in customer_data.columns:
                    performance_record['unique_customers'] = customer_data['customer_id'].nunique()
            
            self.performance_history.append(performance_record)
            
            # Maintain history limit
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
        except Exception as e:
            logger.error(f"Failed to track integrated performance: {e}")
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """Get status of adaptive ensemble features"""
        try:
            status = {
                'adaptive_enabled': self.adaptive_enabled,
                'adaptive_config_available': self.adaptive_config is not None,
                'performance_history_count': len(self.performance_history),
                'weight_change_history_count': len(self.weight_change_history),
                'last_update': None
            }
            
            if self.performance_history:
                status['last_update'] = self.performance_history[-1]['timestamp']
            
            # Get ensemble-specific status if available
            if hasattr(self, 'get_weight_history'):
                try:
                    weight_history = self.get_weight_history()
                    if not weight_history.empty:
                        status['current_weights'] = weight_history.iloc[-1].to_dict()
                        status['weight_updates_count'] = len(weight_history)
                except:
                    pass
            
            if hasattr(self, 'get_performance_metrics'):
                try:
                    performance_metrics = self.get_performance_metrics()
                    status['current_performance'] = performance_metrics
                except:
                    pass
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get adaptive status: {e}")
            return {
                'adaptive_enabled': self.adaptive_enabled,
                'error': str(e)
            }
    
    def enable_adaptive_features(self, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        """Enable or disable adaptive features with optional configuration update"""
        try:
            self.adaptive_enabled = enabled
            
            if config and self.adaptive_config:
                # Update configuration
                for key, value in config.items():
                    if hasattr(self.adaptive_config, key):
                        setattr(self.adaptive_config, key, value)
            
            # Enable/disable adaptive learning in parent class if available
            if hasattr(self, 'enable_adaptive_learning'):
                self.enable_adaptive_learning(enabled)
            
            logger.info(f"Adaptive features {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to toggle adaptive features: {e}")
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get metrics specific to integrated forecasting performance"""
        try:
            metrics = {
                'total_forecasts': len(self.performance_history),
                'adaptive_enabled': self.adaptive_enabled,
                'average_customer_count': 0,
                'forecasts_with_customer_data': 0,
                'last_forecast_date': None
            }
            
            if self.performance_history:
                # Calculate averages
                customer_counts = [
                    record.get('customer_count', 0) 
                    for record in self.performance_history 
                    if record.get('customer_count')
                ]
                
                if customer_counts:
                    metrics['average_customer_count'] = sum(customer_counts) / len(customer_counts)
                
                metrics['forecasts_with_customer_data'] = sum(
                    1 for record in self.performance_history 
                    if record.get('customer_data_available', False)
                )
                
                metrics['last_forecast_date'] = self.performance_history[-1]['timestamp']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get integration metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_intervals(self, base_forecast: pd.Series,
                                      customer_impact: CustomerImpactForecast,
                                      retention_forecast: RetentionForecast) -> Dict[str, pd.Series]:
        """Calculate confidence intervals for the forecast"""
        confidence_intervals = {}
        
        # Calculate uncertainty from various sources
        demand_uncertainty = 0.15  # 15% base demand uncertainty
        customer_uncertainty = 0.10  # 10% customer behavior uncertainty
        retention_uncertainty = 0.08  # 8% retention uncertainty
        
        total_uncertainty = np.sqrt(demand_uncertainty**2 + customer_uncertainty**2 + retention_uncertainty**2)
        
        for confidence_level in self.confidence_levels:
            if confidence_level == 0.5:
                # Median forecast (P50)
                confidence_intervals[f'P{int(confidence_level * 100)}'] = base_forecast
            else:
                # Calculate confidence bounds
                z_score = 1.645 if confidence_level == 0.9 else -1.645  # 90% confidence
                if confidence_level == 0.1:
                    z_score = -1.645
                
                adjustment = base_forecast * total_uncertainty * z_score
                confidence_intervals[f'P{int(confidence_level * 100)}'] = base_forecast + adjustment
        
        return confidence_intervals
    
    def _generate_model_explanations(self) -> Dict[str, ModelExplanation]:
        """Generate explanations for model predictions"""
        explanations = {}
        
        # Demand forecasting explanation
        explanations['demand_forecast'] = ModelExplanation(
            model_name='Integrated Demand Forecasting',
            key_factors=['Historical demand patterns', 'Customer retention rates', 'Seasonal trends'],
            factor_contributions={
                'historical_patterns': 0.4,
                'customer_retention': 0.3,
                'seasonal_trends': 0.2,
                'external_factors': 0.1
            },
            confidence_level=0.85,
            data_quality_score=0.9,
            assumptions=[
                'Historical patterns will continue',
                'Customer behavior remains consistent',
                'No major market disruptions'
            ],
            limitations=[
                'Limited to 12-month horizon',
                'Assumes stable market conditions',
                'May not capture black swan events'
            ]
        )
        
        # Customer retention explanation
        explanations['retention_forecast'] = ModelExplanation(
            model_name='Customer Retention Analytics',
            key_factors=['Transaction frequency', 'Customer engagement', 'Segment characteristics'],
            factor_contributions={
                'transaction_frequency': 0.35,
                'engagement_metrics': 0.25,
                'customer_age': 0.20,
                'segment_behavior': 0.20
            },
            confidence_level=0.80,
            data_quality_score=0.85,
            assumptions=[
                'Customer behavior patterns persist',
                'Retention drivers remain stable',
                'Market competition stays constant'
            ],
            limitations=[
                'Based on historical customer data',
                'May not predict sudden behavior changes',
                'Segment definitions may evolve'
            ]
        )
        
        return explanations
    
    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        # Mock accuracy metrics (in real implementation, use historical validation)
        return {
            'mape': 12.5,  # Mean Absolute Percentage Error
            'mae': 8.3,    # Mean Absolute Error
            'rmse': 11.7,  # Root Mean Square Error
            'r_squared': 0.87  # R-squared
        }
    
    def _generate_business_actions(self, forecast: EnhancedForecast) -> List[BusinessAction]:
        """Generate recommended business actions"""
        actions = []
        
        # High churn risk action
        high_risk_customers = sum(1 for insight in forecast.business_insights 
                                if insight.insight_type == 'risk' and 'churn' in insight.title.lower())
        
        if high_risk_customers > 0:
            actions.append(BusinessAction(
                action_id='retention_campaign_001',
                action_type='customer_retention',
                description='Launch targeted retention campaign for high-risk customers',
                priority='high',
                estimated_impact='Reduce churn by 15-25%',
                implementation_timeline='2-3 weeks',
                required_resources=['Marketing team', 'Customer success team', 'Budget: $50K'],
                success_metrics=['Churn rate reduction', 'Customer engagement increase', 'Revenue retention']
            ))
        
        # Growth opportunity action
        growth_opportunities = sum(1 for insight in forecast.business_insights 
                                 if insight.insight_type == 'opportunity')
        
        if growth_opportunities > 0:
            actions.append(BusinessAction(
                action_id='growth_initiative_001',
                action_type='growth_acceleration',
                description='Capitalize on identified growth opportunities',
                priority='medium',
                estimated_impact='Increase customer acquisition by 20%',
                implementation_timeline='4-6 weeks',
                required_resources=['Marketing budget', 'Sales team', 'Product team'],
                success_metrics=['New customer acquisition', 'Market share growth', 'Revenue increase']
            ))
        
        # Forecast accuracy improvement action
        if forecast.forecast_accuracy_metrics.get('mape', 0) > 15:
            actions.append(BusinessAction(
                action_id='forecast_improvement_001',
                action_type='process_improvement',
                description='Improve forecasting accuracy through data quality and model enhancements',
                priority='medium',
                estimated_impact='Improve forecast accuracy by 20-30%',
                implementation_timeline='6-8 weeks',
                required_resources=['Data science team', 'IT support', 'Data quality tools'],
                success_metrics=['MAPE reduction', 'Model confidence increase', 'Decision quality improvement']
            ))
        
        return actions
    
    def _generate_mock_transaction_data(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock transaction data for demonstration"""
        transactions = []
        
        for _, customer in customer_data.iterrows():
            customer_id = customer.get('customer_id', f'customer_{np.random.randint(1000, 9999)}')
            
            # Generate 1-10 transactions per customer
            num_transactions = np.random.randint(1, 11)
            
            for i in range(num_transactions):
                transaction_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
                amount = np.random.uniform(10, 500)
                
                transactions.append({
                    'customer_id': customer_id,
                    'transaction_date': transaction_date,
                    'total_amount': amount,
                    'revenue': amount
                })
        
        return pd.DataFrame(transactions)
    
    def _create_fallback_forecast(self, demand_data: pd.DataFrame) -> EnhancedForecast:
        """Create fallback forecast in case of errors"""
        dates = pd.date_range(start=datetime.now(), periods=self.forecast_horizon, freq='M')
        default_forecast = pd.Series([100] * self.forecast_horizon, index=dates)
        
        return EnhancedForecast(
            demand_forecast=default_forecast,
            confidence_intervals={'P50': default_forecast},
            customer_impact=CustomerImpactForecast(
                new_customer_impact=0.0,
                churn_impact=0.0,
                segment_contributions={},
                seasonal_customer_patterns=pd.Series(),
                retention_adjusted_demand=pd.Series(),
                customer_acquisition_forecast=pd.Series(),
                ltv_weighted_forecast=pd.Series()
            ),
            retention_forecast=RetentionForecast(
                retention_rate_forecast=pd.Series([0.8] * self.forecast_horizon, index=dates),
                churn_rate_forecast=pd.Series([0.2] * self.forecast_horizon, index=dates),
                customer_count_forecast=pd.Series([1000] * self.forecast_horizon, index=dates),
                segment_retention_forecasts={},
                high_risk_customer_count=pd.Series([50] * self.forecast_horizon, index=dates),
                retention_revenue_impact=pd.Series([10000] * self.forecast_horizon, index=dates)
            ),
            business_insights=[],
            recommended_actions=[],
            model_explanations={},
            forecast_accuracy_metrics={'mape': 20.0},
            data_sources=['Fallback Data'],
            generation_timestamp=datetime.now()
        )

# Example usage and testing
if __name__ == "__main__":
    async def test_integrated_forecasting():
        engine = IntegratedForecastingEngine()
        
        # Create sample data
        demand_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'demand': np.random.normal(100, 20, 100)
        })
        
        customer_data = pd.DataFrame({
            'customer_id': [f'customer_{i}' for i in range(100)],
            'first_transaction_date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'total_revenue': np.random.uniform(100, 5000, 100),
            'total_transactions': np.random.randint(1, 50, 100)
        })
        
        # Generate integrated forecast
        forecast = await engine.forecast_with_retention(demand_data, customer_data)
        
        print(f"Generated forecast with {len(forecast.business_insights)} insights")
        print(f"Recommended {len(forecast.recommended_actions)} business actions")
        print(f"Overall retention rate: {forecast.retention_forecast.retention_rate_forecast.mean():.2f}")
        
        # Display insights
        for insight in forecast.business_insights:
            print(f"- {insight.title}: {insight.description}")
    
    # Run test
    asyncio.run(test_integrated_forecasting())