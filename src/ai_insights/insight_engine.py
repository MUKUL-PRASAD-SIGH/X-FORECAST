"""
AI Insight Generation Engine for Cyberpunk AI Dashboard
Automated business insights, pattern detection, and natural language generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Types of business insights"""
    TREND = "trend"
    ANOMALY = "anomaly"
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    PERFORMANCE = "performance"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"

class Urgency(Enum):
    """Urgency levels for insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BusinessInsight:
    """Business insight with metadata"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    impact_score: float  # 0-10
    urgency: Urgency
    supporting_data: Dict[str, Any]
    recommended_actions: List[str]
    tags: List[str]
    generated_at: datetime
    expires_at: Optional[datetime]

@dataclass
class PatternDetectionResult:
    """Result from pattern detection analysis"""
    pattern_type: str
    pattern_strength: float
    pattern_description: str
    affected_metrics: List[str]
    time_period: str
    statistical_significance: float

class InsightGenerator(ABC):
    """Abstract base class for insight generators"""
    
    @abstractmethod
    def generate_insights(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate insights from data"""
        pass
    
    @abstractmethod
    def get_generator_name(self) -> str:
        """Get the name of this generator"""
        pass

class TrendAnalysisGenerator(InsightGenerator):
    """Generate insights from trend analysis"""
    
    def __init__(self):
        self.trend_templates = {
            'increasing': [
                "{metric} is showing a strong upward trend with {change}% growth over {period}",
                "Significant growth detected in {metric}: {change}% increase in the last {period}",
                "{metric} has been consistently rising, up {change}% from {period} ago"
            ],
            'decreasing': [
                "{metric} is declining with a {change}% decrease over {period}",
                "Concerning downward trend in {metric}: {change}% drop in the last {period}",
                "{metric} has been falling consistently, down {change}% from {period} ago"
            ],
            'stable': [
                "{metric} remains stable with minimal variation over {period}",
                "{metric} shows consistent performance with {change}% variation in the last {period}"
            ],
            'volatile': [
                "{metric} shows high volatility with significant fluctuations over {period}",
                "Unstable pattern detected in {metric} with {volatility}% volatility in the last {period}"
            ]
        }
    
    def generate_insights(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate trend-based insights"""
        insights = []
        
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if len(data[column].dropna()) < 3:
                    continue
                
                # Calculate trend metrics
                values = data[column].dropna().values
                trend_result = self._analyze_trend(values, column)
                
                if trend_result:
                    insight = self._create_trend_insight(trend_result, column, context)
                    if insight:
                        insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error generating trend insights: {e}")
        
        return insights
    
    def _analyze_trend(self, values: np.ndarray, metric_name: str) -> Optional[Dict[str, Any]]:
        """Analyze trend in time series data"""
        if len(values) < 3:
            return None
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate trend strength (R-squared)
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate percentage change
        start_value = values[0]
        end_value = values[-1]
        pct_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
        
        # Calculate volatility
        volatility = np.std(values) / np.mean(values) * 100 if np.mean(values) != 0 else 0
        
        # Determine trend type
        if abs(pct_change) < 5 and volatility < 10:
            trend_type = 'stable'
        elif volatility > 30:
            trend_type = 'volatile'
        elif pct_change > 5:
            trend_type = 'increasing'
        elif pct_change < -5:
            trend_type = 'decreasing'
        else:
            trend_type = 'stable'
        
        return {
            'metric_name': metric_name,
            'trend_type': trend_type,
            'slope': slope,
            'r_squared': r_squared,
            'pct_change': pct_change,
            'volatility': volatility,
            'start_value': start_value,
            'end_value': end_value,
            'period': f"{len(values)} periods"
        }
    
    def _create_trend_insight(self, trend_result: Dict[str, Any], column: str, context: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Create business insight from trend analysis"""
        try:
            trend_type = trend_result['trend_type']
            pct_change = abs(trend_result['pct_change'])
            r_squared = trend_result['r_squared']
            
            # Skip weak trends
            if r_squared < 0.3 and trend_type not in ['volatile']:
                return None
            
            # Generate description
            template = np.random.choice(self.trend_templates[trend_type])
            description = template.format(
                metric=column.replace('_', ' ').title(),
                change=f"{pct_change:.1f}",
                period=trend_result['period'],
                volatility=f"{trend_result['volatility']:.1f}"
            )
            
            # Determine urgency and impact
            if trend_type == 'decreasing' and pct_change > 20:
                urgency = Urgency.HIGH
                impact_score = 8.0
            elif trend_type == 'increasing' and pct_change > 15:
                urgency = Urgency.MEDIUM
                impact_score = 7.0
            elif trend_type == 'volatile':
                urgency = Urgency.MEDIUM
                impact_score = 6.0
            else:
                urgency = Urgency.LOW
                impact_score = 4.0
            
            # Generate recommendations
            recommendations = self._generate_trend_recommendations(trend_type, column, trend_result)
            
            return BusinessInsight(
                insight_id=f"trend_{column}_{int(datetime.now().timestamp())}",
                insight_type=InsightType.TREND,
                title=f"{column.replace('_', ' ').title()} Trend Analysis",
                description=description,
                confidence=min(r_squared + 0.3, 1.0),
                impact_score=impact_score,
                urgency=urgency,
                supporting_data=trend_result,
                recommended_actions=recommendations,
                tags=['trend', 'analysis', column],
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=7)
            )
        
        except Exception as e:
            logger.error(f"Error creating trend insight: {e}")
            return None
    
    def _generate_trend_recommendations(self, trend_type: str, metric: str, trend_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend type"""
        recommendations = []
        
        if trend_type == 'decreasing':
            recommendations.extend([
                f"Investigate root causes of declining {metric}",
                f"Implement corrective measures to reverse the downward trend",
                f"Monitor {metric} closely for further deterioration"
            ])
        elif trend_type == 'increasing':
            recommendations.extend([
                f"Capitalize on the positive trend in {metric}",
                f"Analyze factors driving {metric} growth for replication",
                f"Ensure resources are adequate to sustain growth"
            ])
        elif trend_type == 'volatile':
            recommendations.extend([
                f"Identify sources of volatility in {metric}",
                f"Implement stabilization measures",
                f"Consider risk management strategies"
            ])
        else:  # stable
            recommendations.extend([
                f"Maintain current strategies for {metric}",
                f"Look for opportunities to improve {metric} performance"
            ])
        
        return recommendations
    
    def get_generator_name(self) -> str:
        return "TrendAnalysisGenerator"

class AnomalyDetectionGenerator(InsightGenerator):
    """Generate insights from anomaly detection"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    def generate_insights(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate anomaly-based insights"""
        insights = []
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty or len(numeric_data) < 10:
                return insights
            
            # Prepare data for anomaly detection
            scaled_data = self.scaler.fit_transform(numeric_data.fillna(0))
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(scaled_data)
            anomaly_indices = np.where(anomaly_scores == -1)[0]
            
            if len(anomaly_indices) > 0:
                # Analyze anomalies
                for idx in anomaly_indices:
                    anomaly_insight = self._analyze_anomaly(data.iloc[idx], numeric_data.columns, idx, context)
                    if anomaly_insight:
                        insights.append(anomaly_insight)
        
        except Exception as e:
            logger.error(f"Error generating anomaly insights: {e}")
        
        return insights
    
    def _analyze_anomaly(self, anomaly_row: pd.Series, columns: List[str], index: int, context: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Analyze a specific anomaly"""
        try:
            # Find the most anomalous features
            anomalous_features = []
            for col in columns:
                if pd.notna(anomaly_row[col]):
                    col_data = context.get('full_data', pd.DataFrame())[col].dropna()
                    if len(col_data) > 1:
                        z_score = abs((anomaly_row[col] - col_data.mean()) / col_data.std())
                        if z_score > 2:  # Significant deviation
                            anomalous_features.append({
                                'feature': col,
                                'value': anomaly_row[col],
                                'z_score': z_score,
                                'mean': col_data.mean(),
                                'std': col_data.std()
                            })
            
            if not anomalous_features:
                return None
            
            # Sort by z-score
            anomalous_features.sort(key=lambda x: x['z_score'], reverse=True)
            top_feature = anomalous_features[0]
            
            # Generate description
            description = f"Anomaly detected in {top_feature['feature'].replace('_', ' ')}: value {top_feature['value']:.2f} is {top_feature['z_score']:.1f} standard deviations from the mean ({top_feature['mean']:.2f})"
            
            # Determine impact and urgency
            max_z_score = max(f['z_score'] for f in anomalous_features)
            if max_z_score > 4:
                urgency = Urgency.CRITICAL
                impact_score = 9.0
            elif max_z_score > 3:
                urgency = Urgency.HIGH
                impact_score = 7.0
            else:
                urgency = Urgency.MEDIUM
                impact_score = 5.0
            
            # Generate recommendations
            recommendations = [
                f"Investigate the cause of anomalous {top_feature['feature']} value",
                "Verify data quality and collection processes",
                "Check for system errors or unusual business events",
                "Monitor for similar anomalies in the future"
            ]
            
            return BusinessInsight(
                insight_id=f"anomaly_{index}_{int(datetime.now().timestamp())}",
                insight_type=InsightType.ANOMALY,
                title=f"Anomaly Detected in {top_feature['feature'].replace('_', ' ').title()}",
                description=description,
                confidence=min(max_z_score / 5, 1.0),
                impact_score=impact_score,
                urgency=urgency,
                supporting_data={
                    'anomalous_features': anomalous_features,
                    'row_index': index,
                    'timestamp': anomaly_row.get('timestamp', 'unknown')
                },
                recommended_actions=recommendations,
                tags=['anomaly', 'detection', top_feature['feature']],
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=3)
            )
        
        except Exception as e:
            logger.error(f"Error analyzing anomaly: {e}")
            return None
    
    def get_generator_name(self) -> str:
        return "AnomalyDetectionGenerator"

class OpportunityDetectionGenerator(InsightGenerator):
    """Generate insights about business opportunities"""
    
    def __init__(self):
        self.opportunity_patterns = {
            'growth_acceleration': {
                'description': "Accelerating growth pattern detected in {metric}",
                'conditions': lambda data: self._check_acceleration(data),
                'impact': 8.0,
                'urgency': Urgency.HIGH
            },
            'seasonal_opportunity': {
                'description': "Seasonal opportunity identified in {metric} for {season}",
                'conditions': lambda data: self._check_seasonal_pattern(data),
                'impact': 6.0,
                'urgency': Urgency.MEDIUM
            },
            'correlation_opportunity': {
                'description': "Strong correlation found between {metric1} and {metric2}",
                'conditions': lambda data: self._check_correlations(data),
                'impact': 7.0,
                'urgency': Urgency.MEDIUM
            }
        }
    
    def generate_insights(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate opportunity-based insights"""
        insights = []
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            for pattern_name, pattern_config in self.opportunity_patterns.items():
                opportunity_data = pattern_config['conditions'](numeric_data)
                
                if opportunity_data:
                    insight = self._create_opportunity_insight(
                        pattern_name, pattern_config, opportunity_data, context
                    )
                    if insight:
                        insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error generating opportunity insights: {e}")
        
        return insights
    
    def _check_acceleration(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for accelerating growth patterns"""
        for column in data.columns:
            if len(data[column].dropna()) < 6:
                continue
            
            values = data[column].dropna().values
            
            # Calculate first and second derivatives
            first_diff = np.diff(values)
            second_diff = np.diff(first_diff)
            
            # Check if acceleration is consistently positive
            if len(second_diff) > 0 and np.mean(second_diff[-3:]) > 0:
                acceleration = np.mean(second_diff[-3:])
                if acceleration > np.std(second_diff):
                    return {
                        'metric': column,
                        'acceleration': acceleration,
                        'recent_growth': np.mean(first_diff[-3:]),
                        'pattern_strength': acceleration / np.std(second_diff)
                    }
        
        return None
    
    def _check_seasonal_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for seasonal opportunities"""
        # Simplified seasonal detection
        for column in data.columns:
            if len(data[column].dropna()) < 12:
                continue
            
            values = data[column].dropna().values
            
            # Simple seasonal pattern detection (monthly)
            if len(values) >= 12:
                monthly_avg = np.mean(values.reshape(-1, 12), axis=0)
                seasonal_strength = np.std(monthly_avg) / np.mean(monthly_avg)
                
                if seasonal_strength > 0.2:  # Significant seasonal variation
                    peak_month = np.argmax(monthly_avg)
                    return {
                        'metric': column,
                        'seasonal_strength': seasonal_strength,
                        'peak_month': peak_month + 1,
                        'peak_value': monthly_avg[peak_month],
                        'season': self._get_season_name(peak_month)
                    }
        
        return None
    
    def _check_correlations(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for strong correlations between metrics"""
        if len(data.columns) < 2:
            return None
        
        correlation_matrix = data.corr()
        
        # Find strong correlations (excluding diagonal)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            # Return the strongest correlation
            strongest = max(strong_correlations, key=lambda x: abs(x['correlation']))
            return strongest
        
        return None
    
    def _get_season_name(self, month: int) -> str:
        """Get season name from month number"""
        seasons = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        return seasons.get(month, 'Unknown')
    
    def _create_opportunity_insight(self, pattern_name: str, pattern_config: Dict[str, Any], 
                                  opportunity_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Create opportunity insight"""
        try:
            # Generate description based on pattern
            description_template = pattern_config['description']
            description = description_template.format(**opportunity_data)
            
            # Generate recommendations based on opportunity type
            recommendations = self._generate_opportunity_recommendations(pattern_name, opportunity_data)
            
            return BusinessInsight(
                insight_id=f"opportunity_{pattern_name}_{int(datetime.now().timestamp())}",
                insight_type=InsightType.OPPORTUNITY,
                title=f"Business Opportunity: {pattern_name.replace('_', ' ').title()}",
                description=description,
                confidence=0.8,
                impact_score=pattern_config['impact'],
                urgency=pattern_config['urgency'],
                supporting_data=opportunity_data,
                recommended_actions=recommendations,
                tags=['opportunity', pattern_name, opportunity_data.get('metric', 'unknown')],
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=14)
            )
        
        except Exception as e:
            logger.error(f"Error creating opportunity insight: {e}")
            return None
    
    def _generate_opportunity_recommendations(self, pattern_name: str, opportunity_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for opportunities"""
        recommendations = []
        
        if pattern_name == 'growth_acceleration':
            recommendations.extend([
                f"Invest additional resources in {opportunity_data.get('metric', 'this area')}",
                "Analyze factors driving acceleration for replication",
                "Prepare for increased capacity needs",
                "Monitor sustainability of growth acceleration"
            ])
        elif pattern_name == 'seasonal_opportunity':
            season = opportunity_data.get('season', 'peak season')
            recommendations.extend([
                f"Prepare marketing campaigns for {season}",
                f"Increase inventory levels before {season}",
                f"Optimize staffing for {season} demand",
                "Analyze historical performance during this season"
            ])
        elif pattern_name == 'correlation_opportunity':
            metric1 = opportunity_data.get('metric1', 'metric1')
            metric2 = opportunity_data.get('metric2', 'metric2')
            recommendations.extend([
                f"Leverage the relationship between {metric1} and {metric2}",
                f"Use {metric1} as a leading indicator for {metric2}",
                "Investigate causal relationships",
                "Develop joint optimization strategies"
            ])
        
        return recommendations
    
    def get_generator_name(self) -> str:
        return "OpportunityDetectionGenerator"

class NaturalLanguageGenerator:
    """Generate natural language explanations for insights"""
    
    def __init__(self):
        self.templates = {
            'forecast_explanation': [
                "The forecast indicates {direction} trend with {confidence}% confidence",
                "Based on historical patterns, we expect {metric} to {direction} by {amount}",
                "Our AI models predict {metric} will {direction} over the next {period}"
            ],
            'performance_summary': [
                "{metric} is performing {performance_level} with {current_value}",
                "Current {metric} performance is {performance_level} at {current_value}",
                "{metric} shows {performance_level} performance with a value of {current_value}"
            ],
            'risk_assessment': [
                "Risk level for {metric} is {risk_level} due to {risk_factors}",
                "{metric} presents {risk_level} risk based on {risk_factors}",
                "We assess {risk_level} risk in {metric} considering {risk_factors}"
            ]
        }
    
    def generate_explanation(self, insight: BusinessInsight, template_type: str = 'general') -> str:
        """Generate natural language explanation for an insight"""
        try:
            if template_type in self.templates:
                template = np.random.choice(self.templates[template_type])
                return template.format(**insight.supporting_data)
            else:
                return self._generate_general_explanation(insight)
        
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return insight.description
    
    def _generate_general_explanation(self, insight: BusinessInsight) -> str:
        """Generate general explanation for any insight"""
        explanation_parts = [
            f"Analysis of {insight.title.lower()} reveals:",
            insight.description
        ]
        
        if insight.confidence > 0.8:
            explanation_parts.append("This finding has high statistical confidence.")
        elif insight.confidence > 0.6:
            explanation_parts.append("This finding has moderate statistical confidence.")
        
        if insight.impact_score > 7:
            explanation_parts.append("This has significant business impact.")
        
        if insight.urgency == Urgency.CRITICAL:
            explanation_parts.append("Immediate attention is required.")
        elif insight.urgency == Urgency.HIGH:
            explanation_parts.append("This should be addressed promptly.")
        
        return " ".join(explanation_parts)

class InsightEngine:
    """Main insight generation engine"""
    
    def __init__(self):
        self.generators: List[InsightGenerator] = [
            TrendAnalysisGenerator(),
            AnomalyDetectionGenerator(),
            OpportunityDetectionGenerator()
        ]
        self.nlg = NaturalLanguageGenerator()
        self.insight_cache: Dict[str, BusinessInsight] = {}
        self.max_cache_size = 1000
    
    def generate_business_insights(self, data: pd.DataFrame, context: Dict[str, Any] = None) -> List[BusinessInsight]:
        """Generate comprehensive business insights from data"""
        if context is None:
            context = {}
        
        context['full_data'] = data  # Provide full data context to generators
        
        all_insights = []
        
        try:
            for generator in self.generators:
                logger.info(f"Running {generator.get_generator_name()}")
                insights = generator.generate_insights(data, context)
                all_insights.extend(insights)
            
            # Deduplicate and rank insights
            unique_insights = self._deduplicate_insights(all_insights)
            ranked_insights = self._rank_insights(unique_insights)
            
            # Cache insights
            for insight in ranked_insights:
                self.insight_cache[insight.insight_id] = insight
            
            # Manage cache size
            if len(self.insight_cache) > self.max_cache_size:
                self._cleanup_cache()
            
            logger.info(f"Generated {len(ranked_insights)} unique insights")
            return ranked_insights
        
        except Exception as e:
            logger.error(f"Error generating business insights: {e}")
            return []
    
    def _deduplicate_insights(self, insights: List[BusinessInsight]) -> List[BusinessInsight]:
        """Remove duplicate insights"""
        unique_insights = []
        seen_combinations = set()
        
        for insight in insights:
            # Create a signature for the insight
            signature = (
                insight.insight_type.value,
                insight.title.lower(),
                tuple(sorted(insight.tags))
            )
            
            if signature not in seen_combinations:
                seen_combinations.add(signature)
                unique_insights.append(insight)
        
        return unique_insights
    
    def _rank_insights(self, insights: List[BusinessInsight]) -> List[BusinessInsight]:
        """Rank insights by importance"""
        def insight_score(insight: BusinessInsight) -> float:
            urgency_weights = {
                Urgency.CRITICAL: 4.0,
                Urgency.HIGH: 3.0,
                Urgency.MEDIUM: 2.0,
                Urgency.LOW: 1.0
            }
            
            return (
                insight.impact_score * 0.4 +
                insight.confidence * 0.3 +
                urgency_weights[insight.urgency] * 0.3
            )
        
        return sorted(insights, key=insight_score, reverse=True)
    
    def _cleanup_cache(self):
        """Clean up expired insights from cache"""
        current_time = datetime.now()
        expired_keys = [
            key for key, insight in self.insight_cache.items()
            if insight.expires_at and insight.expires_at < current_time
        ]
        
        for key in expired_keys:
            del self.insight_cache[key]
        
        # If still too large, remove oldest insights
        if len(self.insight_cache) > self.max_cache_size:
            sorted_insights = sorted(
                self.insight_cache.items(),
                key=lambda x: x[1].generated_at
            )
            
            for key, _ in sorted_insights[:len(self.insight_cache) - self.max_cache_size]:
                del self.insight_cache[key]
    
    def get_cached_insights(self, max_age_hours: int = 24) -> List[BusinessInsight]:
        """Get cached insights within specified age"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        return [
            insight for insight in self.insight_cache.values()
            if insight.generated_at >= cutoff_time
        ]
    
    def explain_insight(self, insight_id: str, template_type: str = 'general') -> Optional[str]:
        """Generate natural language explanation for a specific insight"""
        if insight_id in self.insight_cache:
            insight = self.insight_cache[insight_id]
            return self.nlg.generate_explanation(insight, template_type)
        
        return None
    
    def get_insights_by_type(self, insight_type: InsightType, max_count: int = 10) -> List[BusinessInsight]:
        """Get insights filtered by type"""
        filtered_insights = [
            insight for insight in self.insight_cache.values()
            if insight.insight_type == insight_type
        ]
        
        return sorted(filtered_insights, key=lambda x: x.impact_score, reverse=True)[:max_count]
    
    def get_urgent_insights(self, urgency_level: Urgency = Urgency.HIGH) -> List[BusinessInsight]:
        """Get insights with specified urgency level or higher"""
        urgency_order = [Urgency.LOW, Urgency.MEDIUM, Urgency.HIGH, Urgency.CRITICAL]
        min_urgency_index = urgency_order.index(urgency_level)
        
        urgent_insights = [
            insight for insight in self.insight_cache.values()
            if urgency_order.index(insight.urgency) >= min_urgency_index
        ]
        
        return sorted(urgent_insights, key=lambda x: (urgency_order.index(x.urgency), x.impact_score), reverse=True)

# Example usage and testing
if __name__ == "__main__":
    def test_insight_engine():
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.cumsum(np.random.normal(1000, 200, 100)) + np.sin(np.arange(100) * 0.1) * 500,
            'customers': np.random.poisson(50, 100) + np.arange(100) * 0.5,
            'conversion_rate': np.random.beta(2, 8, 100),
            'inventory_level': 1000 - np.cumsum(np.random.normal(5, 2, 100))
        })
        
        # Add some anomalies
        data.loc[50, 'revenue'] = data['revenue'].mean() * 3  # Revenue spike
        data.loc[75, 'customers'] = 5  # Customer drop
        
        # Generate insights
        engine = InsightEngine()
        insights = engine.generate_business_insights(data)
        
        print(f"Generated {len(insights)} insights:")
        for insight in insights[:5]:  # Show top 5
            print(f"\n{insight.title}")
            print(f"Type: {insight.insight_type.value}")
            print(f"Urgency: {insight.urgency.value}")
            print(f"Impact: {insight.impact_score}/10")
            print(f"Confidence: {insight.confidence:.2f}")
            print(f"Description: {insight.description}")
            print(f"Actions: {', '.join(insight.recommended_actions[:2])}")
    
    test_insight_engine()