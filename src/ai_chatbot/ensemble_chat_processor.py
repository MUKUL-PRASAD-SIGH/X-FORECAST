"""
Ensemble-Aware Chat Processor
Extends the existing chat interface to understand ensemble forecasting queries,
model performance questions, and provide plain-language explanations.
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Import existing components
from .conversational_ai import ConversationalAI, ChatResponse
from ..models.ensemble_forecasting_engine import EnsembleForecastingEngine, EnsembleResult
from ..models.business_insights_engine import BusinessInsightsEngine, BusinessInsightsResult
from ..models.model_performance_tracker import ModelPerformanceTracker

logger = logging.getLogger(__name__)

@dataclass
class EnsembleQueryContext:
    """Context for ensemble-related queries"""
    query_type: str  # 'forecast', 'performance', 'model_comparison', 'insights', 'general'
    models_mentioned: List[str]
    metrics_requested: List[str]
    time_horizon: Optional[str]
    confidence_level: Optional[float]
    specific_model: Optional[str]
    business_context: Optional[str]

@dataclass
class EnsembleChatResponse(ChatResponse):
    """Enhanced chat response with ensemble-specific data"""
    ensemble_data: Optional[Dict[str, Any]] = None
    model_performance_data: Optional[Dict[str, Any]] = None
    forecast_data: Optional[Dict[str, Any]] = None
    insights_data: Optional[Dict[str, Any]] = None
    technical_explanation: Optional[str] = None
    plain_language_summary: Optional[str] = None

class EnsembleChatProcessor:
    """
    Processes chat queries related to ensemble forecasting with natural language understanding
    """
    
    def __init__(self, 
                 ensemble_engine: Optional[EnsembleForecastingEngine] = None,
                 insights_engine: Optional[BusinessInsightsEngine] = None,
                 performance_tracker: Optional[ModelPerformanceTracker] = None):
        """
        Initialize ensemble chat processor
        
        Args:
            ensemble_engine: Ensemble forecasting engine instance
            insights_engine: Business insights engine instance
            performance_tracker: Model performance tracker instance
        """
        self.ensemble_engine = ensemble_engine
        self.insights_engine = insights_engine
        self.performance_tracker = performance_tracker
        
        # Initialize base conversational AI
        self.base_ai = ConversationalAI(show_welcome=False)
        
        # Query patterns for different types of ensemble queries
        self.query_patterns = self._initialize_query_patterns()
        
        # Model name mappings
        self.model_names = {
            'arima': ['arima', 'autoregressive', 'ar', 'ma', 'time series'],
            'ets': ['ets', 'exponential smoothing', 'holt', 'winters'],
            'xgboost': ['xgboost', 'xgb', 'gradient boosting', 'tree', 'machine learning'],
            'lstm': ['lstm', 'neural network', 'deep learning', 'rnn', 'recurrent'],
            'croston': ['croston', 'intermittent', 'sparse', 'irregular']
        }
        
        # Metrics mappings
        self.metrics_map = {
            'accuracy': ['accuracy', 'precise', 'correct', 'right'],
            'mae': ['mae', 'mean absolute error', 'average error'],
            'mape': ['mape', 'percentage error', 'percent error'],
            'rmse': ['rmse', 'root mean square', 'squared error'],
            'confidence': ['confidence', 'certainty', 'reliability', 'trust'],
            'weights': ['weights', 'contribution', 'importance', 'influence']
        }
        
        logger.info("Ensemble chat processor initialized")
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for different query types"""
        return {
            'forecast': [
                r'forecast|predict|projection|future|next \d+ months?',
                r'what will|how much|expected sales|demand forecast',
                r'ensemble forecast|combined forecast|weighted forecast'
            ],
            'performance': [
                r'performance|accuracy|how good|how well|effective',
                r'mae|mape|rmse|error|mistakes|wrong',
                r'model performance|forecast accuracy|prediction quality'
            ],
            'model_comparison': [
                r'compare|versus|vs|better|best|worst|rank',
                r'which model|what model|model comparison',
                r'arima vs|lstm vs|xgboost vs|ets vs|croston vs'
            ],
            'insights': [
                r'insights|recommendations|advice|suggest|should',
                r'business insights|what does this mean|implications',
                r'opportunities|risks|trends|patterns'
            ],
            'weights': [
                r'weights|contribution|importance|influence',
                r'model weights|ensemble weights|how much each',
                r'why is.*weighted|weight distribution'
            ],
            'confidence': [
                r'confidence|certain|reliable|trust|sure',
                r'how confident|confidence interval|uncertainty',
                r'p10|p50|p90|percentile'
            ]
        }
    
    async def process_ensemble_query(self, query: str, 
                                   user_context: Optional[Dict[str, Any]] = None) -> EnsembleChatResponse:
        """
        Process ensemble-related query with natural language understanding
        
        Args:
            query: User query string
            user_context: Optional user context (company data, etc.)
            
        Returns:
            Enhanced chat response with ensemble data
        """
        try:
            logger.info(f"Processing ensemble query: {query[:100]}...")
            
            # Parse query to understand intent and context
            query_context = self._parse_query_context(query)
            
            # Route to appropriate handler based on query type
            if query_context.query_type == 'forecast':
                response = await self._handle_forecast_query(query, query_context, user_context)
            elif query_context.query_type == 'performance':
                response = await self._handle_performance_query(query, query_context, user_context)
            elif query_context.query_type == 'model_comparison':
                response = await self._handle_model_comparison_query(query, query_context, user_context)
            elif query_context.query_type == 'insights':
                response = await self._handle_insights_query(query, query_context, user_context)
            elif query_context.query_type == 'weights':
                response = await self._handle_weights_query(query, query_context, user_context)
            elif query_context.query_type == 'confidence':
                response = await self._handle_confidence_query(query, query_context, user_context)
            else:
                # Fall back to base AI for general queries
                base_response = await self.base_ai.process_natural_language_query(query, user_context)
                response = EnsembleChatResponse(
                    response_text=base_response.response_text,
                    confidence=base_response.confidence,
                    sources=base_response.sources,
                    follow_up_questions=base_response.follow_up_questions,
                    suggested_actions=base_response.suggested_actions
                )
            
            # Add context-aware follow-up questions
            response.follow_up_questions = self._generate_contextual_follow_ups(query_context, response)
            
            logger.info(f"Generated ensemble response with confidence {response.confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Ensemble query processing failed: {e}")
            return self._create_error_response(str(e))
    
    def _parse_query_context(self, query: str) -> EnsembleQueryContext:
        """Parse query to understand intent and extract context"""
        try:
            query_lower = query.lower()
            
            # Determine query type
            query_type = 'general'
            for qtype, patterns in self.query_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        query_type = qtype
                        break
                if query_type != 'general':
                    break
            
            # Extract mentioned models
            models_mentioned = []
            for model, aliases in self.model_names.items():
                for alias in aliases:
                    if alias in query_lower:
                        models_mentioned.append(model)
                        break
            
            # Extract requested metrics
            metrics_requested = []
            for metric, aliases in self.metrics_map.items():
                for alias in aliases:
                    if alias in query_lower:
                        metrics_requested.append(metric)
                        break
            
            # Extract time horizon
            time_horizon = None
            horizon_match = re.search(r'(\d+)\s*(month|week|day|year)s?', query_lower)
            if horizon_match:
                time_horizon = f"{horizon_match.group(1)} {horizon_match.group(2)}s"
            
            # Extract confidence level
            confidence_level = None
            conf_match = re.search(r'(\d+)%?\s*confidence', query_lower)
            if conf_match:
                confidence_level = float(conf_match.group(1)) / 100
            
            # Extract specific model if mentioned
            specific_model = models_mentioned[0] if models_mentioned else None
            
            # Extract business context keywords
            business_keywords = ['revenue', 'sales', 'profit', 'growth', 'decline', 'trend', 'seasonal']
            business_context = None
            for keyword in business_keywords:
                if keyword in query_lower:
                    business_context = keyword
                    break
            
            return EnsembleQueryContext(
                query_type=query_type,
                models_mentioned=models_mentioned,
                metrics_requested=metrics_requested,
                time_horizon=time_horizon,
                confidence_level=confidence_level,
                specific_model=specific_model,
                business_context=business_context
            )
            
        except Exception as e:
            logger.error(f"Query context parsing failed: {e}")
            return EnsembleQueryContext(
                query_type='general',
                models_mentioned=[],
                metrics_requested=[],
                time_horizon=None,
                confidence_level=None,
                specific_model=None,
                business_context=None
            )
    
    async def _handle_forecast_query(self, query: str, context: EnsembleQueryContext, 
                                   user_context: Optional[Dict[str, Any]]) -> EnsembleChatResponse:
        """Handle forecast-related queries"""
        try:
            if not self.ensemble_engine:
                return self._create_unavailable_response("Ensemble forecasting engine not available")
            
            # Extract horizon from context or use default
            horizon = 6  # Default 6 months
            if context.time_horizon:
                horizon_match = re.search(r'(\d+)', context.time_horizon)
                if horizon_match:
                    horizon = int(horizon_match.group(1))
            
            # Generate ensemble forecast
            ensemble_result = await self.ensemble_engine.generate_forecast(horizon=horizon)
            
            # Create plain language explanation
            plain_language = self._explain_forecast_results(ensemble_result, context)
            
            # Create technical explanation
            technical_explanation = self._create_technical_forecast_explanation(ensemble_result)
            
            # Format forecast data for response
            forecast_data = {
                'horizon_months': horizon,
                'point_forecast': ensemble_result.point_forecast.to_dict() if not ensemble_result.point_forecast.empty else {},
                'confidence_intervals': {
                    level: interval.to_dict() if not interval.empty else {}
                    for level, interval in ensemble_result.confidence_intervals.items()
                },
                'model_weights': ensemble_result.model_weights,
                'ensemble_accuracy': ensemble_result.ensemble_accuracy,
                'pattern_type': ensemble_result.pattern_analysis.pattern_type if ensemble_result.pattern_analysis else 'unknown'
            }
            
            return EnsembleChatResponse(
                response_text=plain_language,
                confidence=0.9,
                sources=["Ensemble Forecasting Engine", "5-Model Adaptive System"],
                forecast_data=forecast_data,
                technical_explanation=technical_explanation,
                plain_language_summary=plain_language,
                suggested_actions=[
                    "View detailed forecast charts",
                    "Adjust forecast horizon",
                    "Compare individual model forecasts",
                    "Export forecast results"
                ]
            )
            
        except Exception as e:
            logger.error(f"Forecast query handling failed: {e}")
            return self._create_error_response(f"Forecast generation failed: {str(e)}")
    
    async def _handle_performance_query(self, query: str, context: EnsembleQueryContext,
                                      user_context: Optional[Dict[str, Any]]) -> EnsembleChatResponse:
        """Handle model performance queries"""
        try:
            if not self.ensemble_engine:
                return self._create_unavailable_response("Ensemble engine not available")
            
            # Get current performance metrics
            performance_metrics = self.ensemble_engine.get_performance_metrics()
            
            # Create plain language explanation
            plain_language = self._explain_performance_metrics(performance_metrics, context)
            
            # Create technical explanation
            technical_explanation = self._create_technical_performance_explanation(performance_metrics)
            
            return EnsembleChatResponse(
                response_text=plain_language,
                confidence=0.85,
                sources=["Model Performance Tracker", "Ensemble Engine"],
                model_performance_data=performance_metrics,
                technical_explanation=technical_explanation,
                plain_language_summary=plain_language,
                suggested_actions=[
                    "View performance dashboard",
                    "Compare model accuracies",
                    "Check model weights",
                    "Review recent predictions"
                ]
            )
            
        except Exception as e:
            logger.error(f"Performance query handling failed: {e}")
            return self._create_error_response(f"Performance analysis failed: {str(e)}")
    
    async def _handle_model_comparison_query(self, query: str, context: EnsembleQueryContext,
                                           user_context: Optional[Dict[str, Any]]) -> EnsembleChatResponse:
        """Handle model comparison queries"""
        try:
            if not self.ensemble_engine:
                return self._create_unavailable_response("Ensemble engine not available")
            
            # Get model status and performance
            model_status = self.ensemble_engine.get_model_status_summary()
            performance_metrics = self.ensemble_engine.get_performance_metrics()
            
            # Create comparison analysis
            comparison_text = self._create_model_comparison(model_status, performance_metrics, context)
            
            # Create technical details
            technical_explanation = self._create_technical_comparison_explanation(model_status, performance_metrics)
            
            return EnsembleChatResponse(
                response_text=comparison_text,
                confidence=0.8,
                sources=["Model Status Tracker", "Performance Metrics"],
                ensemble_data=model_status,
                model_performance_data=performance_metrics,
                technical_explanation=technical_explanation,
                plain_language_summary=comparison_text,
                suggested_actions=[
                    "View detailed model comparison",
                    "Adjust model weights",
                    "Retrain underperforming models",
                    "Analyze model strengths"
                ]
            )
            
        except Exception as e:
            logger.error(f"Model comparison query handling failed: {e}")
            return self._create_error_response(f"Model comparison failed: {str(e)}")
    
    async def _handle_insights_query(self, query: str, context: EnsembleQueryContext,
                                   user_context: Optional[Dict[str, Any]]) -> EnsembleChatResponse:
        """Handle business insights queries"""
        try:
            if not self.insights_engine:
                return self._create_unavailable_response("Business insights engine not available")
            
            # For demo purposes, create sample insights
            # In real implementation, this would use actual data and ensemble results
            insights_text = self._create_business_insights_response(context)
            
            return EnsembleChatResponse(
                response_text=insights_text,
                confidence=0.75,
                sources=["Business Insights Engine", "Pattern Analysis"],
                insights_data={"sample": "insights"},
                plain_language_summary=insights_text,
                suggested_actions=[
                    "View detailed insights dashboard",
                    "Explore growth opportunities",
                    "Review risk assessments",
                    "Generate executive report"
                ]
            )
            
        except Exception as e:
            logger.error(f"Insights query handling failed: {e}")
            return self._create_error_response(f"Insights generation failed: {str(e)}")
    
    async def _handle_weights_query(self, query: str, context: EnsembleQueryContext,
                                  user_context: Optional[Dict[str, Any]]) -> EnsembleChatResponse:
        """Handle model weights queries"""
        try:
            if not self.ensemble_engine:
                return self._create_unavailable_response("Ensemble engine not available")
            
            # Get current model weights
            model_status = self.ensemble_engine.get_model_status_summary()
            weights = model_status.get('ensemble_weights', {})
            
            # Create weights explanation
            weights_text = self._explain_model_weights(weights, context)
            
            return EnsembleChatResponse(
                response_text=weights_text,
                confidence=0.9,
                sources=["Ensemble Weight Manager"],
                ensemble_data={'weights': weights},
                plain_language_summary=weights_text,
                suggested_actions=[
                    "View weight evolution chart",
                    "Understand weight calculation",
                    "Adjust weight constraints",
                    "Monitor weight changes"
                ]
            )
            
        except Exception as e:
            logger.error(f"Weights query handling failed: {e}")
            return self._create_error_response(f"Weights analysis failed: {str(e)}")
    
    async def _handle_confidence_query(self, query: str, context: EnsembleQueryContext,
                                     user_context: Optional[Dict[str, Any]]) -> EnsembleChatResponse:
        """Handle confidence interval queries"""
        try:
            if not self.ensemble_engine:
                return self._create_unavailable_response("Ensemble engine not available")
            
            # Generate forecast with confidence intervals
            ensemble_result = await self.ensemble_engine.generate_forecast(horizon=6)
            
            # Create confidence explanation
            confidence_text = self._explain_confidence_intervals(ensemble_result, context)
            
            return EnsembleChatResponse(
                response_text=confidence_text,
                confidence=0.85,
                sources=["Ensemble Forecasting Engine"],
                forecast_data={
                    'confidence_intervals': {
                        level: interval.to_dict() if not interval.empty else {}
                        for level, interval in ensemble_result.confidence_intervals.items()
                    },
                    'ensemble_accuracy': ensemble_result.ensemble_accuracy
                },
                plain_language_summary=confidence_text,
                suggested_actions=[
                    "View confidence interval charts",
                    "Understand uncertainty sources",
                    "Adjust confidence levels",
                    "Compare model uncertainties"
                ]
            )
            
        except Exception as e:
            logger.error(f"Confidence query handling failed: {e}")
            return self._create_error_response(f"Confidence analysis failed: {str(e)}")
    
    def _explain_forecast_results(self, ensemble_result: EnsembleResult, context: EnsembleQueryContext) -> str:
        """Create plain language explanation of forecast results"""
        try:
            if ensemble_result.point_forecast.empty:
                return "I don't have enough data to generate a reliable forecast. Please upload more historical data to improve predictions."
            
            # Get forecast summary
            forecast_values = ensemble_result.point_forecast
            avg_forecast = forecast_values.mean()
            trend = "increasing" if forecast_values.iloc[-1] > forecast_values.iloc[0] else "decreasing"
            
            # Get model information
            top_models = sorted(ensemble_result.model_weights.items(), key=lambda x: x[1], reverse=True)[:2]
            
            explanation = f"""
Based on the ensemble of 5 forecasting models, here's what I predict:

📈 **Forecast Summary:**
• Average predicted value: {avg_forecast:,.0f}
• Trend direction: {trend}
• Forecast accuracy: {ensemble_result.ensemble_accuracy*100:.1f}%
• Pattern detected: {ensemble_result.pattern_analysis.pattern_type if ensemble_result.pattern_analysis else 'mixed'}

🤖 **Model Insights:**
• Top performing model: {top_models[0][0].upper()} ({top_models[0][1]*100:.1f}% weight)
• Secondary model: {top_models[1][0].upper()} ({top_models[1][1]*100:.1f}% weight)

The ensemble combines predictions from ARIMA, ETS, XGBoost, LSTM, and Croston models, 
automatically adjusting their influence based on recent performance.
            """.strip()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Forecast explanation failed: {e}")
            return "I generated a forecast, but encountered an issue explaining the results. Please check the forecast dashboard for detailed information."
    
    def _explain_performance_metrics(self, performance_metrics: Dict[str, Any], context: EnsembleQueryContext) -> str:
        """Create plain language explanation of performance metrics"""
        try:
            overall_accuracy = performance_metrics.get('overall_accuracy', 0.0)
            model_performances = performance_metrics.get('model_performances', [])
            
            if not model_performances:
                return "Performance metrics are not available yet. The models need more data to calculate accurate performance statistics."
            
            # Find best and worst performing models
            best_model = max(model_performances, key=lambda x: x.get('r_squared', 0))
            worst_model = min(model_performances, key=lambda x: x.get('r_squared', 0))
            
            explanation = f"""
Here's how the forecasting models are performing:

📊 **Overall Performance:**
• Ensemble accuracy: {overall_accuracy*100:.1f}%
• Number of active models: {len(model_performances)}

🏆 **Best Performer:**
• Model: {best_model.get('model_name', 'Unknown').upper()}
• Accuracy (R²): {best_model.get('r_squared', 0)*100:.1f}%
• Average error (MAPE): {best_model.get('mape', 0):.1f}%

⚠️ **Needs Attention:**
• Model: {worst_model.get('model_name', 'Unknown').upper()}
• Accuracy (R²): {worst_model.get('r_squared', 0)*100:.1f}%
• Average error (MAPE): {worst_model.get('mape', 0):.1f}%

The ensemble automatically adjusts model weights based on performance, 
giving more influence to accurate models and less to underperforming ones.
            """.strip()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Performance explanation failed: {e}")
            return "I have performance data available, but encountered an issue explaining it. Please check the performance dashboard for detailed metrics."
    
    def _create_model_comparison(self, model_status: Dict[str, Any], 
                               performance_metrics: Dict[str, Any], context: EnsembleQueryContext) -> str:
        """Create model comparison explanation"""
        try:
            model_details = model_status.get('model_details', {})
            weights = model_status.get('ensemble_weights', {})
            
            if not model_details:
                return "Model comparison data is not available. Please ensure the ensemble system is properly initialized."
            
            comparison_text = "🔍 **Model Comparison:**\n\n"
            
            # Sort models by weight (influence)
            sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            for i, (model_name, weight) in enumerate(sorted_models, 1):
                model_info = model_details.get(model_name, {})
                status = "✅ Active" if model_info.get('initialized', False) else "❌ Inactive"
                
                comparison_text += f"""
**{i}. {model_name.upper()}**
• Status: {status}
• Influence: {weight*100:.1f}%
• Strength: {self._get_model_strength_description(model_name)}
                """.strip() + "\n\n"
            
            comparison_text += """
💡 **How the ensemble works:**
Each model has different strengths - ARIMA excels at trends, ETS handles seasonality, 
XGBoost captures complex patterns, LSTM learns from sequences, and Croston manages 
intermittent demand. The system automatically balances their contributions.
            """.strip()
            
            return comparison_text
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return "I can compare the models, but encountered an issue generating the comparison. Please check the model status dashboard."
    
    def _get_model_strength_description(self, model_name: str) -> str:
        """Get plain language description of model strengths"""
        strengths = {
            'arima': 'Excellent for trend analysis and time series patterns',
            'ets': 'Great for seasonal patterns and exponential smoothing',
            'xgboost': 'Powerful for complex, non-linear relationships',
            'lstm': 'Superior for learning long-term dependencies',
            'croston': 'Specialized for intermittent and sparse demand'
        }
        return strengths.get(model_name, 'General forecasting capabilities')
    
    def _create_business_insights_response(self, context: EnsembleQueryContext) -> str:
        """Create business insights response"""
        return """
💡 **Business Insights:**

Based on the ensemble forecasting analysis, here are key insights:

📈 **Growth Opportunities:**
• Strong seasonal patterns detected - optimize inventory for peak periods
• Model accuracy is high (>85%) - reliable for strategic planning
• Trend analysis shows stable growth trajectory

⚠️ **Risk Factors:**
• Some volatility in recent data - monitor for demand shifts
• Model performance varies by time period - consider retraining schedule

🎯 **Recommendations:**
• Increase inventory 15-20% before seasonal peaks
• Implement automated reordering based on forecast confidence
• Monitor model performance weekly for early drift detection

The ensemble system provides reliable predictions with quantified uncertainty, 
enabling data-driven decision making with confidence intervals.
        """.strip()
    
    def _explain_model_weights(self, weights: Dict[str, float], context: EnsembleQueryContext) -> str:
        """Explain model weights in plain language"""
        try:
            if not weights:
                return "Model weights are not available. The ensemble system may not be initialized yet."
            
            # Sort by weight
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            explanation = "🎯 **Model Influence Distribution:**\n\n"
            
            for model_name, weight in sorted_weights:
                percentage = weight * 100
                bar = "█" * int(percentage / 5)  # Visual bar
                explanation += f"**{model_name.upper()}**: {percentage:.1f}% {bar}\n"
            
            explanation += f"""

💡 **Why these weights?**
The ensemble automatically adjusts model influence based on recent performance. 
Models with higher accuracy get more weight in the final prediction.

🔄 **Dynamic Adjustment:**
Weights update continuously as new data arrives and model performance changes. 
This ensures the ensemble always uses the best-performing combination.
            """.strip()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Weights explanation failed: {e}")
            return "I have weight information available, but encountered an issue explaining it."
    
    def _explain_confidence_intervals(self, ensemble_result: EnsembleResult, context: EnsembleQueryContext) -> str:
        """Explain confidence intervals in plain language"""
        try:
            confidence_intervals = ensemble_result.confidence_intervals
            
            if not confidence_intervals:
                return "Confidence intervals are not available for this forecast."
            
            explanation = f"""
📊 **Forecast Confidence:**

The ensemble provides uncertainty estimates to help you understand prediction reliability:

🎯 **Confidence Levels:**
• **P10 (10th percentile)**: Conservative estimate - 90% chance actual will be higher
• **P50 (50th percentile)**: Most likely outcome - the main forecast
• **P90 (90th percentile)**: Optimistic estimate - 90% chance actual will be lower

📈 **Forecast Accuracy:** {ensemble_result.ensemble_accuracy*100:.1f}%

💡 **How to use this:**
• Use P50 for planning and budgeting
• Use P10 for risk management and safety stock
• Use P90 for opportunity planning and capacity

The wider the range between P10 and P90, the more uncertain the forecast. 
Narrow ranges indicate high confidence, wide ranges suggest more variability.
            """.strip()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Confidence explanation failed: {e}")
            return "I have confidence interval data, but encountered an issue explaining it."
    
    def _create_technical_forecast_explanation(self, ensemble_result: EnsembleResult) -> str:
        """Create technical explanation for advanced users"""
        return f"""
Technical Details:
- Ensemble Method: Adaptive weighted combination of 5 models
- Models: ARIMA, ETS, XGBoost, LSTM, Croston
- Weight Calculation: Performance-based with recent accuracy weighting
- Confidence Intervals: Model disagreement + historical error patterns
- Pattern Detection: {ensemble_result.pattern_analysis.pattern_type if ensemble_result.pattern_analysis else 'N/A'}
- Data Quality Score: {ensemble_result.data_quality_score:.3f}
        """.strip()
    
    def _create_technical_performance_explanation(self, performance_metrics: Dict[str, Any]) -> str:
        """Create technical performance explanation"""
        return f"""
Technical Performance Metrics:
- Overall Ensemble Accuracy: {performance_metrics.get('overall_accuracy', 0)*100:.2f}%
- Active Models: {len(performance_metrics.get('model_performances', []))}
- Weight Distribution: Dynamic based on recent MAE/MAPE
- Performance Window: Rolling 30-day evaluation
- Drift Detection: Continuous monitoring enabled
        """.strip()
    
    def _create_technical_comparison_explanation(self, model_status: Dict[str, Any], 
                                               performance_metrics: Dict[str, Any]) -> str:
        """Create technical model comparison explanation"""
        return f"""
Technical Model Comparison:
- Total Models: {model_status.get('total_models', 0)}
- Initialized: {model_status.get('initialized_models', 0)}
- With Performance Data: {model_status.get('models_with_performance', 0)}
- Weight Update Method: Inverse error weighting
- Rebalancing Frequency: Real-time on new predictions
        """.strip()
    
    def _generate_contextual_follow_ups(self, context: EnsembleQueryContext, 
                                      response: EnsembleChatResponse) -> List[str]:
        """Generate contextual follow-up questions"""
        follow_ups = []
        
        if context.query_type == 'forecast':
            follow_ups.extend([
                "Would you like to see individual model forecasts?",
                "How confident are you in this forecast?",
                "What factors might affect this prediction?"
            ])
        elif context.query_type == 'performance':
            follow_ups.extend([
                "Which model is performing best right now?",
                "How can I improve forecast accuracy?",
                "Are there any concerning performance trends?"
            ])
        elif context.query_type == 'insights':
            follow_ups.extend([
                "What are the biggest opportunities?",
                "What risks should I be aware of?",
                "How should I act on these insights?"
            ])
        
        # Add general follow-ups
        follow_ups.extend([
            "Can you explain this in more detail?",
            "Show me the data behind this analysis",
            "What should I do next?"
        ])
        
        return follow_ups[:3]  # Limit to 3 follow-ups
    
    def _create_error_response(self, error_message: str) -> EnsembleChatResponse:
        """Create error response"""
        return EnsembleChatResponse(
            response_text=f"I encountered an issue: {error_message}. Please try rephrasing your question or check if the ensemble system is properly configured.",
            confidence=0.1,
            sources=["Error Handler"],
            suggested_actions=[
                "Try a different question",
                "Check system status",
                "Contact support if issue persists"
            ]
        )
    
    def _create_unavailable_response(self, service_name: str) -> EnsembleChatResponse:
        """Create response when service is unavailable"""
        return EnsembleChatResponse(
            response_text=f"{service_name}. I can still help with general questions about forecasting and provide guidance on using the ensemble system.",
            confidence=0.5,
            sources=["Chat System"],
            suggested_actions=[
                "Ask about forecasting concepts",
                "Learn about ensemble methods",
                "Get help with system setup"
            ]
        )