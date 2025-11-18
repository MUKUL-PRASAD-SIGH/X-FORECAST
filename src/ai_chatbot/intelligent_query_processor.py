"""
Intelligent Query Processing System
Handles query understanding, recommendation retrieval, conversation context management,
and clarifying question generation for ensemble forecasting discussions.
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# Import existing components
from .ensemble_chat_processor import EnsembleChatProcessor, EnsembleQueryContext, EnsembleChatResponse
from ..models.business_insights_engine import BusinessInsightsEngine, BusinessInsightsResult

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Enhanced conversation context management for multi-turn ensemble discussions"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    current_topic: Optional[str]
    mentioned_models: List[str]
    discussed_metrics: List[str]
    user_preferences: Dict[str, Any]
    business_context: Dict[str, Any]
    last_query_time: datetime
    context_confidence: float
    # Enhanced fields for multi-turn discussions
    topic_thread: List[str]  # Track topic evolution
    model_focus_history: Dict[str, int]  # Track model mention frequency
    metric_focus_history: Dict[str, int]  # Track metric mention frequency
    question_types: List[str]  # Track types of questions asked
    clarification_history: List[Dict[str, Any]]  # Track clarifications provided
    user_expertise_indicators: Dict[str, float]  # Track expertise level indicators
    conversation_flow_state: str  # 'exploring', 'drilling_down', 'comparing', 'concluding'
    pending_clarifications: List[str]  # Track unresolved clarifications
    context_anchors: List[Dict[str, Any]]  # Important context points to maintain

@dataclass
class QueryUnderstanding:
    """Represents understood query with confidence and ambiguities"""
    original_query: str
    interpreted_intent: str
    confidence_score: float
    ambiguous_terms: List[str]
    missing_context: List[str]
    suggested_clarifications: List[str]
    extracted_entities: Dict[str, Any]
    query_complexity: str  # 'simple', 'moderate', 'complex'

@dataclass
class RecommendationContext:
    """Context for generating recommendations"""
    user_expertise_level: str  # 'beginner', 'intermediate', 'expert'
    business_role: str  # 'analyst', 'manager', 'executive'
    current_focus: str  # 'performance', 'forecasting', 'insights'
    recent_actions: List[str]
    preferred_detail_level: str  # 'summary', 'detailed', 'technical'

class IntelligentQueryProcessor:
    """
    Advanced query processing with context awareness and intelligent clarification
    """
    
    def __init__(self, 
                 ensemble_chat_processor: EnsembleChatProcessor,
                 insights_engine: Optional[BusinessInsightsEngine] = None,
                 max_context_history: int = 50):
        """
        Initialize intelligent query processor
        
        Args:
            ensemble_chat_processor: Ensemble chat processor instance
            insights_engine: Business insights engine for recommendations
            max_context_history: Maximum conversation history to maintain
        """
        self.ensemble_processor = ensemble_chat_processor
        self.insights_engine = insights_engine
        self.max_context_history = max_context_history
        
        # Conversation contexts by session
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Query understanding patterns
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_extractors = self._initialize_entity_extractors()
        self.ambiguity_detectors = self._initialize_ambiguity_detectors()
        
        # Recommendation templates
        self.recommendation_templates = self._initialize_recommendation_templates()
        
        # Context decay settings
        self.context_decay_minutes = 30
        self.topic_transition_threshold = 0.3
        
        logger.info("Intelligent query processor initialized")
    
    async def process_intelligent_query(self, 
                                      query: str,
                                      user_id: str,
                                      session_id: str,
                                      user_context: Optional[Dict[str, Any]] = None) -> EnsembleChatResponse:
        """
        Process query with intelligent understanding and context management
        
        Args:
            query: User query string
            user_id: User identifier
            session_id: Session identifier
            user_context: Additional user context
            
        Returns:
            Enhanced chat response with intelligent processing
        """
        try:
            logger.info(f"Processing intelligent query for user {user_id}: {query[:100]}...")
            
            # Get or create conversation context
            conversation_context = self._get_conversation_context(user_id, session_id)
            
            # Update conversation context with new query
            self._update_conversation_context(conversation_context, query, user_context)
            
            # Understand the query with context
            query_understanding = self._understand_query_with_context(query, conversation_context)
            
            # Check if clarification is needed
            if self._needs_clarification(query_understanding):
                return await self._generate_clarifying_response(query_understanding, conversation_context)
            
            # Enhance query with conversation context
            enhanced_query = self._enhance_query_with_context(query, conversation_context)
            
            # Process with ensemble chat processor
            base_response = await self.ensemble_processor.process_ensemble_query(
                enhanced_query, user_context
            )
            
            # Enhance response with intelligent recommendations
            enhanced_response = await self._enhance_response_with_recommendations(
                base_response, query_understanding, conversation_context
            )
            
            # Update conversation history
            self._update_conversation_history(conversation_context, query, enhanced_response)
            
            logger.info(f"Generated intelligent response with {enhanced_response.confidence:.2f} confidence")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Intelligent query processing failed: {e}")
            return self._create_error_response(str(e))
    
    def _get_conversation_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Enhanced conversation context management for multi-turn discussions"""
        context_key = f"{user_id}_{session_id}"
        
        if context_key not in self.conversation_contexts:
            self.conversation_contexts[context_key] = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                conversation_history=[],
                current_topic=None,
                mentioned_models=[],
                discussed_metrics=[],
                user_preferences={},
                business_context={},
                last_query_time=datetime.now(),
                context_confidence=0.0,
                # Enhanced fields initialization
                topic_thread=[],
                model_focus_history={},
                metric_focus_history={},
                question_types=[],
                clarification_history=[],
                user_expertise_indicators={},
                conversation_flow_state='exploring',
                pending_clarifications=[],
                context_anchors=[]
            )
        
        # Enhanced context decay with preservation of important context
        context = self.conversation_contexts[context_key]
        time_since_last = (datetime.now() - context.last_query_time).total_seconds() / 60
        
        if time_since_last > self.context_decay_minutes:
            # Preserve important context elements during decay
            preserved_elements = {
                'user_preferences': context.user_preferences.copy(),
                'model_focus_history': context.model_focus_history.copy(),
                'metric_focus_history': context.metric_focus_history.copy(),
                'user_expertise_indicators': context.user_expertise_indicators.copy(),
                'context_anchors': [anchor for anchor in context.context_anchors 
                                  if (datetime.now() - datetime.fromisoformat(anchor.get('timestamp', datetime.now().isoformat()))).total_seconds() / 3600 < 24]  # Keep anchors from last 24 hours
            }
            
            # Reset context with preserved elements
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                conversation_history=[],
                current_topic=None,
                mentioned_models=[],
                discussed_metrics=[],
                user_preferences=preserved_elements['user_preferences'],
                business_context={},
                last_query_time=datetime.now(),
                context_confidence=0.0,
                topic_thread=[],
                model_focus_history=preserved_elements['model_focus_history'],
                metric_focus_history=preserved_elements['metric_focus_history'],
                question_types=[],
                clarification_history=[],
                user_expertise_indicators=preserved_elements['user_expertise_indicators'],
                conversation_flow_state='exploring',
                pending_clarifications=[],
                context_anchors=preserved_elements['context_anchors']
            )
            self.conversation_contexts[context_key] = context
        
        return context
    
    def _update_conversation_context(self, context: ConversationContext, 
                                   query: str, user_context: Optional[Dict[str, Any]]):
        """Enhanced conversation context update for multi-turn ensemble discussions"""
        try:
            context.last_query_time = datetime.now()
            query_lower = query.lower()
            
            # Enhanced model tracking with frequency counting
            for model_name in ['arima', 'ets', 'xgboost', 'lstm', 'croston', 'ensemble']:
                if model_name in query_lower:
                    if model_name not in context.mentioned_models:
                        context.mentioned_models.append(model_name)
                    # Track model focus frequency
                    context.model_focus_history[model_name] = context.model_focus_history.get(model_name, 0) + 1
            
            # Enhanced metrics tracking with frequency counting
            metrics = ['accuracy', 'mae', 'mape', 'rmse', 'confidence', 'weights', 
                      'precision', 'recall', 'r2', 'r_squared', 'performance']
            for metric in metrics:
                if metric in query_lower:
                    if metric not in context.discussed_metrics:
                        context.discussed_metrics.append(metric)
                    # Track metric focus frequency
                    context.metric_focus_history[metric] = context.metric_focus_history.get(metric, 0) + 1
            
            # Enhanced question type tracking
            question_type = self._classify_question_type(query)
            context.question_types.append(question_type)
            if len(context.question_types) > 20:  # Keep last 20 question types
                context.question_types = context.question_types[-20:]
            
            # Track forecast-related terms for context continuity
            forecast_terms = ['forecast', 'prediction', 'projection', 'future', 'trend']
            context.business_context['forecast_focus'] = any(term in query_lower for term in forecast_terms)
            
            # Enhanced confidence tracking with intensity scoring
            confidence_terms = ['confidence', 'certain', 'reliable', 'trust', 'uncertainty']
            confidence_intensity = sum(1 for term in confidence_terms if term in query_lower)
            context.business_context['confidence_focus'] = confidence_intensity > 0
            context.business_context['confidence_intensity'] = confidence_intensity
            
            # Update business context from user_context
            if user_context:
                context.business_context.update(user_context)
                
                # Extract company-specific context for personalization
                if 'company_name' in user_context:
                    context.business_context['company_name'] = user_context['company_name']
                if 'industry' in user_context:
                    context.business_context['industry'] = user_context['industry']
            
            # Enhanced topic thread management
            current_topic = self._detect_query_topic(query)
            previous_topic = context.current_topic
            
            # Update topic thread
            context.topic_thread.append(current_topic)
            if len(context.topic_thread) > 15:  # Keep last 15 topics
                context.topic_thread = context.topic_thread[-15:]
            
            if previous_topic and current_topic != previous_topic:
                # Topic transition detected - enhanced context continuity management
                context.context_confidence *= 0.8
                
                # Enhanced topic transition tracking
                if 'topic_transitions' not in context.business_context:
                    context.business_context['topic_transitions'] = []
                
                transition_data = {
                    'from': previous_topic,
                    'to': current_topic,
                    'timestamp': datetime.now().isoformat(),
                    'query_snippet': query[:50],  # Store query snippet for context
                    'transition_type': self._classify_topic_transition(previous_topic, current_topic)
                }
                context.business_context['topic_transitions'].append(transition_data)
                
                # Keep only recent transitions
                if len(context.business_context['topic_transitions']) > 8:
                    context.business_context['topic_transitions'] = context.business_context['topic_transitions'][-8:]
                
                # Update conversation flow state based on transition
                context.conversation_flow_state = self._determine_conversation_flow_state(context)
            else:
                # Same topic - increase confidence and maintain flow
                context.context_confidence = min(context.context_confidence + 0.1, 1.0)
            
            context.current_topic = current_topic
            
            # Enhanced user preference learning with expertise indicators
            self._update_enhanced_user_preferences(context, query)
            
            # Track query complexity patterns for adaptive responses
            complexity = self._assess_query_complexity_simple(query)
            if 'complexity_history' not in context.business_context:
                context.business_context['complexity_history'] = []
            
            context.business_context['complexity_history'].append(complexity)
            if len(context.business_context['complexity_history']) > 15:
                context.business_context['complexity_history'] = context.business_context['complexity_history'][-15:]
            
            # Enhanced conversation flow indicators
            self._update_enhanced_conversation_flow_indicators(context, query)
            
            # Update context anchors for important information
            self._update_context_anchors(context, query, current_topic)
            
            # Clear resolved clarifications
            self._update_pending_clarifications(context, query)
            
        except Exception as e:
            logger.error(f"Enhanced context update failed: {e}")
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question being asked"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how accurate', 'accuracy of', 'reliable']):
            return 'accuracy_inquiry'
        elif any(word in query_lower for word in ['confidence', 'certain', 'uncertainty']):
            return 'confidence_inquiry'
        elif any(word in query_lower for word in ['compare', 'versus', 'better', 'best']):
            return 'comparison_inquiry'
        elif any(word in query_lower for word in ['why', 'how does', 'explain']):
            return 'explanation_inquiry'
        elif any(word in query_lower for word in ['recommend', 'should', 'what to do']):
            return 'recommendation_inquiry'
        elif any(word in query_lower for word in ['forecast', 'predict', 'future']):
            return 'forecast_inquiry'
        else:
            return 'general_inquiry'
    
    def _classify_topic_transition(self, from_topic: str, to_topic: str) -> str:
        """Classify the type of topic transition"""
        transition_map = {
            ('forecasting', 'performance'): 'drill_down',
            ('performance', 'forecasting'): 'zoom_out',
            ('forecasting', 'insights'): 'business_focus',
            ('insights', 'forecasting'): 'technical_focus',
            ('performance', 'technical'): 'deep_dive',
            ('technical', 'performance'): 'practical_focus'
        }
        
        return transition_map.get((from_topic, to_topic), 'lateral_shift')
    
    def _determine_conversation_flow_state(self, context: ConversationContext) -> str:
        """Determine the current conversation flow state"""
        recent_transitions = context.business_context.get('topic_transitions', [])[-3:]
        
        if len(recent_transitions) == 0:
            return 'exploring'
        elif len(recent_transitions) == 1:
            return 'focusing'
        elif all(t.get('transition_type') == 'drill_down' for t in recent_transitions):
            return 'drilling_down'
        elif len(set(t.get('to') for t in recent_transitions)) > 2:
            return 'exploring'
        else:
            return 'comparing'
    
    def _update_enhanced_user_preferences(self, context: ConversationContext, query: str):
        """Enhanced user preference learning with expertise indicators"""
        query_lower = query.lower()
        
        # Enhanced expertise level detection
        technical_indicators = ['algorithm', 'parameter', 'hyperparameter', 'methodology', 'implementation']
        business_indicators = ['roi', 'revenue', 'profit', 'strategy', 'business impact']
        beginner_indicators = ['explain', 'what does', 'how to', 'simple', 'basic']
        
        # Update expertise indicators with scoring
        for indicator in technical_indicators:
            if indicator in query_lower:
                context.user_expertise_indicators['technical'] = context.user_expertise_indicators.get('technical', 0) + 1
        
        for indicator in business_indicators:
            if indicator in query_lower:
                context.user_expertise_indicators['business'] = context.user_expertise_indicators.get('business', 0) + 1
        
        for indicator in beginner_indicators:
            if indicator in query_lower:
                context.user_expertise_indicators['beginner'] = context.user_expertise_indicators.get('beginner', 0) + 1
        
        # Determine overall expertise level based on indicators
        total_technical = context.user_expertise_indicators.get('technical', 0)
        total_business = context.user_expertise_indicators.get('business', 0)
        total_beginner = context.user_expertise_indicators.get('beginner', 0)
        
        if total_technical > total_business and total_technical > total_beginner:
            context.user_preferences['expertise_level'] = 'expert'
        elif total_beginner > total_technical and total_beginner > total_business:
            context.user_preferences['expertise_level'] = 'beginner'
        else:
            context.user_preferences['expertise_level'] = 'intermediate'
        
        # Enhanced detail preference detection
        if any(word in query_lower for word in ['detail', 'comprehensive', 'thorough', 'deep']):
            context.user_preferences['detail_level'] = 'detailed'
        elif any(word in query_lower for word in ['summary', 'brief', 'quick', 'overview']):
            context.user_preferences['detail_level'] = 'summary'
        else:
            context.user_preferences['detail_level'] = 'standard'
        
        # Track communication style preferences
        if any(word in query_lower for word in ['simple', 'plain', 'easy']):
            context.user_preferences['communication_style'] = 'simple'
        elif any(word in query_lower for word in ['technical', 'precise', 'exact']):
            context.user_preferences['communication_style'] = 'technical'
        else:
            context.user_preferences['communication_style'] = 'balanced'
    
    def _update_enhanced_conversation_flow_indicators(self, context: ConversationContext, query: str):
        """Enhanced conversation flow indicators for multi-turn discussions"""
        query_lower = query.lower()
        
        # Enhanced follow-up detection
        follow_up_indicators = ['also', 'additionally', 'furthermore', 'moreover', 'and', 'but', 'however', 'what about']
        context.business_context['is_follow_up'] = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Enhanced clarification request detection
        clarification_indicators = ['clarify', 'explain more', 'what do you mean', 'can you elaborate', 'unclear', 'confusing']
        context.business_context['needs_clarification'] = any(indicator in query_lower for indicator in clarification_indicators)
        
        # Enhanced comparison request detection
        comparison_indicators = ['compare', 'versus', 'vs', 'difference', 'better', 'worse', 'which is']
        context.business_context['is_comparison'] = any(indicator in query_lower for indicator in comparison_indicators)
        
        # Enhanced temporal reference tracking
        temporal_indicators = ['next', 'future', 'upcoming', 'later', 'soon', 'eventually', 'when', 'timeline']
        context.business_context['has_temporal_focus'] = any(indicator in query_lower for indicator in temporal_indicators)
        
        # Track reference to previous conversation
        reference_indicators = ['earlier', 'before', 'previously', 'you mentioned', 'as we discussed']
        context.business_context['references_previous'] = any(indicator in query_lower for indicator in reference_indicators)
        
        # Track drill-down requests
        drill_down_indicators = ['more detail', 'deeper', 'specifically', 'in particular', 'focus on']
        context.business_context['is_drill_down'] = any(indicator in query_lower for indicator in drill_down_indicators)
    
    def _update_context_anchors(self, context: ConversationContext, query: str, topic: str):
        """Update context anchors for important conversation points"""
        # Create anchor for important queries
        if len(query.split()) > 8 or any(word in query.lower() for word in ['important', 'critical', 'key', 'main']):
            anchor = {
                'timestamp': datetime.now().isoformat(),
                'topic': topic,
                'query_summary': query[:100],
                'importance_score': self._calculate_anchor_importance(query, context)
            }
            
            context.context_anchors.append(anchor)
            
            # Keep only top 10 most important anchors
            if len(context.context_anchors) > 10:
                context.context_anchors.sort(key=lambda x: x['importance_score'], reverse=True)
                context.context_anchors = context.context_anchors[:10]
    
    def _calculate_anchor_importance(self, query: str, context: ConversationContext) -> float:
        """Calculate importance score for context anchors"""
        importance = 0.0
        query_lower = query.lower()
        
        # Length factor
        importance += min(len(query.split()) / 20, 0.3)
        
        # Technical complexity
        technical_terms = ['algorithm', 'parameter', 'methodology', 'implementation']
        importance += sum(0.1 for term in technical_terms if term in query_lower)
        
        # Business impact terms
        business_terms = ['revenue', 'profit', 'strategy', 'decision', 'critical']
        importance += sum(0.15 for term in business_terms if term in query_lower)
        
        # Question complexity
        if query_lower.count('?') > 1:
            importance += 0.2
        
        return min(importance, 1.0)
    
    def _update_pending_clarifications(self, context: ConversationContext, query: str):
        """Update pending clarifications based on new query"""
        query_lower = query.lower()
        
        # Check if query addresses pending clarifications
        resolved_clarifications = []
        for clarification in context.pending_clarifications:
            if any(word in query_lower for word in clarification.lower().split()[:3]):
                resolved_clarifications.append(clarification)
        
        # Remove resolved clarifications
        for resolved in resolved_clarifications:
            if resolved in context.pending_clarifications:
                context.pending_clarifications.remove(resolved)
    
    def _assess_query_complexity_simple(self, query: str) -> str:
        """Simple query complexity assessment for context tracking"""
        word_count = len(query.split())
        question_count = query.count('?')
        technical_terms = ['algorithm', 'parameter', 'hyperparameter', 'ensemble', 'methodology']
        
        complexity_score = 0
        if word_count > 15:
            complexity_score += 1
        if question_count > 1:
            complexity_score += 1
        if any(term in query.lower() for term in technical_terms):
            complexity_score += 1
        
        if complexity_score >= 2:
            return 'complex'
        elif complexity_score == 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _update_conversation_flow_indicators(self, context: ConversationContext, query: str):
        """Update indicators for conversation flow and continuity"""
        query_lower = query.lower()
        
        # Track follow-up indicators
        follow_up_indicators = ['also', 'additionally', 'furthermore', 'moreover', 'and', 'but', 'however']
        context.business_context['is_follow_up'] = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Track clarification requests
        clarification_indicators = ['clarify', 'explain more', 'what do you mean', 'can you elaborate']
        context.business_context['needs_clarification'] = any(indicator in query_lower for indicator in clarification_indicators)
        
        # Track comparison requests
        comparison_indicators = ['compare', 'versus', 'vs', 'difference', 'better', 'worse']
        context.business_context['is_comparison'] = any(indicator in query_lower for indicator in comparison_indicators)
        
        # Track temporal references for forecast context
        temporal_indicators = ['next', 'future', 'upcoming', 'later', 'soon', 'eventually']
        context.business_context['has_temporal_focus'] = any(indicator in query_lower for indicator in temporal_indicators)
    
    def _understand_query_with_context(self, query: str, context: ConversationContext) -> QueryUnderstanding:
        """Understand query with conversation context"""
        try:
            query_lower = query.lower()
            
            # Determine intent with context
            intent = self._determine_intent_with_context(query, context)
            
            # Calculate confidence based on clarity and context
            confidence_factors = [
                self._calculate_query_clarity(query),
                self._calculate_context_relevance(query, context),
                context.context_confidence
            ]
            confidence_score = np.mean([f for f in confidence_factors if f is not None])
            
            # Detect ambiguous terms
            ambiguous_terms = self._detect_ambiguous_terms(query, context)
            
            # Identify missing context
            missing_context = self._identify_missing_context(query, intent, context)
            
            # Generate clarification suggestions
            clarifications = self._generate_clarification_suggestions(ambiguous_terms, missing_context)
            
            # Extract entities
            entities = self._extract_entities_with_context(query, context)
            
            # Determine query complexity
            complexity = self._assess_query_complexity(query, entities, context)
            
            return QueryUnderstanding(
                original_query=query,
                interpreted_intent=intent,
                confidence_score=confidence_score,
                ambiguous_terms=ambiguous_terms,
                missing_context=missing_context,
                suggested_clarifications=clarifications,
                extracted_entities=entities,
                query_complexity=complexity
            )
            
        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            return self._create_fallback_understanding(query)
    
    def _needs_clarification(self, understanding: QueryUnderstanding) -> bool:
        """Determine if query needs clarification"""
        return (
            understanding.confidence_score < 0.6 or
            len(understanding.ambiguous_terms) > 2 or
            len(understanding.missing_context) > 1 or
            understanding.query_complexity == 'complex' and understanding.confidence_score < 0.8
        )
    
    async def _generate_clarifying_response(self, understanding: QueryUnderstanding,
                                          context: ConversationContext) -> EnsembleChatResponse:
        """Generate intelligent clarifying response for ambiguous queries"""
        try:
            # Generate context-aware clarification
            clarification_text = await self._generate_intelligent_clarification_text(understanding, context)
            
            # Generate smart clarification options
            options = self._generate_smart_clarification_options(understanding, context)
            
            # Generate contextual follow-up questions
            follow_ups = self._generate_contextual_clarification_questions(understanding, context)
            
            return EnsembleChatResponse(
                response_text=clarification_text,
                confidence=0.9,
                sources=["Intelligent Query Processor"],
                follow_up_questions=follow_ups,
                suggested_actions=options,
                requires_action=True
            )
            
        except Exception as e:
            logger.error(f"Clarifying response generation failed: {e}")
            return self._create_error_response("I need more information to answer your question properly.")
    
    async def _generate_intelligent_clarification_text(self, understanding: QueryUnderstanding,
                                                     context: ConversationContext) -> str:
        """Enhanced intelligent clarification text generation for ambiguous queries"""
        clarification_parts = []
        
        # Enhanced personalized opening based on conversation flow state
        flow_state = context.conversation_flow_state
        if flow_state == 'drilling_down':
            clarification_parts.append("I see you're diving deeper into this topic. Let me make sure I understand exactly what you're looking for.")
        elif flow_state == 'comparing':
            clarification_parts.append("You're comparing different aspects. To give you the most relevant comparison, I need to clarify a few details.")
        elif len(context.conversation_history) > 0:
            clarification_parts.append("Based on our conversation, I want to make sure I understand your question correctly.")
        else:
            clarification_parts.append("I want to provide you with the most accurate information.")
        
        # Enhanced ambiguity resolution with conversation context
        if understanding.ambiguous_terms:
            ambiguous_term = understanding.ambiguous_terms[0]
            
            # Advanced context-aware clarification for ambiguous terms
            if ambiguous_term.lower() in ['it', 'this', 'that']:
                # Use conversation anchors and recent context
                if context.context_anchors:
                    recent_anchor = context.context_anchors[-1]
                    clarification_parts.append(f"When you refer to '{ambiguous_term}', are you asking about {recent_anchor['topic']} from our earlier discussion?")
                elif context.current_topic:
                    clarification_parts.append(f"When you refer to '{ambiguous_term}', are you asking about {context.current_topic}?")
                elif context.mentioned_models:
                    # Use model focus history to prioritize
                    most_discussed_model = max(context.model_focus_history.items(), key=lambda x: x[1])[0] if context.model_focus_history else context.mentioned_models[-1]
                    clarification_parts.append(f"Are you referring to the {most_discussed_model} model?")
                else:
                    clarification_parts.append(f"Could you clarify what '{ambiguous_term}' refers to?")
            
            elif ambiguous_term.lower() in ['they', 'them']:
                if len(context.mentioned_models) > 1:
                    clarification_parts.append(f"When you say '{ambiguous_term}', are you referring to the models {', '.join(context.mentioned_models[-2:])}?")
                else:
                    clarification_parts.append(f"Could you specify what '{ambiguous_term}' refers to?")
            
            elif ambiguous_term.lower() == 'which':
                # Context-aware which clarification
                if context.business_context.get('is_comparison'):
                    clarification_parts.append("Which specific aspect would you like me to compare?")
                elif len(context.mentioned_models) > 1:
                    clarification_parts.append(f"Which of these models are you asking about: {', '.join(context.mentioned_models)}?")
                else:
                    clarification_parts.append("Which specific item are you referring to?")
            
            else:
                # Enhanced clarification for other ambiguous terms
                if ambiguous_term in context.discussed_metrics:
                    clarification_parts.append(f"When you mention '{ambiguous_term}', are you asking about the {ambiguous_term} metric specifically?")
                else:
                    clarification_parts.append(f"Could you clarify what you mean by '{ambiguous_term}'?")
        
        # Enhanced missing context resolution with conversation awareness
        if understanding.missing_context:
            missing = understanding.missing_context[0]
            
            if missing == 'time_horizon':
                # Use business context and conversation history for better suggestions
                if context.business_context.get('industry') == 'retail':
                    clarification_parts.append("For retail forecasting, are you interested in seasonal planning (3-6 months) or annual planning (12+ months)?")
                elif context.business_context.get('has_temporal_focus'):
                    clarification_parts.append("What specific time period are you interested in for the forecast?")
                else:
                    # Use conversation history to suggest appropriate horizons
                    recent_questions = context.question_types[-3:] if len(context.question_types) >= 3 else context.question_types
                    if 'accuracy_inquiry' in recent_questions:
                        clarification_parts.append("For accuracy assessment, are you looking at short-term (1-3 months) or long-term (6+ months) performance?")
                    else:
                        clarification_parts.append("Are you looking for short-term (1-3 months) or long-term (6+ months) predictions?")
            
            elif missing == 'specific_model':
                # Enhanced model clarification using focus history
                if context.model_focus_history:
                    top_models = sorted(context.model_focus_history.items(), key=lambda x: x[1], reverse=True)[:3]
                    model_names = [model[0].upper() for model in top_models]
                    clarification_parts.append(f"Are you asking specifically about {', '.join(model_names)} or the ensemble as a whole?")
                elif len(context.mentioned_models) > 0:
                    clarification_parts.append(f"Are you asking specifically about {', '.join(context.mentioned_models)} or the ensemble as a whole?")
                else:
                    clarification_parts.append("Are you interested in a specific forecasting model or the overall ensemble performance?")
            
            elif missing == 'metric_type':
                # Enhanced metric clarification using discussion history
                if context.metric_focus_history:
                    top_metrics = sorted(context.metric_focus_history.items(), key=lambda x: x[1], reverse=True)[:2]
                    if context.business_context.get('confidence_focus'):
                        clarification_parts.append("Are you looking for confidence intervals, prediction reliability, or uncertainty quantification?")
                    else:
                        clarification_parts.append(f"Which performance aspect interests you: {', '.join([m[0] for m in top_metrics])} or overall accuracy?")
                elif context.business_context.get('confidence_focus'):
                    clarification_parts.append("Are you looking for confidence intervals, accuracy metrics, or error rates?")
                else:
                    clarification_parts.append("Which aspect of performance interests you most: accuracy, reliability, or error analysis?")
        
        # Enhanced context-specific guidance
        if understanding.query_complexity == 'complex':
            if context.user_preferences.get('expertise_level') == 'beginner':
                clarification_parts.append("I can break this down into simpler parts and explain each step if that would be helpful.")
            else:
                clarification_parts.append("I can address each part of your question separately for a more comprehensive response.")
        
        # Add conversation flow guidance
        if context.conversation_flow_state == 'exploring':
            clarification_parts.append("Feel free to ask follow-up questions as we explore this topic together.")
        elif context.conversation_flow_state == 'drilling_down':
            clarification_parts.append("I can provide more technical details once I understand your specific focus area.")
        
        return " ".join(clarification_parts)
    
    def _generate_smart_clarification_options(self, understanding: QueryUnderstanding,
                                            context: ConversationContext) -> List[str]:
        """Enhanced smart clarification options generation based on conversation context"""
        options = []
        
        # Enhanced time horizon options with business context
        if 'time_horizon' in understanding.missing_context:
            if context.business_context.get('industry') == 'retail':
                options.extend(["Next quarter (seasonal planning)", "Next 6 months (inventory planning)", "Next year (strategic planning)"])
            elif context.business_context.get('industry') == 'manufacturing':
                options.extend(["Next month (production planning)", "Next quarter (capacity planning)", "Next 6 months (supply chain planning)"])
            else:
                # Use conversation history to suggest appropriate horizons
                if any(q_type in ['accuracy_inquiry', 'confidence_inquiry'] for q_type in context.question_types[-3:]):
                    options.extend(["Short-term accuracy (1-3 months)", "Medium-term reliability (3-6 months)", "Long-term trends (6+ months)"])
                else:
                    options.extend(["Short-term (1-3 months)", "Medium-term (3-6 months)", "Long-term (6+ months)"])
        
        # Enhanced model-specific options using focus history
        if 'specific_model' in understanding.missing_context:
            if context.model_focus_history:
                # Prioritize models based on discussion frequency
                top_models = sorted(context.model_focus_history.items(), key=lambda x: x[1], reverse=True)[:3]
                options.extend([f"Focus on {model[0].upper()} model" for model in top_models])
            elif context.mentioned_models:
                options.extend([f"Focus on {model.upper()}" for model in context.mentioned_models[:3]])
            else:
                # Provide context-aware model options
                if context.business_context.get('confidence_focus'):
                    options.extend(["Compare model reliability", "Best performing model", "Ensemble confidence"])
                else:
                    options.extend(["Compare all models", "Best performing model", "Ensemble overview"])
        
        # Enhanced metric type options with conversation awareness
        if 'metric_type' in understanding.missing_context:
            if context.business_context.get('confidence_focus'):
                confidence_intensity = context.business_context.get('confidence_intensity', 1)
                if confidence_intensity > 2:  # High confidence focus
                    options.extend(["Detailed confidence intervals", "Uncertainty quantification", "Prediction reliability scores"])
                else:
                    options.extend(["Confidence intervals", "Prediction reliability", "Uncertainty analysis"])
            elif context.metric_focus_history:
                # Suggest related metrics based on discussion history
                discussed_metrics = list(context.metric_focus_history.keys())
                if 'accuracy' in discussed_metrics:
                    options.extend(["Accuracy breakdown", "Error analysis", "Performance trends"])
                elif 'performance' in discussed_metrics:
                    options.extend(["Performance metrics", "Model comparison", "Accuracy analysis"])
                else:
                    options.extend(["Accuracy metrics", "Error analysis", "Performance comparison"])
            else:
                options.extend(["Accuracy metrics", "Error analysis", "Performance comparison"])
        
        # Enhanced intent-specific options with conversation flow awareness
        if understanding.interpreted_intent == 'general':
            flow_state = context.conversation_flow_state
            
            if flow_state == 'drilling_down':
                options.extend(["Technical details", "Implementation specifics", "Advanced configuration"])
            elif flow_state == 'comparing':
                options.extend(["Model comparison", "Performance analysis", "Feature comparison"])
            elif flow_state == 'exploring':
                recent_topics = context.topic_thread[-3:] if len(context.topic_thread) >= 3 else context.topic_thread
                if 'forecasting' in recent_topics:
                    options.extend(["Forecast accuracy", "Model performance", "Business insights"])
                elif 'performance' in recent_topics:
                    options.extend(["Performance metrics", "Model comparison", "Optimization suggestions"])
                else:
                    options.extend(["View forecasts", "Check model status", "Get recommendations"])
            else:
                options.extend(["View forecasts", "Check model status", "Get recommendations"])
        
        # Enhanced ambiguous query options with conversation context
        if not options and understanding.ambiguous_terms:
            ambiguous_term = understanding.ambiguous_terms[0].lower()
            
            if ambiguous_term in ['it', 'this', 'that']:
                # Use conversation anchors and recent context for better options
                if context.context_anchors:
                    recent_topics = [anchor['topic'] for anchor in context.context_anchors[-2:]]
                    options.extend([f"About {topic}" for topic in set(recent_topics)])
                
                if context.current_topic:
                    options.append(f"Current {context.current_topic} discussion")
                
                # Add general options if we don't have enough specific ones
                if len(options) < 3:
                    options.extend([
                        "Forecast accuracy and reliability",
                        "Model performance comparison", 
                        "Business insights and recommendations"
                    ])
            
            elif ambiguous_term in ['which', 'what']:
                # Context-aware which/what clarifications
                if context.mentioned_models:
                    options.extend([f"{model.upper()} model" for model in context.mentioned_models[:3]])
                if context.discussed_metrics:
                    options.extend([f"{metric.title()} metric" for metric in context.discussed_metrics[:2]])
                
                # Add general options
                options.extend(["Specific details", "General overview", "Comparison analysis"])
            
            else:
                # Default enhanced options
                expertise_level = context.user_preferences.get('expertise_level', 'intermediate')
                if expertise_level == 'beginner':
                    options.extend([
                        "Simple explanation",
                        "Step-by-step guide",
                        "Basic concepts overview"
                    ])
                elif expertise_level == 'expert':
                    options.extend([
                        "Technical implementation details",
                        "Advanced configuration options",
                        "Methodology deep-dive"
                    ])
                else:
                    options.extend([
                        "Clarify the specific topic",
                        "Provide more context",
                        "Ask a more specific question"
                    ])
        
        # Add conversation flow continuation options
        if context.conversation_flow_state == 'drilling_down' and len(options) < 4:
            options.append("Continue with more details")
        elif context.conversation_flow_state == 'comparing' and len(options) < 4:
            options.append("Add another comparison aspect")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_options = []
        for option in options:
            if option not in seen:
                seen.add(option)
                unique_options.append(option)
        
        return unique_options[:4]  # Limit to 4 options
    
    def _generate_contextual_clarification_questions(self, understanding: QueryUnderstanding,
                                                   context: ConversationContext) -> List[str]:
        """Generate contextual follow-up questions for clarification"""
        questions = []
        
        # Based on conversation history
        if len(context.conversation_history) > 0:
            last_topic = context.conversation_history[-1].get('topic', '')
            if last_topic and last_topic != context.current_topic:
                questions.append(f"Are you still interested in {last_topic} or focusing on {context.current_topic}?")
        
        # Based on user expertise level
        expertise = context.user_preferences.get('expertise_level', 'intermediate')
        if expertise == 'beginner':
            questions.append("Would you like a simple explanation or detailed technical information?")
        elif expertise == 'expert':
            questions.append("Are you looking for technical details or business implications?")
        
        # Based on query complexity
        if understanding.query_complexity == 'complex':
            questions.append("Should I address each part of your question separately?")
        
        # Based on business context
        if context.business_context.get('is_comparison'):
            questions.append("What specific aspects would you like me to compare?")
        
        return questions[:3]  # Limit to 3 questions
    
    def _enhance_query_with_context(self, query: str, context: ConversationContext) -> str:
        """Enhance query with conversation context"""
        try:
            enhanced_query = query
            
            # Add context from previous conversation
            if context.current_topic and context.current_topic not in query.lower():
                enhanced_query += f" (in context of {context.current_topic})"
            
            # Add mentioned models if relevant
            if context.mentioned_models and not any(model in query.lower() for model in context.mentioned_models):
                if len(context.mentioned_models) == 1:
                    enhanced_query += f" for {context.mentioned_models[0]} model"
            
            # Add business context
            if context.business_context.get('company_name'):
                enhanced_query += f" for {context.business_context['company_name']}"
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query
    
    async def _enhance_response_with_recommendations(self, 
                                                   base_response: EnsembleChatResponse,
                                                   understanding: QueryUnderstanding,
                                                   context: ConversationContext) -> EnsembleChatResponse:
        """Enhance response with intelligent recommendations"""
        try:
            # Create recommendation context
            rec_context = self._create_recommendation_context(understanding, context)
            
            # Generate contextual recommendations
            recommendations = await self._generate_contextual_recommendations(
                understanding, context, rec_context
            )
            
            # Enhance suggested actions
            enhanced_actions = self._enhance_suggested_actions(
                base_response.suggested_actions or [], recommendations, rec_context
            )
            
            # Add personalized follow-ups
            personalized_follow_ups = self._generate_personalized_follow_ups(
                understanding, context, base_response
            )
            
            # Create enhanced response
            enhanced_response = EnsembleChatResponse(
                response_text=base_response.response_text,
                confidence=base_response.confidence,
                sources=base_response.sources,
                follow_up_questions=personalized_follow_ups,
                suggested_actions=enhanced_actions,
                ensemble_data=base_response.ensemble_data,
                model_performance_data=base_response.model_performance_data,
                forecast_data=base_response.forecast_data,
                insights_data=base_response.insights_data,
                technical_explanation=base_response.technical_explanation,
                plain_language_summary=base_response.plain_language_summary,
                requires_action=base_response.requires_action
            )
            
            # Add recommendations to response text if appropriate
            if recommendations and rec_context.preferred_detail_level != 'summary':
                enhanced_response.response_text += "\n\n" + self._format_recommendations(recommendations, rec_context)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Response enhancement failed: {e}")
            return base_response
    
    def _update_conversation_history(self, context: ConversationContext, 
                                   query: str, response: EnsembleChatResponse):
        """Enhanced conversation history update with multi-turn context tracking"""
        try:
            # Determine query intent for history tracking
            query_intent = self._determine_intent_with_context(query, context)
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response_summary': response.plain_language_summary or response.response_text[:200],
                'confidence': response.confidence,
                'topic': context.current_topic,
                'intent': query_intent,
                'models_discussed': context.mentioned_models.copy(),
                'metrics_discussed': context.discussed_metrics.copy(),
                'question_type': context.question_types[-1] if context.question_types else 'general_inquiry',
                'conversation_flow_state': context.conversation_flow_state,
                'clarifications_provided': len(response.suggested_actions or []) > 0,
                'follow_up_generated': len(response.follow_up_questions or []) > 0
            }
            
            context.conversation_history.append(history_entry)
            
            # Maintain history size limit
            if len(context.conversation_history) > self.max_context_history:
                context.conversation_history = context.conversation_history[-self.max_context_history:]
            
            # Update context confidence based on successful interaction
            if response.confidence > 0.7:
                context.context_confidence = min(context.context_confidence + 0.15, 1.0)
            elif response.confidence > 0.5:
                context.context_confidence = min(context.context_confidence + 0.1, 1.0)
            else:
                context.context_confidence = max(context.context_confidence - 0.05, 0.0)
            
            # Track clarification resolution
            if response.requires_action and response.suggested_actions:
                # Add to pending clarifications if this was a clarifying response
                for action in response.suggested_actions:
                    if action not in context.pending_clarifications:
                        context.pending_clarifications.append(action)
            
            # Update conversation flow state based on response
            self._update_conversation_flow_state_post_response(context, response)
            
        except Exception as e:
            logger.error(f"Enhanced history update failed: {e}")
    
    def _update_conversation_flow_state_post_response(self, context: ConversationContext, 
                                                    response: EnsembleChatResponse):
        """Update conversation flow state after generating response"""
        try:
            # Update flow state based on response characteristics
            if response.requires_action:
                # If we're asking for clarification, we're in exploring mode
                context.conversation_flow_state = 'exploring'
            elif len(response.follow_up_questions or []) > 2:
                # Multiple follow-ups suggest we're drilling down
                context.conversation_flow_state = 'drilling_down'
            elif response.confidence > 0.8 and len(context.conversation_history) > 3:
                # High confidence with history suggests we might be concluding
                recent_confidences = [entry.get('confidence', 0) for entry in context.conversation_history[-3:]]
                if all(conf > 0.7 for conf in recent_confidences):
                    context.conversation_flow_state = 'concluding'
            
        except Exception as e:
            logger.error(f"Flow state update failed: {e}")
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize enhanced intent detection patterns for forecast accuracy and confidence"""
        return {
            'forecast_accuracy': [
                r'how accurate|accuracy of|reliable|trust.*forecast',
                r'confidence.*forecast|forecast.*confidence',
                r'how good.*prediction|prediction.*quality',
                r'can.*trust|should.*believe|reliable.*forecast',
                r'accuracy.*rate|error.*rate|prediction.*error',
                r'how.*precise|precision.*forecast|forecast.*precision',
                # Enhanced patterns for forecast accuracy
                r'forecast.*performance|performance.*forecast',
                r'how.*well.*predict|prediction.*success',
                r'forecast.*quality|quality.*forecast',
                r'accurate.*ensemble|ensemble.*accurate',
                r'reliable.*model|model.*reliable',
                r'trustworthy.*forecast|forecast.*trustworthy',
                r'forecast.*validity|validity.*forecast',
                r'prediction.*reliability|reliability.*prediction'
            ],
            'confidence_intervals': [
                r'confidence.*interval|uncertainty.*band|prediction.*interval',
                r'p10|p50|p90|percentile|confidence.*level',
                r'how.*certain|certainty.*level|uncertainty.*forecast',
                r'range.*prediction|forecast.*range|upper.*lower.*bound',
                r'margin.*error|error.*margin|confidence.*bound',
                # Enhanced patterns for confidence questions
                r'uncertainty.*analysis|analysis.*uncertainty',
                r'confidence.*score|score.*confidence',
                r'prediction.*uncertainty|uncertainty.*prediction',
                r'forecast.*certainty|certainty.*forecast',
                r'how.*confident|confident.*forecast',
                r'reliability.*measure|measure.*reliability',
                r'statistical.*confidence|confidence.*statistical',
                r'prediction.*bounds|bounds.*prediction',
                r'forecast.*variance|variance.*forecast',
                r'error.*bounds|bounds.*error'
            ],
            'model_performance': [
                r'performance.*model|model.*performance',
                r'which.*better|best.*model|compare.*model',
                r'mae|mape|rmse|error.*rate',
                r'model.*ranking|rank.*model|top.*performing',
                r'accuracy.*comparison|performance.*metric'
            ],
            'forecast_explanation': [
                r'why.*forecast|explain.*prediction|how.*calculated',
                r'what.*mean|interpret.*result|understand.*forecast',
                r'methodology.*used|approach.*taken|algorithm.*behind',
                r'how.*work|explain.*method|forecast.*logic'
            ],
            'business_impact': [
                r'business.*impact|revenue.*effect|profit.*change',
                r'what.*should.*do|recommend|action.*take',
                r'strategic.*implication|business.*decision|operational.*impact',
                r'opportunity|risk|threat|advantage'
            ],
            'technical_details': [
                r'technical.*detail|algorithm.*used|method.*work',
                r'parameter|weight|coefficient|calculation',
                r'hyperparameter|configuration|model.*setting',
                r'ensemble.*method|weight.*calculation|model.*combination'
            ],
            'trend_analysis': [
                r'trend|trending|direction|growth|decline',
                r'seasonal|seasonality|pattern|cycle',
                r'increasing|decreasing|stable|volatile',
                r'future.*direction|forecast.*trend|trend.*forecast'
            ],
            'recommendation_request': [
                r'recommend|suggestion|advice|should.*do',
                r'best.*practice|what.*next|action.*plan',
                r'improve|optimize|enhance|better',
                r'strategy|approach|solution|fix'
            ]
        }
    
    def _initialize_entity_extractors(self) -> Dict[str, str]:
        """Initialize entity extraction patterns"""
        return {
            'time_horizon': r'(\d+)\s*(month|week|day|year)s?',
            'confidence_level': r'(\d+)%?\s*confidence',
            'model_name': r'(arima|ets|xgboost|lstm|croston)',
            'metric_type': r'(mae|mape|rmse|accuracy|confidence|weight)',
            'business_metric': r'(revenue|sales|profit|growth|decline)',
            'comparison_type': r'(compare|versus|vs|better|best|worst)'
        }
    
    def _initialize_ambiguity_detectors(self) -> List[str]:
        """Initialize ambiguity detection patterns"""
        return [
            r'\bit\b',  # Ambiguous pronoun
            r'\bthis\b',  # Ambiguous demonstrative
            r'\bthat\b',  # Ambiguous demonstrative
            r'\bthey\b',  # Ambiguous pronoun
            r'\bwhich\b.*\?',  # Ambiguous question
            r'\bhow\b.*\?',  # Potentially ambiguous question
        ]
    
    def _initialize_recommendation_templates(self) -> Dict[str, List[str]]:
        """Initialize recommendation templates"""
        return {
            'beginner': [
                "Start by reviewing the forecast dashboard to understand the predictions",
                "Check the model performance metrics to see which models are working best",
                "Look at the confidence intervals to understand prediction uncertainty"
            ],
            'intermediate': [
                "Analyze the ensemble weights to understand model contributions",
                "Review pattern detection results for business insights",
                "Compare individual model forecasts to identify strengths"
            ],
            'expert': [
                "Examine model drift detection for performance degradation",
                "Optimize ensemble parameters based on recent performance",
                "Implement custom weight constraints for domain knowledge"
            ]
        }
    
    def _detect_query_topic(self, query: str) -> str:
        """Detect the main topic of the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['forecast', 'predict', 'future']):
            return 'forecasting'
        elif any(word in query_lower for word in ['performance', 'accuracy', 'error']):
            return 'performance'
        elif any(word in query_lower for word in ['insight', 'recommend', 'business']):
            return 'insights'
        elif any(word in query_lower for word in ['model', 'algorithm', 'technical']):
            return 'technical'
        else:
            return 'general'
    
    def _update_user_preferences(self, context: ConversationContext, query: str):
        """Update user preferences based on query patterns"""
        query_lower = query.lower()
        
        # Detect expertise level
        if any(word in query_lower for word in ['technical', 'algorithm', 'parameter']):
            context.user_preferences['expertise_level'] = 'expert'
        elif any(word in query_lower for word in ['explain', 'what does', 'how to']):
            context.user_preferences['expertise_level'] = 'beginner'
        else:
            context.user_preferences['expertise_level'] = 'intermediate'
        
        # Detect detail preference
        if any(word in query_lower for word in ['detail', 'comprehensive', 'thorough']):
            context.user_preferences['detail_level'] = 'detailed'
        elif any(word in query_lower for word in ['summary', 'brief', 'quick']):
            context.user_preferences['detail_level'] = 'summary'
        else:
            context.user_preferences['detail_level'] = 'standard'
    
    def _determine_intent_with_context(self, query: str, context: ConversationContext) -> str:
        """Determine query intent using enhanced context and pattern matching"""
        query_lower = query.lower()
        
        # Check explicit patterns first with confidence scoring
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        # If we have pattern matches, use the highest scoring intent
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Enhanced context-based intent inference
        if context.current_topic:
            # Forecast accuracy and confidence specific inference
            if context.current_topic == 'forecasting':
                if any(word in query_lower for word in ['accurate', 'confidence', 'reliable', 'trust']):
                    return 'forecast_accuracy'
                elif any(word in query_lower for word in ['interval', 'uncertainty', 'range', 'bound']):
                    return 'confidence_intervals'
                elif any(word in query_lower for word in ['how', 'why', 'what', 'explain']):
                    return 'forecast_explanation'
            
            elif context.current_topic == 'performance':
                if any(word in query_lower for word in ['improve', 'better', 'optimize']):
                    return 'recommendation_request'
                elif any(word in query_lower for word in ['compare', 'versus', 'best', 'worst']):
                    return 'model_performance'
            
            elif context.current_topic == 'insights':
                if any(word in query_lower for word in ['recommend', 'should', 'action']):
                    return 'recommendation_request'
                else:
                    return 'business_impact'
        
        # Use conversation history for intent inference
        if context.conversation_history:
            recent_topics = [entry.get('topic', '') for entry in context.conversation_history[-2:]]
            
            if 'performance' in recent_topics and any(word in query_lower for word in ['accuracy', 'error']):
                return 'forecast_accuracy'
            elif 'forecasting' in recent_topics and any(word in query_lower for word in ['confidence', 'certain']):
                return 'confidence_intervals'
        
        # Business context inference
        if context.business_context.get('confidence_focus') and any(word in query_lower for word in ['forecast', 'prediction']):
            return 'confidence_intervals'
        elif context.business_context.get('forecast_focus') and any(word in query_lower for word in ['accurate', 'good', 'reliable']):
            return 'forecast_accuracy'
        
        # Model-specific context
        if context.mentioned_models and any(word in query_lower for word in ['performance', 'accuracy', 'error']):
            return 'model_performance'
        
        # Question type inference
        if query_lower.startswith('how accurate') or query_lower.startswith('how reliable'):
            return 'forecast_accuracy'
        elif query_lower.startswith('how confident') or 'confidence' in query_lower:
            return 'confidence_intervals'
        elif query_lower.startswith('why') or query_lower.startswith('how does'):
            return 'forecast_explanation'
        elif query_lower.startswith('what should') or 'recommend' in query_lower:
            return 'recommendation_request'
        
        return 'general'
    
    def _calculate_query_clarity(self, query: str) -> float:
        """Calculate query clarity score"""
        try:
            clarity_factors = []
            
            # Length factor (not too short, not too long)
            length = len(query.split())
            if 3 <= length <= 20:
                clarity_factors.append(1.0)
            elif length < 3:
                clarity_factors.append(0.5)
            else:
                clarity_factors.append(0.8)
            
            # Question structure
            if query.strip().endswith('?'):
                clarity_factors.append(1.0)
            elif any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where']):
                clarity_factors.append(0.9)
            else:
                clarity_factors.append(0.7)
            
            # Specific terms
            specific_terms = ['forecast', 'model', 'accuracy', 'performance', 'prediction']
            if any(term in query.lower() for term in specific_terms):
                clarity_factors.append(1.0)
            else:
                clarity_factors.append(0.6)
            
            return np.mean(clarity_factors)
            
        except Exception as e:
            logger.error(f"Clarity calculation failed: {e}")
            return 0.5
    
    def _calculate_context_relevance(self, query: str, context: ConversationContext) -> float:
        """Calculate how relevant the query is to current context"""
        try:
            if not context.current_topic:
                return 0.5
            
            query_lower = query.lower()
            relevance_score = 0.0
            
            # Topic continuity
            if context.current_topic in query_lower:
                relevance_score += 0.4
            
            # Model continuity
            if context.mentioned_models:
                for model in context.mentioned_models:
                    if model in query_lower:
                        relevance_score += 0.3
                        break
            
            # Metric continuity
            if context.discussed_metrics:
                for metric in context.discussed_metrics:
                    if metric in query_lower:
                        relevance_score += 0.3
                        break
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Context relevance calculation failed: {e}")
            return 0.5
    
    def _detect_ambiguous_terms(self, query: str, context: ConversationContext) -> List[str]:
        """Detect ambiguous terms in query"""
        ambiguous = []
        
        for pattern in self.ambiguity_detectors:
            matches = re.findall(pattern, query, re.IGNORECASE)
            ambiguous.extend(matches)
        
        return list(set(ambiguous))
    
    def _identify_missing_context(self, query: str, intent: str, context: ConversationContext) -> List[str]:
        """Identify missing context for proper understanding"""
        missing = []
        query_lower = query.lower()
        
        # Check for missing time horizon in forecast queries
        if intent in ['forecast_accuracy', 'forecast_explanation'] and not re.search(r'\d+\s*(month|week|day)', query_lower):
            missing.append('time_horizon')
        
        # Check for missing model specification in comparison queries
        if 'compare' in query_lower and len(context.mentioned_models) < 2:
            missing.append('specific_model')
        
        # Check for missing metric type in performance queries
        if intent == 'model_performance' and not any(metric in query_lower for metric in ['mae', 'mape', 'rmse', 'accuracy']):
            missing.append('metric_type')
        
        return missing
    
    def _generate_clarification_suggestions(self, ambiguous_terms: List[str], missing_context: List[str]) -> List[str]:
        """Generate clarification suggestions"""
        suggestions = []
        
        for missing in missing_context:
            if missing == 'time_horizon':
                suggestions.append("What time period are you interested in?")
            elif missing == 'specific_model':
                suggestions.append("Which models would you like to compare?")
            elif missing == 'metric_type':
                suggestions.append("Which performance metric interests you?")
        
        if ambiguous_terms:
            suggestions.append(f"Could you clarify what you mean by '{ambiguous_terms[0]}'?")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _extract_entities_with_context(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Extract entities using context"""
        entities = {}
        
        for entity_type, pattern in self.entity_extractors.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        # Use context to fill missing entities
        if 'model_name' not in entities and len(context.mentioned_models) == 1:
            entities['model_name'] = context.mentioned_models[0]
        
        return entities
    
    def _assess_query_complexity(self, query: str, entities: Dict[str, Any], context: ConversationContext) -> str:
        """Assess query complexity"""
        complexity_score = 0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 1
        
        # Multiple entities
        if len(entities) > 3:
            complexity_score += 1
        
        # Multiple questions
        if query.count('?') > 1:
            complexity_score += 1
        
        # Technical terms
        technical_terms = ['algorithm', 'parameter', 'coefficient', 'optimization']
        if any(term in query.lower() for term in technical_terms):
            complexity_score += 1
        
        if complexity_score >= 3:
            return 'complex'
        elif complexity_score >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _create_recommendation_context(self, understanding: QueryUnderstanding, 
                                     context: ConversationContext) -> RecommendationContext:
        """Create recommendation context"""
        return RecommendationContext(
            user_expertise_level=context.user_preferences.get('expertise_level', 'intermediate'),
            business_role=context.business_context.get('role', 'analyst'),
            current_focus=context.current_topic or 'general',
            recent_actions=[entry.get('topic', '') for entry in context.conversation_history[-3:]],
            preferred_detail_level=context.user_preferences.get('detail_level', 'standard')
        )
    
    async def _generate_contextual_recommendations(self, understanding: QueryUnderstanding,
                                                 context: ConversationContext,
                                                 rec_context: RecommendationContext) -> List[str]:
        """Generate contextual recommendations based on business insights"""
        recommendations = []
        
        try:
            # Get base recommendations for expertise level
            base_recs = self.recommendation_templates.get(rec_context.user_expertise_level, [])
            recommendations.extend(base_recs[:1])  # Only take 1 base recommendation
            
            # Generate intelligent recommendations based on query intent and context
            if understanding.interpreted_intent == 'forecast_accuracy':
                recommendations.extend(await self._generate_accuracy_recommendations(context, rec_context))
            elif understanding.interpreted_intent == 'model_performance':
                recommendations.extend(await self._generate_performance_recommendations(context, rec_context))
            elif understanding.interpreted_intent == 'business_impact':
                recommendations.extend(await self._generate_business_recommendations(context, rec_context))
            elif understanding.interpreted_intent == 'forecast_explanation':
                recommendations.extend(await self._generate_explanation_recommendations(context, rec_context))
            elif understanding.interpreted_intent == 'technical_details':
                recommendations.extend(await self._generate_technical_recommendations(context, rec_context))
            else:
                # General recommendations based on conversation history
                recommendations.extend(await self._generate_general_recommendations(context, rec_context))
            
            # Add insights-based recommendations if insights engine is available
            if self.insights_engine:
                insights_recs = await self._retrieve_insights_based_recommendations(understanding, context)
                recommendations.extend(insights_recs)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)
            
            return unique_recommendations[:4]  # Limit to 4 recommendations
            
        except Exception as e:
            logger.error(f"Contextual recommendations generation failed: {e}")
            return ["Review the forecast dashboard for detailed insights"]
    
    def _enhance_suggested_actions(self, base_actions: List[str], 
                                 recommendations: List[str],
                                 rec_context: RecommendationContext) -> List[str]:
        """Enhance suggested actions with recommendations"""
        enhanced_actions = base_actions.copy()
        
        # Add recommendation-based actions
        for rec in recommendations:
            if rec not in enhanced_actions:
                enhanced_actions.append(rec)
        
        # Prioritize based on user context
        if rec_context.user_expertise_level == 'beginner':
            enhanced_actions.insert(0, "View getting started guide")
        elif rec_context.user_expertise_level == 'expert':
            enhanced_actions.append("Access advanced configuration options")
        
        return enhanced_actions[:5]  # Limit to 5 actions
    
    def _generate_personalized_follow_ups(self, understanding: QueryUnderstanding,
                                        context: ConversationContext,
                                        response: EnsembleChatResponse) -> List[str]:
        """Generate personalized follow-up questions"""
        follow_ups = response.follow_up_questions or []
        
        # Add context-aware follow-ups
        if context.current_topic == 'forecasting':
            follow_ups.append("Would you like to see how individual models contributed?")
        elif context.current_topic == 'performance':
            follow_ups.append("Are you interested in improving model performance?")
        
        # Add expertise-level appropriate follow-ups
        expertise = context.user_preferences.get('expertise_level', 'intermediate')
        if expertise == 'beginner':
            follow_ups.append("Would you like me to explain this in simpler terms?")
        elif expertise == 'expert':
            follow_ups.append("Do you want to see the technical implementation details?")
        
        return follow_ups[:3]  # Limit to 3 follow-ups
    
    def _format_recommendations(self, recommendations: List[str], rec_context: RecommendationContext) -> str:
        """Format recommendations for display"""
        if not recommendations:
            return ""
        
        formatted = "\n💡 **Recommendations:**\n"
        for i, rec in enumerate(recommendations, 1):
            formatted += f"{i}. {rec}\n"
        
        return formatted.strip()
    
    def _generate_clarification_options(self, understanding: QueryUnderstanding,
                                      context: ConversationContext) -> List[str]:
        """Generate clarification options"""
        options = []
        
        if 'time_horizon' in understanding.missing_context:
            options.extend(["Next 3 months", "Next 6 months", "Next year"])
        
        if 'specific_model' in understanding.missing_context:
            options.extend(["Compare all models", "Focus on best performer", "Analyze specific model"])
        
        if 'metric_type' in understanding.missing_context:
            options.extend(["Accuracy metrics", "Error rates", "Confidence levels"])
        
        return options[:4]  # Limit to 4 options
    
    def _create_fallback_understanding(self, query: str) -> QueryUnderstanding:
        """Create fallback understanding when parsing fails"""
        return QueryUnderstanding(
            original_query=query,
            interpreted_intent='general',
            confidence_score=0.3,
            ambiguous_terms=[],
            missing_context=[],
            suggested_clarifications=["Could you rephrase your question?"],
            extracted_entities={},
            query_complexity='simple'
        )
    
    async def _generate_accuracy_recommendations(self, context: ConversationContext, 
                                               rec_context: RecommendationContext) -> List[str]:
        """Generate recommendations for forecast accuracy queries"""
        recommendations = []
        
        if rec_context.user_expertise_level == 'beginner':
            recommendations.extend([
                "Check the confidence intervals to understand prediction reliability",
                "Compare ensemble accuracy with individual model performance"
            ])
        elif rec_context.user_expertise_level == 'expert':
            recommendations.extend([
                "Analyze model residuals for systematic bias patterns",
                "Review cross-validation metrics for out-of-sample performance"
            ])
        else:
            recommendations.extend([
                "Review recent accuracy trends to identify performance changes",
                "Examine confidence intervals for uncertainty assessment"
            ])
        
        # Add context-specific recommendations
        if 'arima' in context.mentioned_models:
            recommendations.append("Consider ARIMA model diagnostics for trend accuracy")
        if 'lstm' in context.mentioned_models:
            recommendations.append("Check LSTM sequence learning performance")
        
        return recommendations[:2]
    
    async def _generate_performance_recommendations(self, context: ConversationContext,
                                                  rec_context: RecommendationContext) -> List[str]:
        """Generate recommendations for model performance queries"""
        recommendations = []
        
        if rec_context.current_focus == 'performance':
            recommendations.extend([
                "Monitor model drift indicators for performance degradation",
                "Compare recent vs historical performance metrics"
            ])
        
        if len(context.mentioned_models) > 1:
            recommendations.append("Analyze individual model strengths for ensemble optimization")
        else:
            recommendations.append("Review ensemble weight distribution for model contributions")
        
        # Add expertise-specific recommendations
        if rec_context.user_expertise_level == 'expert':
            recommendations.append("Implement custom performance metrics for domain-specific evaluation")
        
        return recommendations[:2]
    
    async def _generate_business_recommendations(self, context: ConversationContext,
                                              rec_context: RecommendationContext) -> List[str]:
        """Generate business impact recommendations"""
        recommendations = []
        
        business_context = context.business_context.get('industry', 'general')
        
        if rec_context.business_role == 'executive':
            recommendations.extend([
                "Generate executive summary with key performance indicators",
                "Focus on revenue impact and strategic implications"
            ])
        elif rec_context.business_role == 'manager':
            recommendations.extend([
                "Review operational metrics and resource allocation needs",
                "Identify process improvement opportunities from forecasts"
            ])
        else:
            recommendations.extend([
                "Analyze forecast implications for daily operations",
                "Generate actionable insights for tactical decisions"
            ])
        
        return recommendations[:2]
    
    async def _generate_explanation_recommendations(self, context: ConversationContext,
                                                  rec_context: RecommendationContext) -> List[str]:
        """Generate recommendations for forecast explanation queries"""
        recommendations = []
        
        if rec_context.preferred_detail_level == 'summary':
            recommendations.extend([
                "View plain-language forecast summary",
                "Focus on key trends and patterns"
            ])
        elif rec_context.preferred_detail_level == 'detailed':
            recommendations.extend([
                "Explore detailed model contributions and methodology",
                "Review pattern detection and seasonality analysis"
            ])
        else:
            recommendations.extend([
                "Check forecast assumptions and input parameters",
                "Review confidence intervals and uncertainty sources"
            ])
        
        return recommendations[:2]
    
    async def _generate_technical_recommendations(self, context: ConversationContext,
                                                rec_context: RecommendationContext) -> List[str]:
        """Generate technical detail recommendations"""
        recommendations = []
        
        if rec_context.user_expertise_level == 'expert':
            recommendations.extend([
                "Access model hyperparameters and configuration details",
                "Review ensemble weight calculation methodology"
            ])
        else:
            recommendations.extend([
                "Learn about ensemble forecasting concepts",
                "Understand model selection and weighting principles"
            ])
        
        if context.current_topic == 'technical':
            recommendations.append("Explore advanced configuration options")
        
        return recommendations[:2]
    
    async def _generate_general_recommendations(self, context: ConversationContext,
                                             rec_context: RecommendationContext) -> List[str]:
        """Generate general recommendations based on conversation history"""
        recommendations = []
        
        # Analyze conversation patterns
        recent_topics = [entry.get('topic', '') for entry in context.conversation_history[-3:]]
        
        if 'forecasting' in recent_topics:
            recommendations.append("Explore forecast accuracy and confidence metrics")
        if 'performance' in recent_topics:
            recommendations.append("Review model comparison and ensemble weights")
        if not recent_topics:
            recommendations.extend([
                "Start with forecast overview and key metrics",
                "Explore model performance dashboard"
            ])
        
        return recommendations[:2]
    
    async def _retrieve_insights_based_recommendations(self, understanding: QueryUnderstanding,
                                                     context: ConversationContext) -> List[str]:
        """Enhanced recommendation retrieval system based on business insights"""
        try:
            if not self.insights_engine:
                return await self._generate_fallback_insights_recommendations(understanding, context)
            
            insights_recommendations = []
            
            # Enhanced insights-based recommendations by intent
            if understanding.interpreted_intent == 'forecast_accuracy':
                insights_recommendations.extend([
                    "Review ensemble weight distribution to identify top-performing models",
                    "Analyze recent accuracy trends to detect performance degradation",
                    "Compare cross-validation results with live forecast performance"
                ])
                
                # Add model-specific accuracy insights
                if context.mentioned_models:
                    for model in context.mentioned_models[:2]:  # Limit to 2 models
                        insights_recommendations.append(f"Check {model.upper()} model diagnostics for accuracy optimization")
                        
            elif understanding.interpreted_intent == 'confidence_intervals':
                insights_recommendations.extend([
                    "Examine prediction interval coverage for reliability assessment",
                    "Review uncertainty quantification across different forecast horizons",
                    "Analyze confidence interval width trends for stability insights"
                ])
                
                # Add confidence-specific insights based on business context
                if context.business_context.get('industry') == 'retail':
                    insights_recommendations.append("Consider seasonal confidence variations for retail planning")
                elif context.business_context.get('industry') == 'manufacturing':
                    insights_recommendations.append("Account for supply chain uncertainty in confidence assessment")
                    
            elif understanding.interpreted_intent == 'model_performance':
                insights_recommendations.extend([
                    "Compare individual model contributions to ensemble performance",
                    "Identify models showing performance drift for retraining priority",
                    "Analyze model complementarity for ensemble optimization"
                ])
                
            elif understanding.interpreted_intent == 'business_impact':
                insights_recommendations.extend([
                    "Focus on high-impact forecast scenarios for strategic planning",
                    "Prioritize accuracy improvements in key business segments",
                    "Leverage forecast insights for proactive decision making"
                ])
                
                # Add role-specific business recommendations
                business_role = context.business_context.get('role', 'analyst')
                if business_role == 'executive':
                    insights_recommendations.append("Generate executive dashboard with key forecast KPIs")
                elif business_role == 'manager':
                    insights_recommendations.append("Create operational forecast reports for team planning")
                    
            elif understanding.interpreted_intent == 'recommendation_request':
                insights_recommendations.extend([
                    "Implement automated model retraining based on performance thresholds",
                    "Set up forecast accuracy monitoring alerts for proactive management",
                    "Configure ensemble weight constraints based on domain expertise"
                ])
                
            # Add context-aware insights based on conversation history
            if len(context.conversation_history) > 2:
                recent_intents = [entry.get('intent', '') for entry in context.conversation_history[-3:]]
                if 'forecast_accuracy' in recent_intents and 'confidence_intervals' in recent_intents:
                    insights_recommendations.append("Consider implementing adaptive confidence intervals based on accuracy trends")
                    
            # Add temporal insights based on query timing
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                insights_recommendations.append("Schedule regular forecast review meetings for continuous improvement")
            
            return insights_recommendations[:3]  # Limit to 3 insights recommendations
            
        except Exception as e:
            logger.error(f"Enhanced insights-based recommendations failed: {e}")
            return await self._generate_fallback_insights_recommendations(understanding, context)
    
    async def _generate_fallback_insights_recommendations(self, understanding: QueryUnderstanding,
                                                        context: ConversationContext) -> List[str]:
        """Generate fallback recommendations when insights engine is unavailable"""
        fallback_recommendations = []
        
        # Intent-based fallback recommendations
        intent_fallbacks = {
            'forecast_accuracy': [
                "Review model performance metrics in the dashboard",
                "Check recent forecast vs actual comparisons"
            ],
            'confidence_intervals': [
                "Examine prediction intervals in forecast visualizations",
                "Review uncertainty metrics in model performance section"
            ],
            'model_performance': [
                "Compare individual model accuracy scores",
                "Analyze ensemble weight evolution over time"
            ],
            'business_impact': [
                "Generate business summary report from forecast results",
                "Review key performance indicators in insights dashboard"
            ]
        }
        
        fallback_recommendations.extend(
            intent_fallbacks.get(understanding.interpreted_intent, ["Explore the forecast dashboard for detailed insights"])
        )
        
        return fallback_recommendations[:2]
    
    async def _generate_technical_recommendations(self, context: ConversationContext,
                                                  rec_context: RecommendationContext) -> List[str]:
        """Generate technical recommendations for advanced users"""
        recommendations = []
        
        if rec_context.user_expertise_level == 'expert':
            recommendations.extend([
                "Review ensemble weight optimization algorithms",
                "Analyze model hyperparameter sensitivity",
                "Implement custom performance metrics for domain-specific evaluation"
            ])
        elif rec_context.user_expertise_level == 'intermediate':
            recommendations.extend([
                "Explore model configuration options",
                "Review technical performance diagnostics",
                "Understand ensemble methodology details"
            ])
        else:
            recommendations.extend([
                "Learn about forecasting model basics",
                "Understand ensemble approach benefits",
                "Review technical glossary for key terms"
            ])
        
        # Add context-specific technical recommendations
        if context.mentioned_models:
            for model in context.mentioned_models[:1]:  # Focus on one model
                recommendations.append(f"Deep dive into {model.upper()} model technical specifications")
        
        return recommendations[:2]

    def _create_error_response(self, error_message: str) -> EnsembleChatResponse:
        """Create error response"""
        return EnsembleChatResponse(
            response_text=f"I encountered an issue processing your query: {error_message}. Please try rephrasing your question.",
            confidence=0.1,
            sources=["Error Handler"],
            suggested_actions=[
                "Try a simpler question",
                "Check your query for typos",
                "Ask for help with specific topics"
            ]
        )