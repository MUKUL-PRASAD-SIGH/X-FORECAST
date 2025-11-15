"""
FVA Analysis and Reporting System
Provides comprehensive FVA analysis with automated insights and recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from .fva_tracker import FVATracker, FVAMetrics, FVAReport
from .fva_database import FVADataAccessLayer

logger = logging.getLogger(__name__)

class ReportType(Enum):
    USER_PERFORMANCE = "user_performance"
    PRODUCT_ANALYSIS = "product_analysis"
    TREND_ANALYSIS = "trend_analysis"
    EXECUTIVE_SUMMARY = "executive_summary"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class FVATrendAnalysis:
    """FVA trend analysis results"""
    metric_name: str
    trend_direction: str  # improving, declining, stable
    trend_strength: float  # 0-1
    trend_significance: float  # p-value
    seasonal_component: bool
    forecast_next_period: float
    confidence_interval: Tuple[float, float]

@dataclass
class UserFVAProfile:
    """Comprehensive user FVA profile"""
    user_id: str
    user_name: str
    department: str
    role: str
    
    # Performance metrics
    overall_fva_score: float
    accuracy_improvement: float
    positive_fva_rate: float
    override_frequency: float
    confidence_calibration: float
    
    # Behavioral patterns
    preferred_override_types: List[str]
    peak_activity_hours: List[int]
    seasonal_patterns: Dict[str, float]
    
    # Comparative analysis
    peer_group_ranking: int
    percentile_rank: float
    improvement_trend: str
    
    # Recommendations
    strengths: List[str]
    improvement_areas: List[str]
    training_recommendations: List[str]

@dataclass
class ProductFVAAnalysis:
    """Product-level FVA analysis"""
    sku: str
    product_name: str
    category: str
    
    # FVA metrics
    override_frequency: float
    avg_fva_impact: float
    forecast_difficulty_score: float
    
    # Pattern analysis
    systematic_bias_detected: bool
    seasonal_override_patterns: Dict[str, float]
    user_consensus_score: float  # How much users agree on adjustments
    
    # Business impact
    revenue_impact: float
    inventory_impact: float
    service_level_impact: float
    
    # Recommendations
    model_improvement_suggestions: List[str]
    override_guidelines: List[str]

class FVAReportingEngine:
    """Advanced FVA reporting and analysis engine"""
    
    def __init__(self, fva_tracker: FVATracker):
        self.fva_tracker = fva_tracker
        self.data_access = fva_tracker.data_access
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_user_performance_report(self, user_id: str, period_days: int = 90) -> UserFVAProfile:
        """Generate comprehensive user performance report"""
        
        # Get user overrides
        user_overrides = self.data_acc