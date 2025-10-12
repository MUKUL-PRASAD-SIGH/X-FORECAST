"""
Predictive Maintenance Engine for Cyberpunk AI Dashboard
Monitors system health and predicts potential failures before they occur
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import psutil
import time
import json
from abc import ABC, abstractmethod
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of system failures that can be predicted"""
    CPU_OVERLOAD = "cpu_overload"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_SPACE_CRITICAL = "disk_space_critical"
    DATABASE_PERFORMANCE = "database_performance"
    API_RESPONSE_DEGRADATION = "api_response_degradation"
    MODEL_ACCURACY_DECLINE = "model_accuracy_decline"
    DATA_QUALITY_ISSUES = "data_quality_issues"
    NETWORK_LATENCY = "network_latency"

class Severity(Enum):
    """Severity levels for predictions and alerts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MaintenanceAction(Enum):
    """Types of maintenance actions"""
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    CLEAR_CACHE = "clear_cache"
    OPTIMIZE_DATABASE = "optimize_database"
    RETRAIN_MODEL = "retrain_model"
    CLEAN_DATA = "clean_data"
    UPDATE_CONFIGURATION = "update_configuration"
    ALERT_ADMIN = "alert_admin"

@dataclass
class SystemMetrics:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    database_metrics: Dict[str, float]
    api_response_times: Dict[str, float]
    model_performance: Dict[str, float]
    data_quality_scores: Dict[str, float]
    active_users: int
    request_rate: float
    error_rate: float

@dataclass
class FailurePrediction:
    """Prediction of potential system failure"""
    failure_type: FailureType
    probability: float
    severity: Severity
    predicted_time: datetime
    confidence_interval: Tuple[datetime, datetime]
    contributing_factors: List[str]
    recommended_actions: List[MaintenanceAction]
    impact_assessment: str
    prevention_window: timedelta

@dataclass
class PerformanceAnalysis:
    """Analysis of system performance trends"""
    metric_name: str
    current_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1
    anomaly_score: float
    historical_percentile: float
    predicted_values: List[float]
    recommendations: List[str]

@dataclass
class MaintenanceSchedule:
    """Scheduled maintenance activities"""
    maintenance_id: str
    scheduled_time: datetime
    duration_estimate: timedelta
    actions: List[MaintenanceAction]
    priority: Severity
    affected_services: List[str]
    rollback_plan: str
    success_criteria: List[str]
    notification_list: List[str]

@dataclass
class ResourceAnalysis:
    """Analysis of resource utilization"""
    resource_type: str
    current_utilization: float
    peak_utilization: float
    average_utilization: float
    utilization_trend: str
    capacity_remaining: float
    projected_exhaustion: Optional[datetime]
    scaling_recommendations: List[str]

@dataclass
class CapacityRecommendations:
    """Capacity planning recommendations"""
    resource_type: str
    current_capacity: float
    recommended_capacity: float
    scaling_timeline: str
    cost_impact: str
    risk_assessment: str
    implementation_steps: List[str]

class MetricsCollector:
    """Collect comprehensive system health metrics"""
    
    def __init__(self):
        self.collection_interval = 60  # seconds
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 10000
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system health metrics"""
        try:
            # CPU and Memory metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Mock database metrics (replace with actual database monitoring)
            database_metrics = {
                "connection_count": np.random.randint(10, 100),
                "query_response_time": np.random.uniform(0.1, 2.0),
                "cache_hit_ratio": np.random.uniform(0.8, 0.99),
                "lock_wait_time": np.random.uniform(0.0, 0.5)
            }
            
            # Mock API response times
            api_response_times = {
                "forecast_endpoint": np.random.uniform(0.1, 1.0),
                "analytics_endpoint": np.random.uniform(0.2, 1.5),
                "dashboard_endpoint": np.random.uniform(0.05, 0.3)
            }
            
            # Mock model performance metrics
            model_performance = {
                "forecast_accuracy": np.random.uniform(0.85, 0.95),
                "prediction_latency": np.random.uniform(0.1, 0.5),
                "model_drift_score": np.random.uniform(0.0, 0.3)
            }
            
            # Mock data quality scores
            data_quality_scores = {
                "completeness": np.random.uniform(0.9, 1.0),
                "consistency": np.random.uniform(0.85, 0.98),
                "accuracy": np.random.uniform(0.9, 0.99),
                "timeliness": np.random.uniform(0.8, 1.0)
            }
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io=network_io,
                database_metrics=database_metrics,
                api_response_times=api_response_times,
                model_performance=model_performance,
                data_quality_scores=data_quality_scores,
                active_users=np.random.randint(1, 50),
                request_rate=np.random.uniform(10, 100),
                error_rate=np.random.uniform(0.0, 0.05)
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics in case of error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                database_metrics={},
                api_response_times={},
                model_performance={},
                data_quality_scores={},
                active_users=0,
                request_rate=0.0,
                error_rate=0.0
            )
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

class TrendAnalyzer:
    """Analyze performance trends and patterns"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def analyze_metric_trend(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze trend for a specific metric"""
        if len(values) < 3:
            return {"trend": "insufficient_data", "strength": 0.0}
        
        # Calculate trend using linear regression
        x = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        # Simple linear trend calculation
        slope = np.polyfit(x.flatten(), y, 1)[0]
        
        # Determine trend direction and strength
        if abs(slope) < 0.01:
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "increasing"
            trend_strength = min(abs(slope) * 10, 1.0)
        else:
            trend_direction = "decreasing"
            trend_strength = min(abs(slope) * 10, 1.0)
        
        # Calculate volatility
        volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        return {
            "trend": trend_direction,
            "strength": trend_strength,
            "slope": slope,
            "volatility": volatility,
            "r_squared": self._calculate_r_squared(x.flatten(), y)
        }
    
    def detect_anomalies(self, metrics_history: List[SystemMetrics]) -> List[Dict[str, Any]]:
        """Detect anomalies in system metrics"""
        if len(metrics_history) < 10:
            return []
        
        # Prepare data for anomaly detection
        features = []
        for metrics in metrics_history:
            feature_vector = [
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                metrics.request_rate,
                metrics.error_rate,
                metrics.database_metrics.get("query_response_time", 0),
                metrics.api_response_times.get("forecast_endpoint", 0)
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Fit anomaly detector
        self.anomaly_detector.fit(features_array)
        anomaly_scores = self.anomaly_detector.decision_function(features_array)
        anomalies = self.anomaly_detector.predict(features_array)
        
        # Identify anomalous points
        anomalous_metrics = []
        for i, (is_anomaly, score) in enumerate(zip(anomalies, anomaly_scores)):
            if is_anomaly == -1:  # Anomaly detected
                anomalous_metrics.append({
                    "timestamp": metrics_history[i].timestamp,
                    "anomaly_score": score,
                    "metrics": metrics_history[i]
                })
        
        return anomalous_metrics
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared for trend analysis"""
        try:
            correlation_matrix = np.corrcoef(x, y)
            correlation = correlation_matrix[0, 1]
            return correlation ** 2
        except:
            return 0.0

class FailurePredictionModel:
    """Machine learning model for predicting system failures"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            "cpu_usage", "memory_usage", "disk_usage", "request_rate", 
            "error_rate", "db_response_time", "api_response_time"
        ]
        self.is_trained = False
    
    def train_models(self, metrics_history: List[SystemMetrics], 
                    failure_history: List[Dict[str, Any]]):
        """Train failure prediction models"""
        if len(metrics_history) < 100:
            logger.warning("Insufficient data for training failure prediction models")
            return
        
        # Prepare training data
        X, y = self._prepare_training_data(metrics_history, failure_history)
        
        if len(X) == 0:
            logger.warning("No training data available")
            return
        
        # Train models for each failure type
        for failure_type in FailureType:
            try:
                # Create binary classification problem
                y_binary = (y == failure_type.value).astype(int)
                
                if np.sum(y_binary) < 5:  # Need at least 5 positive examples
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model (using Isolation Forest for anomaly detection)
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X_train_scaled[y_train == 0])  # Train on normal data
                
                # Store model and scaler
                self.models[failure_type.value] = model
                self.scalers[failure_type.value] = scaler
                
                logger.info(f"Trained model for {failure_type.value}")
                
            except Exception as e:
                logger.error(f"Error training model for {failure_type.value}: {e}")
        
        self.is_trained = True
    
    def predict_failures(self, current_metrics: SystemMetrics, 
                        metrics_history: List[SystemMetrics]) -> List[FailurePrediction]:
        """Predict potential system failures"""
        if not self.is_trained or not self.models:
            return self._generate_rule_based_predictions(current_metrics, metrics_history)
        
        predictions = []
        
        # Prepare current metrics for prediction
        features = self._extract_features(current_metrics)
        
        for failure_type_str, model in self.models.items():
            try:
                failure_type = FailureType(failure_type_str)
                scaler = self.scalers[failure_type_str]
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Get anomaly score
                anomaly_score = model.decision_function(features_scaled)[0]
                is_anomaly = model.predict(features_scaled)[0] == -1
                
                if is_anomaly:
                    # Convert anomaly score to probability
                    probability = min(abs(anomaly_score) * 0.5, 0.95)
                    
                    # Determine severity
                    if probability > 0.8:
                        severity = Severity.CRITICAL
                    elif probability > 0.6:
                        severity = Severity.HIGH
                    elif probability > 0.4:
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW
                    
                    # Create prediction
                    prediction = FailurePrediction(
                        failure_type=failure_type,
                        probability=probability,
                        severity=severity,
                        predicted_time=datetime.now() + timedelta(hours=2),
                        confidence_interval=(
                            datetime.now() + timedelta(hours=1),
                            datetime.now() + timedelta(hours=4)
                        ),
                        contributing_factors=self._identify_contributing_factors(
                            failure_type, current_metrics
                        ),
                        recommended_actions=self._get_recommended_actions(failure_type),
                        impact_assessment=self._assess_impact(failure_type),
                        prevention_window=timedelta(hours=1)
                    )
                    
                    predictions.append(prediction)
                    
            except Exception as e:
                logger.error(f"Error predicting {failure_type_str}: {e}")
        
        return predictions
    
    def _prepare_training_data(self, metrics_history: List[SystemMetrics], 
                             failure_history: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from metrics and failure history"""
        # This is a simplified implementation
        # In practice, you would need actual failure labels
        X = []
        y = []
        
        for metrics in metrics_history:
            features = self._extract_features(metrics)
            X.append(features)
            
            # Mock failure labels based on thresholds
            if metrics.cpu_usage > 90:
                y.append(FailureType.CPU_OVERLOAD.value)
            elif metrics.memory_usage > 95:
                y.append(FailureType.MEMORY_EXHAUSTION.value)
            elif metrics.disk_usage > 90:
                y.append(FailureType.DISK_SPACE_CRITICAL.value)
            else:
                y.append("normal")
        
        return np.array(X), np.array(y)
    
    def _extract_features(self, metrics: SystemMetrics) -> List[float]:
        """Extract features from system metrics"""
        return [
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.disk_usage,
            metrics.request_rate,
            metrics.error_rate,
            metrics.database_metrics.get("query_response_time", 0),
            metrics.api_response_times.get("forecast_endpoint", 0)
        ]
    
    def _generate_rule_based_predictions(self, current_metrics: SystemMetrics,
                                       metrics_history: List[SystemMetrics]) -> List[FailurePrediction]:
        """Generate predictions using rule-based approach when ML models aren't available"""
        predictions = []
        
        # CPU overload prediction
        if current_metrics.cpu_usage > 85:
            predictions.append(FailurePrediction(
                failure_type=FailureType.CPU_OVERLOAD,
                probability=min((current_metrics.cpu_usage - 85) / 15, 0.95),
                severity=Severity.HIGH if current_metrics.cpu_usage > 95 else Severity.MEDIUM,
                predicted_time=datetime.now() + timedelta(minutes=30),
                confidence_interval=(
                    datetime.now() + timedelta(minutes=15),
                    datetime.now() + timedelta(hours=1)
                ),
                contributing_factors=["High CPU utilization", "Increased request load"],
                recommended_actions=[MaintenanceAction.SCALE_RESOURCES, MaintenanceAction.RESTART_SERVICE],
                impact_assessment="System performance degradation, potential service interruption",
                prevention_window=timedelta(minutes=15)
            ))
        
        # Memory exhaustion prediction
        if current_metrics.memory_usage > 90:
            predictions.append(FailurePrediction(
                failure_type=FailureType.MEMORY_EXHAUSTION,
                probability=min((current_metrics.memory_usage - 90) / 10, 0.95),
                severity=Severity.CRITICAL if current_metrics.memory_usage > 98 else Severity.HIGH,
                predicted_time=datetime.now() + timedelta(minutes=20),
                confidence_interval=(
                    datetime.now() + timedelta(minutes=10),
                    datetime.now() + timedelta(minutes=45)
                ),
                contributing_factors=["Memory leak", "Increased data processing"],
                recommended_actions=[MaintenanceAction.RESTART_SERVICE, MaintenanceAction.CLEAR_CACHE],
                impact_assessment="Critical system failure, service unavailability",
                prevention_window=timedelta(minutes=10)
            ))
        
        # Disk space critical prediction
        if current_metrics.disk_usage > 85:
            predictions.append(FailurePrediction(
                failure_type=FailureType.DISK_SPACE_CRITICAL,
                probability=min((current_metrics.disk_usage - 85) / 15, 0.95),
                severity=Severity.HIGH if current_metrics.disk_usage > 95 else Severity.MEDIUM,
                predicted_time=datetime.now() + timedelta(hours=2),
                confidence_interval=(
                    datetime.now() + timedelta(hours=1),
                    datetime.now() + timedelta(hours=6)
                ),
                contributing_factors=["Log file growth", "Data accumulation"],
                recommended_actions=[MaintenanceAction.CLEAN_DATA, MaintenanceAction.SCALE_RESOURCES],
                impact_assessment="Data loss risk, service degradation",
                prevention_window=timedelta(hours=1)
            ))
        
        return predictions
    
    def _identify_contributing_factors(self, failure_type: FailureType, 
                                     metrics: SystemMetrics) -> List[str]:
        """Identify contributing factors for a specific failure type"""
        factors = []
        
        if failure_type == FailureType.CPU_OVERLOAD:
            if metrics.request_rate > 50:
                factors.append("High request rate")
            if metrics.active_users > 30:
                factors.append("High user load")
        elif failure_type == FailureType.MEMORY_EXHAUSTION:
            if metrics.database_metrics.get("connection_count", 0) > 80:
                factors.append("High database connections")
            factors.append("Memory leak in application")
        elif failure_type == FailureType.DISK_SPACE_CRITICAL:
            factors.append("Log file accumulation")
            factors.append("Data growth")
        
        return factors
    
    def _get_recommended_actions(self, failure_type: FailureType) -> List[MaintenanceAction]:
        """Get recommended actions for a failure type"""
        action_map = {
            FailureType.CPU_OVERLOAD: [MaintenanceAction.SCALE_RESOURCES, MaintenanceAction.RESTART_SERVICE],
            FailureType.MEMORY_EXHAUSTION: [MaintenanceAction.RESTART_SERVICE, MaintenanceAction.CLEAR_CACHE],
            FailureType.DISK_SPACE_CRITICAL: [MaintenanceAction.CLEAN_DATA, MaintenanceAction.SCALE_RESOURCES],
            FailureType.DATABASE_PERFORMANCE: [MaintenanceAction.OPTIMIZE_DATABASE, MaintenanceAction.RESTART_SERVICE],
            FailureType.API_RESPONSE_DEGRADATION: [MaintenanceAction.RESTART_SERVICE, MaintenanceAction.SCALE_RESOURCES],
            FailureType.MODEL_ACCURACY_DECLINE: [MaintenanceAction.RETRAIN_MODEL, MaintenanceAction.CLEAN_DATA]
        }
        
        return action_map.get(failure_type, [MaintenanceAction.ALERT_ADMIN])
    
    def _assess_impact(self, failure_type: FailureType) -> str:
        """Assess the impact of a potential failure"""
        impact_map = {
            FailureType.CPU_OVERLOAD: "Performance degradation, slower response times",
            FailureType.MEMORY_EXHAUSTION: "Critical system failure, service unavailability",
            FailureType.DISK_SPACE_CRITICAL: "Data loss risk, service interruption",
            FailureType.DATABASE_PERFORMANCE: "Data access delays, transaction failures",
            FailureType.API_RESPONSE_DEGRADATION: "User experience degradation, timeout errors",
            FailureType.MODEL_ACCURACY_DECLINE: "Forecast quality reduction, business impact"
        }
        
        return impact_map.get(failure_type, "Unknown impact")

class CapacityPlanner:
    """Plan system capacity based on usage trends"""
    
    def __init__(self):
        self.growth_models = {}
    
    def analyze_capacity_needs(self, metrics_history: List[SystemMetrics]) -> List[CapacityRecommendations]:
        """Analyze capacity needs based on historical trends"""
        if len(metrics_history) < 24:  # Need at least 24 hours of data
            return []
        
        recommendations = []
        
        # Analyze CPU capacity
        cpu_values = [m.cpu_usage for m in metrics_history]
        cpu_recommendation = self._analyze_resource_capacity(
            "CPU", cpu_values, threshold=80, critical_threshold=95
        )
        if cpu_recommendation:
            recommendations.append(cpu_recommendation)
        
        # Analyze Memory capacity
        memory_values = [m.memory_usage for m in metrics_history]
        memory_recommendation = self._analyze_resource_capacity(
            "Memory", memory_values, threshold=85, critical_threshold=98
        )
        if memory_recommendation:
            recommendations.append(memory_recommendation)
        
        # Analyze Disk capacity
        disk_values = [m.disk_usage for m in metrics_history]
        disk_recommendation = self._analyze_resource_capacity(
            "Disk", disk_values, threshold=80, critical_threshold=95
        )
        if disk_recommendation:
            recommendations.append(disk_recommendation)
        
        return recommendations
    
    def _analyze_resource_capacity(self, resource_type: str, values: List[float],
                                 threshold: float, critical_threshold: float) -> Optional[CapacityRecommendations]:
        """Analyze capacity for a specific resource type"""
        if not values:
            return None
        
        current_usage = values[-1]
        max_usage = max(values)
        avg_usage = np.mean(values)
        
        # Calculate trend
        if len(values) >= 10:
            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
        else:
            trend_slope = 0
        
        # Determine if scaling is needed
        if current_usage > threshold or max_usage > critical_threshold or trend_slope > 1:
            # Calculate recommended capacity
            if trend_slope > 0:
                # Project future usage
                hours_ahead = 168  # 1 week
                projected_usage = current_usage + (trend_slope * hours_ahead)
                recommended_capacity = max(projected_usage * 1.2, current_usage * 1.5)
            else:
                recommended_capacity = max_usage * 1.3
            
            return CapacityRecommendations(
                resource_type=resource_type,
                current_capacity=100.0,  # Assuming 100% is full capacity
                recommended_capacity=recommended_capacity,
                scaling_timeline="Within 1-2 weeks" if trend_slope > 2 else "Within 1 month",
                cost_impact="Medium" if recommended_capacity < current_usage * 2 else "High",
                risk_assessment="High" if current_usage > critical_threshold else "Medium",
                implementation_steps=[
                    f"Monitor {resource_type.lower()} usage trends",
                    f"Plan {resource_type.lower()} scaling strategy",
                    f"Implement {resource_type.lower()} scaling",
                    "Validate performance improvements"
                ]
            )
        
        return None

class SystemHealthAnalyzer:
    """Analyze overall system health and performance"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.trend_analyzer = TrendAnalyzer()
        self.capacity_planner = CapacityPlanner()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system health metrics"""
        return self.metrics_collector.collect_system_metrics()
    
    def analyze_resource_utilization(self, metrics: List[SystemMetrics]) -> List[ResourceAnalysis]:
        """Analyze resource utilization patterns and predict needs"""
        if not metrics:
            return []
        
        analyses = []
        
        # Analyze CPU utilization
        cpu_values = [m.cpu_usage for m in metrics]
        cpu_analysis = ResourceAnalysis(
            resource_type="CPU",
            current_utilization=cpu_values[-1],
            peak_utilization=max(cpu_values),
            average_utilization=np.mean(cpu_values),
            utilization_trend=self._determine_trend(cpu_values),
            capacity_remaining=100 - cpu_values[-1],
            projected_exhaustion=self._project_exhaustion(cpu_values, 95),
            scaling_recommendations=self._get_scaling_recommendations("CPU", cpu_values)
        )
        analyses.append(cpu_analysis)
        
        # Analyze Memory utilization
        memory_values = [m.memory_usage for m in metrics]
        memory_analysis = ResourceAnalysis(
            resource_type="Memory",
            current_utilization=memory_values[-1],
            peak_utilization=max(memory_values),
            average_utilization=np.mean(memory_values),
            utilization_trend=self._determine_trend(memory_values),
            capacity_remaining=100 - memory_values[-1],
            projected_exhaustion=self._project_exhaustion(memory_values, 98),
            scaling_recommendations=self._get_scaling_recommendations("Memory", memory_values)
        )
        analyses.append(memory_analysis)
        
        return analyses
    
    def generate_capacity_recommendations(self, usage_trends: Dict[str, List[float]]) -> List[CapacityRecommendations]:
        """Generate capacity planning recommendations"""
        recommendations = []
        
        for resource_type, values in usage_trends.items():
            if not values:
                continue
            
            current_usage = values[-1]
            trend = self._determine_trend(values)
            
            if current_usage > 80 or trend == "increasing":
                recommendation = CapacityRecommendations(
                    resource_type=resource_type,
                    current_capacity=100.0,
                    recommended_capacity=current_usage * 1.5,
                    scaling_timeline="1-2 weeks" if current_usage > 90 else "1 month",
                    cost_impact="Medium",
                    risk_assessment="High" if current_usage > 90 else "Medium",
                    implementation_steps=[
                        f"Monitor {resource_type} usage",
                        f"Plan {resource_type} scaling",
                        "Implement scaling solution",
                        "Validate improvements"
                    ]
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _determine_trend(self, values: List[float]) -> str:
        """Determine trend direction for a series of values"""
        if len(values) < 3:
            return "stable"
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope > 1:
            return "increasing"
        elif slope < -1:
            return "decreasing"
        else:
            return "stable"
    
    def _project_exhaustion(self, values: List[float], threshold: float) -> Optional[datetime]:
        """Project when a resource might reach exhaustion"""
        if len(values) < 5:
            return None
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope <= 0:
            return None  # Not increasing
        
        current_value = values[-1]
        if current_value >= threshold:
            return datetime.now()  # Already at threshold
        
        # Calculate hours until threshold
        hours_to_threshold = (threshold - current_value) / slope
        
        if hours_to_threshold > 0 and hours_to_threshold < 8760:  # Within a year
            return datetime.now() + timedelta(hours=hours_to_threshold)
        
        return None
    
    def _get_scaling_recommendations(self, resource_type: str, values: List[float]) -> List[str]:
        """Get scaling recommendations for a resource"""
        current_usage = values[-1]
        trend = self._determine_trend(values)
        
        recommendations = []
        
        if current_usage > 90:
            recommendations.append(f"Immediate {resource_type.lower()} scaling required")
        elif current_usage > 80:
            recommendations.append(f"Plan {resource_type.lower()} scaling within 1-2 weeks")
        
        if trend == "increasing":
            recommendations.append(f"Monitor {resource_type.lower()} growth trend closely")
            recommendations.append(f"Consider auto-scaling for {resource_type.lower()}")
        
        return recommendations

class MaintenanceScheduler:
    """Schedule and manage maintenance activities"""
    
    def __init__(self):
        self.scheduled_maintenance: List[MaintenanceSchedule] = []
        self.maintenance_history: List[Dict[str, Any]] = []
    
    def schedule_preventive_maintenance(self, predictions: List[FailurePrediction]) -> List[MaintenanceSchedule]:
        """Schedule optimal maintenance windows based on predictions"""
        scheduled_items = []
        
        for prediction in predictions:
            # Determine optimal maintenance time
            maintenance_time = self._calculate_optimal_time(prediction)
            
            # Create maintenance schedule
            schedule = MaintenanceSchedule(
                maintenance_id=f"maint_{prediction.failure_type.value}_{int(time.time())}",
                scheduled_time=maintenance_time,
                duration_estimate=self._estimate_duration(prediction.recommended_actions),
                actions=prediction.recommended_actions,
                priority=prediction.severity,
                affected_services=self._identify_affected_services(prediction.failure_type),
                rollback_plan=self._create_rollback_plan(prediction.recommended_actions),
                success_criteria=self._define_success_criteria(prediction.failure_type),
                notification_list=self._get_notification_list(prediction.severity)
            )
            
            scheduled_items.append(schedule)
            self.scheduled_maintenance.append(schedule)
        
        return scheduled_items
    
    def _calculate_optimal_time(self, prediction: FailurePrediction) -> datetime:
        """Calculate optimal maintenance time"""
        # Schedule maintenance before predicted failure time
        optimal_time = prediction.predicted_time - prediction.prevention_window
        
        # Ensure it's not in the past
        if optimal_time <= datetime.now():
            optimal_time = datetime.now() + timedelta(minutes=30)
        
        # Try to schedule during low-usage hours (2 AM - 6 AM)
        if optimal_time.hour not in range(2, 6):
            # Move to next 2 AM
            next_2am = optimal_time.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_2am <= optimal_time:
                next_2am += timedelta(days=1)
            
            # Only use 2 AM if it's within the prevention window
            if next_2am < prediction.predicted_time:
                optimal_time = next_2am
        
        return optimal_time
    
    def _estimate_duration(self, actions: List[MaintenanceAction]) -> timedelta:
        """Estimate maintenance duration based on actions"""
        duration_map = {
            MaintenanceAction.RESTART_SERVICE: timedelta(minutes=10),
            MaintenanceAction.SCALE_RESOURCES: timedelta(minutes=30),
            MaintenanceAction.CLEAR_CACHE: timedelta(minutes=5),
            MaintenanceAction.OPTIMIZE_DATABASE: timedelta(hours=1),
            MaintenanceAction.RETRAIN_MODEL: timedelta(hours=2),
            MaintenanceAction.CLEAN_DATA: timedelta(minutes=20),
            MaintenanceAction.UPDATE_CONFIGURATION: timedelta(minutes=15),
            MaintenanceAction.ALERT_ADMIN: timedelta(minutes=1)
        }
        
        total_duration = timedelta()
        for action in actions:
            total_duration += duration_map.get(action, timedelta(minutes=30))
        
        return total_duration
    
    def _identify_affected_services(self, failure_type: FailureType) -> List[str]:
        """Identify services affected by maintenance"""
        service_map = {
            FailureType.CPU_OVERLOAD: ["API", "Dashboard", "Analytics"],
            FailureType.MEMORY_EXHAUSTION: ["API", "Dashboard", "ML Models"],
            FailureType.DISK_SPACE_CRITICAL: ["Database", "Logging", "Data Processing"],
            FailureType.DATABASE_PERFORMANCE: ["API", "Analytics", "Reporting"],
            FailureType.API_RESPONSE_DEGRADATION: ["API", "Dashboard"],
            FailureType.MODEL_ACCURACY_DECLINE: ["ML Models", "Forecasting"]
        }
        
        return service_map.get(failure_type, ["All Services"])
    
    def _create_rollback_plan(self, actions: List[MaintenanceAction]) -> str:
        """Create rollback plan for maintenance actions"""
        rollback_steps = []
        
        for action in actions:
            if action == MaintenanceAction.RESTART_SERVICE:
                rollback_steps.append("Restart service if issues occur")
            elif action == MaintenanceAction.SCALE_RESOURCES:
                rollback_steps.append("Scale back to original resource levels")
            elif action == MaintenanceAction.UPDATE_CONFIGURATION:
                rollback_steps.append("Restore previous configuration")
        
        return "; ".join(rollback_steps) if rollback_steps else "Monitor system and restart services if needed"
    
    def _define_success_criteria(self, failure_type: FailureType) -> List[str]:
        """Define success criteria for maintenance"""
        criteria_map = {
            FailureType.CPU_OVERLOAD: ["CPU usage below 80%", "Response times improved"],
            FailureType.MEMORY_EXHAUSTION: ["Memory usage below 85%", "No memory leaks detected"],
            FailureType.DISK_SPACE_CRITICAL: ["Disk usage below 80%", "Sufficient free space available"],
            FailureType.DATABASE_PERFORMANCE: ["Query response times improved", "No connection issues"],
            FailureType.API_RESPONSE_DEGRADATION: ["API response times below 1s", "Error rate below 1%"],
            FailureType.MODEL_ACCURACY_DECLINE: ["Model accuracy above 90%", "Predictions within expected range"]
        }
        
        return criteria_map.get(failure_type, ["System stability restored", "No error alerts"])
    
    def _get_notification_list(self, severity: Severity) -> List[str]:
        """Get notification list based on severity"""
        if severity == Severity.CRITICAL:
            return ["admin@company.com", "oncall@company.com", "cto@company.com"]
        elif severity == Severity.HIGH:
            return ["admin@company.com", "oncall@company.com"]
        else:
            return ["admin@company.com"]

class PredictiveMaintenanceEngine:
    """Main predictive maintenance engine"""
    
    def __init__(self):
        self.health_monitor = SystemHealthAnalyzer()
        self.failure_predictor = FailurePredictionModel()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.anomaly_detector = TrendAnalyzer()
        
        # Configuration
        self.monitoring_interval = 300  # 5 minutes
        self.prediction_horizon = 24  # hours
        self.is_running = False
        
        # Storage
        self.metrics_history: List[SystemMetrics] = []
        self.predictions_history: List[FailurePrediction] = []
    
    async def start_monitoring(self):
        """Start continuous system monitoring"""
        self.is_running = True
        logger.info("Starting predictive maintenance monitoring")
        
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = self.health_monitor.collect_system_metrics()
                self.metrics_history.append(current_metrics)
                
                # Keep only recent history
                cutoff_time = datetime.now() - timedelta(hours=72)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp >= cutoff_time
                ]
                
                # Generate predictions
                predictions = await self.predict_system_failures(current_metrics)
                
                # Schedule maintenance if needed
                if predictions:
                    scheduled_maintenance = self.schedule_preventive_maintenance(predictions)
                    if scheduled_maintenance:
                        logger.info(f"Scheduled {len(scheduled_maintenance)} maintenance activities")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("Stopping predictive maintenance monitoring")
    
    async def predict_system_failures(self, current_metrics: SystemMetrics) -> List[FailurePrediction]:
        """Predict potential system failures before they occur"""
        try:
            # Get recent metrics for context
            recent_metrics = self.metrics_history[-100:] if len(self.metrics_history) > 100 else self.metrics_history
            
            # Generate predictions
            predictions = self.failure_predictor.predict_failures(current_metrics, recent_metrics)
            
            # Store predictions
            self.predictions_history.extend(predictions)
            
            # Keep only recent predictions
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.predictions_history = [
                p for p in self.predictions_history if p.predicted_time >= cutoff_time
            ]
            
            # Log high-priority predictions
            for prediction in predictions:
                if prediction.severity in [Severity.HIGH, Severity.CRITICAL]:
                    logger.warning(
                        f"High-priority failure prediction: {prediction.failure_type.value} "
                        f"(probability: {prediction.probability:.2f}, "
                        f"predicted time: {prediction.predicted_time})"
                    )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting system failures: {e}")
            return []
    
    def analyze_performance_trends(self, system_metrics: List[SystemMetrics]) -> List[PerformanceAnalysis]:
        """Analyze system performance trends and predict degradation"""
        if len(system_metrics) < 10:
            return []
        
        analyses = []
        
        # Analyze CPU performance
        cpu_values = [m.cpu_usage for m in system_metrics]
        cpu_analysis = self._analyze_metric_performance("CPU Usage", cpu_values)
        analyses.append(cpu_analysis)
        
        # Analyze Memory performance
        memory_values = [m.memory_usage for m in system_metrics]
        memory_analysis = self._analyze_metric_performance("Memory Usage", memory_values)
        analyses.append(memory_analysis)
        
        # Analyze API response times
        api_times = [m.api_response_times.get("forecast_endpoint", 0) for m in system_metrics]
        api_analysis = self._analyze_metric_performance("API Response Time", api_times)
        analyses.append(api_analysis)
        
        return analyses
    
    def schedule_preventive_maintenance(self, predictions: List[FailurePrediction]) -> List[MaintenanceSchedule]:
        """Schedule optimal maintenance windows based on predictions"""
        return self.maintenance_scheduler.schedule_preventive_maintenance(predictions)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}
        
        current_metrics = self.metrics_history[-1]
        recent_predictions = [
            p for p in self.predictions_history 
            if p.predicted_time >= datetime.now()
        ]
        
        # Calculate health score
        health_score = self._calculate_health_score(current_metrics)
        
        return {
            "timestamp": current_metrics.timestamp,
            "health_score": health_score,
            "status": self._determine_system_status(health_score),
            "current_metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "disk_usage": current_metrics.disk_usage,
                "error_rate": current_metrics.error_rate
            },
            "active_predictions": len(recent_predictions),
            "critical_predictions": len([p for p in recent_predictions if p.severity == Severity.CRITICAL]),
            "scheduled_maintenance": len(self.maintenance_scheduler.scheduled_maintenance),
            "recommendations": self._generate_health_recommendations(current_metrics, recent_predictions)
        }
    
    def _analyze_metric_performance(self, metric_name: str, values: List[float]) -> PerformanceAnalysis:
        """Analyze performance for a specific metric"""
        if not values:
            return PerformanceAnalysis(
                metric_name=metric_name,
                current_value=0.0,
                trend_direction="unknown",
                trend_strength=0.0,
                anomaly_score=0.0,
                historical_percentile=0.0,
                predicted_values=[],
                recommendations=[]
            )
        
        current_value = values[-1]
        trend_analysis = self.anomaly_detector.analyze_metric_trend(values, [])
        
        # Calculate anomaly score
        if len(values) > 10:
            mean_val = np.mean(values[:-5])  # Exclude recent values
            std_val = np.std(values[:-5])
            anomaly_score = abs(current_value - mean_val) / (std_val + 1e-6)
        else:
            anomaly_score = 0.0
        
        # Calculate historical percentile
        historical_percentile = (np.sum(np.array(values) <= current_value) / len(values)) * 100
        
        # Generate simple predictions
        predicted_values = []
        if len(values) >= 5:
            slope = np.polyfit(range(len(values)), values, 1)[0]
            for i in range(1, 6):  # Predict next 5 points
                predicted_values.append(current_value + slope * i)
        
        # Generate recommendations
        recommendations = []
        if anomaly_score > 2:
            recommendations.append(f"Investigate {metric_name.lower()} anomaly")
        if trend_analysis.get("trend") == "increasing" and current_value > 80:
            recommendations.append(f"Monitor {metric_name.lower()} growth trend")
        
        return PerformanceAnalysis(
            metric_name=metric_name,
            current_value=current_value,
            trend_direction=trend_analysis.get("trend", "unknown"),
            trend_strength=trend_analysis.get("strength", 0.0),
            anomaly_score=anomaly_score,
            historical_percentile=historical_percentile,
            predicted_values=predicted_values,
            recommendations=recommendations
        )
    
    def _calculate_health_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall system health score (0-100)"""
        scores = []
        
        # CPU health (lower is better)
        cpu_score = max(0, 100 - metrics.cpu_usage)
        scores.append(cpu_score)
        
        # Memory health (lower is better)
        memory_score = max(0, 100 - metrics.memory_usage)
        scores.append(memory_score)
        
        # Disk health (lower is better)
        disk_score = max(0, 100 - metrics.disk_usage)
        scores.append(disk_score)
        
        # Error rate health (lower is better)
        error_score = max(0, 100 - (metrics.error_rate * 2000))  # Scale error rate
        scores.append(error_score)
        
        # API performance health
        avg_response_time = np.mean(list(metrics.api_response_times.values()))
        api_score = max(0, 100 - (avg_response_time * 50))  # Scale response time
        scores.append(api_score)
        
        return np.mean(scores)
    
    def _determine_system_status(self, health_score: float) -> str:
        """Determine system status based on health score"""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 80:
            return "good"
        elif health_score >= 70:
            return "fair"
        elif health_score >= 60:
            return "poor"
        else:
            return "critical"
    
    def _generate_health_recommendations(self, current_metrics: SystemMetrics, 
                                       predictions: List[FailurePrediction]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        if current_metrics.cpu_usage > 80:
            recommendations.append("Consider CPU scaling or optimization")
        
        if current_metrics.memory_usage > 85:
            recommendations.append("Monitor memory usage and consider scaling")
        
        if current_metrics.disk_usage > 80:
            recommendations.append("Clean up disk space or expand storage")
        
        if current_metrics.error_rate > 0.02:
            recommendations.append("Investigate and resolve error sources")
        
        if predictions:
            recommendations.append(f"Address {len(predictions)} predicted failure(s)")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    async def test_predictive_maintenance():
        engine = PredictiveMaintenanceEngine()
        
        # Collect some metrics
        for _ in range(10):
            metrics = engine.health_monitor.collect_system_metrics()
            engine.metrics_history.append(metrics)
            await asyncio.sleep(1)
        
        # Generate predictions
        current_metrics = engine.metrics_history[-1]
        predictions = await engine.predict_system_failures(current_metrics)
        
        print(f"Generated {len(predictions)} predictions")
        for prediction in predictions:
            print(f"- {prediction.failure_type.value}: {prediction.probability:.2f} probability")
        
        # Get health summary
        health_summary = engine.get_system_health_summary()
        print(f"System health score: {health_summary['health_score']:.1f}")
        print(f"System status: {health_summary['status']}")
    
    # Run test
    asyncio.run(test_predictive_maintenance())