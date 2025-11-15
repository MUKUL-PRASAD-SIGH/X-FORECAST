"""
Automated Governance Workflow Engine
Manages configurable demand planning cycles with exception detection and approval routing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import uuid
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExceptionType(Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    FORECAST_ACCURACY_DECLINE = "forecast_accuracy_decline"
    INVENTORY_IMBALANCE = "inventory_imbalance"
    DEMAND_VOLATILITY = "demand_volatility"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    CROSS_CATEGORY_IMPACT = "cross_category_impact"
    SUPPLY_CONSTRAINT = "supply_constraint"

class ExceptionSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationType(Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DASHBOARD = "dashboard"
    WEBHOOK = "webhook"

@dataclass
class WorkflowTemplate:
    """Template for configurable workflow definitions"""
    template_id: str
    name: str
    description: str
    
    # Cycle configuration
    cycle_frequency: str  # daily, weekly, monthly
    cycle_duration_hours: int
    
    # Stages and transitions
    stages: List[Dict[str, Any]]
    transitions: Dict[str, List[str]]
    
    # Exception handling
    exception_rules: List[Dict[str, Any]]
    escalation_rules: List[Dict[str, Any]]
    
    # Stakeholder configuration
    stakeholder_roles: Dict[str, List[str]]
    approval_matrix: Dict[str, Dict[str, Any]]
    
    # Notification settings
    notification_templates: Dict[str, str]
    reminder_schedule: List[int]  # hours
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowInstance:
    """Active workflow instance"""
    instance_id: str
    template_id: str
    name: str
    
    # State management
    current_state: WorkflowState
    current_stage: str
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data context
    context_data: Dict[str, Any] = field(default_factory=dict)
    forecast_data: Optional[pd.DataFrame] = None
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    due_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Stakeholders
    assigned_users: List[str] = field(default_factory=list)
    approvers: List[str] = field(default_factory=list)
    
    # Exception tracking
    detected_exceptions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Audit trail
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ForecastException:
    """Detected forecast exception requiring attention"""
    exception_id: str
    exception_type: ExceptionType
    severity: ExceptionSeverity
    
    # Location information
    sku: Optional[str] = None
    location: Optional[str] = None
    channel: Optional[str] = None
    
    # Exception details
    description: str = ""
    detected_value: float = 0.0
    threshold_value: float = 0.0
    deviation_magnitude: float = 0.0
    
    # Context
    detection_method: str = ""
    confidence_score: float = 0.0
    business_impact: str = ""
    
    # Routing
    assigned_to: Optional[str] = None
    escalation_level: int = 0
    
    # Timestamps
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
class
 WorkflowEngine:
    """
    Automated governance workflow engine for demand planning cycles
    """
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.active_instances: Dict[str, WorkflowInstance] = {}
        self.exception_detector = ExceptionDetector()
        self.approval_router = ApprovalRouter()
        self.notification_service = NotificationService()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_template(self, template_config: Dict[str, Any]) -> WorkflowTemplate:
        """Create a new workflow template"""
        template = WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            name=template_config['name'],
            description=template_config.get('description', ''),
            cycle_frequency=template_config['cycle_frequency'],
            cycle_duration_hours=template_config['cycle_duration_hours'],
            stages=template_config['stages'],
            transitions=template_config['transitions'],
            exception_rules=template_config.get('exception_rules', []),
            escalation_rules=template_config.get('escalation_rules', []),
            stakeholder_roles=template_config.get('stakeholder_roles', {}),
            approval_matrix=template_config.get('approval_matrix', {}),
            notification_templates=template_config.get('notification_templates', {}),
            reminder_schedule=template_config.get('reminder_schedule', [24, 48, 72])
        )
        
        self.templates[template.template_id] = template
        logger.info(f"Created workflow template: {template.name}")
        return template
    
    def start_workflow(self, template_id: str, context_data: Dict[str, Any] = None) -> WorkflowInstance:
        """Start a new workflow instance"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        instance = WorkflowInstance(
            instance_id=str(uuid.uuid4()),
            template_id=template_id,
            name=f"{template.name} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            current_state=WorkflowState.CREATED,
            current_stage=template.stages[0]['name'] if template.stages else 'start',
            context_data=context_data or {},
            due_at=datetime.utcnow() + timedelta(hours=template.cycle_duration_hours)
        )
        
        # Add initial state to history
        instance.state_history.append({
            'state': WorkflowState.CREATED.value,
            'stage': instance.current_stage,
            'timestamp': datetime.utcnow().isoformat(),
            'user': 'system',
            'notes': 'Workflow instance created'
        })
        
        self.active_instances[instance.instance_id] = instance
        
        # Start execution
        self.executor.submit(self._execute_workflow, instance.instance_id)
        
        logger.info(f"Started workflow instance: {instance.instance_id}")
        return instance
    
    def _execute_workflow(self, instance_id: str):
        """Execute workflow instance"""
        try:
            instance = self.active_instances[instance_id]
            template = self.templates[instance.template_id]
            
            # Update state to running
            self._update_instance_state(instance_id, WorkflowState.RUNNING)
            
            # Execute stages
            for stage in template.stages:
                if instance.current_state in [WorkflowState.FAILED, WorkflowState.CANCELLED]:
                    break
                
                self._execute_stage(instance_id, stage)
                
                # Check for exceptions
                exceptions = self._detect_exceptions(instance_id, stage)
                if exceptions:
                    instance.detected_exceptions.extend(exceptions)
                    
                    # Route for approval if needed
                    if self._requires_approval(exceptions):
                        self._route_for_approval(instance_id, exceptions)
                        break
            
            # Complete workflow if no exceptions or all resolved
            if instance.current_state == WorkflowState.RUNNING:
                self._update_instance_state(instance_id, WorkflowState.COMPLETED)
                
        except Exception as e:
            logger.error(f"Workflow execution failed for {instance_id}: {e}")
            self._update_instance_state(instance_id, WorkflowState.FAILED)
    
    def _execute_stage(self, instance_id: str, stage: Dict[str, Any]):
        """Execute a workflow stage"""
        instance = self.active_instances[instance_id]
        
        # Update current stage
        instance.current_stage = stage['name']
        
        # Log stage execution
        instance.audit_log.append({
            'action': 'stage_started',
            'stage': stage['name'],
            'timestamp': datetime.utcnow().isoformat(),
            'details': stage.get('description', '')
        })
        
        # Execute stage actions
        if 'actions' in stage:
            for action in stage['actions']:
                self._execute_action(instance_id, action)
        
        # Wait for stage completion
        if stage.get('wait_for_completion', False):
            # Implementation would depend on specific stage requirements
            pass
    
    def _execute_action(self, instance_id: str, action: Dict[str, Any]):
        """Execute a workflow action"""
        action_type = action.get('type')
        
        if action_type == 'forecast_generation':
            self._execute_forecast_generation(instance_id, action)
        elif action_type == 'data_validation':
            self._execute_data_validation(instance_id, action)
        elif action_type == 'exception_detection':
            self._execute_exception_detection(instance_id, action)
        elif action_type == 'notification':
            self._execute_notification(instance_id, action)
        else:
            logger.warning(f"Unknown action type: {action_type}")
    
    def _detect_exceptions(self, instance_id: str, stage: Dict[str, Any]) -> List[ForecastException]:
        """Detect exceptions in current stage"""
        instance = self.active_instances[instance_id]
        template = self.templates[instance.template_id]
        
        exceptions = []
        
        # Apply exception rules
        for rule in template.exception_rules:
            if rule.get('stage') == stage['name'] or rule.get('stage') == 'all':
                detected = self.exception_detector.detect_exceptions(
                    instance.forecast_data,
                    rule
                )
                exceptions.extend(detected)
        
        return exceptions
    
    def _requires_approval(self, exceptions: List[ForecastException]) -> bool:
        """Check if exceptions require approval"""
        return any(
            exc.severity in [ExceptionSeverity.HIGH, ExceptionSeverity.CRITICAL]
            for exc in exceptions
        )
    
    def _route_for_approval(self, instance_id: str, exceptions: List[ForecastException]):
        """Route workflow for approval"""
        instance = self.active_instances[instance_id]
        template = self.templates[instance.template_id]
        
        # Update state
        self._update_instance_state(instance_id, WorkflowState.WAITING_APPROVAL)
        
        # Determine approvers
        approvers = self.approval_router.get_approvers(exceptions, template.approval_matrix)
        instance.approvers = approvers
        
        # Send notifications
        self.notification_service.send_approval_request(instance, exceptions)
        
        # Schedule reminders
        self._schedule_reminders(instance_id)
    
    def _update_instance_state(self, instance_id: str, new_state: WorkflowState, user: str = 'system', notes: str = ''):
        """Update workflow instance state"""
        instance = self.active_instances[instance_id]
        old_state = instance.current_state
        
        instance.current_state = new_state
        instance.state_history.append({
            'from_state': old_state.value,
            'to_state': new_state.value,
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'notes': notes
        })
        
        if new_state in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED]:
            instance.completed_at = datetime.utcnow()
        
        logger.info(f"Workflow {instance_id} state changed: {old_state.value} -> {new_state.value}")
    
    def approve_workflow(self, instance_id: str, approver: str, decision: str, notes: str = ''):
        """Approve or reject workflow"""
        if instance_id not in self.active_instances:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self.active_instances[instance_id]
        
        if instance.current_state != WorkflowState.WAITING_APPROVAL:
            raise ValueError(f"Instance {instance_id} is not waiting for approval")
        
        if approver not in instance.approvers:
            raise ValueError(f"User {approver} is not authorized to approve this workflow")
        
        # Record decision
        instance.audit_log.append({
            'action': 'approval_decision',
            'user': approver,
            'decision': decision,
            'notes': notes,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if decision.lower() == 'approve':
            self._update_instance_state(instance_id, WorkflowState.APPROVED, approver, notes)
            # Continue execution
            self.executor.submit(self._continue_workflow, instance_id)
        else:
            self._update_instance_state(instance_id, WorkflowState.REJECTED, approver, notes)
    
    def _continue_workflow(self, instance_id: str):
        """Continue workflow execution after approval"""
        # Resume from current stage
        self._execute_workflow(instance_id)
    
    def _schedule_reminders(self, instance_id: str):
        """Schedule reminder notifications"""
        instance = self.active_instances[instance_id]
        template = self.templates[instance.template_id]
        
        for hours in template.reminder_schedule:
            # Schedule reminder (implementation would use actual scheduler)
            reminder_time = datetime.utcnow() + timedelta(hours=hours)
            logger.info(f"Reminder scheduled for {instance_id} at {reminder_time}")
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get workflow instance status"""
        if instance_id not in self.active_instances:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self.active_instances[instance_id]
        
        return {
            'instance_id': instance.instance_id,
            'name': instance.name,
            'current_state': instance.current_state.value,
            'current_stage': instance.current_stage,
            'started_at': instance.started_at.isoformat(),
            'due_at': instance.due_at.isoformat() if instance.due_at else None,
            'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
            'assigned_users': instance.assigned_users,
            'approvers': instance.approvers,
            'exceptions_count': len(instance.detected_exceptions),
            'state_history': instance.state_history[-5:]  # Last 5 state changes
        }
    
    def _execute_forecast_generation(self, instance_id: str, action: Dict[str, Any]):
        """Execute forecast generation action"""
        # Implementation would integrate with forecasting models
        logger.info(f"Executing forecast generation for {instance_id}")
    
    def _execute_data_validation(self, instance_id: str, action: Dict[str, Any]):
        """Execute data validation action"""
        # Implementation would validate data quality
        logger.info(f"Executing data validation for {instance_id}")
    
    def _execute_exception_detection(self, instance_id: str, action: Dict[str, Any]):
        """Execute exception detection action"""
        # Implementation would run exception detection
        logger.info(f"Executing exception detection for {instance_id}")
    
    def _execute_notification(self, instance_id: str, action: Dict[str, Any]):
        """Execute notification action"""
        # Implementation would send notifications
        logger.info(f"Executing notification for {instance_id}")


class ExceptionDetector:
    """
    Exception detection engine for identifying forecasts requiring human review
    """
    
    def __init__(self):
        self.statistical_detector = StatisticalExceptionDetector()
        self.ml_detector = MLExceptionDetector()
        self.business_rule_detector = BusinessRuleDetector()
    
    def detect_exceptions(self, forecast_data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect exceptions based on rule configuration"""
        exceptions = []
        
        rule_type = rule.get('type')
        
        if rule_type == 'statistical':
            exceptions.extend(self.statistical_detector.detect(forecast_data, rule))
        elif rule_type == 'ml_anomaly':
            exceptions.extend(self.ml_detector.detect(forecast_data, rule))
        elif rule_type == 'business_rule':
            exceptions.extend(self.business_rule_detector.detect(forecast_data, rule))
        
        return exceptions


class StatisticalExceptionDetector:
    """Statistical threshold-based exception detection"""
    
    def detect(self, data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect statistical outliers"""
        exceptions = []
        
        if data is None or data.empty:
            return exceptions
        
        threshold_type = rule.get('threshold_type', 'zscore')
        threshold_value = rule.get('threshold_value', 3.0)
        
        if threshold_type == 'zscore':
            exceptions.extend(self._detect_zscore_outliers(data, threshold_value))
        elif threshold_type == 'iqr':
            exceptions.extend(self._detect_iqr_outliers(data, threshold_value))
        elif threshold_type == 'percentage_change':
            exceptions.extend(self._detect_percentage_change_outliers(data, threshold_value))
        
        return exceptions
    
    def _detect_zscore_outliers(self, data: pd.DataFrame, threshold: float) -> List[ForecastException]:
        """Detect Z-score based outliers"""
        exceptions = []
        
        if 'forecast' not in data.columns:
            return exceptions
        
        z_scores = np.abs((data['forecast'] - data['forecast'].mean()) / data['forecast'].std())
        outliers = data[z_scores > threshold]
        
        for idx, row in outliers.iterrows():
            exception = ForecastException(
                exception_id=str(uuid.uuid4()),
                exception_type=ExceptionType.STATISTICAL_OUTLIER,
                severity=self._determine_severity(z_scores.loc[idx]),
                sku=row.get('sku'),
                location=row.get('location'),
                channel=row.get('channel'),
                description=f"Z-score outlier detected: {z_scores.loc[idx]:.2f}",
                detected_value=row['forecast'],
                threshold_value=threshold,
                deviation_magnitude=z_scores.loc[idx],
                detection_method='zscore',
                confidence_score=min(z_scores.loc[idx] / 5.0, 1.0)
            )
            exceptions.append(exception)
        
        return exceptions
    
    def _detect_iqr_outliers(self, data: pd.DataFrame, multiplier: float) -> List[ForecastException]:
        """Detect IQR-based outliers"""
        exceptions = []
        
        if 'forecast' not in data.columns:
            return exceptions
        
        Q1 = data['forecast'].quantile(0.25)
        Q3 = data['forecast'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = data[(data['forecast'] < lower_bound) | (data['forecast'] > upper_bound)]
        
        for idx, row in outliers.iterrows():
            deviation = max(abs(row['forecast'] - lower_bound), abs(row['forecast'] - upper_bound))
            
            exception = ForecastException(
                exception_id=str(uuid.uuid4()),
                exception_type=ExceptionType.STATISTICAL_OUTLIER,
                severity=self._determine_severity(deviation / IQR),
                sku=row.get('sku'),
                location=row.get('location'),
                channel=row.get('channel'),
                description=f"IQR outlier detected: {row['forecast']:.2f} outside [{lower_bound:.2f}, {upper_bound:.2f}]",
                detected_value=row['forecast'],
                threshold_value=multiplier,
                deviation_magnitude=deviation,
                detection_method='iqr',
                confidence_score=min(deviation / (2 * IQR), 1.0)
            )
            exceptions.append(exception)
        
        return exceptions
    
    def _detect_percentage_change_outliers(self, data: pd.DataFrame, threshold: float) -> List[ForecastException]:
        """Detect large percentage changes"""
        exceptions = []
        
        if 'forecast' not in data.columns or 'historical' not in data.columns:
            return exceptions
        
        pct_change = ((data['forecast'] - data['historical']) / data['historical'].abs()) * 100
        outliers = data[pct_change.abs() > threshold]
        
        for idx, row in outliers.iterrows():
            change = pct_change.loc[idx]
            
            exception = ForecastException(
                exception_id=str(uuid.uuid4()),
                exception_type=ExceptionType.FORECAST_ACCURACY_DECLINE,
                severity=self._determine_severity(abs(change) / 100),
                sku=row.get('sku'),
                location=row.get('location'),
                channel=row.get('channel'),
                description=f"Large forecast change: {change:.1f}%",
                detected_value=row['forecast'],
                threshold_value=threshold,
                deviation_magnitude=abs(change),
                detection_method='percentage_change',
                confidence_score=min(abs(change) / 200, 1.0)
            )
            exceptions.append(exception)
        
        return exceptions
    
    def _determine_severity(self, magnitude: float) -> ExceptionSeverity:
        """Determine exception severity based on magnitude"""
        if magnitude > 5.0:
            return ExceptionSeverity.CRITICAL
        elif magnitude > 3.0:
            return ExceptionSeverity.HIGH
        elif magnitude > 2.0:
            return ExceptionSeverity.MEDIUM
        else:
            return ExceptionSeverity.LOW


class MLExceptionDetector:
    """Machine learning-based anomaly detection"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def detect(self, data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect ML-based anomalies"""
        exceptions = []
        
        if data is None or data.empty:
            return exceptions
        
        # Prepare features
        features = self._prepare_features(data)
        if features.empty:
            return exceptions
        
        # Fit model if not already fitted
        if not self.is_fitted:
            self._fit_model(features)
        
        # Detect anomalies
        anomaly_scores = self.isolation_forest.decision_function(features)
        anomalies = self.isolation_forest.predict(features) == -1
        
        anomaly_data = data[anomalies]
        anomaly_scores_filtered = anomaly_scores[anomalies]
        
        for idx, (_, row) in enumerate(anomaly_data.iterrows()):
            exception = ForecastException(
                exception_id=str(uuid.uuid4()),
                exception_type=ExceptionType.DEMAND_VOLATILITY,
                severity=self._determine_ml_severity(anomaly_scores_filtered[idx]),
                sku=row.get('sku'),
                location=row.get('location'),
                channel=row.get('channel'),
                description=f"ML anomaly detected with score: {anomaly_scores_filtered[idx]:.3f}",
                detected_value=row.get('forecast', 0.0),
                deviation_magnitude=abs(anomaly_scores_filtered[idx]),
                detection_method='isolation_forest',
                confidence_score=min(abs(anomaly_scores_filtered[idx]) * 2, 1.0)
            )
            exceptions.append(exception)
        
        return exceptions
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML detection"""
        features = []
        
        # Basic forecast features
        if 'forecast' in data.columns:
            features.append('forecast')
        
        # Historical comparison features
        if 'historical' in data.columns:
            features.append('historical')
            if 'forecast' in data.columns:
                data['forecast_change'] = data['forecast'] - data['historical']
                data['forecast_pct_change'] = (data['forecast'] - data['historical']) / data['historical'].abs()
                features.extend(['forecast_change', 'forecast_pct_change'])
        
        # Seasonality features
        if 'date' in data.columns:
            data['month'] = pd.to_datetime(data['date']).dt.month
            data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
            features.extend(['month', 'day_of_week'])
        
        return data[features].fillna(0)
    
    def _fit_model(self, features: pd.DataFrame):
        """Fit the ML model"""
        if not features.empty:
            scaled_features = self.scaler.fit_transform(features)
            self.isolation_forest.fit(scaled_features)
            self.is_fitted = True
    
    def _determine_ml_severity(self, anomaly_score: float) -> ExceptionSeverity:
        """Determine severity based on ML anomaly score"""
        if anomaly_score < -0.5:
            return ExceptionSeverity.CRITICAL
        elif anomaly_score < -0.3:
            return ExceptionSeverity.HIGH
        elif anomaly_score < -0.1:
            return ExceptionSeverity.MEDIUM
        else:
            return ExceptionSeverity.LOW


class BusinessRuleDetector:
    """Business rule-based exception detection"""
    
    def detect(self, data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect business rule violations"""
        exceptions = []
        
        if data is None or data.empty:
            return exceptions
        
        rule_name = rule.get('rule_name')
        
        if rule_name == 'inventory_imbalance':
            exceptions.extend(self._detect_inventory_imbalance(data, rule))
        elif rule_name == 'supply_constraint':
            exceptions.extend(self._detect_supply_constraints(data, rule))
        elif rule_name == 'seasonal_anomaly':
            exceptions.extend(self._detect_seasonal_anomalies(data, rule))
        
        return exceptions
    
    def _detect_inventory_imbalance(self, data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect inventory imbalance issues"""
        exceptions = []
        
        if 'forecast' not in data.columns or 'inventory' not in data.columns:
            return exceptions
        
        # Calculate days of supply
        data['days_of_supply'] = data['inventory'] / (data['forecast'] / 30)  # Assuming monthly forecast
        
        min_days = rule.get('min_days_supply', 7)
        max_days = rule.get('max_days_supply', 90)
        
        imbalanced = data[(data['days_of_supply'] < min_days) | (data['days_of_supply'] > max_days)]
        
        for _, row in imbalanced.iterrows():
            severity = ExceptionSeverity.CRITICAL if row['days_of_supply'] < 3 else ExceptionSeverity.HIGH
            
            exception = ForecastException(
                exception_id=str(uuid.uuid4()),
                exception_type=ExceptionType.INVENTORY_IMBALANCE,
                severity=severity,
                sku=row.get('sku'),
                location=row.get('location'),
                channel=row.get('channel'),
                description=f"Inventory imbalance: {row['days_of_supply']:.1f} days of supply",
                detected_value=row['days_of_supply'],
                threshold_value=min_days if row['days_of_supply'] < min_days else max_days,
                detection_method='business_rule',
                business_impact="Potential stockout or excess inventory"
            )
            exceptions.append(exception)
        
        return exceptions
    
    def _detect_supply_constraints(self, data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect supply constraint violations"""
        exceptions = []
        
        if 'forecast' not in data.columns or 'supply_capacity' not in data.columns:
            return exceptions
        
        capacity_threshold = rule.get('capacity_threshold', 0.9)
        
        # Calculate capacity utilization
        data['capacity_utilization'] = data['forecast'] / data['supply_capacity']
        constrained = data[data['capacity_utilization'] > capacity_threshold]
        
        for _, row in constrained.iterrows():
            exception = ForecastException(
                exception_id=str(uuid.uuid4()),
                exception_type=ExceptionType.SUPPLY_CONSTRAINT,
                severity=ExceptionSeverity.HIGH,
                sku=row.get('sku'),
                location=row.get('location'),
                channel=row.get('channel'),
                description=f"Supply constraint: {row['capacity_utilization']:.1%} capacity utilization",
                detected_value=row['capacity_utilization'],
                threshold_value=capacity_threshold,
                detection_method='business_rule',
                business_impact="Potential supply shortage"
            )
            exceptions.append(exception)
        
        return exceptions
    
    def _detect_seasonal_anomalies(self, data: pd.DataFrame, rule: Dict[str, Any]) -> List[ForecastException]:
        """Detect seasonal pattern anomalies"""
        exceptions = []
        
        if 'forecast' not in data.columns or 'date' not in data.columns:
            return exceptions
        
        # Simple seasonal anomaly detection (would be more sophisticated in practice)
        data['month'] = pd.to_datetime(data['date']).dt.month
        monthly_avg = data.groupby('month')['forecast'].mean()
        
        for _, row in data.iterrows():
            month = pd.to_datetime(row['date']).month
            expected = monthly_avg[month]
            deviation = abs(row['forecast'] - expected) / expected
            
            if deviation > rule.get('seasonal_threshold', 0.5):
                exception = ForecastException(
                    exception_id=str(uuid.uuid4()),
                    exception_type=ExceptionType.SEASONAL_ANOMALY,
                    severity=ExceptionSeverity.MEDIUM,
                    sku=row.get('sku'),
                    location=row.get('location'),
                    channel=row.get('channel'),
                    description=f"Seasonal anomaly: {deviation:.1%} deviation from expected",
                    detected_value=row['forecast'],
                    threshold_value=expected,
                    deviation_magnitude=deviation,
                    detection_method='business_rule'
                )
                exceptions.append(exception)
        
        return exceptions


class ApprovalRouter:
    """
    Approval routing and escalation system
    """
    
    def get_approvers(self, exceptions: List[ForecastException], approval_matrix: Dict[str, Dict[str, Any]]) -> List[str]:
        """Determine approvers based on exceptions and approval matrix"""
        approvers = set()
        
        for exception in exceptions:
            # Get approvers based on exception type and severity
            exception_key = f"{exception.exception_type.value}_{exception.severity.value}"
            
            if exception_key in approval_matrix:
                approvers.update(approval_matrix[exception_key].get('approvers', []))
            
            # Add default approvers for high severity
            if exception.severity in [ExceptionSeverity.HIGH, ExceptionSeverity.CRITICAL]:
                approvers.update(approval_matrix.get('default_high_severity', {}).get('approvers', []))
        
        return list(approvers)
    
    def escalate_approval(self, instance_id: str, current_approvers: List[str], approval_matrix: Dict[str, Dict[str, Any]]) -> List[str]:
        """Escalate approval to higher level"""
        escalated_approvers = set()
        
        for approver in current_approvers:
            # Find escalation path
            if approver in approval_matrix.get('escalation_paths', {}):
                escalated_approvers.update(approval_matrix['escalation_paths'][approver])
        
        # Add default escalation
        escalated_approvers.update(approval_matrix.get('default_escalation', {}).get('approvers', []))
        
        return list(escalated_approvers)


class NotificationService:
    """
    Notification service for workflow alerts and reminders
    """
    
    def __init__(self):
        self.notification_handlers = {
            NotificationType.EMAIL: self._send_email,
            NotificationType.SMS: self._send_sms,
            NotificationType.SLACK: self._send_slack,
            NotificationType.DASHBOARD: self._send_dashboard,
            NotificationType.WEBHOOK: self._send_webhook
        }
    
    def send_approval_request(self, instance: WorkflowInstance, exceptions: List[ForecastException]):
        """Send approval request notifications"""
        message = self._create_approval_message(instance, exceptions)
        
        for approver in instance.approvers:
            # Send to all configured notification types
            for notification_type in [NotificationType.EMAIL, NotificationType.DASHBOARD]:
                self._send_notification(notification_type, approver, message)
    
    def send_reminder(self, instance: WorkflowInstance):
        """Send reminder notifications"""
        message = self._create_reminder_message(instance)
        
        for approver in instance.approvers:
            self._send_notification(NotificationType.EMAIL, approver, message)
    
    def send_escalation(self, instance: WorkflowInstance, escalated_approvers: List[str]):
        """Send escalation notifications"""
        message = self._create_escalation_message(instance)
        
        for approver in escalated_approvers:
            self._send_notification(NotificationType.EMAIL, approver, message)
    
    def _send_notification(self, notification_type: NotificationType, recipient: str, message: str):
        """Send notification using specified type"""
        handler = self.notification_handlers.get(notification_type)
        if handler:
            handler(recipient, message)
        else:
            logger.warning(f"No handler for notification type: {notification_type}")
    
    def _create_approval_message(self, instance: WorkflowInstance, exceptions: List[ForecastException]) -> str:
        """Create approval request message"""
        return f"""
        Workflow Approval Required: {instance.name}
        
        Instance ID: {instance.instance_id}
        Due Date: {instance.due_at}
        
        Exceptions Detected: {len(exceptions)}
        - Critical: {sum(1 for e in exceptions if e.severity == ExceptionSeverity.CRITICAL)}
        - High: {sum(1 for e in exceptions if e.severity == ExceptionSeverity.HIGH)}
        - Medium: {sum(1 for e in exceptions if e.severity == ExceptionSeverity.MEDIUM)}
        
        Please review and approve/reject this workflow.
        """
    
    def _create_reminder_message(self, instance: WorkflowInstance) -> str:
        """Create reminder message"""
        return f"""
        Reminder: Workflow Approval Pending
        
        Workflow: {instance.name}
        Instance ID: {instance.instance_id}
        Due Date: {instance.due_at}
        
        This workflow is still waiting for your approval.
        """
    
    def _create_escalation_message(self, instance: WorkflowInstance) -> str:
        """Create escalation message"""
        return f"""
        Workflow Escalated: {instance.name}
        
        Instance ID: {instance.instance_id}
        Original Due Date: {instance.due_at}
        
        This workflow has been escalated due to delayed approval.
        """
    
    def _send_email(self, recipient: str, message: str):
        """Send email notification"""
        logger.info(f"Sending email to {recipient}: {message[:100]}...")
    
    def _send_sms(self, recipient: str, message: str):
        """Send SMS notification"""
        logger.info(f"Sending SMS to {recipient}: {message[:50]}...")
    
    def _send_slack(self, recipient: str, message: str):
        """Send Slack notification"""
        logger.info(f"Sending Slack message to {recipient}: {message[:100]}...")
    
    def _send_dashboard(self, recipient: str, message: str):
        """Send dashboard notification"""
        logger.info(f"Sending dashboard notification to {recipient}")
    
    def _send_webhook(self, recipient: str, message: str):
        """Send webhook notification"""
        logger.info(f"Sending webhook to {recipient}")