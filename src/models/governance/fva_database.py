"""
FVA Database Schema and Data Access Layer
SQLAlchemy models for FVA tracking system
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class ForecastOverrideDB(Base):
    """Database model for forecast overrides"""
    __tablename__ = 'forecast_overrides'
    
    # Primary key
    override_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Forecast identification
    sku = Column(String(100), nullable=False, index=True)
    location = Column(String(100), nullable=False, index=True)
    channel = Column(String(100), nullable=False, index=True)
    period = Column(String(20), nullable=False, index=True)
    
    # Override details
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    original_forecast = Column(Float, nullable=False)
    adjusted_forecast = Column(Float, nullable=False)
    override_type = Column(String(50), nullable=False, index=True)
    reason = Column(Text)
    confidence = Column(Float, default=0.5)
    business_justification = Column(Text)
    expected_impact = Column(Float)
    
    # Approval workflow
    approval_status = Column(String(20), default='pending', index=True)
    approver_id = Column(String(100), index=True)
    approval_timestamp = Column(DateTime, index=True)
    
    # Performance tracking
    actual_demand = Column(Float)  # Filled when actual data becomes available
    accuracy_improvement = Column(Float)  # Calculated post-facto
    
    # Relationships
    user_metadata = relationship("UserMetadataDB", back_populates="overrides")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_sku_location_period', 'sku', 'location', 'period'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_approval_status', 'approval_status'),
    )

class UserMetadataDB(Base):
    """Database model for user metadata and performance tracking"""
    __tablename__ = 'user_metadata'
    
    user_id = Column(String(100), primary_key=True)
    user_name = Column(String(200))
    department = Column(String(100), index=True)
    role = Column(String(100), index=True)
    experience_level = Column(String(50))  # junior, senior, expert
    
    # Performance metrics
    total_overrides = Column(Integer, default=0)
    positive_fva_rate = Column(Float, default=0.0)
    avg_accuracy_improvement = Column(Float, default=0.0)
    fva_score = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    overrides = relationship("ForecastOverrideDB", back_populates="user_metadata")

class FVAMetricsDB(Base):
    """Database model for aggregated FVA metrics"""
    __tablename__ = 'fva_metrics'
    
    metric_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Aggregation dimensions
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    aggregation_level = Column(String(50), nullable=False)  # user, sku, location, global
    aggregation_key = Column(String(200), index=True)  # specific user_id, sku, etc.
    
    # FVA metrics
    total_overrides = Column(Integer, default=0)
    override_rate = Column(Float, default=0.0)
    avg_override_magnitude = Column(Float, default=0.0)
    fva_accuracy_improvement = Column(Float, default=0.0)
    fva_bias_reduction = Column(Float, default=0.0)
    positive_fva_rate = Column(Float, default=0.0)
    
    # Performance metrics
    time_to_decision = Column(Float, default=0.0)  # hours
    override_persistence = Column(Float, default=0.0)
    
    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_period_aggregation', 'period_start', 'period_end', 'aggregation_level'),
        Index('idx_aggregation_key', 'aggregation_key'),
    )

class FVAAlertDB(Base):
    """Database model for FVA alerts and notifications"""
    __tablename__ = 'fva_alerts'
    
    alert_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Alert details
    alert_type = Column(String(50), nullable=False, index=True)  # negative_fva, low_accuracy, etc.
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Target information
    user_id = Column(String(100), index=True)
    sku = Column(String(100), index=True)
    location = Column(String(100), index=True)
    
    # Alert metrics
    threshold_value = Column(Float)
    actual_value = Column(Float)
    deviation = Column(Float)
    
    # Status tracking
    status = Column(String(20), default='active', index=True)  # active, acknowledged, resolved
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    resolved_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_status_created', 'status', 'created_at'),
    )

class FVADataAccessLayer:
    """Data access layer for FVA tracking system"""
    
    def __init__(self, database_url: str = "sqlite:///fva_tracking.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
    def create_override(self, override_data: dict) -> str:
        """Create a new forecast override record"""
        try:
            override = ForecastOverrideDB(**override_data)
            self.session.add(override)
            self.session.commit()
            
            # Update user metadata
            self._update_user_metadata(override_data['user_id'])
            
            logger.info(f"Created override record {override.override_id}")
            return override.override_id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to create override: {e}")
            raise
    
    def update_override_actual(self, override_id: str, actual_demand: float) -> bool:
        """Update override with actual demand and calculate accuracy improvement"""
        try:
            override = self.session.query(ForecastOverrideDB).filter_by(override_id=override_id).first()
            if not override:
                return False
            
            override.actual_demand = actual_demand
            
            # Calculate accuracy improvement
            original_error = abs(actual_demand - override.original_forecast)
            adjusted_error = abs(actual_demand - override.adjusted_forecast)
            
            if original_error > 0:
                override.accuracy_improvement = (original_error - adjusted_error) / original_error
            else:
                override.accuracy_improvement = 0.0
            
            self.session.commit()
            
            # Update user performance metrics
            self._update_user_performance(override.user_id)
            
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to update override actual: {e}")
            return False
    
    def get_user_overrides(self, user_id: str, days_back: int = 30) -> list:
        """Get overrides for a specific user"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        overrides = self.session.query(ForecastOverrideDB).filter(
            ForecastOverrideDB.user_id == user_id,
            ForecastOverrideDB.timestamp >= cutoff_date
        ).order_by(ForecastOverrideDB.timestamp.desc()).all()
        
        return [self._override_to_dict(override) for override in overrides]
    
    def get_fva_metrics(self, period_start: datetime, period_end: datetime, 
                       aggregation_level: str = 'global', aggregation_key: str = None) -> dict:
        """Get FVA metrics for specified period and aggregation"""
        
        # Check if metrics already calculated
        existing_metrics = self.session.query(FVAMetricsDB).filter(
            FVAMetricsDB.period_start == period_start,
            FVAMetricsDB.period_end == period_end,
            FVAMetricsDB.aggregation_level == aggregation_level,
            FVAMetricsDB.aggregation_key == aggregation_key
        ).first()
        
        if existing_metrics:
            return self._metrics_to_dict(existing_metrics)
        
        # Calculate metrics
        metrics = self._calculate_fva_metrics(period_start, period_end, aggregation_level, aggregation_key)
        
        # Store calculated metrics
        metrics_record = FVAMetricsDB(
            period_start=period_start,
            period_end=period_end,
            aggregation_level=aggregation_level,
            aggregation_key=aggregation_key,
            **metrics
        )
        
        self.session.add(metrics_record)
        self.session.commit()
        
        return metrics
    
    def create_alert(self, alert_data: dict) -> str:
        """Create FVA alert"""
        try:
            alert = FVAAlertDB(**alert_data)
            self.session.add(alert)
            self.session.commit()
            
            logger.info(f"Created FVA alert {alert.alert_id}")
            return alert.alert_id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to create alert: {e}")
            raise
    
    def get_active_alerts(self, user_id: str = None) -> list:
        """Get active FVA alerts"""
        query = self.session.query(FVAAlertDB).filter(FVAAlertDB.status == 'active')
        
        if user_id:
            query = query.filter(FVAAlertDB.user_id == user_id)
        
        alerts = query.order_by(FVAAlertDB.created_at.desc()).all()
        return [self._alert_to_dict(alert) for alert in alerts]
    
    def _update_user_metadata(self, user_id: str):
        """Update or create user metadata"""
        user_metadata = self.session.query(UserMetadataDB).filter_by(user_id=user_id).first()
        
        if not user_metadata:
            user_metadata = UserMetadataDB(user_id=user_id)
            self.session.add(user_metadata)
        
        # Update override count
        user_metadata.total_overrides = self.session.query(ForecastOverrideDB).filter_by(user_id=user_id).count()
        user_metadata.last_updated = datetime.utcnow()
        
        self.session.commit()
    
    def _update_user_performance(self, user_id: str):
        """Update user performance metrics"""
        user_metadata = self.session.query(UserMetadataDB).filter_by(user_id=user_id).first()
        if not user_metadata:
            return
        
        # Calculate performance metrics
        overrides_with_actuals = self.session.query(ForecastOverrideDB).filter(
            ForecastOverrideDB.user_id == user_id,
            ForecastOverrideDB.actual_demand.isnot(None)
        ).all()
        
        if overrides_with_actuals:
            improvements = [o.accuracy_improvement for o in overrides_with_actuals if o.accuracy_improvement is not None]
            
            if improvements:
                user_metadata.avg_accuracy_improvement = sum(improvements) / len(improvements)
                user_metadata.positive_fva_rate = sum(1 for imp in improvements if imp > 0) / len(improvements)
                
                # Calculate composite FVA score
                user_metadata.fva_score = min(100, max(0, 
                    user_metadata.avg_accuracy_improvement * 50 + user_metadata.positive_fva_rate * 50
                ))
        
        user_metadata.last_updated = datetime.utcnow()
        self.session.commit()
    
    def _calculate_fva_metrics(self, period_start: datetime, period_end: datetime,
                              aggregation_level: str, aggregation_key: str) -> dict:
        """Calculate FVA metrics for specified period"""
        
        # Base query for the period
        query = self.session.query(ForecastOverrideDB).filter(
            ForecastOverrideDB.timestamp >= period_start,
            ForecastOverrideDB.timestamp <= period_end
        )
        
        # Apply aggregation filter
        if aggregation_level == 'user' and aggregation_key:
            query = query.filter(ForecastOverrideDB.user_id == aggregation_key)
        elif aggregation_level == 'sku' and aggregation_key:
            query = query.filter(ForecastOverrideDB.sku == aggregation_key)
        elif aggregation_level == 'location' and aggregation_key:
            query = query.filter(ForecastOverrideDB.location == aggregation_key)
        
        overrides = query.all()
        
        if not overrides:
            return {
                'total_overrides': 0,
                'override_rate': 0.0,
                'avg_override_magnitude': 0.0,
                'fva_accuracy_improvement': 0.0,
                'fva_bias_reduction': 0.0,
                'positive_fva_rate': 0.0,
                'time_to_decision': 0.0,
                'override_persistence': 0.0
            }
        
        # Calculate metrics
        total_overrides = len(overrides)
        
        # Override magnitude
        magnitudes = []
        for override in overrides:
            if override.original_forecast != 0:
                magnitude = abs(override.adjusted_forecast - override.original_forecast) / abs(override.original_forecast)
                magnitudes.append(magnitude)
        
        avg_override_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
        
        # Accuracy improvement (only for overrides with actuals)
        improvements = [o.accuracy_improvement for o in overrides if o.accuracy_improvement is not None]
        fva_accuracy_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        positive_fva_rate = sum(1 for imp in improvements if imp > 0) / len(improvements) if improvements else 0.0
        
        # Estimate total forecasts for override rate calculation
        estimated_total_forecasts = total_overrides * 10  # Rough estimate
        override_rate = total_overrides / estimated_total_forecasts if estimated_total_forecasts > 0 else 0.0
        
        return {
            'total_overrides': total_overrides,
            'override_rate': override_rate,
            'avg_override_magnitude': avg_override_magnitude,
            'fva_accuracy_improvement': fva_accuracy_improvement,
            'fva_bias_reduction': 0.0,  # Would need more complex calculation
            'positive_fva_rate': positive_fva_rate,
            'time_to_decision': 2.5,  # Mock data
            'override_persistence': 0.75  # Mock data
        }
    
    def _override_to_dict(self, override: ForecastOverrideDB) -> dict:
        """Convert override DB object to dictionary"""
        return {
            'override_id': override.override_id,
            'sku': override.sku,
            'location': override.location,
            'channel': override.channel,
            'period': override.period,
            'timestamp': override.timestamp.isoformat(),
            'user_id': override.user_id,
            'original_forecast': override.original_forecast,
            'adjusted_forecast': override.adjusted_forecast,
            'override_type': override.override_type,
            'reason': override.reason,
            'confidence': override.confidence,
            'approval_status': override.approval_status,
            'actual_demand': override.actual_demand,
            'accuracy_improvement': override.accuracy_improvement
        }
    
    def _metrics_to_dict(self, metrics: FVAMetricsDB) -> dict:
        """Convert metrics DB object to dictionary"""
        return {
            'total_overrides': metrics.total_overrides,
            'override_rate': metrics.override_rate,
            'avg_override_magnitude': metrics.avg_override_magnitude,
            'fva_accuracy_improvement': metrics.fva_accuracy_improvement,
            'fva_bias_reduction': metrics.fva_bias_reduction,
            'positive_fva_rate': metrics.positive_fva_rate,
            'time_to_decision': metrics.time_to_decision,
            'override_persistence': metrics.override_persistence,
            'calculated_at': metrics.calculated_at.isoformat()
        }
    
    def _alert_to_dict(self, alert: FVAAlertDB) -> dict:
        """Convert alert DB object to dictionary"""
        return {
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'title': alert.title,
            'description': alert.description,
            'user_id': alert.user_id,
            'sku': alert.sku,
            'location': alert.location,
            'threshold_value': alert.threshold_value,
            'actual_value': alert.actual_value,
            'deviation': alert.deviation,
            'status': alert.status,
            'created_at': alert.created_at.isoformat()
        }
    
    def close(self):
        """Close database session"""
        self.session.close()