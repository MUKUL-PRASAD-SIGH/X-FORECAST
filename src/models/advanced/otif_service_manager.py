"""
OTIF (On-Time In-Full) Service Level Management System
Tracks delivery performance and optimizes service-inventory trade-offs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FailureCategory(Enum):
    INVENTORY_SHORTAGE = "inventory_shortage"
    CAPACITY_CONSTRAINT = "capacity_constraint"
    DEMAND_SPIKE = "demand_spike"
    SUPPLIER_DELAY = "supplier_delay"
    TRANSPORTATION_DELAY = "transportation_delay"
    QUALITY_ISSUE = "quality_issue"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"

class ServiceLevelTarget(Enum):
    BASIC = 0.90      # 90% OTIF
    STANDARD = 0.95   # 95% OTIF
    PREMIUM = 0.98    # 98% OTIF
    EXCELLENCE = 0.995 # 99.5% OTIF

@dataclass
class OrderRecord:
    """Individual order record for OTIF tracking"""
    order_id: str
    customer_id: str
    sku: str
    location: str
    
    # Order details
    order_date: datetime
    requested_delivery_date: datetime
    actual_delivery_date: Optional[datetime]
    
    # Quantities
    ordered_quantity: float
    delivered_quantity: float
    
    # OTIF metrics
    on_time: bool = False
    in_full: bool = False
    otif: bool = False
    
    # Tolerance parameters
    time_tolerance_hours: float = 24.0
    quantity_tolerance_pct: float = 0.02  # 2% tolerance
    
    # Failure analysis
    failure_reasons: List[FailureCategory] = field(default_factory=list)
    root_cause: Optional[FailureCategory] = None

@dataclass
class OTIFMetrics:
    """OTIF performance metrics"""
    period_start: datetime
    period_end: datetime
    
    # Basic metrics
    total_orders: int
    on_time_orders: int
    in_full_orders: int
    otif_orders: int
    
    # Performance rates
    on_time_rate: float
    in_full_rate: float
    otif_rate: float
    
    # Detailed breakdowns
    failure_breakdown: Dict[FailureCategory, int]
    customer_performance: Dict[str, float]
    sku_performance: Dict[str, float]
    location_performance: Dict[str, float]

@dataclass
class RootCauseAnalysisResult:
    """Result of root cause analysis"""
    analysis_id: str
    period: Tuple[datetime, datetime]
    
    # Primary failure causes
    top_failure_causes: List[Tuple[FailureCategory, float]]  # (cause, impact_percentage)
    
    # Detailed analysis
    inventory_related_failures: Dict[str, Any]
    capacity_related_failures: Dict[str, Any]
    demand_related_failures: Dict[str, Any]
    
    # Recommendations
    improvement_recommendations: List[str]
    estimated_impact: Dict[str, float]

@dataclass
class ServiceInventoryOptimizationResult:
    """Result of service-inventory optimization"""
    optimization_id: str
    
    # Current state
    current_otif_rate: float
    current_inventory_cost: float
    
    # Optimized state
    target_otif_rate: float
    optimized_inventory_levels: Dict[str, float]
    optimized_safety_stocks: Dict[str, float]
    
    # Trade-off analysis
    inventory_cost_increase: float
    service_level_improvement: float
    roi_estimate: float
    
    # Implementation plan
    implementation_steps: List[str]
    expected_timeline_days: int

class OTIFServiceManager:
    """
    OTIF Service Level Management System
    """
    
    def __init__(self):
        self.order_records = []
        self.otif_metrics_history = []
        self.failure_classifier = None
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.service_optimizer = ServiceInventoryOptimizer()
        self.scaler = StandardScaler()
        
        # Configuration
        self.default_time_tolerance_hours = 24.0
        self.default_quantity_tolerance_pct = 0.02
        
    def track_order_performance(self, orders_data: pd.DataFrame) -> List[OrderRecord]:
        """
        Track OTIF performance for orders
        Expected columns: order_id, customer_id, sku, location, order_date, 
                         requested_delivery_date, actual_delivery_date, 
                         ordered_quantity, delivered_quantity
        """
        logger.info(f"Tracking OTIF performance for {len(orders_data)} orders...")
        
        order_records = []
        
        for _, row in orders_data.iterrows():
            # Create order record
            order = OrderRecord(
                order_id=row['order_id'],
                customer_id=row['customer_id'],
                sku=row['sku'],
                location=row['location'],
                order_date=pd.to_datetime(row['order_date']),
                requested_delivery_date=pd.to_datetime(row['requested_delivery_date']),
                actual_delivery_date=pd.to_datetime(row['actual_delivery_date']) if pd.notna(row['actual_delivery_date']) else None,
                ordered_quantity=row['ordered_quantity'],
                delivered_quantity=row.get('delivered_quantity', 0),
                time_tolerance_hours=row.get('time_tolerance_hours', self.default_time_tolerance_hours),
                quantity_tolerance_pct=row.get('quantity_tolerance_pct', self.default_quantity_tolerance_pct)
            )
            
            # Calculate OTIF metrics
            self._calculate_otif_metrics(order)
            
            order_records.append(order)
        
        self.order_records.extend(order_records)
        logger.info(f"Processed {len(order_records)} orders")
        
        return order_records
    
    def _calculate_otif_metrics(self, order: OrderRecord):
        """Calculate OTIF metrics for an order"""
        
        # On-Time calculation
        if order.actual_delivery_date is not None:
            delivery_delay_hours = (order.actual_delivery_date - order.requested_delivery_date).total_seconds() / 3600
            order.on_time = delivery_delay_hours <= order.time_tolerance_hours
        else:
            order.on_time = False  # Not delivered yet
        
        # In-Full calculation
        if order.ordered_quantity > 0:
            fill_rate = order.delivered_quantity / order.ordered_quantity
            tolerance = order.quantity_tolerance_pct
            order.in_full = fill_rate >= (1 - tolerance)
        else:
            order.in_full = False
        
        # OTIF calculation
        order.otif = order.on_time and order.in_full
    
    def calculate_otif_metrics(self, start_date: datetime, end_date: datetime) -> OTIFMetrics:
        """
        Calculate OTIF metrics for a specific period
        """
        
        # Filter orders for the period
        period_orders = [
            order for order in self.order_records
            if start_date <= order.order_date <= end_date
        ]
        
        if not period_orders:
            return OTIFMetrics(
                period_start=start_date,
                period_end=end_date,
                total_orders=0,
                on_time_orders=0,
                in_full_orders=0,
                otif_orders=0,
                on_time_rate=0.0,
                in_full_rate=0.0,
                otif_rate=0.0,
                failure_breakdown={},
                customer_performance={},
                sku_performance={},
                location_performance={}
            )
        
        # Calculate basic metrics
        total_orders = len(period_orders)
        on_time_orders = sum(1 for order in period_orders if order.on_time)
        in_full_orders = sum(1 for order in period_orders if order.in_full)
        otif_orders = sum(1 for order in period_orders if order.otif)
        
        # Calculate rates
        on_time_rate = on_time_orders / total_orders
        in_full_rate = in_full_orders / total_orders
        otif_rate = otif_orders / total_orders
        
        # Calculate failure breakdown
        failure_breakdown = {}
        for order in period_orders:
            if not order.otif:
                for failure_reason in order.failure_reasons:
                    failure_breakdown[failure_reason] = failure_breakdown.get(failure_reason, 0) + 1
        
        # Calculate performance by dimension
        customer_performance = self._calculate_dimension_performance(period_orders, 'customer_id')
        sku_performance = self._calculate_dimension_performance(period_orders, 'sku')
        location_performance = self._calculate_dimension_performance(period_orders, 'location')
        
        metrics = OTIFMetrics(
            period_start=start_date,
            period_end=end_date,
            total_orders=total_orders,
            on_time_orders=on_time_orders,
            in_full_orders=in_full_orders,
            otif_orders=otif_orders,
            on_time_rate=on_time_rate,
            in_full_rate=in_full_rate,
            otif_rate=otif_rate,
            failure_breakdown=failure_breakdown,
            customer_performance=customer_performance,
            sku_performance=sku_performance,
            location_performance=location_performance
        )
        
        self.otif_metrics_history.append(metrics)
        return metrics
    
    def _calculate_dimension_performance(self, orders: List[OrderRecord], dimension: str) -> Dict[str, float]:
        """Calculate OTIF performance by dimension"""
        dimension_groups = {}
        
        for order in orders:
            key = getattr(order, dimension)
            if key not in dimension_groups:
                dimension_groups[key] = {'total': 0, 'otif': 0}
            
            dimension_groups[key]['total'] += 1
            if order.otif:
                dimension_groups[key]['otif'] += 1
        
        return {
            key: group['otif'] / group['total'] if group['total'] > 0 else 0.0
            for key, group in dimension_groups.items()
        }
    
    def detect_failure_patterns(self, orders_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect patterns in OTIF failures using machine learning
        """
        logger.info("Detecting OTIF failure patterns...")
        
        if len(self.order_records) < 100:
            logger.warning("Insufficient data for pattern detection")
            return {"status": "insufficient_data", "patterns": []}
        
        # Prepare features for ML analysis
        features_df = self._prepare_failure_features(orders_data)
        
        if features_df.empty:
            return {"status": "no_features", "patterns": []}
        
        # Train failure classifier
        X = features_df.drop(['otif_failure'], axis=1)
        y = features_df['otif_failure']
        
        if y.sum() < 10:  # Need at least 10 failures
            return {"status": "insufficient_failures", "patterns": []}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest classifier
        self.failure_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.failure_classifier.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, self.failure_classifier.feature_importances_))
        
        # Identify anomalous patterns
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = isolation_forest.fit_predict(X_train_scaled)
        
        patterns = {
            "status": "success",
            "model_accuracy": self.failure_classifier.score(X_test_scaled, y_test),
            "feature_importance": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True),
            "anomaly_detection": {
                "total_anomalies": sum(1 for score in anomaly_scores if score == -1),
                "anomaly_rate": sum(1 for score in anomaly_scores if score == -1) / len(anomaly_scores)
            }
        }
        
        return patterns
    
    def _prepare_failure_features(self, orders_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for failure pattern analysis"""
        
        features = []
        
        for order in self.order_records:
            if order.actual_delivery_date is None:
                continue
                
            # Time-based features
            order_day_of_week = order.order_date.weekday()
            order_hour = order.order_date.hour
            delivery_lead_time = (order.actual_delivery_date - order.order_date).days
            
            # Quantity features
            quantity_ratio = order.delivered_quantity / order.ordered_quantity if order.ordered_quantity > 0 else 0
            
            # Create feature vector
            feature_row = {
                'order_day_of_week': order_day_of_week,
                'order_hour': order_hour,
                'delivery_lead_time': delivery_lead_time,
                'ordered_quantity': order.ordered_quantity,
                'quantity_ratio': quantity_ratio,
                'otif_failure': not order.otif
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def predict_service_risk(self, future_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Predict OTIF risk for future orders
        """
        if self.failure_classifier is None:
            logger.warning("Failure classifier not trained. Run detect_failure_patterns first.")
            return future_orders.copy()
        
        # Prepare features for prediction
        prediction_features = []
        
        for _, row in future_orders.iterrows():
            order_date = pd.to_datetime(row['order_date'])
            
            feature_row = {
                'order_day_of_week': order_date.weekday(),
                'order_hour': order_date.hour,
                'delivery_lead_time': row.get('expected_lead_time', 3),
                'ordered_quantity': row['ordered_quantity'],
                'quantity_ratio': 1.0  # Assume full delivery initially
            }
            
            prediction_features.append(feature_row)
        
        features_df = pd.DataFrame(prediction_features)
        features_scaled = self.scaler.transform(features_df)
        
        # Predict failure probability
        failure_probabilities = self.failure_classifier.predict_proba(features_scaled)[:, 1]
        
        # Add predictions to orders
        result_df = future_orders.copy()
        result_df['otif_failure_risk'] = failure_probabilities
        result_df['risk_category'] = pd.cut(
            failure_probabilities, 
            bins=[0, 0.2, 0.5, 0.8, 1.0], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return result_df


class RootCauseAnalyzer:
    """
    Root cause analysis engine for OTIF failures
    """
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_failures(self, orders: List[OrderRecord], 
                        inventory_data: Optional[pd.DataFrame] = None,
                        capacity_data: Optional[pd.DataFrame] = None) -> RootCauseAnalysisResult:
        """
        Perform comprehensive root cause analysis of OTIF failures
        """
        
        failed_orders = [order for order in orders if not order.otif]
        
        if not failed_orders:
            return RootCauseAnalysisResult(
                analysis_id=f"rca_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                period=(min(order.order_date for order in orders), 
                       max(order.order_date for order in orders)),
                top_failure_causes=[],
                inventory_related_failures={},
                capacity_related_failures={},
                demand_related_failures={},
                improvement_recommendations=[],
                estimated_impact={}
            )
        
        # Analyze failure causes
        failure_counts = {}
        for order in failed_orders:
            for failure_reason in order.failure_reasons:
                failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
        
        total_failures = len(failed_orders)
        top_failure_causes = [
            (cause, count / total_failures) 
            for cause, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Detailed analysis by category
        inventory_failures = self._analyze_inventory_failures(failed_orders, inventory_data)
        capacity_failures = self._analyze_capacity_failures(failed_orders, capacity_data)
        demand_failures = self._analyze_demand_failures(failed_orders)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(top_failure_causes, inventory_failures, capacity_failures, demand_failures)
        
        # Estimate impact
        impact_estimates = self._estimate_improvement_impact(top_failure_causes, len(orders))
        
        result = RootCauseAnalysisResult(
            analysis_id=f"rca_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            period=(min(order.order_date for order in orders), 
                   max(order.order_date for order in orders)),
            top_failure_causes=top_failure_causes,
            inventory_related_failures=inventory_failures,
            capacity_related_failures=capacity_failures,
            demand_related_failures=demand_failures,
            improvement_recommendations=recommendations,
            estimated_impact=impact_estimates
        )
        
        self.analysis_history.append(result)
        return result
    
    def _analyze_inventory_failures(self, failed_orders: List[OrderRecord], 
                                   inventory_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze inventory-related failures"""
        
        inventory_failures = [
            order for order in failed_orders 
            if FailureCategory.INVENTORY_SHORTAGE in order.failure_reasons
        ]
        
        analysis = {
            "total_inventory_failures": len(inventory_failures),
            "affected_skus": list(set(order.sku for order in inventory_failures)),
            "affected_locations": list(set(order.location for order in inventory_failures)),
            "average_shortage_quantity": np.mean([
                order.ordered_quantity - order.delivered_quantity 
                for order in inventory_failures
            ]) if inventory_failures else 0
        }
        
        if inventory_data is not None:
            # Additional analysis with inventory data
            analysis["stockout_correlation"] = self._calculate_stockout_correlation(
                inventory_failures, inventory_data
            )
        
        return analysis
    
    def _analyze_capacity_failures(self, failed_orders: List[OrderRecord], 
                                  capacity_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze capacity-related failures"""
        
        capacity_failures = [
            order for order in failed_orders 
            if FailureCategory.CAPACITY_CONSTRAINT in order.failure_reasons
        ]
        
        analysis = {
            "total_capacity_failures": len(capacity_failures),
            "peak_failure_days": self._identify_peak_failure_days(capacity_failures),
            "affected_locations": list(set(order.location for order in capacity_failures))
        }
        
        return analysis
    
    def _analyze_demand_failures(self, failed_orders: List[OrderRecord]) -> Dict[str, Any]:
        """Analyze demand-related failures"""
        
        demand_failures = [
            order for order in failed_orders 
            if FailureCategory.DEMAND_SPIKE in order.failure_reasons
        ]
        
        analysis = {
            "total_demand_failures": len(demand_failures),
            "spike_patterns": self._identify_demand_spike_patterns(demand_failures)
        }
        
        return analysis
    
    def _calculate_stockout_correlation(self, failures: List[OrderRecord], 
                                      inventory_data: pd.DataFrame) -> float:
        """Calculate correlation between failures and stockouts"""
        # Simplified correlation calculation
        return 0.75  # Placeholder
    
    def _identify_peak_failure_days(self, failures: List[OrderRecord]) -> List[str]:
        """Identify days with highest failure rates"""
        failure_days = {}
        for order in failures:
            day = order.order_date.strftime('%A')
            failure_days[day] = failure_days.get(day, 0) + 1
        
        return sorted(failure_days.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _identify_demand_spike_patterns(self, failures: List[OrderRecord]) -> Dict[str, Any]:
        """Identify patterns in demand spike failures"""
        return {
            "seasonal_spikes": ["Q4", "Holiday periods"],
            "promotional_spikes": ["Black Friday", "End of month"]
        }
    
    def _generate_recommendations(self, top_causes: List[Tuple[FailureCategory, float]], 
                                inventory_failures: Dict[str, Any],
                                capacity_failures: Dict[str, Any],
                                demand_failures: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        for cause, impact in top_causes[:3]:  # Top 3 causes
            if cause == FailureCategory.INVENTORY_SHORTAGE:
                recommendations.append(f"Increase safety stock for {len(inventory_failures['affected_skus'])} SKUs with frequent stockouts")
                recommendations.append("Implement dynamic reorder point calculation based on demand variability")
            
            elif cause == FailureCategory.CAPACITY_CONSTRAINT:
                recommendations.append("Add capacity during peak demand periods identified in analysis")
                recommendations.append("Implement load balancing across locations")
            
            elif cause == FailureCategory.DEMAND_SPIKE:
                recommendations.append("Improve demand sensing and early warning systems")
                recommendations.append("Create buffer capacity for unexpected demand spikes")
        
        return recommendations
    
    def _estimate_improvement_impact(self, top_causes: List[Tuple[FailureCategory, float]], 
                                   total_orders: int) -> Dict[str, float]:
        """Estimate impact of addressing top failure causes"""
        
        impact_estimates = {}
        
        for cause, failure_rate in top_causes[:3]:
            if cause == FailureCategory.INVENTORY_SHORTAGE:
                # Assume 70% of inventory failures can be prevented
                impact_estimates["inventory_improvement"] = failure_rate * 0.7
            
            elif cause == FailureCategory.CAPACITY_CONSTRAINT:
                # Assume 50% of capacity failures can be prevented
                impact_estimates["capacity_improvement"] = failure_rate * 0.5
            
            elif cause == FailureCategory.DEMAND_SPIKE:
                # Assume 30% of demand spike failures can be prevented
                impact_estimates["demand_improvement"] = failure_rate * 0.3
        
        return impact_estimates


class ServiceInventoryOptimizer:
    """
    Service-inventory trade-off optimization engine
    """
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_service_inventory_tradeoff(self, 
                                          current_metrics: OTIFMetrics,
                                          target_service_level: float,
                                          inventory_data: pd.DataFrame,
                                          cost_parameters: Dict[str, float]) -> ServiceInventoryOptimizationResult:
        """
        Optimize inventory levels to achieve target service level
        """
        
        logger.info(f"Optimizing for target OTIF rate: {target_service_level}")
        
        # Current state
        current_otif = current_metrics.otif_rate
        current_inventory_cost = self._calculate_current_inventory_cost(inventory_data, cost_parameters)
        
        if current_otif >= target_service_level:
            logger.info("Current service level already meets target")
            return ServiceInventoryOptimizationResult(
                optimization_id=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                current_otif_rate=current_otif,
                current_inventory_cost=current_inventory_cost,
                target_otif_rate=target_service_level,
                optimized_inventory_levels={},
                optimized_safety_stocks={},
                inventory_cost_increase=0.0,
                service_level_improvement=0.0,
                roi_estimate=0.0,
                implementation_steps=[],
                expected_timeline_days=0
            )
        
        # Calculate required service level improvement
        service_gap = target_service_level - current_otif
        
        # Optimize inventory levels
        optimized_levels = self._optimize_inventory_levels(
            inventory_data, service_gap, cost_parameters
        )
        
        # Calculate costs and benefits
        new_inventory_cost = self._calculate_optimized_inventory_cost(
            optimized_levels, cost_parameters
        )
        
        cost_increase = new_inventory_cost - current_inventory_cost
        
        # Estimate ROI
        revenue_improvement = self._estimate_revenue_improvement(
            service_gap, cost_parameters.get('revenue_per_order', 100)
        )
        roi_estimate = (revenue_improvement - cost_increase) / cost_increase if cost_increase > 0 else 0
        
        # Generate implementation plan
        implementation_steps = self._generate_implementation_plan(optimized_levels)
        
        result = ServiceInventoryOptimizationResult(
            optimization_id=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            current_otif_rate=current_otif,
            current_inventory_cost=current_inventory_cost,
            target_otif_rate=target_service_level,
            optimized_inventory_levels=optimized_levels['inventory_levels'],
            optimized_safety_stocks=optimized_levels['safety_stocks'],
            inventory_cost_increase=cost_increase,
            service_level_improvement=service_gap,
            roi_estimate=roi_estimate,
            implementation_steps=implementation_steps,
            expected_timeline_days=30
        )
        
        self.optimization_history.append(result)
        return result
    
    def _calculate_current_inventory_cost(self, inventory_data: pd.DataFrame, 
                                        cost_parameters: Dict[str, float]) -> float:
        """Calculate current inventory holding cost"""
        
        holding_cost_rate = cost_parameters.get('holding_cost_rate', 0.25)  # 25% annual
        
        total_inventory_value = (inventory_data['quantity'] * inventory_data['unit_cost']).sum()
        annual_holding_cost = total_inventory_value * holding_cost_rate
        
        return annual_holding_cost
    
    def _optimize_inventory_levels(self, inventory_data: pd.DataFrame, 
                                 service_gap: float, 
                                 cost_parameters: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Optimize inventory levels using mathematical optimization"""
        
        # Simplified optimization - increase safety stock proportionally
        safety_stock_multiplier = 1 + (service_gap * 2)  # Heuristic relationship
        
        optimized_levels = {
            'inventory_levels': {},
            'safety_stocks': {}
        }
        
        for _, row in inventory_data.iterrows():
            sku_location = f"{row['sku']}_{row['location']}"
            
            current_level = row['quantity']
            current_safety_stock = row.get('safety_stock', current_level * 0.2)
            
            # Optimize safety stock
            new_safety_stock = current_safety_stock * safety_stock_multiplier
            new_inventory_level = current_level + (new_safety_stock - current_safety_stock)
            
            optimized_levels['inventory_levels'][sku_location] = new_inventory_level
            optimized_levels['safety_stocks'][sku_location] = new_safety_stock
        
        return optimized_levels
    
    def _calculate_optimized_inventory_cost(self, optimized_levels: Dict[str, Dict[str, float]], 
                                          cost_parameters: Dict[str, float]) -> float:
        """Calculate cost of optimized inventory levels"""
        
        holding_cost_rate = cost_parameters.get('holding_cost_rate', 0.25)
        average_unit_cost = cost_parameters.get('average_unit_cost', 10)
        
        total_inventory_value = sum(optimized_levels['inventory_levels'].values()) * average_unit_cost
        annual_holding_cost = total_inventory_value * holding_cost_rate
        
        return annual_holding_cost
    
    def _estimate_revenue_improvement(self, service_improvement: float, 
                                    revenue_per_order: float) -> float:
        """Estimate revenue improvement from service level increase"""
        
        # Assume linear relationship between service improvement and revenue
        # This is a simplified model - in practice, this would be more complex
        estimated_additional_orders = service_improvement * 1000  # Heuristic
        revenue_improvement = estimated_additional_orders * revenue_per_order
        
        return revenue_improvement
    
    def _generate_implementation_plan(self, optimized_levels: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate implementation plan for inventory optimization"""
        
        steps = [
            "1. Review and approve optimized inventory levels",
            "2. Update safety stock parameters in inventory management system",
            "3. Place additional purchase orders for increased inventory levels",
            "4. Monitor service level improvements over 30-day period",
            "5. Fine-tune inventory levels based on actual performance",
            "6. Implement automated reorder point adjustments"
        ]
        
        return steps
    
    def create_service_level_dashboard_data(self, metrics_history: List[OTIFMetrics]) -> Dict[str, Any]:
        """
        Create data structure for OTIF dashboard visualization
        """
        
        if not metrics_history:
            return {"status": "no_data"}
        
        # Time series data
        dates = [metrics.period_start for metrics in metrics_history]
        otif_rates = [metrics.otif_rate for metrics in metrics_history]
        on_time_rates = [metrics.on_time_rate for metrics in metrics_history]
        in_full_rates = [metrics.in_full_rate for metrics in metrics_history]
        
        # Current performance
        latest_metrics = metrics_history[-1]
        
        # Trend analysis
        if len(otif_rates) >= 2:
            trend = "improving" if otif_rates[-1] > otif_rates[-2] else "declining"
        else:
            trend = "stable"
        
        dashboard_data = {
            "status": "success",
            "current_performance": {
                "otif_rate": latest_metrics.otif_rate,
                "on_time_rate": latest_metrics.on_time_rate,
                "in_full_rate": latest_metrics.in_full_rate,
                "total_orders": latest_metrics.total_orders,
                "trend": trend
            },
            "time_series": {
                "dates": [d.isoformat() for d in dates],
                "otif_rates": otif_rates,
                "on_time_rates": on_time_rates,
                "in_full_rates": in_full_rates
            },
            "failure_breakdown": dict(latest_metrics.failure_breakdown),
            "performance_by_dimension": {
                "customers": dict(latest_metrics.customer_performance),
                "skus": dict(latest_metrics.sku_performance),
                "locations": dict(latest_metrics.location_performance)
            }
        }
        
        return dashboard_dataders
        in_full_rate = in_full_orders / total_or