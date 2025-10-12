"""
Multi-Echelon Buffer Optimization System
Optimizes inventory buffers across supply network considering service levels and costs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import networkx as nx
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, poisson
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class NodeType(Enum):
    SUPPLIER = "supplier"
    MANUFACTURING = "manufacturing"
    DISTRIBUTION_CENTER = "distribution_center"
    REGIONAL_WAREHOUSE = "regional_warehouse"
    RETAIL_STORE = "retail_store"
    CUSTOMER = "customer"

class ServiceLevelType(Enum):
    FILL_RATE = "fill_rate"
    CYCLE_SERVICE_LEVEL = "cycle_service_level"
    READY_RATE = "ready_rate"

class DisruptionType(Enum):
    SUPPLIER_DELAY = "supplier_delay"
    TRANSPORTATION_DELAY = "transportation_delay"
    CAPACITY_CONSTRAINT = "capacity_constraint"
    DEMAND_SPIKE = "demand_spike"
    QUALITY_ISSUE = "quality_issue"

@dataclass
class NetworkNode:
    """Represents a node in the supply network"""
    node_id: str
    node_type: NodeType
    location: str
    
    # Capacity and constraints
    capacity: float
    holding_cost_rate: float
    ordering_cost: float
    
    # Lead time characteristics
    lead_time_mean: float
    lead_time_std: float
    lead_time_distribution: str = "normal"
    
    # Demand characteristics (for end nodes)
    demand_mean: Optional[float] = None
    demand_std: Optional[float] = None
    demand_distribution: str = "normal"
    
    # Service level requirements
    target_service_level: float = 0.95
    service_level_type: ServiceLevelType = ServiceLevelType.FILL_RATE
    
    # Current state
    current_inventory: float = 0.0
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    order_quantity: float = 0.0

@dataclass
class NetworkLink:
    """Represents a connection between nodes"""
    link_id: str
    from_node: str
    to_node: str
    
    # Transportation characteristics
    transport_time_mean: float
    transport_time_std: float
    transport_cost_per_unit: float
    transport_capacity: float
    
    # Reliability
    reliability: float = 0.99
    
    # Current state
    in_transit_inventory: float = 0.0

@dataclass
class BufferOptimizationResult:
    """Result of buffer optimization"""
    node_id: str
    optimal_safety_stock: float
    optimal_reorder_point: float
    optimal_order_quantity: float
    expected_service_level: float
    expected_holding_cost: float
    expected_shortage_cost: float
    total_cost: float

@dataclass
class NetworkOptimizationResult:
    """Result of network-wide optimization"""
    optimization_id: str
    node_results: Dict[str, BufferOptimizationResult]
    total_network_cost: float
    average_service_level: float
    optimization_method: str
    convergence_info: Dict[str, Any]

@dataclass
class DisruptionScenario:
    """Disruption scenario for analysis"""
    scenario_id: str
    disruption_type: DisruptionType
    affected_nodes: List[str]
    affected_links: List[str]
    impact_magnitude: float
    duration_days: int
    probability: float

@dataclass
class DisruptionImpactResult:
    """Result of disruption impact analysis"""
    scenario_id: str
    service_level_impact: Dict[str, float]
    cost_impact: Dict[str, float]
    recovery_time_days: int
    mitigation_recommendations: List[str]

class MultiEchelonOptimizer:
    """
    Multi-echelon buffer optimization system
    """
    
    def __init__(self):
        self.network = nx.DiGraph()
        self.nodes = {}
        self.links = {}
        self.optimization_results = {}
        
        # Optimization parameters
        self.shortage_cost_multiplier = 10.0  # Shortage cost as multiple of holding cost
        self.service_level_penalty = 1000.0  # Penalty for not meeting service level
        
    def build_supply_network(self, nodes_config: List[Dict[str, Any]], 
                           links_config: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build supply network from configuration
        """
        logger.info("Building supply network...")
        
        # Create nodes
        for node_config in nodes_config:
            node = NetworkNode(
                node_id=node_config['node_id'],
                node_type=NodeType(node_config['node_type']),
                location=node_config.get('location', ''),
                capacity=node_config.get('capacity', float('inf')),
                holding_cost_rate=node_config.get('holding_cost_rate', 0.2),
                ordering_cost=node_config.get('ordering_cost', 100.0),
                lead_time_mean=node_config.get('lead_time_mean', 7.0),
                lead_time_std=node_config.get('lead_time_std', 1.0),
                demand_mean=node_config.get('demand_mean'),
                demand_std=node_config.get('demand_std'),
                target_service_level=node_config.get('target_service_level', 0.95),
                current_inventory=node_config.get('current_inventory', 0.0)
            )
            
            self.nodes[node.node_id] = node
            self.network.add_node(node.node_id, **node.__dict__)
        
        # Create links
        for link_config in links_config:
            link = NetworkLink(
                link_id=link_config['link_id'],
                from_node=link_config['from_node'],
                to_node=link_config['to_node'],
                transport_time_mean=link_config.get('transport_time_mean', 2.0),
                transport_time_std=link_config.get('transport_time_std', 0.5),
                transport_cost_per_unit=link_config.get('transport_cost_per_unit', 1.0),
                transport_capacity=link_config.get('transport_capacity', float('inf')),
                reliability=link_config.get('reliability', 0.99)
            )
            
            self.links[link.link_id] = link
            self.network.add_edge(
                link.from_node, 
                link.to_node, 
                **link.__dict__
            )
        
        logger.info(f"Built network with {len(self.nodes)} nodes and {len(self.links)} links")
        return self.network
    
    def optimize_buffer_levels(self, optimization_method: str = "metric") -> NetworkOptimizationResult:
        """
        Optimize buffer levels across the network
        """
        logger.info(f"Optimizing buffer levels using {optimization_method} method...")
        
        if optimization_method == "metric":
            return self._optimize_using_metric()
        elif optimization_method == "simulation":
            return self._optimize_using_simulation()
        elif optimization_method == "analytical":
            return self._optimize_using_analytical()
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def _optimize_using_metric(self) -> NetworkOptimizationResult:
        """
        Optimize using METRIC (Multi-Echelon Technique for Recoverable Item Control)
        """
        
        node_results = {}
        total_cost = 0.0
        service_levels = []
        
        # Process nodes in reverse topological order (from customers to suppliers)
        try:
            topo_order = list(nx.topological_sort(self.network))
            reverse_order = topo_order[::-1]
        except:
            # If network has cycles, use arbitrary order
            reverse_order = list(self.nodes.keys())
        
        for node_id in reverse_order:
            node = self.nodes[node_id]
            
            # Calculate optimal buffer levels for this node
            result = self._optimize_single_node_metric(node_id)
            node_results[node_id] = result
            
            total_cost += result.total_cost
            service_levels.append(result.expected_service_level)
        
        avg_service_level = np.mean(service_levels) if service_levels else 0.0
        
        return NetworkOptimizationResult(
            optimization_id=f"metric_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            node_results=node_results,
            total_network_cost=total_cost,
            average_service_level=avg_service_level,
            optimization_method="metric",
            convergence_info={"status": "completed", "iterations": len(reverse_order)}
        )
    
    def _optimize_single_node_metric(self, node_id: str) -> BufferOptimizationResult:
        """
        Optimize buffer levels for a single node using METRIC approach
        """
        
        node = self.nodes[node_id]
        
        # Calculate effective demand and lead time
        effective_demand_mean, effective_demand_std = self._calculate_effective_demand(node_id)
        effective_lead_time_mean, effective_lead_time_std = self._calculate_effective_lead_time(node_id)
        
        # Calculate demand during lead time
        ddlt_mean = effective_demand_mean * effective_lead_time_mean
        ddlt_std = np.sqrt(
            (effective_demand_std ** 2) * effective_lead_time_mean +
            (effective_demand_mean ** 2) * (effective_lead_time_std ** 2)
        )
        
        # Calculate optimal safety stock using newsvendor model
        target_service_level = node.target_service_level
        
        if node.service_level_type == ServiceLevelType.FILL_RATE:
            # Fill rate optimization
            z_score = norm.ppf(target_service_level)
            safety_stock = z_score * ddlt_std
        else:
            # Cycle service level optimization
            z_score = norm.ppf(target_service_level)
            safety_stock = z_score * ddlt_std
        
        # Calculate reorder point
        reorder_point = ddlt_mean + safety_stock
        
        # Calculate optimal order quantity using EOQ
        if effective_demand_mean > 0:
            eoq = np.sqrt(2 * node.ordering_cost * effective_demand_mean / node.holding_cost_rate)
        else:
            eoq = 0.0
        
        # Calculate expected costs
        holding_cost = (safety_stock + eoq / 2) * node.holding_cost_rate
        
        # Estimate shortage cost (simplified)
        shortage_probability = 1 - target_service_level
        shortage_cost = shortage_probability * effective_demand_mean * node.holding_cost_rate * self.shortage_cost_multiplier
        
        total_cost = holding_cost + shortage_cost
        
        # Calculate expected service level
        if ddlt_std > 0:
            expected_service_level = norm.cdf((safety_stock) / ddlt_std)
        else:
            expected_service_level = 1.0
        
        return BufferOptimizationResult(
            node_id=node_id,
            optimal_safety_stock=max(0, safety_stock),
            optimal_reorder_point=max(0, reorder_point),
            optimal_order_quantity=max(0, eoq),
            expected_service_level=expected_service_level,
            expected_holding_cost=holding_cost,
            expected_shortage_cost=shortage_cost,
            total_cost=total_cost
        )
    
    def _calculate_effective_demand(self, node_id: str) -> Tuple[float, float]:
        """
        Calculate effective demand for a node considering downstream demand
        """
        
        node = self.nodes[node_id]
        
        # If node has direct demand (end node)
        if node.demand_mean is not None:
            return node.demand_mean, node.demand_std or 0.0
        
        # Calculate demand from downstream nodes
        downstream_nodes = list(self.network.successors(node_id))
        
        if not downstream_nodes:
            return 0.0, 0.0
        
        total_demand_mean = 0.0
        total_demand_var = 0.0
        
        for downstream_node in downstream_nodes:
            downstream_demand_mean, downstream_demand_std = self._calculate_effective_demand(downstream_node)
            
            # Add demand from this downstream node
            total_demand_mean += downstream_demand_mean
            total_demand_var += downstream_demand_std ** 2
        
        return total_demand_mean, np.sqrt(total_demand_var)
    
    def _calculate_effective_lead_time(self, node_id: str) -> Tuple[float, float]:
        """
        Calculate effective lead time for a node
        """
        
        node = self.nodes[node_id]
        
        # Start with node's own lead time
        total_lead_time_mean = node.lead_time_mean
        total_lead_time_var = node.lead_time_std ** 2
        
        # Add transportation time from upstream nodes
        upstream_nodes = list(self.network.predecessors(node_id))
        
        for upstream_node in upstream_nodes:
            # Find link from upstream node
            edge_data = self.network.get_edge_data(upstream_node, node_id)
            if edge_data:
                transport_time_mean = edge_data.get('transport_time_mean', 0)
                transport_time_std = edge_data.get('transport_time_std', 0)
                
                total_lead_time_mean += transport_time_mean
                total_lead_time_var += transport_time_std ** 2
        
        return total_lead_time_mean, np.sqrt(total_lead_time_var)
    
    def _optimize_using_simulation(self) -> NetworkOptimizationResult:
        """
        Optimize using discrete event simulation
        """
        logger.info("Simulation-based optimization not fully implemented - using METRIC as fallback")
        return self._optimize_using_metric()
    
    def _optimize_using_analytical(self) -> NetworkOptimizationResult:
        """
        Optimize using analytical guaranteed service model
        """
        
        # Define optimization variables: safety stock levels for each node
        node_ids = list(self.nodes.keys())
        n_nodes = len(node_ids)
        
        # Initial guess: current safety stock levels
        x0 = [self.nodes[node_id].safety_stock for node_id in node_ids]
        
        # Bounds: safety stock >= 0
        bounds = [(0, None) for _ in range(n_nodes)]
        
        # Constraints: service level requirements
        constraints = []
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            
            def service_level_constraint(x, node_idx=i, target_sl=node.target_service_level):
                safety_stock = x[node_idx]
                _, ddlt_std = self._get_demand_during_lead_time_params(node_ids[node_idx])
                
                if ddlt_std > 0:
                    achieved_sl = norm.cdf(safety_stock / ddlt_std)
                else:
                    achieved_sl = 1.0
                
                return achieved_sl - target_sl  # >= 0
            
            constraints.append({'type': 'ineq', 'fun': service_level_constraint})
        
        # Objective function: minimize total network cost
        def objective(x):
            total_cost = 0.0
            
            for i, node_id in enumerate(node_ids):
                node = self.nodes[node_id]
                safety_stock = x[i]
                
                # Holding cost
                holding_cost = safety_stock * node.holding_cost_rate
                
                # Shortage cost (simplified)
                _, ddlt_std = self._get_demand_during_lead_time_params(node_id)
                if ddlt_std > 0:
                    shortage_prob = 1 - norm.cdf(safety_stock / ddlt_std)
                else:
                    shortage_prob = 0.0
                
                demand_mean, _ = self._calculate_effective_demand(node_id)
                shortage_cost = shortage_prob * demand_mean * node.holding_cost_rate * self.shortage_cost_multiplier
                
                total_cost += holding_cost + shortage_cost
            
            return total_cost
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_safety_stocks = result.x
            else:
                logger.warning("Optimization did not converge, using METRIC fallback")
                return self._optimize_using_metric()
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}, using METRIC fallback")
            return self._optimize_using_metric()
        
        # Create results
        node_results = {}
        total_cost = 0.0
        service_levels = []
        
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            safety_stock = optimal_safety_stocks[i]
            
            # Calculate other parameters
            ddlt_mean, ddlt_std = self._get_demand_during_lead_time_params(node_id)
            reorder_point = ddlt_mean + safety_stock
            
            demand_mean, _ = self._calculate_effective_demand(node_id)
            if demand_mean > 0:
                eoq = np.sqrt(2 * node.ordering_cost * demand_mean / node.holding_cost_rate)
            else:
                eoq = 0.0
            
            # Calculate costs
            holding_cost = safety_stock * node.holding_cost_rate
            if ddlt_std > 0:
                shortage_prob = 1 - norm.cdf(safety_stock / ddlt_std)
                expected_service_level = norm.cdf(safety_stock / ddlt_std)
            else:
                shortage_prob = 0.0
                expected_service_level = 1.0
            
            shortage_cost = shortage_prob * demand_mean * node.holding_cost_rate * self.shortage_cost_multiplier
            
            node_result = BufferOptimizationResult(
                node_id=node_id,
                optimal_safety_stock=safety_stock,
                optimal_reorder_point=reorder_point,
                optimal_order_quantity=eoq,
                expected_service_level=expected_service_level,
                expected_holding_cost=holding_cost,
                expected_shortage_cost=shortage_cost,
                total_cost=holding_cost + shortage_cost
            )
            
            node_results[node_id] = node_result
            total_cost += node_result.total_cost
            service_levels.append(expected_service_level)
        
        return NetworkOptimizationResult(
            optimization_id=f"analytical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            node_results=node_results,
            total_network_cost=total_cost,
            average_service_level=np.mean(service_levels),
            optimization_method="analytical",
            convergence_info={
                "status": "success" if result.success else "failed",
                "iterations": result.nit,
                "function_evaluations": result.nfev
            }
        )
    
    def _get_demand_during_lead_time_params(self, node_id: str) -> Tuple[float, float]:
        """Get demand during lead time parameters for a node"""
        demand_mean, demand_std = self._calculate_effective_demand(node_id)
        lead_time_mean, lead_time_std = self._calculate_effective_lead_time(node_id)
        
        ddlt_mean = demand_mean * lead_time_mean
        ddlt_std = np.sqrt(
            (demand_std ** 2) * lead_time_mean +
            (demand_mean ** 2) * (lead_time_std ** 2)
        )
        
        return ddlt_mean, ddlt_std
    
    def analyze_disruption_impact(self, disruption_scenarios: List[DisruptionScenario]) -> List[DisruptionImpactResult]:
        """
        Analyze impact of supply chain disruptions
        """
        logger.info(f"Analyzing {len(disruption_scenarios)} disruption scenarios...")
        
        results = []
        
        for scenario in disruption_scenarios:
            impact_result = self._analyze_single_disruption(scenario)
            results.append(impact_result)
        
        return results
    
    def _analyze_single_disruption(self, scenario: DisruptionScenario) -> DisruptionImpactResult:
        """
        Analyze impact of a single disruption scenario
        """
        
        service_level_impact = {}
        cost_impact = {}
        mitigation_recommendations = []
        
        # Create disrupted network
        disrupted_network = self.network.copy()
        
        if scenario.disruption_type == DisruptionType.SUPPLIER_DELAY:
            # Increase lead times for affected nodes
            for node_id in scenario.affected_nodes:
                if node_id in self.nodes:
                    original_lead_time = self.nodes[node_id].lead_time_mean
                    disrupted_lead_time = original_lead_time * (1 + scenario.impact_magnitude)
                    
                    # Calculate impact on service level
                    original_ddlt_mean, original_ddlt_std = self._get_demand_during_lead_time_params(node_id)
                    
                    # Simulate disrupted parameters
                    demand_mean, demand_std = self._calculate_effective_demand(node_id)
                    disrupted_ddlt_mean = demand_mean * disrupted_lead_time
                    disrupted_ddlt_std = np.sqrt(
                        (demand_std ** 2) * disrupted_lead_time +
                        (demand_mean ** 2) * (self.nodes[node_id].lead_time_std ** 2)
                    )
                    
                    # Calculate service level impact
                    current_safety_stock = self.nodes[node_id].safety_stock
                    if original_ddlt_std > 0:
                        original_sl = norm.cdf(current_safety_stock / original_ddlt_std)
                    else:
                        original_sl = 1.0
                    
                    if disrupted_ddlt_std > 0:
                        disrupted_sl = norm.cdf(current_safety_stock / disrupted_ddlt_std)
                    else:
                        disrupted_sl = 1.0
                    
                    service_level_impact[node_id] = disrupted_sl - original_sl
                    
                    # Calculate cost impact (additional holding cost for mitigation)
                    if disrupted_ddlt_std > 0:
                        required_safety_stock = norm.ppf(self.nodes[node_id].target_service_level) * disrupted_ddlt_std
                        additional_safety_stock = max(0, required_safety_stock - current_safety_stock)
                        additional_cost = additional_safety_stock * self.nodes[node_id].holding_cost_rate
                        cost_impact[node_id] = additional_cost
                    
                    # Mitigation recommendations
                    if service_level_impact[node_id] < -0.05:  # Significant impact
                        mitigation_recommendations.append(
                            f"Increase safety stock for {node_id} by {additional_safety_stock:.1f} units"
                        )
        
        elif scenario.disruption_type == DisruptionType.DEMAND_SPIKE:
            # Increase demand for affected nodes
            for node_id in scenario.affected_nodes:
                if node_id in self.nodes and self.nodes[node_id].demand_mean:
                    original_demand = self.nodes[node_id].demand_mean
                    disrupted_demand = original_demand * (1 + scenario.impact_magnitude)
                    
                    # Calculate service level impact
                    current_safety_stock = self.nodes[node_id].safety_stock
                    lead_time_mean, lead_time_std = self._calculate_effective_lead_time(node_id)
                    
                    original_ddlt_mean = original_demand * lead_time_mean
                    disrupted_ddlt_mean = disrupted_demand * lead_time_mean
                    
                    demand_std = self.nodes[node_id].demand_std or original_demand * 0.2
                    disrupted_demand_std = demand_std * (1 + scenario.impact_magnitude * 0.5)
                    
                    original_ddlt_std = np.sqrt(
                        (demand_std ** 2) * lead_time_mean +
                        (original_demand ** 2) * (lead_time_std ** 2)
                    )
                    
                    disrupted_ddlt_std = np.sqrt(
                        (disrupted_demand_std ** 2) * lead_time_mean +
                        (disrupted_demand ** 2) * (lead_time_std ** 2)
                    )
                    
                    if original_ddlt_std > 0:
                        original_sl = norm.cdf(current_safety_stock / original_ddlt_std)
                    else:
                        original_sl = 1.0
                    
                    if disrupted_ddlt_std > 0:
                        disrupted_sl = norm.cdf(current_safety_stock / disrupted_ddlt_std)
                    else:
                        disrupted_sl = 1.0
                    
                    service_level_impact[node_id] = disrupted_sl - original_sl
                    
                    # Cost impact
                    if disrupted_ddlt_std > 0:
                        required_safety_stock = norm.ppf(self.nodes[node_id].target_service_level) * disrupted_ddlt_std
                        additional_safety_stock = max(0, required_safety_stock - current_safety_stock)
                        additional_cost = additional_safety_stock * self.nodes[node_id].holding_cost_rate
                        cost_impact[node_id] = additional_cost
                    
                    if service_level_impact[node_id] < -0.05:
                        mitigation_recommendations.append(
                            f"Increase safety stock for {node_id} to handle demand spike"
                        )
        
        # Estimate recovery time (simplified)
        recovery_time_days = scenario.duration_days + 7  # Disruption duration + recovery buffer
        
        return DisruptionImpactResult(
            scenario_id=scenario.scenario_id,
            service_level_impact=service_level_impact,
            cost_impact=cost_impact,
            recovery_time_days=recovery_time_days,
            mitigation_recommendations=mitigation_recommendations
        )
    
    def rebalance_inventory_for_disruption(self, disruption_scenario: DisruptionScenario) -> Dict[str, float]:
        """
        Dynamically rebalance inventory for disruption scenario
        """
        logger.info(f"Rebalancing inventory for disruption: {disruption_scenario.scenario_id}")
        
        rebalancing_plan = {}
        
        # Analyze disruption impact
        impact_result = self._analyze_single_disruption(disruption_scenario)
        
        # For each affected node, calculate optimal rebalancing
        for node_id in disruption_scenario.affected_nodes:
            if node_id in self.nodes:
                current_inventory = self.nodes[node_id].current_inventory
                current_safety_stock = self.nodes[node_id].safety_stock
                
                # Calculate required safety stock under disruption
                if disruption_scenario.disruption_type == DisruptionType.SUPPLIER_DELAY:
                    # Increase safety stock proportionally to lead time increase
                    multiplier = 1 + disruption_scenario.impact_magnitude
                    required_safety_stock = current_safety_stock * np.sqrt(multiplier)
                
                elif disruption_scenario.disruption_type == DisruptionType.DEMAND_SPIKE:
                    # Increase safety stock for demand spike
                    multiplier = 1 + disruption_scenario.impact_magnitude
                    required_safety_stock = current_safety_stock * multiplier
                
                else:
                    # Default increase
                    required_safety_stock = current_safety_stock * 1.2
                
                # Calculate rebalancing amount
                additional_inventory_needed = max(0, required_safety_stock - current_inventory)
                rebalancing_plan[node_id] = additional_inventory_needed
        
        return rebalancing_plan
    
    def calculate_service_level_projections(self, time_horizon_days: int = 90) -> Dict[str, List[float]]:
        """
        Project service levels over time horizon
        """
        logger.info(f"Calculating service level projections for {time_horizon_days} days")
        
        projections = {}
        
        for node_id, node in self.nodes.items():
            daily_projections = []
            
            # Get current parameters
            demand_mean, demand_std = self._calculate_effective_demand(node_id)
            lead_time_mean, lead_time_std = self._calculate_effective_lead_time(node_id)
            current_safety_stock = node.safety_stock
            
            # Project service level for each day
            for day in range(time_horizon_days):
                # Simple projection assuming gradual degradation due to demand uncertainty
                degradation_factor = 1 - (day * 0.001)  # 0.1% degradation per day
                
                effective_safety_stock = current_safety_stock * degradation_factor
                
                ddlt_mean = demand_mean * lead_time_mean
                ddlt_std = np.sqrt(
                    (demand_std ** 2) * lead_time_mean +
                    (demand_mean ** 2) * (lead_time_std ** 2)
                )
                
                if ddlt_std > 0:
                    projected_service_level = norm.cdf(effective_safety_stock / ddlt_std)
                else:
                    projected_service_level = 1.0
                
                daily_projections.append(max(0.0, min(1.0, projected_service_level)))
            
            projections[node_id] = daily_projections
        
        return projections
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive network metrics
        """
        
        if not self.nodes:
            return {"error": "No network configured"}
        
        # Basic network statistics
        total_nodes = len(self.nodes)
        total_links = len(self.links)
        
        # Node type distribution
        node_type_counts = {}
        for node in self.nodes.values():
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        # Service level statistics
        service_levels = [node.target_service_level for node in self.nodes.values()]
        avg_service_level = np.mean(service_levels)
        min_service_level = np.min(service_levels)
        max_service_level = np.max(service_levels)
        
        # Cost statistics
        holding_costs = [node.holding_cost_rate for node in self.nodes.values()]
        avg_holding_cost = np.mean(holding_costs)
        
        # Lead time statistics
        lead_times = [node.lead_time_mean for node in self.nodes.values()]
        avg_lead_time = np.mean(lead_times)
        max_lead_time = np.max(lead_times)
        
        # Network topology metrics
        if self.network.number_of_nodes() > 0:
            try:
                # Calculate network metrics
                density = nx.density(self.network)
                
                # Find longest path (supply chain depth)
                if nx.is_directed_acyclic_graph(self.network):
                    longest_path = nx.dag_longest_path_length(self.network)
                else:
                    longest_path = 0
                
                # Centrality measures
                in_degree_centrality = nx.in_degree_centrality(self.network)
                out_degree_centrality = nx.out_degree_centrality(self.network)
                
                # Find critical nodes (high centrality)
                critical_nodes = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                
            except:
                density = 0
                longest_path = 0
                critical_nodes = []
        else:
            density = 0
            longest_path = 0
            critical_nodes = []
        
        return {
            "network_structure": {
                "total_nodes": total_nodes,
                "total_links": total_links,
                "node_type_distribution": node_type_counts,
                "network_density": density,
                "supply_chain_depth": longest_path,
                "critical_nodes": critical_nodes
            },
            "service_levels": {
                "average": avg_service_level,
                "minimum": min_service_level,
                "maximum": max_service_level,
                "distribution": service_levels
            },
            "cost_structure": {
                "average_holding_cost_rate": avg_holding_cost,
                "holding_cost_distribution": holding_costs
            },
            "lead_times": {
                "average": avg_lead_time,
                "maximum": max_lead_time,
                "distribution": lead_times
            },
            "optimization_status": {
                "last_optimization": len(self.optimization_results) > 0,
                "optimization_methods_used": list(set(
                    result.optimization_method for result in self.optimization_results.values()
                ))
            }
        }