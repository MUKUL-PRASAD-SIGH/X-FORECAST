import pandas as pd
import numpy as np
from scipy import stats

class InventoryOptimizer:
    def __init__(self):
        self.service_levels = {'high': 0.95, 'medium': 0.90, 'low': 0.85}
        self.holding_cost_rate = 0.25  # 25% annual holding cost
        
        # Multi-echelon parameters
        self.echelon_structure = {}
        self.transportation_costs = {}
        self.capacity_constraints = {}
        self.service_time_targets = {
            'critical': 24,    # hours
            'standard': 72,    # hours
            'economy': 168     # hours (1 week)
        }
    
    def calculate_safety_stock(self, forecast: pd.Series, forecast_error: float, 
                              lead_time: int, service_level: float = 0.95) -> dict:
        """Calculate optimal safety stock"""
        # Demand during lead time
        lead_time_demand = forecast.head(lead_time).sum()
        
        # Variability during lead time
        lead_time_std = forecast_error * np.sqrt(lead_time)
        
        # Safety factor for service level
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * lead_time_std
        
        return {
            'safety_stock': max(0, safety_stock),
            'lead_time_demand': lead_time_demand,
            'reorder_point': lead_time_demand + safety_stock,
            'service_level': service_level,
            'z_score': z_score
        }
    
    def calculate_economic_order_quantity(self, annual_demand: float, 
                                        ordering_cost: float, 
                                        holding_cost_per_unit: float) -> dict:
        """Calculate EOQ for optimal order quantity"""
        if holding_cost_per_unit <= 0:
            return {'eoq': annual_demand / 12, 'total_cost': float('inf')}
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        
        # Calculate total costs
        ordering_cost_annual = (annual_demand / eoq) * ordering_cost
        holding_cost_annual = (eoq / 2) * holding_cost_per_unit
        total_cost = ordering_cost_annual + holding_cost_annual
        
        return {
            'eoq': eoq,
            'ordering_cost_annual': ordering_cost_annual,
            'holding_cost_annual': holding_cost_annual,
            'total_cost': total_cost,
            'order_frequency': annual_demand / eoq
        }
    
    def optimize_inventory_policy(self, forecast: pd.Series, 
                                 forecast_error: float,
                                 lead_time: int,
                                 unit_cost: float,
                                 ordering_cost: float = 100,
                                 service_level: float = 0.95) -> dict:
        """Comprehensive inventory optimization"""
        annual_demand = forecast.sum() * (365 / len(forecast))
        holding_cost_per_unit = unit_cost * self.holding_cost_rate
        
        # Calculate EOQ
        eoq_result = self.calculate_economic_order_quantity(
            annual_demand, ordering_cost, holding_cost_per_unit
        )
        
        # Calculate safety stock
        safety_stock_result = self.calculate_safety_stock(
            forecast, forecast_error, lead_time, service_level
        )
        
        # Calculate inventory metrics
        avg_inventory = eoq_result['eoq'] / 2 + safety_stock_result['safety_stock']
        inventory_turns = annual_demand / avg_inventory if avg_inventory > 0 else 0
        
        return {
            **eoq_result,
            **safety_stock_result,
            'avg_inventory': avg_inventory,
            'inventory_turns': inventory_turns,
            'total_inventory_cost': eoq_result['total_cost'] + 
                                   (safety_stock_result['safety_stock'] * holding_cost_per_unit)
        }
    
    def simulate_inventory_scenarios(self, base_policy: dict, 
                                   demand_scenarios: list) -> pd.DataFrame:
        """Simulate inventory performance under different demand scenarios"""
        results = []
        
        for i, scenario_demand in enumerate(demand_scenarios):
            # Simulate inventory levels
            stockouts = 0
            excess_inventory = 0
            
            current_inventory = base_policy['reorder_point']
            
            for daily_demand in scenario_demand:
                if current_inventory < daily_demand:
                    stockouts += daily_demand - current_inventory
                    current_inventory = 0
                else:
                    current_inventory -= daily_demand
                
                # Reorder when hitting reorder point
                if current_inventory <= base_policy['reorder_point']:
                    current_inventory += base_policy['eoq']
            
            # Calculate scenario metrics
            service_level_achieved = 1 - (stockouts / sum(scenario_demand))
            avg_inventory = current_inventory / len(scenario_demand)
            
            results.append({
                'scenario': i + 1,
                'stockouts': stockouts,
                'service_level_achieved': service_level_achieved,
                'avg_inventory': avg_inventory,
                'total_cost': avg_inventory * base_policy.get('holding_cost_per_unit', 1)
            })
        
        return pd.DataFrame(results)
    
    def generate_replenishment_plan(self, forecast: pd.Series, 
                                   current_inventory: float,
                                   optimal_policy: dict) -> pd.DataFrame:
        """Generate replenishment plan"""
        plan = []
        inventory_level = current_inventory
        
        for day, demand in enumerate(forecast):
            # Check if reorder needed
            if inventory_level <= optimal_policy['reorder_point']:
                order_qty = optimal_policy['eoq']
                plan.append({
                    'day': day,
                    'action': 'ORDER',
                    'quantity': order_qty,
                    'inventory_before': inventory_level,
                    'inventory_after': inventory_level + order_qty - demand
                })
                inventory_level += order_qty
            
            inventory_level -= demand
            
            if day % 7 == 0:  # Weekly status
                plan.append({
                    'day': day,
                    'action': 'STATUS',
                    'quantity': 0,
                    'inventory_before': inventory_level + demand,
                    'inventory_after': inventory_level
                })
        
        return pd.DataFrame(plan)    
def optimize_multi_echelon_inventory(self, demand_forecasts: Dict[str, pd.Series],
                                       echelon_structure: Dict[str, Any],
                                       service_targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Multi-echelon inventory optimization
        Optimizes inventory placement across upstream and downstream locations
        """
        
        optimization_results = {}
        
        # Define echelon levels
        echelons = {
            'distribution_center': {'level': 1, 'serves': ['regional_warehouse']},
            'regional_warehouse': {'level': 2, 'serves': ['local_store']},
            'local_store': {'level': 3, 'serves': ['customer']}
        }
        
        # For each SKU, optimize inventory placement
        for forecast_key, forecast in demand_forecasts.items():
            try:
                # Parse forecast key
                parts = forecast_key.split('_')
                sku = parts[0] if len(parts) > 0 else forecast_key
                location = parts[1] if len(parts) > 1 else 'default'
                
                # Calculate optimal inventory for each echelon
                echelon_inventory = self._optimize_echelon_placement(
                    sku, location, forecast, echelons, service_targets
                )
                
                optimization_results[forecast_key] = echelon_inventory
                
            except Exception as e:
                logger.warning(f"Multi-echelon optimization failed for {forecast_key}: {e}")
        
        return optimization_results
    
    def _optimize_echelon_placement(self, sku: str, location: str, forecast: pd.Series,
                                  echelons: Dict[str, Any], service_targets: Dict[str, float]) -> Dict[str, Any]:
        """Optimize inventory placement for a specific SKU across echelons"""
        
        # Calculate demand statistics
        mean_demand = forecast.mean()
        demand_std = forecast.std()
        
        # Get service target for this SKU (assume 'standard' if not specified)
        service_target = service_targets.get(f"{sku}_{location}", 0.95)
        
        # Calculate optimal inventory for each echelon
        echelon_results = {}
        
        for echelon_name, echelon_info in echelons.items():
            level = echelon_info['level']
            
            # Lead time increases with echelon level
            lead_time = level * 2  # days
            
            # Calculate safety stock for this echelon
            lead_time_demand = mean_demand * lead_time
            lead_time_std = demand_std * np.sqrt(lead_time)
            
            # Service level adjustment based on echelon (upstream can have lower service)
            adjusted_service_level = service_target - (0.02 * (level - 1))  # Reduce by 2% per upstream level
            adjusted_service_level = max(0.85, adjusted_service_level)
            
            z_score = stats.norm.ppf(adjusted_service_level)
            safety_stock = z_score * lead_time_std
            
            # Reorder point
            reorder_point = lead_time_demand + safety_stock
            
            # Calculate holding costs (higher for upstream due to longer holding)
            holding_cost_multiplier = 1.0 + (0.1 * (level - 1))
            adjusted_holding_cost = self.holding_cost_rate * holding_cost_multiplier
            
            echelon_results[echelon_name] = {
                'safety_stock': max(0, safety_stock),
                'reorder_point': max(0, reorder_point),
                'service_level': adjusted_service_level,
                'lead_time_days': lead_time,
                'holding_cost_rate': adjusted_holding_cost,
                'recommended_stock_level': max(0, reorder_point + safety_stock)
            }
        
        # Calculate total system cost
        total_system_cost = sum(
            result['recommended_stock_level'] * result['holding_cost_rate'] * mean_demand
            for result in echelon_results.values()
        )
        
        # Calculate service level achievement
        system_service_level = min(result['service_level'] for result in echelon_results.values())
        
        return {
            'sku': sku,
            'location': location,
            'echelon_inventory': echelon_results,
            'total_system_cost': total_system_cost,
            'system_service_level': system_service_level,
            'optimization_date': datetime.now().isoformat()
        }
    
    def calculate_otif_impact(self, inventory_policy: Dict[str, Any], 
                            demand_scenarios: List[pd.Series]) -> Dict[str, float]:
        """
        Calculate On-Time-In-Full (OTIF) performance under different scenarios
        """
        
        otif_results = {}
        
        for scenario_idx, scenario_demand in enumerate(demand_scenarios):
            scenario_name = f"scenario_{scenario_idx + 1}"
            
            # Simulate inventory performance
            stockouts = 0
            total_orders = len(scenario_demand)
            on_time_deliveries = 0
            in_full_deliveries = 0
            
            current_inventory = inventory_policy.get('reorder_point', 0)
            
            for day, daily_demand in enumerate(scenario_demand):
                # Check if we can fulfill demand
                if current_inventory >= daily_demand:
                    # Full delivery possible
                    in_full_deliveries += 1
                    current_inventory -= daily_demand
                    
                    # Check if delivery is on time (assume 2-day lead time)
                    if current_inventory > inventory_policy.get('safety_stock', 0):
                        on_time_deliveries += 1
                else:
                    # Stockout
                    stockouts += 1
                    current_inventory = max(0, current_inventory - daily_demand)
                
                # Reorder when hitting reorder point
                if current_inventory <= inventory_policy.get('reorder_point', 0):
                    order_quantity = inventory_policy.get('order_quantity', daily_demand * 7)
                    # Simulate lead time (inventory arrives after lead time)
                    current_inventory += order_quantity
            
            # Calculate OTIF metrics
            in_full_rate = in_full_deliveries / total_orders if total_orders > 0 else 0
            on_time_rate = on_time_deliveries / total_orders if total_orders > 0 else 0
            otif_rate = min(in_full_rate, on_time_rate)  # OTIF is the minimum of both
            
            otif_results[scenario_name] = {
                'otif_rate': otif_rate,
                'on_time_rate': on_time_rate,
                'in_full_rate': in_full_rate,
                'stockout_rate': stockouts / total_orders if total_orders > 0 else 0,
                'service_level_achieved': 1 - (stockouts / total_orders) if total_orders > 0 else 1
            }
        
        # Calculate average OTIF across scenarios
        avg_otif = np.mean([result['otif_rate'] for result in otif_results.values()])
        avg_service_level = np.mean([result['service_level_achieved'] for result in otif_results.values()])
        
        return {
            'scenario_results': otif_results,
            'average_otif_rate': avg_otif,
            'average_service_level': avg_service_level,
            'otif_variability': np.std([result['otif_rate'] for result in otif_results.values()]),
            'recommendation': 'increase_safety_stock' if avg_otif < 0.95 else 'maintain_current_policy'
        }