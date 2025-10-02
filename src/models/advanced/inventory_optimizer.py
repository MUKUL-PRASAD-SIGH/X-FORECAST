import pandas as pd
import numpy as np
from scipy import stats

class InventoryOptimizer:
    def __init__(self):
        self.service_levels = {'high': 0.95, 'medium': 0.90, 'low': 0.85}
        self.holding_cost_rate = 0.25  # 25% annual holding cost
    
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