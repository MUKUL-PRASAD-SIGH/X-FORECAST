import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PromotionEngine:
    def __init__(self):
        self.promotion_effects = {}
        self.baseline_models = {}
    
    def detect_promotions(self, data: pd.DataFrame, threshold_multiplier=1.5) -> pd.DataFrame:
        """Detect promotion periods in historical data"""
        data = data.copy()
        
        # Calculate rolling baseline
        data['baseline'] = data['demand'].rolling(14, center=True).median()
        data['promotion_flag'] = (data['demand'] > data['baseline'] * threshold_multiplier).astype(int)
        
        # Group consecutive promotion days
        data['promo_group'] = (data['promotion_flag'] != data['promotion_flag'].shift()).cumsum()
        
        return data
    
    def calculate_promotion_uplift(self, data: pd.DataFrame) -> dict:
        """Calculate promotion uplift effects"""
        promo_data = data[data['promotion_flag'] == 1]
        baseline_data = data[data['promotion_flag'] == 0]
        
        if len(promo_data) == 0 or len(baseline_data) == 0:
            return {'uplift_factor': 1.0, 'duration_effect': 0}
        
        avg_promo_demand = promo_data['demand'].mean()
        avg_baseline_demand = baseline_data['demand'].mean()
        
        uplift_factor = avg_promo_demand / avg_baseline_demand if avg_baseline_demand > 0 else 1.0
        
        # Calculate duration effects
        promo_groups = promo_data.groupby('promo_group')
        avg_duration = promo_groups.size().mean()
        
        return {
            'uplift_factor': uplift_factor,
            'avg_duration': avg_duration,
            'peak_uplift': promo_data['demand'].max() / avg_baseline_demand if avg_baseline_demand > 0 else 1.0,
            'total_incremental': (avg_promo_demand - avg_baseline_demand) * len(promo_data)
        }
    
    def apply_promotion_forecast(self, baseline_forecast: pd.Series, 
                                promotion_schedule: pd.DataFrame) -> pd.Series:
        """Apply promotion effects to baseline forecast"""
        enhanced_forecast = baseline_forecast.copy()
        
        for _, promo in promotion_schedule.iterrows():
            start_idx = promo.get('start_day', 0)
            duration = promo.get('duration', 1)
            uplift = promo.get('uplift_factor', 1.5)
            
            end_idx = min(start_idx + duration, len(enhanced_forecast))
            
            if start_idx < len(enhanced_forecast):
                enhanced_forecast.iloc[start_idx:end_idx] *= uplift
        
        return enhanced_forecast
    
    def optimize_promotion_calendar(self, baseline_forecast: pd.Series, 
                                   budget_constraint: float) -> pd.DataFrame:
        """Optimize promotion calendar for maximum impact"""
        # Simple greedy optimization
        forecast_periods = len(baseline_forecast)
        promotion_schedule = []
        
        # Find periods with highest baseline demand for promotion
        sorted_periods = baseline_forecast.sort_values(ascending=False)
        
        remaining_budget = budget_constraint
        promo_cost = budget_constraint / 4  # Assume 4 promotions max
        
        for period_idx in sorted_periods.head(4).index:
            if remaining_budget >= promo_cost:
                promotion_schedule.append({
                    'start_day': period_idx,
                    'duration': 3,
                    'uplift_factor': 1.8,
                    'cost': promo_cost,
                    'expected_incremental': baseline_forecast.iloc[period_idx] * 0.8 * 3
                })
                remaining_budget -= promo_cost
        
        return pd.DataFrame(promotion_schedule)
    
    def calculate_roi(self, baseline_forecast: pd.Series, 
                     promoted_forecast: pd.Series, 
                     promotion_cost: float, 
                     unit_margin: float) -> dict:
        """Calculate promotion ROI"""
        incremental_units = (promoted_forecast - baseline_forecast).sum()
        incremental_revenue = incremental_units * unit_margin
        roi = (incremental_revenue - promotion_cost) / promotion_cost if promotion_cost > 0 else 0
        
        return {
            'incremental_units': incremental_units,
            'incremental_revenue': incremental_revenue,
            'promotion_cost': promotion_cost,
            'roi': roi,
            'payback_ratio': incremental_revenue / promotion_cost if promotion_cost > 0 else 0
        }