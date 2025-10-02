import unittest
import pandas as pd
import numpy as np
from src.models.advanced.npi_engine import NPIEngine
from src.models.advanced.promotion_engine import PromotionEngine
from src.models.advanced.inventory_optimizer import InventoryOptimizer

class TestPhase3(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'demand': np.random.randint(50, 200, 100),
            'product_id': 'SKU001'
        })
    
    def test_npi_engine(self):
        engine = NPIEngine()
        profile = engine.create_product_profile(self.sample_data)
        
        self.assertIn('avg_demand', profile)
        self.assertIn('volatility', profile)
        self.assertGreater(profile['avg_demand'], 0)
    
    def test_promotion_engine(self):
        engine = PromotionEngine()
        promo_data = engine.detect_promotions(self.sample_data)
        
        self.assertIn('promotion_flag', promo_data.columns)
        self.assertIn('baseline', promo_data.columns)
    
    def test_inventory_optimizer(self):
        optimizer = InventoryOptimizer()
        forecast = pd.Series([100, 105, 110, 95, 120])
        
        safety_stock = optimizer.calculate_safety_stock(
            forecast, forecast_error=10, lead_time=7
        )
        
        self.assertIn('safety_stock', safety_stock)
        self.assertIn('reorder_point', safety_stock)
        self.assertGreater(safety_stock['safety_stock'], 0)
    
    def test_eoq_calculation(self):
        optimizer = InventoryOptimizer()
        eoq_result = optimizer.calculate_economic_order_quantity(
            annual_demand=10000, ordering_cost=100, holding_cost_per_unit=5
        )
        
        self.assertIn('eoq', eoq_result)
        self.assertIn('total_cost', eoq_result)
        self.assertGreater(eoq_result['eoq'], 0)

if __name__ == '__main__':
    unittest.main()