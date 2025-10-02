import unittest
import pandas as pd
import numpy as np
from src.models.classical.arima import ARIMAForecaster
from src.models.classical.ets import ETSForecaster
from src.models.intermittent.croston import CrostonForecaster
from src.models.ensemble import EnsembleForecaster
from src.feature_store.feature_engineer import FeatureEngineer

class TestModels(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.Series([10, 12, 13, 12, 15, 16, 18, 20, 22, 25])
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.sample_df = pd.DataFrame({
            'date': dates,
            'demand': self.sample_data,
            'product_id': 'SKU001'
        })
    
    def test_arima_forecast(self):
        model = ARIMAForecaster()
        model.fit(self.sample_data)
        forecast = model.forecast(3)
        self.assertEqual(len(forecast), 3)
    
    def test_ets_forecast(self):
        model = ETSForecaster()
        model.fit(self.sample_data)
        forecast = model.forecast(3)
        self.assertEqual(len(forecast), 3)
    
    def test_croston_forecast(self):
        model = CrostonForecaster()
        model.fit(self.sample_data)
        forecast = model.forecast(3)
        self.assertEqual(len(forecast), 3)
    
    def test_ensemble_forecast(self):
        model = EnsembleForecaster()
        model.fit(self.sample_data)
        forecast = model.forecast(3)
        self.assertEqual(len(forecast), 3)
    
    def test_feature_engineering(self):
        engineer = FeatureEngineer()
        features_df = engineer.create_features(self.sample_df)
        
        # Check that features were created
        feature_cols = [col for col in features_df.columns if col not in ['date', 'demand', 'product_id']]
        self.assertGreater(len(feature_cols), 5)  # Should have multiple features
        
        # Check specific features exist
        self.assertIn('demand_ma_7', features_df.columns)
        self.assertIn('day_of_week', features_df.columns)
    
    def test_enhanced_ensemble(self):
        # Test ensemble with feature engineering
        engineer = FeatureEngineer()
        features_df = engineer.create_features(self.sample_df)
        
        forecaster = EnsembleForecaster(use_ml=False)  # Don't require ML libs for test
        forecaster.fit(self.sample_data, features_df)
        forecast = forecaster.forecast(3)
        
        self.assertEqual(len(forecast), 3)
        self.assertIsInstance(forecast, pd.Series)

if __name__ == '__main__':
    unittest.main()