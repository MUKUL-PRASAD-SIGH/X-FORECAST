import unittest
import pandas as pd
import numpy as np
from src.models.classical.arima import ARIMAForecaster
from src.models.classical.ets import ETSForecaster
from src.models.intermittent.croston import CrostonForecaster

class TestModels(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.Series([10, 12, 13, 12, 15, 16, 18, 20, 22, 25])
    
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

if __name__ == '__main__':
    unittest.main()