import pandas as pd
import numpy as np
from .classical.arima import ARIMAForecaster
from .classical.ets import ETSForecaster
from .intermittent.croston import CrostonForecaster

class EnsembleForecaster:
    def __init__(self):
        self.models = {
            'arima': ARIMAForecaster(),
            'ets': ETSForecaster(),
            'croston': CrostonForecaster()
        }
        self.weights = {'arima': 0.4, 'ets': 0.4, 'croston': 0.2}
    
    def fit(self, data: pd.Series):
        for model in self.models.values():
            try:
                model.fit(data)
            except:
                pass
        return self
    
    def forecast(self, steps: int) -> pd.Series:
        forecasts = {}
        for name, model in self.models.items():
            try:
                forecasts[name] = model.forecast(steps)
            except:
                forecasts[name] = pd.Series([0] * steps)
        
        ensemble_forecast = sum(
            forecasts[name] * self.weights[name] 
            for name in forecasts
        )
        return ensemble_forecast
    
    def export_results(self, forecast: pd.Series, filename: str):
        forecast.to_csv(filename, header=['forecast'])