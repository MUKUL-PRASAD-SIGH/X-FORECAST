import pandas as pd
import numpy as np
from .classical.arima import ARIMAForecaster
from .classical.ets import ETSForecaster
from .intermittent.croston import CrostonForecaster

try:
    from .ml_deep.xgboost_model import XGBoostForecaster
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from .ml_deep.lstm_model import LSTMForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

class EnsembleForecaster:
    def __init__(self, use_ml=False):
        self.models = {
            'arima': ARIMAForecaster(),
            'ets': ETSForecaster(),
            'croston': CrostonForecaster()
        }
        
        if use_ml and XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBoostForecaster()
            
        if use_ml and LSTM_AVAILABLE:
            self.models['lstm'] = LSTMForecaster()
        
        # Adjust weights based on available models
        num_models = len(self.models)
        base_weight = 1.0 / num_models
        self.weights = {name: base_weight for name in self.models.keys()}
    
    def fit(self, data, features_df=None):
        """Fit all models. Use features_df for ML models if available"""
        for name, model in self.models.items():
            try:
                if name in ['xgboost'] and features_df is not None:
                    model.fit(features_df)
                else:
                    if hasattr(data, 'values'):
                        model.fit(data)
                    else:
                        model.fit(pd.Series(data))
            except Exception as e:
                print(f"Warning: {name} model failed to fit: {e}")
        return self
    
    def forecast(self, steps: int, features_df=None) -> pd.Series:
        """Generate ensemble forecast"""
        forecasts = {}
        for name, model in self.models.items():
            try:
                if name in ['xgboost'] and features_df is not None:
                    forecasts[name] = model.forecast(features_df, steps)
                else:
                    forecasts[name] = model.forecast(steps)
            except Exception as e:
                print(f"Warning: {name} model failed to forecast: {e}")
                forecasts[name] = pd.Series([0] * steps)
        
        ensemble_forecast = sum(
            forecasts[name] * self.weights[name] 
            for name in forecasts if name in forecasts
        )
        return ensemble_forecast
    
    def export_results(self, forecast: pd.Series, filename: str):
        forecast.to_csv(filename, header=['forecast'])
    
    def get_model_weights(self) -> dict:
        """Return current model weights"""
        return self.weights.copy()
    
    def set_model_weights(self, weights: dict):
        """Set custom model weights"""
        for name in weights:
            if name in self.models:
                self.weights[name] = weights[name]