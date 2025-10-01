import pandas as pd
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ETSForecaster:
    def __init__(self, trend='add', seasonal='add', seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
    
    def fit(self, data: pd.Series):
        model = ETSModel(data, trend=self.trend, seasonal=self.seasonal, 
                        seasonal_periods=self.seasonal_periods)
        self.fitted_model = model.fit()
        return self
    
    def forecast(self, steps: int) -> pd.Series:
        forecast = self.fitted_model.forecast(steps=steps)
        return pd.Series(forecast)
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict:
        return {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }