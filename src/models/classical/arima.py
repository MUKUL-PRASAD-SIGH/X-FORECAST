import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ARIMAForecaster:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, data: pd.Series):
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
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