import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CrostonForecaster:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.demand_level = None
        self.interval_level = None
    
    def fit(self, data: pd.Series):
        non_zero_demands = data[data > 0]
        intervals = []
        
        last_demand_idx = -1
        for i, value in enumerate(data):
            if value > 0:
                if last_demand_idx >= 0:
                    intervals.append(i - last_demand_idx)
                last_demand_idx = i
        
        self.demand_level = non_zero_demands.mean() if len(non_zero_demands) > 0 else 0
        self.interval_level = np.mean(intervals) if intervals else 1
        return self
    
    def forecast(self, steps: int) -> pd.Series:
        if self.demand_level is None or self.interval_level is None:
            return pd.Series([0] * steps)
        
        forecast_value = self.demand_level / self.interval_level
        return pd.Series([forecast_value] * steps)
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict:
        return {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100 if actual.sum() > 0 else 0
        }