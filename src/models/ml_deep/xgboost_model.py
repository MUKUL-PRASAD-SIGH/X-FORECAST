import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class XGBoostForecaster:
    def __init__(self, n_estimators=100, max_depth=6):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.feature_cols = None
    
    def prepare_data(self, data: pd.DataFrame, target_col='demand'):
        """Prepare data for ML training"""
        # Exclude non-numeric columns and target/date columns
        feature_cols = []
        for col in data.columns:
            if col not in [target_col, 'date', 'product_id']:
                # Only include numeric columns
                if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
        
        X = data[feature_cols].fillna(0)
        y = data[target_col]
        self.feature_cols = feature_cols
        return X, y
    
    def fit(self, data: pd.DataFrame, target_col='demand'):
        X, y = self.prepare_data(data, target_col)
        self.model.fit(X, y)
        return self
    
    def forecast(self, data: pd.DataFrame, steps: int = 1) -> pd.Series:
        if self.feature_cols is None:
            raise ValueError("Model not fitted")
        
        X = data[self.feature_cols].fillna(0)
        predictions = self.model.predict(X)
        return pd.Series(predictions[-steps:])
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict:
        return {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }
    
    def get_feature_importance(self) -> dict:
        if self.feature_cols is None:
            return {}
        return dict(zip(self.feature_cols, self.model.feature_importances_))