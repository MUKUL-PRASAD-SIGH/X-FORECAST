import pandas as pd
import numpy as np
from typing import Dict, List

class FeatureEngineer:
    def __init__(self):
        self.features = {}
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        df = data.copy()
        
        # Rolling statistics
        df['demand_ma_7'] = df['demand'].rolling(7).mean()
        df['demand_ma_30'] = df['demand'].rolling(30).mean()
        df['demand_std_7'] = df['demand'].rolling(7).std()
        df['demand_cv'] = df['demand_std_7'] / df['demand_ma_7']
        
        # Lag features
        df['demand_lag_1'] = df['demand'].shift(1)
        df['demand_lag_7'] = df['demand'].shift(7)
        df['demand_lag_30'] = df['demand'].shift(30)
        
        # Seasonality features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Growth features
        df['demand_growth_7'] = df['demand'].pct_change(7)
        df['demand_growth_30'] = df['demand'].pct_change(30)
        
        # Volatility features
        df['demand_volatility'] = df['demand'].rolling(14).std()
        
        return df.fillna(0)
    
    def get_feature_importance(self, features: List[str]) -> Dict[str, float]:
        """Calculate feature importance scores"""
        return {f: np.random.random() for f in features}  # Placeholder
    
    def save_features(self, df: pd.DataFrame, filename: str):
        """Save features to CSV"""
        df.to_csv(f'./data/processed/{filename}', index=False)