import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class NPIEngine:
    def __init__(self):
        self.similarity_threshold = 0.7
        self.scaler = StandardScaler()
        self.product_profiles = {}
    
    def create_product_profile(self, data: pd.DataFrame) -> dict:
        """Create product similarity profile"""
        profile = {
            'avg_demand': data['demand'].mean(),
            'demand_std': data['demand'].std(),
            'seasonality_strength': self._calculate_seasonality(data),
            'trend_strength': self._calculate_trend(data),
            'volatility': data['demand'].std() / data['demand'].mean(),
            'launch_velocity': data['demand'].head(30).mean() if len(data) >= 30 else data['demand'].mean()
        }
        return profile
    
    def _calculate_seasonality(self, data: pd.DataFrame) -> float:
        """Calculate seasonality strength"""
        if len(data) < 14:
            return 0.0
        weekly_pattern = data['demand'].rolling(7).mean().std()
        return min(weekly_pattern / data['demand'].mean(), 1.0)
    
    def _calculate_trend(self, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        if len(data) < 10:
            return 0.0
        x = np.arange(len(data))
        trend = np.polyfit(x, data['demand'], 1)[0]
        return min(abs(trend) / data['demand'].mean(), 1.0)
    
    def find_similar_products(self, new_product_profile: dict, existing_profiles: dict) -> list:
        """Find similar products for NPI forecasting"""
        similarities = []
        
        for product_id, profile in existing_profiles.items():
            similarity = self._calculate_similarity(new_product_profile, profile)
            if similarity > self.similarity_threshold:
                similarities.append((product_id, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_similarity(self, profile1: dict, profile2: dict) -> float:
        """Calculate similarity between product profiles"""
        features1 = np.array(list(profile1.values())).reshape(1, -1)
        features2 = np.array(list(profile2.values())).reshape(1, -1)
        
        # Normalize features
        combined = np.vstack([features1, features2])
        normalized = self.scaler.fit_transform(combined)
        
        similarity = cosine_similarity(normalized[0:1], normalized[1:2])[0][0]
        return max(0, similarity)
    
    def generate_npi_forecast(self, similar_products_data: dict, forecast_horizon: int) -> pd.Series:
        """Generate NPI forecast using similar products"""
        if not similar_products_data:
            return pd.Series([0] * forecast_horizon)
        
        weighted_forecast = np.zeros(forecast_horizon)
        total_weight = 0
        
        for product_id, (similarity, data) in similar_products_data.items():
            if len(data) >= forecast_horizon:
                early_demand = data['demand'].head(forecast_horizon).values
                weighted_forecast += early_demand * similarity
                total_weight += similarity
        
        if total_weight > 0:
            weighted_forecast /= total_weight
        
        return pd.Series(weighted_forecast)