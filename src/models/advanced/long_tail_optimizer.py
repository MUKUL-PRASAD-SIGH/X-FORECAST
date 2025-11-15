"""
Long-Tail Optimization System
Specialized forecasting for sparse and intermittent demand items
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SparsityLevel(Enum):
    DENSE = "dense"           # Regular demand pattern
    SPARSE = "sparse"         # Some zero periods
    VERY_SPARSE = "very_sparse"  # Many zero periods
    INTERMITTENT = "intermittent"  # Irregular, sporadic demand

class ForecastMethod(Enum):
    STANDARD = "standard"
    CROSTON = "croston"
    TSB = "tsb"
    SBA = "sba"
    POOLED = "pooled"
    HIERARCHICAL_BORROWING = "hierarchical_borrowing"

@dataclass
class SparsityMetrics:
    """Metrics for characterizing demand sparsity"""
    sku: str
    category: str
    
    # Basic sparsity metrics
    zero_periods: int
    total_periods: int
    zero_ratio: float
    
    # Advanced sparsity metrics
    adi: float  # Average Demand Interval
    cv_squared: float  # Coefficient of Variation squared
    
    # Classification
    sparsity_level: SparsityLevel
    recommended_method: ForecastMethod
    
    # Additional metrics
    demand_variability: float
    seasonality_strength: float
    trend_strength: float

@dataclass
class ClusterResult:
    """Result of sparse item clustering"""
    cluster_id: int
    cluster_name: str
    items: List[str]
    cluster_characteristics: Dict[str, float]
    representative_item: str
    pooling_weights: Dict[str, float]

@dataclass
class IntermittentForecastResult:
    """Result from intermittent demand forecasting"""
    sku: str
    method_used: ForecastMethod
    forecast_values: List[float]
    forecast_intervals: List[Tuple[float, float]]
    method_parameters: Dict[str, float]
    forecast_accuracy_metrics: Dict[str, float]

@dataclass
class HierarchicalBorrowingResult:
    """Result from hierarchical borrowing system"""
    target_sku: str
    similar_items: List[Tuple[str, float]]  # (sku, similarity_score)
    borrowed_patterns: Dict[str, float]
    pooled_forecast: List[float]
    confidence_score: float

class LongTailOptimizer:
    """
    Long-tail optimization system for sparse and intermittent demand forecasting
    """
    
    def __init__(self):
        self.sparsity_metrics = {}
        self.clusters = {}
        self.similarity_matrix = None
        self.pooling_weights = {}
        self.scaler = StandardScaler()
        
        # Sparsity classification thresholds
        self.sparsity_thresholds = {
            'zero_ratio': {'sparse': 0.2, 'very_sparse': 0.5, 'intermittent': 0.7},
            'adi': {'sparse': 1.5, 'very_sparse': 3.0, 'intermittent': 6.0},
            'cv_squared': {'sparse': 1.0, 'very_sparse': 2.0, 'intermittent': 4.0}
        }
    
    def analyze_sparsity(self, data: pd.DataFrame) -> Dict[str, SparsityMetrics]:
        """
        Analyze sparsity patterns for all items
        Expected columns: sku, date, demand, category
        """
        logger.info("Analyzing demand sparsity patterns...")
        
        sparsity_results = {}
        
        # Group by SKU
        for sku, sku_data in data.groupby('sku'):
            metrics = self._calculate_sparsity_metrics(sku, sku_data)
            sparsity_results[sku] = metrics
        
        self.sparsity_metrics = sparsity_results
        logger.info(f"Analyzed sparsity for {len(sparsity_results)} items")
        
        return sparsity_results
    
    def _calculate_sparsity_metrics(self, sku: str, sku_data: pd.DataFrame) -> SparsityMetrics:
        """Calculate sparsity metrics for a single SKU"""
        
        # Sort by date
        sku_data = sku_data.sort_values('date')
        demand_series = sku_data['demand'].values
        
        # Basic metrics
        total_periods = len(demand_series)
        zero_periods = np.sum(demand_series == 0)
        zero_ratio = zero_periods / total_periods if total_periods > 0 else 1.0
        
        # Average Demand Interval (ADI)
        non_zero_indices = np.where(demand_series > 0)[0]
        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
            adi = np.mean(intervals) if len(intervals) > 0 else total_periods
        else:
            adi = total_periods
        
        # Coefficient of Variation squared (CV²)
        non_zero_demand = demand_series[demand_series > 0]
        if len(non_zero_demand) > 1:
            cv_squared = (np.std(non_zero_demand) / np.mean(non_zero_demand)) ** 2
        else:
            cv_squared = 0.0
        
        # Demand variability
        demand_variability = np.std(demand_series) / np.mean(demand_series) if np.mean(demand_series) > 0 else 0
        
        # Simple seasonality and trend detection
        seasonality_strength = self._detect_seasonality(demand_series)
        trend_strength = self._detect_trend(demand_series)
        
        # Classify sparsity level
        sparsity_level = self._classify_sparsity_level(zero_ratio, adi, cv_squared)
        
        # Recommend forecasting method
        recommended_method = self._recommend_forecast_method(sparsity_level, adi, cv_squared)
        
        # Get category
        category = sku_data['category'].iloc[0] if 'category' in sku_data.columns else 'unknown'
        
        return SparsityMetrics(
            sku=sku,
            category=category,
            zero_periods=zero_periods,
            total_periods=total_periods,
            zero_ratio=zero_ratio,
            adi=adi,
            cv_squared=cv_squared,
            sparsity_level=sparsity_level,
            recommended_method=recommended_method,
            demand_variability=demand_variability,
            seasonality_strength=seasonality_strength,
            trend_strength=trend_strength
        )
    
    def _detect_seasonality(self, demand_series: np.ndarray) -> float:
        """Simple seasonality detection"""
        if len(demand_series) < 24:  # Need at least 2 years of monthly data
            return 0.0
        
        try:
            # Simple approach: compare variance within seasons vs between seasons
            monthly_means = []
            for month in range(12):
                month_data = demand_series[month::12]
                if len(month_data) > 0:
                    monthly_means.append(np.mean(month_data))
            
            if len(monthly_means) >= 12:
                overall_mean = np.mean(monthly_means)
                seasonal_variance = np.var(monthly_means)
                return seasonal_variance / (overall_mean ** 2) if overall_mean > 0 else 0.0
            
            return 0.0
        except:
            return 0.0
    
    def _detect_trend(self, demand_series: np.ndarray) -> float:
        """Simple trend detection"""
        if len(demand_series) < 6:
            return 0.0
        
        try:
            # Linear trend using least squares
            x = np.arange(len(demand_series))
            coeffs = np.polyfit(x, demand_series, 1)
            trend_slope = coeffs[0]
            
            # Normalize by mean demand
            mean_demand = np.mean(demand_series)
            normalized_trend = abs(trend_slope) / mean_demand if mean_demand > 0 else 0
            
            return min(normalized_trend, 1.0)  # Cap at 1.0
        except:
            return 0.0
    
    def _classify_sparsity_level(self, zero_ratio: float, adi: float, cv_squared: float) -> SparsityLevel:
        """Classify sparsity level based on metrics"""
        
        # Use multiple criteria
        criteria_scores = {
            SparsityLevel.DENSE: 0,
            SparsityLevel.SPARSE: 0,
            SparsityLevel.VERY_SPARSE: 0,
            SparsityLevel.INTERMITTENT: 0
        }
        
        # Zero ratio criteria
        if zero_ratio < self.sparsity_thresholds['zero_ratio']['sparse']:
            criteria_scores[SparsityLevel.DENSE] += 1
        elif zero_ratio < self.sparsity_thresholds['zero_ratio']['very_sparse']:
            criteria_scores[SparsityLevel.SPARSE] += 1
        elif zero_ratio < self.sparsity_thresholds['zero_ratio']['intermittent']:
            criteria_scores[SparsityLevel.VERY_SPARSE] += 1
        else:
            criteria_scores[SparsityLevel.INTERMITTENT] += 1
        
        # ADI criteria
        if adi < self.sparsity_thresholds['adi']['sparse']:
            criteria_scores[SparsityLevel.DENSE] += 1
        elif adi < self.sparsity_thresholds['adi']['very_sparse']:
            criteria_scores[SparsityLevel.SPARSE] += 1
        elif adi < self.sparsity_thresholds['adi']['intermittent']:
            criteria_scores[SparsityLevel.VERY_SPARSE] += 1
        else:
            criteria_scores[SparsityLevel.INTERMITTENT] += 1
        
        # CV² criteria
        if cv_squared < self.sparsity_thresholds['cv_squared']['sparse']:
            criteria_scores[SparsityLevel.DENSE] += 1
        elif cv_squared < self.sparsity_thresholds['cv_squared']['very_sparse']:
            criteria_scores[SparsityLevel.SPARSE] += 1
        elif cv_squared < self.sparsity_thresholds['cv_squared']['intermittent']:
            criteria_scores[SparsityLevel.VERY_SPARSE] += 1
        else:
            criteria_scores[SparsityLevel.INTERMITTENT] += 1
        
        # Return level with highest score
        return max(criteria_scores.items(), key=lambda x: x[1])[0]
    
    def _recommend_forecast_method(self, sparsity_level: SparsityLevel, adi: float, cv_squared: float) -> ForecastMethod:
        """Recommend appropriate forecasting method"""
        
        if sparsity_level == SparsityLevel.DENSE:
            return ForecastMethod.STANDARD
        elif sparsity_level == SparsityLevel.SPARSE:
            return ForecastMethod.POOLED if cv_squared > 2.0 else ForecastMethod.STANDARD
        elif sparsity_level == SparsityLevel.VERY_SPARSE:
            return ForecastMethod.TSB if cv_squared > 1.5 else ForecastMethod.CROSTON
        else:  # INTERMITTENT
            if adi > 8:
                return ForecastMethod.HIERARCHICAL_BORROWING
            elif cv_squared > 3.0:
                return ForecastMethod.SBA
            else:
                return ForecastMethod.TSB
    
    def cluster_sparse_items(self, data: pd.DataFrame) -> Dict[int, ClusterResult]:
        """
        Cluster sparse items by demand patterns for pooling
        """
        logger.info("Clustering sparse items...")
        
        # Filter sparse items
        sparse_items = [sku for sku, metrics in self.sparsity_metrics.items() 
                       if metrics.sparsity_level in [SparsityLevel.SPARSE, SparsityLevel.VERY_SPARSE, SparsityLevel.INTERMITTENT]]
        
        if len(sparse_items) < 3:
            logger.warning("Not enough sparse items for clustering")
            return {}
        
        # Prepare feature matrix
        feature_matrix = self._prepare_clustering_features(data, sparse_items)
        
        if feature_matrix.empty:
            return {}
        
        # Perform clustering
        n_clusters = min(max(2, len(sparse_items) // 5), 10)  # Adaptive number of clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix.values)
        
        # Create cluster results
        clusters = {}
        
        for cluster_id in range(n_clusters):
            cluster_items = [sparse_items[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if len(cluster_items) > 0:
                cluster_result = self._create_cluster_result(cluster_id, cluster_items, feature_matrix, data)
                clusters[cluster_id] = cluster_result
        
        self.clusters = clusters
        logger.info(f"Created {len(clusters)} clusters for sparse items")
        
        return clusters
    
    def _prepare_clustering_features(self, data: pd.DataFrame, sparse_items: List[str]) -> pd.DataFrame:
        """Prepare features for clustering sparse items"""
        
        features = []
        
        for sku in sparse_items:
            sku_data = data[data['sku'] == sku].sort_values('date')
            
            if len(sku_data) < 6:  # Need minimum data
                continue
            
            demand_series = sku_data['demand'].values
            metrics = self.sparsity_metrics[sku]
            
            # Feature vector
            feature_dict = {
                'sku': sku,
                'zero_ratio': metrics.zero_ratio,
                'adi': metrics.adi,
                'cv_squared': metrics.cv_squared,
                'demand_variability': metrics.demand_variability,
                'seasonality_strength': metrics.seasonality_strength,
                'trend_strength': metrics.trend_strength,
                'mean_demand': np.mean(demand_series[demand_series > 0]) if np.any(demand_series > 0) else 0,
                'max_demand': np.max(demand_series),
                'demand_frequency': np.sum(demand_series > 0) / len(demand_series)
            }
            
            # Add category features if available
            if 'category' in sku_data.columns:
                category = sku_data['category'].iloc[0]
                feature_dict['category'] = category
            
            features.append(feature_dict)
        
        feature_df = pd.DataFrame(features)
        
        if feature_df.empty:
            return feature_df
        
        # Select numeric features for clustering
        numeric_features = ['zero_ratio', 'adi', 'cv_squared', 'demand_variability', 
                           'seasonality_strength', 'trend_strength', 'mean_demand', 
                           'max_demand', 'demand_frequency']
        
        clustering_features = feature_df[numeric_features].fillna(0)
        
        # Normalize features
        clustering_features_scaled = pd.DataFrame(
            self.scaler.fit_transform(clustering_features),
            columns=clustering_features.columns,
            index=clustering_features.index
        )
        
        return clustering_features_scaled
    
    def _create_cluster_result(self, cluster_id: int, cluster_items: List[str], 
                              feature_matrix: pd.DataFrame, data: pd.DataFrame) -> ClusterResult:
        """Create cluster result with characteristics and pooling weights"""
        
        # Calculate cluster characteristics
        cluster_indices = [i for i, sku in enumerate(feature_matrix.index) if feature_matrix.iloc[i]['sku'] in cluster_items]
        cluster_features = feature_matrix.iloc[cluster_indices]
        
        characteristics = {}
        for col in cluster_features.columns:
            if col != 'sku':
                characteristics[col] = cluster_features[col].mean()
        
        # Find representative item (closest to cluster center)
        cluster_center = cluster_features.drop('sku', axis=1).mean()
        distances = []
        
        for idx, row in cluster_features.iterrows():
            distance = np.sqrt(np.sum((row.drop('sku') - cluster_center) ** 2))
            distances.append((row['sku'], distance))
        
        representative_item = min(distances, key=lambda x: x[1])[0]
        
        # Calculate pooling weights based on similarity to cluster center
        pooling_weights = {}
        total_weight = 0
        
        for sku, distance in distances:
            # Inverse distance weighting
            weight = 1 / (1 + distance)
            pooling_weights[sku] = weight
            total_weight += weight
        
        # Normalize weights
        for sku in pooling_weights:
            pooling_weights[sku] /= total_weight
        
        # Generate cluster name based on characteristics
        cluster_name = self._generate_cluster_name(characteristics)
        
        return ClusterResult(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            items=cluster_items,
            cluster_characteristics=characteristics,
            representative_item=representative_item,
            pooling_weights=pooling_weights
        )
    
    def _generate_cluster_name(self, characteristics: Dict[str, float]) -> str:
        """Generate descriptive name for cluster"""
        
        zero_ratio = characteristics.get('zero_ratio', 0)
        adi = characteristics.get('adi', 0)
        seasonality = characteristics.get('seasonality_strength', 0)
        
        name_parts = []
        
        # Sparsity level
        if zero_ratio > 0.7:
            name_parts.append("Highly_Intermittent")
        elif zero_ratio > 0.5:
            name_parts.append("Very_Sparse")
        elif zero_ratio > 0.2:
            name_parts.append("Sparse")
        else:
            name_parts.append("Regular")
        
        # Demand interval
        if adi > 6:
            name_parts.append("Long_Intervals")
        elif adi > 3:
            name_parts.append("Medium_Intervals")
        else:
            name_parts.append("Short_Intervals")
        
        # Seasonality
        if seasonality > 0.3:
            name_parts.append("Seasonal")
        
        return "_".join(name_parts)
    
    def forecast_intermittent_demand(self, data: pd.DataFrame, sku: str, 
                                   horizon: int = 12, method: Optional[ForecastMethod] = None) -> IntermittentForecastResult:
        """
        Forecast intermittent demand using specialized methods
        """
        
        if sku not in self.sparsity_metrics:
            raise ValueError(f"Sparsity metrics not available for SKU: {sku}")
        
        metrics = self.sparsity_metrics[sku]
        
        # Use provided method or recommended method
        forecast_method = method or metrics.recommended_method
        
        # Get SKU data
        sku_data = data[data['sku'] == sku].sort_values('date')
        demand_series = sku_data['demand'].values
        
        if len(demand_series) < 6:
            # Not enough data, return zero forecast
            return IntermittentForecastResult(
                sku=sku,
                method_used=forecast_method,
                forecast_values=[0.0] * horizon,
                forecast_intervals=[(0.0, 0.0)] * horizon,
                method_parameters={},
                forecast_accuracy_metrics={}
            )
        
        # Apply appropriate forecasting method
        if forecast_method == ForecastMethod.CROSTON:
            return self._croston_forecast(sku, demand_series, horizon)
        elif forecast_method == ForecastMethod.TSB:
            return self._tsb_forecast(sku, demand_series, horizon)
        elif forecast_method == ForecastMethod.SBA:
            return self._sba_forecast(sku, demand_series, horizon)
        elif forecast_method == ForecastMethod.POOLED:
            return self._pooled_forecast(sku, data, horizon)
        elif forecast_method == ForecastMethod.HIERARCHICAL_BORROWING:
            return self._hierarchical_borrowing_forecast(sku, data, horizon)
        else:
            # Standard forecasting (simple exponential smoothing)
            return self._standard_forecast(sku, demand_series, horizon)
    
    def _croston_forecast(self, sku: str, demand_series: np.ndarray, horizon: int) -> IntermittentForecastResult:
        """Croston's method for intermittent demand"""
        
        # Croston's method parameters
        alpha = 0.1  # Smoothing parameter for demand size
        beta = 0.1   # Smoothing parameter for demand interval
        
        # Initialize
        non_zero_demands = demand_series[demand_series > 0]
        if len(non_zero_demands) == 0:
            return IntermittentForecastResult(
                sku=sku,
                method_used=ForecastMethod.CROSTON,
                forecast_values=[0.0] * horizon,
                forecast_intervals=[(0.0, 0.0)] * horizon,
                method_parameters={'alpha': alpha, 'beta': beta},
                forecast_accuracy_metrics={}
            )
        
        # Initial estimates
        z_hat = np.mean(non_zero_demands)  # Average non-zero demand
        x_hat = np.mean(np.diff(np.where(demand_series > 0)[0])) if len(non_zero_demands) > 1 else 1.0  # Average interval
        
        # Update estimates
        for t, demand in enumerate(demand_series):
            if demand > 0:
                z_hat = alpha * demand + (1 - alpha) * z_hat
                
                # Find interval since last non-zero demand
                prev_non_zero = np.where(demand_series[:t] > 0)[0]
                if len(prev_non_zero) > 0:
                    interval = t - prev_non_zero[-1]
                    x_hat = beta * interval + (1 - beta) * x_hat
        
        # Generate forecast
        forecast_value = z_hat / x_hat if x_hat > 0 else 0
        forecast_values = [forecast_value] * horizon
        
        # Simple confidence intervals (±50% of forecast)
        forecast_intervals = [(max(0, f * 0.5), f * 1.5) for f in forecast_values]
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.CROSTON,
            forecast_values=forecast_values,
            forecast_intervals=forecast_intervals,
            method_parameters={'alpha': alpha, 'beta': beta, 'z_hat': z_hat, 'x_hat': x_hat},
            forecast_accuracy_metrics={}
        )
    
    def _tsb_forecast(self, sku: str, demand_series: np.ndarray, horizon: int) -> IntermittentForecastResult:
        """Teunter-Syntetos-Babai (TSB) method"""
        
        # TSB parameters
        alpha = 0.1  # Smoothing parameter for demand
        beta = 0.1   # Smoothing parameter for probability
        
        # Initialize
        non_zero_periods = np.sum(demand_series > 0)
        total_periods = len(demand_series)
        
        if non_zero_periods == 0:
            return IntermittentForecastResult(
                sku=sku,
                method_used=ForecastMethod.TSB,
                forecast_values=[0.0] * horizon,
                forecast_intervals=[(0.0, 0.0)] * horizon,
                method_parameters={'alpha': alpha, 'beta': beta},
                forecast_accuracy_metrics={}
            )
        
        # Initial estimates
        demand_hat = np.mean(demand_series)  # Average demand (including zeros)
        prob_hat = non_zero_periods / total_periods  # Probability of non-zero demand
        
        # Update estimates
        for demand in demand_series:
            demand_hat = alpha * demand + (1 - alpha) * demand_hat
            prob_indicator = 1 if demand > 0 else 0
            prob_hat = beta * prob_indicator + (1 - beta) * prob_hat
        
        # Generate forecast
        forecast_value = demand_hat * prob_hat
        forecast_values = [forecast_value] * horizon
        
        # Confidence intervals based on probability
        lower_bound = forecast_value * (1 - prob_hat)
        upper_bound = forecast_value * (1 + prob_hat)
        forecast_intervals = [(lower_bound, upper_bound)] * horizon
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.TSB,
            forecast_values=forecast_values,
            forecast_intervals=forecast_intervals,
            method_parameters={'alpha': alpha, 'beta': beta, 'demand_hat': demand_hat, 'prob_hat': prob_hat},
            forecast_accuracy_metrics={}
        )
    
    def _sba_forecast(self, sku: str, demand_series: np.ndarray, horizon: int) -> IntermittentForecastResult:
        """Syntetos-Boylan Approximation (SBA) method"""
        
        # First apply Croston's method
        croston_result = self._croston_forecast(sku, demand_series, horizon)
        
        # Apply SBA bias correction
        x_hat = croston_result.method_parameters.get('x_hat', 1.0)
        alpha = croston_result.method_parameters.get('alpha', 0.1)
        
        # SBA correction factor
        correction_factor = (2 - alpha) / (2 - alpha + alpha * x_hat) if (2 - alpha + alpha * x_hat) > 0 else 1.0
        
        # Apply correction
        corrected_forecasts = [f * correction_factor for f in croston_result.forecast_values]
        corrected_intervals = [(l * correction_factor, u * correction_factor) 
                              for l, u in croston_result.forecast_intervals]
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.SBA,
            forecast_values=corrected_forecasts,
            forecast_intervals=corrected_intervals,
            method_parameters={**croston_result.method_parameters, 'sba_correction': correction_factor},
            forecast_accuracy_metrics={}
        )
    
    def _pooled_forecast(self, sku: str, data: pd.DataFrame, horizon: int) -> IntermittentForecastResult:
        """Enhanced category-level pooled forecasting with dynamic weights"""
        
        # Find cluster for this SKU
        sku_cluster = None
        for cluster_id, cluster in self.clusters.items():
            if sku in cluster.items:
                sku_cluster = cluster
                break
        
        if sku_cluster is None:
            # Try to create ad-hoc pooling based on category
            return self._create_adhoc_pooled_forecast(sku, data, horizon)
        
        # Enhanced pooling with multiple strategies
        pooling_strategies = ['weighted_average', 'similarity_weighted', 'performance_weighted']
        best_forecast = None
        best_confidence = 0.0
        
        for strategy in pooling_strategies:
            try:
                forecast_result = self._apply_pooling_strategy(
                    sku, sku_cluster, data, horizon, strategy
                )
                
                # Calculate strategy confidence
                strategy_confidence = self._calculate_pooling_confidence(
                    sku, sku_cluster, data, strategy
                )
                
                if strategy_confidence > best_confidence:
                    best_confidence = strategy_confidence
                    best_forecast = forecast_result
                    best_forecast.method_parameters['pooling_strategy'] = strategy
                    best_forecast.method_parameters['pooling_confidence'] = strategy_confidence
                    
            except Exception as e:
                logger.warning(f"Pooling strategy {strategy} failed for {sku}: {e}")
                continue
        
        # Fall back to standard forecast if all strategies fail
        if best_forecast is None:
            sku_data = data[data['sku'] == sku].sort_values('date')
            return self._standard_forecast(sku, sku_data['demand'].values, horizon)
        
        return best_forecast
    
    def _create_adhoc_pooled_forecast(self, sku: str, data: pd.DataFrame, 
                                    horizon: int) -> IntermittentForecastResult:
        """Create ad-hoc pooling when no cluster exists"""
        
        sku_data = data[data['sku'] == sku].sort_values('date')
        
        if len(sku_data) == 0:
            return IntermittentForecastResult(
                sku=sku,
                method_used=ForecastMethod.POOLED,
                forecast_values=[0.0] * horizon,
                forecast_intervals=[(0.0, 0.0)] * horizon,
                method_parameters={'pooling_type': 'adhoc', 'pooling_items': 0},
                forecast_accuracy_metrics={}
            )
        
        # Find items in same category for ad-hoc pooling
        category_items = []
        if 'category' in sku_data.columns:
            target_category = sku_data['category'].iloc[0]
            category_data = data[data['category'] == target_category]
            category_items = category_data['sku'].unique().tolist()
        
        if len(category_items) <= 1:
            # No other items to pool with
            return self._standard_forecast(sku, sku_data['demand'].values, horizon)
        
        # Create temporary pooling weights based on similarity
        pooling_weights = {}
        target_metrics = self.sparsity_metrics.get(sku)
        
        if target_metrics:
            total_weight = 0.0
            for item_sku in category_items:
                if item_sku == sku:
                    continue
                
                item_metrics = self.sparsity_metrics.get(item_sku)
                if item_metrics:
                    similarity = self._calculate_item_similarity(target_metrics, item_metrics)
                    if similarity > 0.2:  # Minimum threshold for ad-hoc pooling
                        pooling_weights[item_sku] = similarity
                        total_weight += similarity
            
            # Normalize weights
            if total_weight > 0:
                for item_sku in pooling_weights:
                    pooling_weights[item_sku] /= total_weight
        
        if not pooling_weights:
            return self._standard_forecast(sku, sku_data['demand'].values, horizon)
        
        # Pool demand using calculated weights
        pooled_demand = np.zeros(len(sku_data))
        
        for item_sku, weight in pooling_weights.items():
            item_data = data[data['sku'] == item_sku].sort_values('date')
            if len(item_data) > 0:
                # Align series lengths
                min_len = min(len(pooled_demand), len(item_data))
                pooled_demand[-min_len:] += item_data['demand'].values[-min_len:] * weight
        
        # Apply TSB method to pooled demand
        tsb_result = self._tsb_forecast(sku, pooled_demand, horizon)
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.POOLED,
            forecast_values=tsb_result.forecast_values,
            forecast_intervals=tsb_result.forecast_intervals,
            method_parameters={
                'pooling_type': 'adhoc',
                'pooling_items': len(pooling_weights),
                'pooling_weights': pooling_weights
            },
            forecast_accuracy_metrics={}
        )
    
    def _apply_pooling_strategy(self, sku: str, cluster: ClusterResult, 
                              data: pd.DataFrame, horizon: int, 
                              strategy: str) -> IntermittentForecastResult:
        """Apply specific pooling strategy"""
        
        if strategy == 'weighted_average':
            return self._weighted_average_pooling(sku, cluster, data, horizon)
        elif strategy == 'similarity_weighted':
            return self._similarity_weighted_pooling(sku, cluster, data, horizon)
        elif strategy == 'performance_weighted':
            return self._performance_weighted_pooling(sku, cluster, data, horizon)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
    
    def _weighted_average_pooling(self, sku: str, cluster: ClusterResult,
                                data: pd.DataFrame, horizon: int) -> IntermittentForecastResult:
        """Traditional weighted average pooling"""
        
        pooled_demand = []
        
        for item_sku in cluster.items:
            item_data = data[data['sku'] == item_sku].sort_values('date')
            if len(item_data) > 0:
                weight = cluster.pooling_weights.get(item_sku, 1.0 / len(cluster.items))
                weighted_demand = item_data['demand'].values * weight
                
                if len(pooled_demand) == 0:
                    pooled_demand = weighted_demand
                else:
                    min_len = min(len(pooled_demand), len(weighted_demand))
                    pooled_demand = pooled_demand[-min_len:] + weighted_demand[-min_len:]
        
        if len(pooled_demand) == 0:
            raise ValueError("No pooled demand data available")
        
        tsb_result = self._tsb_forecast(sku, np.array(pooled_demand), horizon)
        individual_weight = cluster.pooling_weights.get(sku, 1.0 / len(cluster.items))
        
        adjusted_forecasts = [f * individual_weight for f in tsb_result.forecast_values]
        adjusted_intervals = [(l * individual_weight, u * individual_weight) 
                             for l, u in tsb_result.forecast_intervals]
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.POOLED,
            forecast_values=adjusted_forecasts,
            forecast_intervals=adjusted_intervals,
            method_parameters={'individual_weight': individual_weight, 'cluster_id': cluster.cluster_id},
            forecast_accuracy_metrics={}
        )
    
    def _similarity_weighted_pooling(self, sku: str, cluster: ClusterResult,
                                   data: pd.DataFrame, horizon: int) -> IntermittentForecastResult:
        """Similarity-weighted pooling based on dynamic similarity calculation"""
        
        target_metrics = self.sparsity_metrics.get(sku)
        if not target_metrics:
            raise ValueError(f"No metrics available for target SKU: {sku}")
        
        # Recalculate weights based on current similarity
        similarity_weights = {}
        total_weight = 0.0
        
        for item_sku in cluster.items:
            if item_sku == sku:
                continue
            
            item_metrics = self.sparsity_metrics.get(item_sku)
            if item_metrics:
                similarity = self._calculate_item_similarity(target_metrics, item_metrics)
                similarity_weights[item_sku] = similarity
                total_weight += similarity
        
        # Normalize weights
        if total_weight > 0:
            for item_sku in similarity_weights:
                similarity_weights[item_sku] /= total_weight
        
        # Pool demand using similarity weights
        pooled_demand = []
        
        for item_sku, weight in similarity_weights.items():
            item_data = data[data['sku'] == item_sku].sort_values('date')
            if len(item_data) > 0:
                weighted_demand = item_data['demand'].values * weight
                
                if len(pooled_demand) == 0:
                    pooled_demand = weighted_demand
                else:
                    min_len = min(len(pooled_demand), len(weighted_demand))
                    pooled_demand = pooled_demand[-min_len:] + weighted_demand[-min_len:]
        
        if len(pooled_demand) == 0:
            raise ValueError("No similarity-weighted pooled demand available")
        
        tsb_result = self._tsb_forecast(sku, np.array(pooled_demand), horizon)
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.POOLED,
            forecast_values=tsb_result.forecast_values,
            forecast_intervals=tsb_result.forecast_intervals,
            method_parameters={
                'similarity_weights': similarity_weights,
                'cluster_id': cluster.cluster_id
            },
            forecast_accuracy_metrics={}
        )
    
    def _performance_weighted_pooling(self, sku: str, cluster: ClusterResult,
                                    data: pd.DataFrame, horizon: int) -> IntermittentForecastResult:
        """Performance-weighted pooling based on historical forecast accuracy"""
        
        # For now, fall back to similarity weighting since we don't have historical accuracy data
        # In a full implementation, this would use stored forecast accuracy metrics
        return self._similarity_weighted_pooling(sku, cluster, data, horizon)
    
    def _calculate_pooling_confidence(self, sku: str, cluster: ClusterResult,
                                    data: pd.DataFrame, strategy: str) -> float:
        """Calculate confidence score for pooling strategy"""
        
        base_confidence = 0.5
        
        # Cluster size factor
        size_factor = min(1.0, len(cluster.items) / 5)
        
        # Data availability factor
        data_factor = 0.0
        for item_sku in cluster.items:
            item_data = data[data['sku'] == item_sku]
            if len(item_data) > 12:  # At least 1 year of data
                data_factor += 1.0
        
        data_factor = min(1.0, data_factor / len(cluster.items))
        
        # Strategy-specific adjustments
        strategy_factor = 1.0
        if strategy == 'similarity_weighted':
            strategy_factor = 1.1  # Slight preference for similarity weighting
        elif strategy == 'performance_weighted':
            strategy_factor = 1.2  # Highest preference if performance data available
        
        confidence = (base_confidence + 0.3 * size_factor + 0.2 * data_factor) * strategy_factor
        
        return min(1.0, confidence)
    
    def _hierarchical_borrowing_forecast(self, sku: str, data: pd.DataFrame, horizon: int) -> IntermittentForecastResult:
        """Hierarchical borrowing from similar products"""
        
        # Find similar items
        borrowing_result = self.find_similar_items_for_borrowing(sku, data)
        
        if not borrowing_result.similar_items:
            # Fall back to individual forecasting
            sku_data = data[data['sku'] == sku].sort_values('date')
            return self._standard_forecast(sku, sku_data['demand'].values, horizon)
        
        # Generate forecasts from similar items
        similar_forecasts = []
        
        for similar_sku, similarity_score in borrowing_result.similar_items[:5]:  # Top 5 similar items
            try:
                similar_data = data[data['sku'] == similar_sku].sort_values('date')
                if len(similar_data) > 6:
                    # Use TSB method for similar item
                    similar_result = self._tsb_forecast(similar_sku, similar_data['demand'].values, horizon)
                    
                    # Weight by similarity
                    weighted_forecast = [f * similarity_score for f in similar_result.forecast_values]
                    similar_forecasts.append(weighted_forecast)
            except:
                continue
        
        if not similar_forecasts:
            return IntermittentForecastResult(
                sku=sku,
                method_used=ForecastMethod.HIERARCHICAL_BORROWING,
                forecast_values=[0.0] * horizon,
                forecast_intervals=[(0.0, 0.0)] * horizon,
                method_parameters={},
                forecast_accuracy_metrics={}
            )
        
        # Combine forecasts
        combined_forecast = np.mean(similar_forecasts, axis=0)
        
        # Calculate confidence intervals based on forecast variance
        forecast_std = np.std(similar_forecasts, axis=0) if len(similar_forecasts) > 1 else combined_forecast * 0.2
        forecast_intervals = [(max(0, f - 1.96 * s), f + 1.96 * s) 
                             for f, s in zip(combined_forecast, forecast_std)]
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.HIERARCHICAL_BORROWING,
            forecast_values=combined_forecast.tolist(),
            forecast_intervals=forecast_intervals,
            method_parameters={'num_similar_items': len(borrowing_result.similar_items)},
            forecast_accuracy_metrics={}
        )
    
    def _standard_forecast(self, sku: str, demand_series: np.ndarray, horizon: int) -> IntermittentForecastResult:
        """Standard exponential smoothing forecast"""
        
        if len(demand_series) == 0:
            return IntermittentForecastResult(
                sku=sku,
                method_used=ForecastMethod.STANDARD,
                forecast_values=[0.0] * horizon,
                forecast_intervals=[(0.0, 0.0)] * horizon,
                method_parameters={},
                forecast_accuracy_metrics={}
            )
        
        # Simple exponential smoothing
        alpha = 0.3
        forecast_value = demand_series[-1]  # Start with last observation
        
        # Update with exponential smoothing
        for demand in demand_series:
            forecast_value = alpha * demand + (1 - alpha) * forecast_value
        
        forecast_values = [forecast_value] * horizon
        
        # Confidence intervals based on historical variance
        demand_std = np.std(demand_series)
        forecast_intervals = [(max(0, forecast_value - 1.96 * demand_std), 
                              forecast_value + 1.96 * demand_std)] * horizon
        
        return IntermittentForecastResult(
            sku=sku,
            method_used=ForecastMethod.STANDARD,
            forecast_values=forecast_values,
            forecast_intervals=forecast_intervals,
            method_parameters={'alpha': alpha},
            forecast_accuracy_metrics={}
        )
    
    def find_similar_items_for_borrowing(self, target_sku: str, data: pd.DataFrame) -> HierarchicalBorrowingResult:
        """
        Enhanced hierarchical borrowing with multi-level hierarchy support
        """
        
        if target_sku not in self.sparsity_metrics:
            return HierarchicalBorrowingResult(
                target_sku=target_sku,
                similar_items=[],
                borrowed_patterns={},
                pooled_forecast=[],
                confidence_score=0.0
            )
        
        target_metrics = self.sparsity_metrics[target_sku]
        target_data = data[data['sku'] == target_sku].sort_values('date')
        
        if len(target_data) == 0:
            return HierarchicalBorrowingResult(
                target_sku=target_sku,
                similar_items=[],
                borrowed_patterns={},
                pooled_forecast=[],
                confidence_score=0.0
            )
        
        # Multi-level hierarchy analysis
        hierarchy_groups = self._analyze_hierarchy_levels(target_sku, target_data, data)
        
        # Calculate similarity scores with hierarchy-aware weighting
        similarity_scores = []
        
        for candidate_sku in self.sparsity_metrics.keys():
            if candidate_sku == target_sku:
                continue
            
            candidate_metrics = self.sparsity_metrics[candidate_sku]
            candidate_data = data[data['sku'] == candidate_sku]
            
            if len(candidate_data) == 0:
                continue
            
            # Base similarity from sparsity metrics
            base_similarity = self._calculate_item_similarity(target_metrics, candidate_metrics)
            
            # Hierarchy-based similarity boost
            hierarchy_boost = self._calculate_hierarchy_boost(
                target_sku, candidate_sku, hierarchy_groups, candidate_data.iloc[0]
            )
            
            # Demand pattern similarity
            pattern_similarity = self._calculate_demand_pattern_similarity(
                target_data, candidate_data
            )
            
            # Combined similarity score
            combined_similarity = (
                0.4 * base_similarity +
                0.4 * hierarchy_boost +
                0.2 * pattern_similarity
            )
            
            # Adaptive threshold based on target item sparsity
            min_threshold = self._get_adaptive_similarity_threshold(target_metrics)
            
            if combined_similarity > min_threshold:
                similarity_scores.append((candidate_sku, combined_similarity))
        
        # Sort by similarity and apply diversity filtering
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        similar_items = self._apply_diversity_filtering(similarity_scores, data)
        
        # Enhanced borrowed patterns calculation
        borrowed_patterns = self._calculate_enhanced_borrowed_patterns(
            target_sku, similar_items, data
        )
        
        # Dynamic confidence scoring
        confidence_score = self._calculate_dynamic_confidence(
            similar_items, target_metrics, hierarchy_groups
        )
        
        return HierarchicalBorrowingResult(
            target_sku=target_sku,
            similar_items=similar_items,
            borrowed_patterns=borrowed_patterns,
            pooled_forecast=[],  # Would be calculated during forecasting
            confidence_score=confidence_score
        )
    
    def _analyze_hierarchy_levels(self, target_sku: str, target_data: pd.DataFrame, 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze multi-level hierarchy for the target item"""
        
        hierarchy_groups = {
            'category': None,
            'subcategory': None,
            'brand': None,
            'product_family': None
        }
        
        # Extract hierarchy information from data columns or SKU patterns
        if 'category' in target_data.columns:
            hierarchy_groups['category'] = target_data['category'].iloc[0]
        
        if 'subcategory' in target_data.columns:
            hierarchy_groups['subcategory'] = target_data['subcategory'].iloc[0]
        
        if 'brand' in target_data.columns:
            hierarchy_groups['brand'] = target_data['brand'].iloc[0]
        
        # Try to infer hierarchy from SKU patterns
        sku_parts = target_sku.split('_')
        if len(sku_parts) > 1:
            hierarchy_groups['product_family'] = sku_parts[0]
        
        return hierarchy_groups
    
    def _calculate_hierarchy_boost(self, target_sku: str, candidate_sku: str,
                                  target_hierarchy: Dict[str, Any], 
                                  candidate_row: pd.Series) -> float:
        """Calculate similarity boost based on hierarchy levels"""
        
        boost = 0.0
        
        # Category level boost
        if (target_hierarchy['category'] and 
            'category' in candidate_row and
            target_hierarchy['category'] == candidate_row['category']):
            boost += 0.3
        
        # Subcategory level boost
        if (target_hierarchy['subcategory'] and 
            'subcategory' in candidate_row and
            target_hierarchy['subcategory'] == candidate_row['subcategory']):
            boost += 0.2
        
        # Brand level boost
        if (target_hierarchy['brand'] and 
            'brand' in candidate_row and
            target_hierarchy['brand'] == candidate_row['brand']):
            boost += 0.15
        
        # Product family boost (from SKU patterns)
        if target_hierarchy['product_family']:
            candidate_parts = candidate_sku.split('_')
            if (len(candidate_parts) > 1 and 
                candidate_parts[0] == target_hierarchy['product_family']):
                boost += 0.1
        
        return min(1.0, boost)
    
    def _calculate_demand_pattern_similarity(self, target_data: pd.DataFrame,
                                           candidate_data: pd.DataFrame) -> float:
        """Calculate similarity based on demand patterns"""
        
        try:
            target_demand = target_data['demand'].values
            candidate_demand = candidate_data['demand'].values
            
            # Align series lengths
            min_len = min(len(target_demand), len(candidate_demand))
            if min_len < 6:  # Need minimum data
                return 0.0
            
            target_aligned = target_demand[-min_len:]
            candidate_aligned = candidate_demand[-min_len:]
            
            # Calculate correlation
            if np.std(target_aligned) == 0 or np.std(candidate_aligned) == 0:
                return 0.0
            
            correlation = np.corrcoef(target_aligned, candidate_aligned)[0, 1]
            
            # Handle NaN correlation
            if np.isnan(correlation):
                return 0.0
            
            # Convert correlation to similarity (0 to 1)
            return max(0.0, (correlation + 1) / 2)
            
        except Exception:
            return 0.0
    
    def _get_adaptive_similarity_threshold(self, target_metrics: SparsityMetrics) -> float:
        """Get adaptive similarity threshold based on target item characteristics"""
        
        base_threshold = 0.3
        
        # Lower threshold for very sparse items (need more borrowing candidates)
        if target_metrics.sparsity_level == SparsityLevel.INTERMITTENT:
            return base_threshold * 0.7
        elif target_metrics.sparsity_level == SparsityLevel.VERY_SPARSE:
            return base_threshold * 0.8
        elif target_metrics.sparsity_level == SparsityLevel.SPARSE:
            return base_threshold * 0.9
        else:
            return base_threshold
    
    def _apply_diversity_filtering(self, similarity_scores: List[Tuple[str, float]], 
                                  data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Apply diversity filtering to avoid selecting too similar items"""
        
        if len(similarity_scores) <= 5:
            return similarity_scores[:10]  # Return all if few candidates
        
        selected_items = []
        selected_items.append(similarity_scores[0])  # Always include most similar
        
        for candidate_sku, similarity in similarity_scores[1:]:
            if len(selected_items) >= 10:
                break
            
            # Check diversity against already selected items
            is_diverse = True
            candidate_data = data[data['sku'] == candidate_sku]
            
            if len(candidate_data) == 0:
                continue
            
            for selected_sku, _ in selected_items:
                selected_data = data[data['sku'] == selected_sku]
                
                if len(selected_data) > 0:
                    # Check if too similar to already selected item
                    pattern_sim = self._calculate_demand_pattern_similarity(
                        candidate_data, selected_data
                    )
                    
                    if pattern_sim > 0.9:  # Too similar
                        is_diverse = False
                        break
            
            if is_diverse:
                selected_items.append((candidate_sku, similarity))
        
        return selected_items
    
    def _calculate_enhanced_borrowed_patterns(self, target_sku: str,
                                            similar_items: List[Tuple[str, float]],
                                            data: pd.DataFrame) -> Dict[str, float]:
        """Calculate enhanced borrowed patterns with dynamic weighting"""
        
        borrowed_patterns = {}
        total_weight = 0.0
        
        for similar_sku, similarity in similar_items:
            # Base weight from similarity
            base_weight = similarity
            
            # Adjust weight based on data quality
            similar_data = data[data['sku'] == similar_sku]
            if len(similar_data) > 0:
                data_quality_factor = min(1.0, len(similar_data) / 24)  # Prefer items with more data
                adjusted_weight = base_weight * data_quality_factor
                
                borrowed_patterns[similar_sku] = adjusted_weight
                total_weight += adjusted_weight
        
        # Normalize weights
        if total_weight > 0:
            for sku in borrowed_patterns:
                borrowed_patterns[sku] /= total_weight
        
        return borrowed_patterns
    
    def _calculate_dynamic_confidence(self, similar_items: List[Tuple[str, float]],
                                    target_metrics: SparsityMetrics,
                                    hierarchy_groups: Dict[str, Any]) -> float:
        """Calculate dynamic confidence score"""
        
        if not similar_items:
            return 0.0
        
        # Base confidence from similarity scores
        avg_similarity = np.mean([sim for _, sim in similar_items])
        similarity_confidence = avg_similarity
        
        # Number of similar items factor
        count_factor = min(1.0, len(similar_items) / 5)
        
        # Hierarchy support factor
        hierarchy_factor = 0.5  # Base factor
        if hierarchy_groups['category']:
            hierarchy_factor += 0.2
        if hierarchy_groups['subcategory']:
            hierarchy_factor += 0.15
        if hierarchy_groups['brand']:
            hierarchy_factor += 0.15
        
        hierarchy_factor = min(1.0, hierarchy_factor)
        
        # Sparsity penalty (very sparse items are harder to borrow for)
        sparsity_penalty = 1.0
        if target_metrics.sparsity_level == SparsityLevel.INTERMITTENT:
            sparsity_penalty = 0.8
        elif target_metrics.sparsity_level == SparsityLevel.VERY_SPARSE:
            sparsity_penalty = 0.9
        
        # Combined confidence
        confidence = (
            0.4 * similarity_confidence +
            0.3 * count_factor +
            0.3 * hierarchy_factor
        ) * sparsity_penalty
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_item_similarity(self, metrics1: SparsityMetrics, metrics2: SparsityMetrics) -> float:
        """Enhanced similarity calculation between two items based on sparsity metrics"""
        
        # Normalize metrics for comparison
        features1 = np.array([
            metrics1.zero_ratio,
            min(metrics1.adi, 10) / 10,  # Cap ADI at 10
            min(metrics1.cv_squared, 5) / 5,  # Cap CV² at 5
            metrics1.demand_variability,
            metrics1.seasonality_strength,
            metrics1.trend_strength
        ])
        
        features2 = np.array([
            metrics2.zero_ratio,
            min(metrics2.adi, 10) / 10,
            min(metrics2.cv_squared, 5) / 5,
            metrics2.demand_variability,
            metrics2.seasonality_strength,
            metrics2.trend_strength
        ])
        
        # Calculate multiple similarity measures
        
        # 1. Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (norm1 * norm2)
        
        # 2. Euclidean distance similarity (inverted and normalized)
        euclidean_dist = np.linalg.norm(features1 - features2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 3. Sparsity level similarity bonus
        sparsity_bonus = 0.0
        if metrics1.sparsity_level == metrics2.sparsity_level:
            sparsity_bonus = 0.2
        
        # 4. Method compatibility bonus
        method_bonus = 0.0
        if metrics1.recommended_method == metrics2.recommended_method:
            method_bonus = 0.1
        
        # Combine similarities with weights
        combined_similarity = (
            0.5 * max(0.0, cosine_sim) +
            0.3 * euclidean_sim +
            sparsity_bonus +
            method_bonus
        )
        
        # Ensure similarity is between 0 and 1
        return min(1.0, max(0.0, combined_similarity))
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of long-tail optimization results"""
        
        if not self.sparsity_metrics:
            return {"error": "No sparsity analysis available"}
        
        # Count items by sparsity level
        sparsity_counts = {}
        method_recommendations = {}
        
        for sku, metrics in self.sparsity_metrics.items():
            level = metrics.sparsity_level.value
            method = metrics.recommended_method.value
            
            sparsity_counts[level] = sparsity_counts.get(level, 0) + 1
            method_recommendations[method] = method_recommendations.get(method, 0) + 1
        
        # Cluster summary
        cluster_summary = {}
        if self.clusters:
            for cluster_id, cluster in self.clusters.items():
                cluster_summary[f"cluster_{cluster_id}"] = {
                    "name": cluster.cluster_name,
                    "item_count": len(cluster.items),
                    "representative_item": cluster.representative_item
                }
        
        return {
            "total_items_analyzed": len(self.sparsity_metrics),
            "sparsity_distribution": sparsity_counts,
            "method_recommendations": method_recommendations,
            "clusters_created": len(self.clusters),
            "cluster_details": cluster_summary,
            "optimization_coverage": {
                "sparse_items": sum(1 for m in self.sparsity_metrics.values() 
                                  if m.sparsity_level != SparsityLevel.DENSE),
                "clustered_items": sum(len(c.items) for c in self.clusters.values()),
                "borrowing_candidates": sum(1 for m in self.sparsity_metrics.values() 
                                          if m.recommended_method == ForecastMethod.HIERARCHICAL_BORROWING)
            }
        }