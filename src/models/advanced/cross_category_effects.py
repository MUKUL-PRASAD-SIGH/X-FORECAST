"""
Cross-Category Effects Engine
Models cannibalization and halo effects for promotion and pricing decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from scipy import stats
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EffectType(Enum):
    CANNIBALIZATION = "cannibalization"
    HALO = "halo"
    SUBSTITUTION = "substitution"
    COMPLEMENTARY = "complementary"

class EffectStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class CategoryRelationship:
    """Represents relationship between two categories"""
    source_category: str
    target_category: str
    effect_type: EffectType
    effect_strength: EffectStrength
    elasticity: float  # % change in target per 1% change in source
    confidence: float
    supporting_evidence: Dict[str, Any]
    last_updated: datetime

@dataclass
class CrossCategoryImpact:
    """Impact of changes in one category on others"""
    source_category: str
    source_change_pct: float
    affected_categories: Dict[str, float]  # category -> impact percentage
    total_network_effect: float
    confidence_score: float
    time_horizon: int  # periods for effect to materialize

@dataclass
class PromotionImpactAnalysis:
    """Analysis of promotion impact across categories"""
    promoted_category: str
    promotion_lift_pct: float
    direct_impact: float
    cannibalization_impact: Dict[str, float]
    halo_impact: Dict[str, float]
    net_incremental_impact: float
    roi_adjustment_factor: float

@dataclass
class MLModelResult:
    """Result from ML-based effect detection"""
    model_type: str
    feature_importance: Dict[str, float]
    cross_val_score: float
    prediction_accuracy: float
    detected_relationships: List[Tuple[str, str, float]]

@dataclass
class DynamicElasticityResult:
    """Result from dynamic elasticity estimation"""
    category_pair: str
    time_varying_elasticity: pd.Series
    regime_changes: List[datetime]
    current_elasticity: float
    elasticity_forecast: List[float]
    confidence_intervals: List[Tuple[float, float]]

@dataclass
class PortfolioSimulationResult:
    """Result from promotion portfolio simulation"""
    simulation_id: str
    promoted_categories: List[str]
    base_scenario: Dict[str, float]
    simulated_outcomes: List[Dict[str, float]]
    expected_outcome: Dict[str, float]
    risk_metrics: Dict[str, float]
    confidence_level: float

class CrossCategoryEngine:
    """
    Models cross-category effects including cannibalization and halo effects
    """
    
    def __init__(self):
        self.category_relationships = {}
        self.effect_network = nx.DiGraph()
        self.elasticity_models = {}
        self.ml_models = {}
        self.dynamic_elasticity_models = {}
        self.scaler = StandardScaler()
        
        # Thresholds for effect classification
        self.effect_thresholds = {
            EffectStrength.WEAK: 0.05,        # 0-5% elasticity
            EffectStrength.MODERATE: 0.15,    # 5-15% elasticity
            EffectStrength.STRONG: 0.30,      # 15-30% elasticity
            EffectStrength.VERY_STRONG: 1.0   # >30% elasticity
        }
    
    def discover_category_relationships(self, data: pd.DataFrame) -> Dict[str, CategoryRelationship]:
        """
        Discover relationships between categories using historical data
        Expected columns: date, category, sku, demand, price, promotion_flag
        """
        
        logger.info("Discovering cross-category relationships...")
        
        relationships = {}
        
        # Get category-level aggregated data
        category_data = self._aggregate_by_category(data)
        categories = category_data['category'].unique()
        
        # Analyze each category pair
        for source_cat in categories:
            for target_cat in categories:
                if source_cat != target_cat:
                    relationship = self._analyze_category_pair(category_data, source_cat, target_cat)
                    if relationship:
                        key = f"{source_cat}_{target_cat}"
                        relationships[key] = relationship
                        
                        # Add to network graph
                        self.effect_network.add_edge(
                            source_cat, 
                            target_cat, 
                            weight=abs(relationship.elasticity),
                            effect_type=relationship.effect_type.value
                        )
        
        self.category_relationships = relationships
        logger.info(f"Discovered {len(relationships)} category relationships")
        
        return relationships
    
    def _aggregate_by_category(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by category and date"""
        
        # Create category mapping if not exists
        if 'category' not in data.columns:
            # Simple category mapping based on SKU prefix
            data['category'] = data['sku'].str.split('-').str[0]
        
        # Aggregate by category and date
        agg_data = data.groupby(['date', 'category']).agg({
            'demand': 'sum',
            'price': 'mean',
            'promotion_flag': 'max'  # 1 if any SKU in category is promoted
        }).reset_index()
        
        return agg_data
    
    def _analyze_category_pair(self, data: pd.DataFrame, source_cat: str, target_cat: str) -> Optional[CategoryRelationship]:
        """Analyze relationship between two categories"""
        
        # Get data for both categories
        source_data = data[data['category'] == source_cat].set_index('date')
        target_data = data[data['category'] == target_cat].set_index('date')
        
        # Align data
        common_dates = source_data.index.intersection(target_data.index)
        if len(common_dates) < 12:  # Need at least 12 periods
            return None
        
        source_aligned = source_data.loc[common_dates]
        target_aligned = target_data.loc[common_dates]
        
        # Calculate elasticity using multiple approaches
        elasticity_estimates = []
        
        # 1. Price elasticity approach
        price_elasticity = self._calculate_price_cross_elasticity(source_aligned, target_aligned)
        if price_elasticity is not None:
            elasticity_estimates.append(price_elasticity)
        
        # 2. Promotion elasticity approach
        promo_elasticity = self._calculate_promotion_cross_elasticity(source_aligned, target_aligned)
        if promo_elasticity is not None:
            elasticity_estimates.append(promo_elasticity)
        
        # 3. Demand correlation approach
        demand_elasticity = self._calculate_demand_correlation_elasticity(source_aligned, target_aligned)
        if demand_elasticity is not None:
            elasticity_estimates.append(demand_elasticity)
        
        if not elasticity_estimates:
            return None
        
        # Average elasticity estimates
        avg_elasticity = np.mean(elasticity_estimates)
        elasticity_std = np.std(elasticity_estimates) if len(elasticity_estimates) > 1 else 0
        
        # Determine effect type and strength
        effect_type = self._classify_effect_type(avg_elasticity)
        effect_strength = self._classify_effect_strength(abs(avg_elasticity))
        
        # Calculate confidence based on consistency of estimates
        confidence = max(0.1, 1.0 - (elasticity_std / abs(avg_elasticity)) if avg_elasticity != 0 else 0.1)
        confidence = min(0.95, confidence)
        
        # Gather supporting evidence
        supporting_evidence = {
            "elasticity_estimates": elasticity_estimates,
            "data_points": len(common_dates),
            "price_correlation": source_aligned['price'].corr(target_aligned['demand']),
            "demand_correlation": source_aligned['demand'].corr(target_aligned['demand']),
            "promotion_overlap": ((source_aligned['promotion_flag'] == 1) & (target_aligned['promotion_flag'] == 1)).sum()
        }
        
        return CategoryRelationship(
            source_category=source_cat,
            target_category=target_cat,
            effect_type=effect_type,
            effect_strength=effect_strength,
            elasticity=avg_elasticity,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            last_updated=datetime.now()
        )
    
    def _calculate_price_cross_elasticity(self, source_data: pd.DataFrame, target_data: pd.DataFrame) -> Optional[float]:
        """Calculate cross-price elasticity between categories"""
        
        try:
            # Calculate percentage changes
            source_price_pct_change = source_data['price'].pct_change().dropna()
            target_demand_pct_change = target_data['demand'].pct_change().dropna()
            
            # Align data
            common_index = source_price_pct_change.index.intersection(target_demand_pct_change.index)
            if len(common_index) < 6:
                return None
            
            source_aligned = source_price_pct_change.loc[common_index]
            target_aligned = target_demand_pct_change.loc[common_index]
            
            # Remove outliers (beyond 3 standard deviations)
            source_z = np.abs(stats.zscore(source_aligned))
            target_z = np.abs(stats.zscore(target_aligned))
            mask = (source_z < 3) & (target_z < 3)
            
            if mask.sum() < 4:
                return None
            
            source_clean = source_aligned[mask]
            target_clean = target_aligned[mask]
            
            # Calculate elasticity using linear regression
            if len(source_clean) > 0 and source_clean.std() > 0:
                model = LinearRegression()
                X = source_clean.values.reshape(-1, 1)
                y = target_clean.values
                model.fit(X, y)
                
                return model.coef_[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Price cross-elasticity calculation failed: {e}")
            return None
    
    def _calculate_promotion_cross_elasticity(self, source_data: pd.DataFrame, target_data: pd.DataFrame) -> Optional[float]:
        """Calculate cross-promotion elasticity between categories"""
        
        try:
            # Find periods where source category was promoted
            source_promo_periods = source_data[source_data['promotion_flag'] == 1]
            source_non_promo_periods = source_data[source_data['promotion_flag'] == 0]
            
            if len(source_promo_periods) < 3 or len(source_non_promo_periods) < 3:
                return None
            
            # Calculate average target demand during source promotions vs non-promotions
            target_demand_during_source_promo = target_data.loc[source_promo_periods.index]['demand'].mean()
            target_demand_during_source_non_promo = target_data.loc[source_non_promo_periods.index]['demand'].mean()
            
            if target_demand_during_source_non_promo == 0:
                return None
            
            # Calculate elasticity as percentage change
            elasticity = (target_demand_during_source_promo - target_demand_during_source_non_promo) / target_demand_during_source_non_promo
            
            return elasticity
            
        except Exception as e:
            logger.warning(f"Promotion cross-elasticity calculation failed: {e}")
            return None
    
    def _calculate_demand_correlation_elasticity(self, source_data: pd.DataFrame, target_data: pd.DataFrame) -> Optional[float]:
        """Calculate elasticity based on demand correlation"""
        
        try:
            # Calculate percentage changes in demand
            source_demand_pct = source_data['demand'].pct_change().dropna()
            target_demand_pct = target_data['demand'].pct_change().dropna()
            
            # Align data
            common_index = source_demand_pct.index.intersection(target_demand_pct.index)
            if len(common_index) < 6:
                return None
            
            source_aligned = source_demand_pct.loc[common_index]
            target_aligned = target_demand_pct.loc[common_index]
            
            # Calculate correlation coefficient as proxy for elasticity
            correlation = source_aligned.corr(target_aligned)
            
            if pd.isna(correlation):
                return None
            
            # Convert correlation to elasticity estimate
            # This is a simplified approach - in practice, more sophisticated methods would be used
            elasticity = correlation * (target_aligned.std() / source_aligned.std()) if source_aligned.std() > 0 else 0
            
            return elasticity
            
        except Exception as e:
            logger.warning(f"Demand correlation elasticity calculation failed: {e}")
            return None
    
    def _classify_effect_type(self, elasticity: float) -> EffectType:
        """Classify the type of cross-category effect"""
        
        if elasticity < -0.05:
            return EffectType.CANNIBALIZATION  # Negative effect
        elif elasticity > 0.05:
            return EffectType.HALO  # Positive effect
        else:
            return EffectType.SUBSTITUTION  # Minimal effect
    
    def _classify_effect_strength(self, abs_elasticity: float) -> EffectStrength:
        """Classify the strength of the effect"""
        
        if abs_elasticity >= self.effect_thresholds[EffectStrength.VERY_STRONG]:
            return EffectStrength.VERY_STRONG
        elif abs_elasticity >= self.effect_thresholds[EffectStrength.STRONG]:
            return EffectStrength.STRONG
        elif abs_elasticity >= self.effect_thresholds[EffectStrength.MODERATE]:
            return EffectStrength.MODERATE
        else:
            return EffectStrength.WEAK
    
    def calculate_cross_category_impact(self, source_category: str, change_pct: float, 
                                      time_horizon: int = 12) -> CrossCategoryImpact:
        """Calculate impact of changes in one category on all others"""
        
        affected_categories = {}
        total_network_effect = 0.0
        confidence_scores = []
        
        # Direct effects
        for key, relationship in self.category_relationships.items():
            if relationship.source_category == source_category:
                target_category = relationship.target_category
                
                # Calculate impact
                impact = change_pct * relationship.elasticity
                affected_categories[target_category] = impact
                total_network_effect += abs(impact)
                confidence_scores.append(relationship.confidence)
        
        # Second-order effects (effects of affected categories on others)
        second_order_effects = {}
        for affected_cat, first_order_impact in affected_categories.items():
            for key, relationship in self.category_relationships.items():
                if relationship.source_category == affected_cat:
                    target_category = relationship.target_category
                    if target_category != source_category:  # Avoid circular effects
                        
                        # Second-order impact (dampened)
                        second_order_impact = first_order_impact * relationship.elasticity * 0.5  # Damping factor
                        
                        if target_category in second_order_effects:
                            second_order_effects[target_category] += second_order_impact
                        else:
                            second_order_effects[target_category] = second_order_impact
        
        # Combine first and second order effects
        for category, impact in second_order_effects.items():
            if category in affected_categories:
                affected_categories[category] += impact
            else:
                affected_categories[category] = impact
            
            total_network_effect += abs(impact)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return CrossCategoryImpact(
            source_category=source_category,
            source_change_pct=change_pct,
            affected_categories=affected_categories,
            total_network_effect=total_network_effect,
            confidence_score=overall_confidence,
            time_horizon=time_horizon
        )
    
    def analyze_promotion_impact(self, promoted_category: str, promotion_lift_pct: float) -> PromotionImpactAnalysis:
        """Analyze the full impact of a promotion including cross-category effects"""
        
        # Calculate cross-category impact
        cross_impact = self.calculate_cross_category_impact(promoted_category, promotion_lift_pct)
        
        # Separate cannibalization and halo effects
        cannibalization_impact = {}
        halo_impact = {}
        
        for category, impact in cross_impact.affected_categories.items():
            if impact < 0:
                cannibalization_impact[category] = impact
            else:
                halo_impact[category] = impact
        
        # Calculate net incremental impact
        total_cannibalization = sum(cannibalization_impact.values())
        total_halo = sum(halo_impact.values())
        net_incremental_impact = promotion_lift_pct + total_cannibalization + total_halo
        
        # Calculate ROI adjustment factor
        # If cannibalization is high, ROI should be adjusted down
        roi_adjustment_factor = net_incremental_impact / promotion_lift_pct if promotion_lift_pct != 0 else 1.0
        
        return PromotionImpactAnalysis(
            promoted_category=promoted_category,
            promotion_lift_pct=promotion_lift_pct,
            direct_impact=promotion_lift_pct,
            cannibalization_impact=cannibalization_impact,
            halo_impact=halo_impact,
            net_incremental_impact=net_incremental_impact,
            roi_adjustment_factor=roi_adjustment_factor
        )
    
    def optimize_promotion_portfolio(self, available_categories: List[str], 
                                   budget_constraint: float,
                                   category_costs: Dict[str, float]) -> Dict[str, Any]:
        """Optimize promotion portfolio considering cross-category effects"""
        
        # Simple greedy optimization considering network effects
        promotion_plan = {}
        remaining_budget = budget_constraint
        
        # Calculate ROI for each category considering cross-effects
        category_roi = {}
        for category in available_categories:
            if category in category_costs and category_costs[category] <= remaining_budget:
                # Assume 20% promotion lift
                analysis = self.analyze_promotion_impact(category, 0.20)
                
                # Calculate adjusted ROI
                base_roi = 0.15  # Assume 15% base ROI
                adjusted_roi = base_roi * analysis.roi_adjustment_factor
                category_roi[category] = adjusted_roi
        
        # Select categories with highest adjusted ROI
        sorted_categories = sorted(category_roi.items(), key=lambda x: x[1], reverse=True)
        
        for category, roi in sorted_categories:
            cost = category_costs.get(category, 0)
            if cost <= remaining_budget:
                promotion_plan[category] = {
                    "promotion_lift": 0.20,
                    "cost": cost,
                    "expected_roi": roi,
                    "cross_category_effects": self.calculate_cross_category_impact(category, 0.20).affected_categories
                }
                remaining_budget -= cost
        
        return {
            "promotion_plan": promotion_plan,
            "total_cost": budget_constraint - remaining_budget,
            "remaining_budget": remaining_budget,
            "expected_total_roi": sum(plan["expected_roi"] for plan in promotion_plan.values()),
            "optimization_method": "greedy_roi_maximization"
        }
    
    def get_category_network_metrics(self) -> Dict[str, Any]:
        """Get network analysis metrics for category relationships"""
        
        if not self.effect_network.nodes():
            return {"error": "No category network available"}
        
        # Calculate network metrics
        centrality = nx.degree_centrality(self.effect_network)
        betweenness = nx.betweenness_centrality(self.effect_network)
        
        # Find most influential categories
        most_influential = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Find categories with strongest effects
        strongest_effects = []
        for source, target, data in self.effect_network.edges(data=True):
            strongest_effects.append((source, target, data['weight'], data['effect_type']))
        
        strongest_effects.sort(key=lambda x: x[2], reverse=True)
        
        return {
            "total_categories": len(self.effect_network.nodes()),
            "total_relationships": len(self.effect_network.edges()),
            "most_influential_categories": most_influential,
            "strongest_effects": strongest_effects[:10],
            "network_density": nx.density(self.effect_network),
            "average_clustering": nx.average_clustering(self.effect_network.to_undirected())
        } 
   
    def ml_based_effect_detection(self, data: pd.DataFrame) -> MLModelResult:
        """
        Use machine learning to detect cross-category relationships
        """
        logger.info("Starting ML-based cross-category effect detection...")
        
        # Prepare feature matrix
        feature_data = self._prepare_ml_features(data)
        
        if feature_data.empty:
            return MLModelResult("none", {}, 0.0, 0.0, [])
        
        # Try multiple ML approaches
        ml_results = {}
        
        # Random Forest approach
        rf_result = self._train_random_forest_model(feature_data)
        ml_results['random_forest'] = rf_result
        
        # XGBoost approach
        xgb_result = self._train_xgboost_model(feature_data)
        ml_results['xgboost'] = xgb_result
        
        # Select best performing model
        best_model = max(ml_results.items(), key=lambda x: x[1].cross_val_score)
        
        logger.info(f"Best ML model: {best_model[0]} with CV score: {best_model[1].cross_val_score:.3f}")
        
        return best_model[1]
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for ML models"""
        
        # Aggregate by category and date
        category_data = self._aggregate_by_category(data)
        
        # Create pivot table for categories
        demand_pivot = category_data.pivot(index='date', columns='category', values='demand').fillna(0)
        price_pivot = category_data.pivot(index='date', columns='category', values='price').fillna(method='ffill')
        promo_pivot = category_data.pivot(index='date', columns='category', values='promotion_flag').fillna(0)
        
        categories = demand_pivot.columns.tolist()
        feature_rows = []
        
        # Create feature vectors for each category pair
        for target_cat in categories:
            for source_cat in categories:
                if source_cat != target_cat:
                    # Extract features for this pair
                    features = self._extract_pair_features(
                        demand_pivot, price_pivot, promo_pivot, 
                        source_cat, target_cat
                    )
                    
                    if features is not None:
                        features['source_category'] = source_cat
                        features['target_category'] = target_cat
                        feature_rows.append(features)
        
        return pd.DataFrame(feature_rows)
    
    def _extract_pair_features(self, demand_pivot: pd.DataFrame, price_pivot: pd.DataFrame, 
                              promo_pivot: pd.DataFrame, source_cat: str, target_cat: str) -> Optional[Dict]:
        """Extract features for a category pair"""
        
        if source_cat not in demand_pivot.columns or target_cat not in demand_pivot.columns:
            return None
        
        source_demand = demand_pivot[source_cat]
        target_demand = demand_pivot[target_cat]
        source_price = price_pivot[source_cat]
        target_price = price_pivot[target_cat]
        source_promo = promo_pivot[source_cat]
        target_promo = promo_pivot[target_cat]
        
        # Calculate various features
        features = {}
        
        # Basic correlation features
        features['demand_correlation'] = source_demand.corr(target_demand)
        features['price_correlation'] = source_price.corr(target_price)
        
        # Lagged correlations
        for lag in [1, 2, 3]:
            if len(source_demand) > lag:
                features[f'demand_correlation_lag_{lag}'] = source_demand.corr(target_demand.shift(lag))
        
        # Price elasticity features
        source_price_change = source_price.pct_change().dropna()
        target_demand_change = target_demand.pct_change().dropna()
        
        if len(source_price_change) > 5 and source_price_change.std() > 0:
            # Align series
            common_idx = source_price_change.index.intersection(target_demand_change.index)
            if len(common_idx) > 3:
                features['price_elasticity'] = source_price_change.loc[common_idx].corr(
                    target_demand_change.loc[common_idx]
                )
        
        # Promotion impact features
        promo_periods = source_promo == 1
        non_promo_periods = source_promo == 0
        
        if promo_periods.sum() > 2 and non_promo_periods.sum() > 2:
            target_during_promo = target_demand[promo_periods].mean()
            target_during_non_promo = target_demand[non_promo_periods].mean()
            
            if target_during_non_promo > 0:
                features['promotion_impact'] = (target_during_promo - target_during_non_promo) / target_during_non_promo
        
        # Volatility features
        features['source_demand_volatility'] = source_demand.std() / source_demand.mean() if source_demand.mean() > 0 else 0
        features['target_demand_volatility'] = target_demand.std() / target_demand.mean() if target_demand.mean() > 0 else 0
        
        # Seasonal features
        if len(source_demand) >= 12:
            # Simple seasonality measure
            monthly_avg = source_demand.groupby(source_demand.index.month).mean()
            features['source_seasonality'] = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
        
        # Target variable (elasticity estimate)
        if 'price_elasticity' in features:
            features['target_elasticity'] = features['price_elasticity']
        elif 'promotion_impact' in features:
            features['target_elasticity'] = features['promotion_impact']
        else:
            features['target_elasticity'] = features.get('demand_correlation', 0)
        
        return features
    
    def _train_random_forest_model(self, feature_data: pd.DataFrame) -> MLModelResult:
        """Train Random Forest model for effect detection"""
        
        # Prepare features and target
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['source_category', 'target_category', 'target_elasticity']]
        
        X = feature_data[feature_cols].fillna(0)
        y = feature_data['target_elasticity'].fillna(0)
        
        if len(X) < 10:
            return MLModelResult("random_forest", {}, 0.0, 0.0, [])
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X, y, cv=min(5, len(X)//2), scoring='r2')
        cv_score = cv_scores.mean()
        
        # Fit full model
        rf_model.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
        
        # Detect relationships
        predictions = rf_model.predict(X)
        detected_relationships = []
        
        for i, (_, row) in enumerate(feature_data.iterrows()):
            if abs(predictions[i]) > 0.05:  # Threshold for significant relationship
                detected_relationships.append((
                    row['source_category'], 
                    row['target_category'], 
                    predictions[i]
                ))
        
        return MLModelResult(
            model_type="random_forest",
            feature_importance=feature_importance,
            cross_val_score=cv_score,
            prediction_accuracy=rf_model.score(X, y),
            detected_relationships=detected_relationships
        )
    
    def _train_xgboost_model(self, feature_data: pd.DataFrame) -> MLModelResult:
        """Train XGBoost model for effect detection"""
        
        # Prepare features and target
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['source_category', 'target_category', 'target_elasticity']]
        
        X = feature_data[feature_cols].fillna(0)
        y = feature_data['target_elasticity'].fillna(0)
        
        if len(X) < 10:
            return MLModelResult("xgboost", {}, 0.0, 0.0, [])
        
        # Train model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42,
            verbosity=0
        )
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, X, y, cv=min(5, len(X)//2), scoring='r2')
        cv_score = cv_scores.mean()
        
        # Fit full model
        xgb_model.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
        
        # Detect relationships
        predictions = xgb_model.predict(X)
        detected_relationships = []
        
        for i, (_, row) in enumerate(feature_data.iterrows()):
            if abs(predictions[i]) > 0.05:  # Threshold for significant relationship
                detected_relationships.append((
                    row['source_category'], 
                    row['target_category'], 
                    predictions[i]
                ))
        
        return MLModelResult(
            model_type="xgboost",
            feature_importance=feature_importance,
            cross_val_score=cv_score,
            prediction_accuracy=xgb_model.score(X, y),
            detected_relationships=detected_relationships
        )
    
    def build_dynamic_elasticity_models(self, data: pd.DataFrame) -> Dict[str, DynamicElasticityResult]:
        """
        Build time-varying coefficient models for dynamic elasticity estimation
        """
        logger.info("Building dynamic elasticity models...")
        
        results = {}
        category_data = self._aggregate_by_category(data)
        categories = category_data['category'].unique()
        
        for source_cat in categories:
            for target_cat in categories:
                if source_cat != target_cat:
                    pair_key = f"{source_cat}_{target_cat}"
                    
                    # Build dynamic model for this pair
                    dynamic_result = self._build_dynamic_elasticity_model(
                        category_data, source_cat, target_cat
                    )
                    
                    if dynamic_result:
                        results[pair_key] = dynamic_result
                        self.dynamic_elasticity_models[pair_key] = dynamic_result
        
        logger.info(f"Built {len(results)} dynamic elasticity models")
        return results
    
    def _build_dynamic_elasticity_model(self, data: pd.DataFrame, source_cat: str, target_cat: str) -> Optional[DynamicElasticityResult]:
        """Build dynamic elasticity model for a category pair"""
        
        # Get data for both categories
        source_data = data[data['category'] == source_cat].set_index('date').sort_index()
        target_data = data[data['category'] == target_cat].set_index('date').sort_index()
        
        # Align data
        common_dates = source_data.index.intersection(target_data.index)
        if len(common_dates) < 24:  # Need at least 24 periods
            return None
        
        source_aligned = source_data.loc[common_dates]
        target_aligned = target_data.loc[common_dates]
        
        # Calculate rolling elasticity
        window_size = 12  # 12-period rolling window
        time_varying_elasticity = pd.Series(index=common_dates, dtype=float)
        
        for i in range(window_size, len(common_dates)):
            window_start = i - window_size
            window_end = i
            
            # Get window data
            source_window = source_aligned.iloc[window_start:window_end]
            target_window = target_aligned.iloc[window_start:window_end]
            
            # Calculate elasticity for this window
            elasticity = self._calculate_window_elasticity(source_window, target_window)
            time_varying_elasticity.iloc[i] = elasticity
        
        # Remove NaN values
        time_varying_elasticity = time_varying_elasticity.dropna()
        
        if len(time_varying_elasticity) < 5:
            return None
        
        # Detect regime changes using change point detection
        regime_changes = self._detect_regime_changes(time_varying_elasticity)
        
        # Current elasticity (most recent)
        current_elasticity = time_varying_elasticity.iloc[-1]
        
        # Forecast elasticity
        elasticity_forecast, confidence_intervals = self._forecast_elasticity(time_varying_elasticity)
        
        return DynamicElasticityResult(
            category_pair=f"{source_cat}_{target_cat}",
            time_varying_elasticity=time_varying_elasticity,
            regime_changes=regime_changes,
            current_elasticity=current_elasticity,
            elasticity_forecast=elasticity_forecast,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_window_elasticity(self, source_data: pd.DataFrame, target_data: pd.DataFrame) -> float:
        """Calculate elasticity for a time window"""
        
        try:
            # Use price changes if available
            if 'price' in source_data.columns:
                source_change = source_data['price'].pct_change().dropna()
                target_change = target_data['demand'].pct_change().dropna()
            else:
                # Use demand changes
                source_change = source_data['demand'].pct_change().dropna()
                target_change = target_data['demand'].pct_change().dropna()
            
            # Align changes
            common_idx = source_change.index.intersection(target_change.index)
            if len(common_idx) < 3:
                return 0.0
            
            source_aligned = source_change.loc[common_idx]
            target_aligned = target_change.loc[common_idx]
            
            # Calculate elasticity using linear regression
            if source_aligned.std() > 0:
                model = LinearRegression()
                X = source_aligned.values.reshape(-1, 1)
                y = target_aligned.values
                model.fit(X, y)
                return model.coef_[0]
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_regime_changes(self, elasticity_series: pd.Series) -> List[datetime]:
        """Detect regime changes in elasticity time series"""
        
        regime_changes = []
        
        if len(elasticity_series) < 10:
            return regime_changes
        
        # Simple change point detection using rolling statistics
        window = 6
        rolling_mean = elasticity_series.rolling(window=window).mean()
        rolling_std = elasticity_series.rolling(window=window).std()
        
        # Detect significant changes in mean
        mean_changes = rolling_mean.diff().abs()
        std_threshold = rolling_std.mean() * 2  # 2 standard deviations
        
        change_points = mean_changes[mean_changes > std_threshold].index
        
        # Convert to datetime list
        for change_point in change_points:
            if isinstance(change_point, pd.Timestamp):
                regime_changes.append(change_point.to_pydatetime())
            else:
                regime_changes.append(change_point)
        
        return regime_changes
    
    def _forecast_elasticity(self, elasticity_series: pd.Series, periods: int = 6) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Forecast future elasticity values"""
        
        if len(elasticity_series) < 5:
            return [elasticity_series.iloc[-1]] * periods, [(0, 0)] * periods
        
        # Simple exponential smoothing for elasticity forecast
        alpha = 0.3  # Smoothing parameter
        
        forecasts = []
        current_level = elasticity_series.iloc[-1]
        
        # Calculate forecast error for confidence intervals
        residuals = elasticity_series.diff().dropna()
        forecast_error = residuals.std()
        
        confidence_intervals = []
        
        for i in range(periods):
            # Forecast
            forecast = current_level
            forecasts.append(forecast)
            
            # Confidence interval (assuming normal distribution)
            margin = 1.96 * forecast_error * np.sqrt(i + 1)  # 95% confidence
            confidence_intervals.append((forecast - margin, forecast + margin))
        
        return forecasts, confidence_intervals
    
    def simulate_promotion_portfolio(self, promotion_scenarios: List[Dict[str, Any]], 
                                   num_simulations: int = 1000) -> List[PortfolioSimulationResult]:
        """
        Monte Carlo simulation for promotion portfolio optimization
        """
        logger.info(f"Running portfolio simulation with {num_simulations} iterations...")
        
        results = []
        
        for scenario_idx, scenario in enumerate(promotion_scenarios):
            simulation_result = self._run_portfolio_simulation(scenario, num_simulations)
            simulation_result.simulation_id = f"scenario_{scenario_idx}"
            results.append(simulation_result)
        
        return results
    
    def _run_portfolio_simulation(self, scenario: Dict[str, Any], num_simulations: int) -> PortfolioSimulationResult:
        """Run Monte Carlo simulation for a single scenario"""
        
        promoted_categories = scenario.get('promoted_categories', [])
        base_lifts = scenario.get('promotion_lifts', {})
        
        # Base scenario (no uncertainty)
        base_scenario = {}
        for category in promoted_categories:
            base_lift = base_lifts.get(category, 0.2)  # Default 20% lift
            impact_analysis = self.analyze_promotion_impact(category, base_lift)
            base_scenario[category] = impact_analysis.net_incremental_impact
        
        # Run simulations
        simulated_outcomes = []
        
        for sim in range(num_simulations):
            sim_outcome = {}
            
            for category in promoted_categories:
                # Add uncertainty to promotion lift
                base_lift = base_lifts.get(category, 0.2)
                
                # Assume 20% coefficient of variation in promotion effectiveness
                lift_std = base_lift * 0.2
                simulated_lift = np.random.normal(base_lift, lift_std)
                simulated_lift = max(0, simulated_lift)  # Ensure non-negative
                
                # Add uncertainty to cross-category effects
                impact_analysis = self.analyze_promotion_impact(category, simulated_lift)
                
                # Add noise to cross-category effects
                noise_factor = np.random.normal(1.0, 0.1)  # 10% noise
                simulated_impact = impact_analysis.net_incremental_impact * noise_factor
                
                sim_outcome[category] = simulated_impact
            
            simulated_outcomes.append(sim_outcome)
        
        # Calculate expected outcome and risk metrics
        expected_outcome = {}
        risk_metrics = {}
        
        for category in promoted_categories:
            category_outcomes = [outcome[category] for outcome in simulated_outcomes]
            
            expected_outcome[category] = np.mean(category_outcomes)
            
            risk_metrics[f"{category}_std"] = np.std(category_outcomes)
            risk_metrics[f"{category}_var"] = np.var(category_outcomes)
            risk_metrics[f"{category}_5th_percentile"] = np.percentile(category_outcomes, 5)
            risk_metrics[f"{category}_95th_percentile"] = np.percentile(category_outcomes, 95)
        
        # Overall portfolio risk metrics
        total_outcomes = [sum(outcome.values()) for outcome in simulated_outcomes]
        risk_metrics['portfolio_expected_return'] = np.mean(total_outcomes)
        risk_metrics['portfolio_std'] = np.std(total_outcomes)
        risk_metrics['portfolio_var'] = np.var(total_outcomes)
        risk_metrics['portfolio_5th_percentile'] = np.percentile(total_outcomes, 5)
        risk_metrics['portfolio_95th_percentile'] = np.percentile(total_outcomes, 95)
        
        # Calculate confidence level (probability of positive outcome)
        positive_outcomes = sum(1 for outcome in total_outcomes if outcome > 0)
        confidence_level = positive_outcomes / len(total_outcomes)
        
        return PortfolioSimulationResult(
            simulation_id="",  # Will be set by caller
            promoted_categories=promoted_categories,
            base_scenario=base_scenario,
            simulated_outcomes=simulated_outcomes,
            expected_outcome=expected_outcome,
            risk_metrics=risk_metrics,
            confidence_level=confidence_level
        )
    
    def optimize_promotion_portfolio_advanced(self, available_categories: List[str],
                                            budget_constraint: float,
                                            category_costs: Dict[str, float],
                                            risk_tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Advanced promotion portfolio optimization considering cross-effects and risk
        """
        logger.info("Running advanced portfolio optimization...")
        
        # Generate all possible combinations within budget
        from itertools import combinations
        
        feasible_combinations = []
        
        # Check all possible combinations
        for r in range(1, len(available_categories) + 1):
            for combo in combinations(available_categories, r):
                total_cost = sum(category_costs.get(cat, 0) for cat in combo)
                if total_cost <= budget_constraint:
                    feasible_combinations.append(combo)
        
        if not feasible_combinations:
            return {"error": "No feasible combinations within budget"}
        
        # Evaluate each combination
        combination_results = []
        
        for combo in feasible_combinations:
            # Create scenario
            scenario = {
                'promoted_categories': list(combo),
                'promotion_lifts': {cat: 0.2 for cat in combo}  # 20% lift for all
            }
            
            # Run simulation
            sim_result = self._run_portfolio_simulation(scenario, 500)  # Reduced simulations for speed
            
            # Calculate risk-adjusted return
            expected_return = sim_result.risk_metrics['portfolio_expected_return']
            portfolio_std = sim_result.risk_metrics['portfolio_std']
            
            # Sharpe-like ratio (return per unit of risk)
            risk_adjusted_return = expected_return / portfolio_std if portfolio_std > 0 else 0
            
            # Check risk tolerance
            downside_risk = abs(sim_result.risk_metrics['portfolio_5th_percentile'])
            meets_risk_tolerance = downside_risk <= risk_tolerance
            
            combination_results.append({
                'categories': combo,
                'total_cost': sum(category_costs.get(cat, 0) for cat in combo),
                'expected_return': expected_return,
                'risk_adjusted_return': risk_adjusted_return,
                'portfolio_std': portfolio_std,
                'confidence_level': sim_result.confidence_level,
                'meets_risk_tolerance': meets_risk_tolerance,
                'simulation_result': sim_result
            })
        
        # Filter by risk tolerance and sort by risk-adjusted return
        acceptable_combinations = [r for r in combination_results if r['meets_risk_tolerance']]
        
        if not acceptable_combinations:
            # If no combinations meet risk tolerance, take the least risky
            acceptable_combinations = sorted(combination_results, key=lambda x: x['portfolio_std'])[:3]
        
        # Sort by risk-adjusted return
        best_combinations = sorted(acceptable_combinations, key=lambda x: x['risk_adjusted_return'], reverse=True)
        
        optimal_combination = best_combinations[0]
        
        return {
            'optimal_portfolio': {
                'categories': optimal_combination['categories'],
                'total_cost': optimal_combination['total_cost'],
                'expected_return': optimal_combination['expected_return'],
                'risk_adjusted_return': optimal_combination['risk_adjusted_return'],
                'confidence_level': optimal_combination['confidence_level']
            },
            'alternative_portfolios': best_combinations[1:4],  # Top 3 alternatives
            'simulation_details': optimal_combination['simulation_result'],
            'optimization_method': 'monte_carlo_risk_adjusted'
        }