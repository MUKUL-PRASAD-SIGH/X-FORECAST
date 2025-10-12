"""
Hierarchical Forecasting Engine for X-FORECAST
Implements SKU × Location × Channel structure with MinT reconciliation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance

logger = logging.getLogger(__name__)

class HierarchyLevel(Enum):
    TOTAL = "total"
    REGION = "region"
    LOCATION = "location"
    CATEGORY = "category"
    CHANNEL = "channel"
    SKU = "sku"
    SKU_LOCATION = "sku_location"
    SKU_CHANNEL = "sku_channel"
    SKU_LOCATION_CHANNEL = "sku_location_channel"

class CovarianceMethod(Enum):
    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    SAMPLE = "sample"
    SHRINKAGE_LW = "ledoit_wolf"
    SHRINKAGE_OAS = "oracle_approximating_shrinkage"
    SHRINKAGE_BASIC = "basic_shrinkage"
    STRUCTURED = "structured"

class ReconciliationMethod(Enum):
    MINT = "mint"  # Minimum Trace
    OLS = "ols"    # Ordinary Least Squares
    WLS = "wls"    # Weighted Least Squares
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"

@dataclass
class HierarchyNode:
    """Represents a node in the forecasting hierarchy"""
    level: HierarchyLevel
    key: str
    parent: Optional[str]
    children: List[str]
    aggregation_weights: Dict[str, float]
    forecast: Optional[pd.Series] = None
    forecast_error: Optional[float] = None

@dataclass
class CovarianceEstimationResult:
    """Result of covariance matrix estimation"""
    covariance_matrix: np.ndarray
    method_used: CovarianceMethod
    shrinkage_parameter: Optional[float]
    condition_number: float
    estimation_error: Optional[float]

@dataclass
class ReconciliationValidationResult:
    """Result of reconciliation method validation"""
    method: ReconciliationMethod
    is_mathematically_correct: bool
    coherence_preserved: bool
    numerical_stability: float
    computation_time: float
    validation_errors: List[str]

@dataclass
class HierarchicalForecast:
    """Complete hierarchical forecast with reconciliation"""
    base_forecasts: Dict[str, pd.Series]
    reconciled_forecasts: Dict[str, pd.Series]
    reconciliation_method: ReconciliationMethod
    coherence_score: float
    hierarchy_structure: Dict[str, HierarchyNode]
    forecast_errors: Dict[str, float]
    aggregation_matrix: np.ndarray
    covariance_estimation: Optional[CovarianceEstimationResult] = None
    validation_result: Optional[ReconciliationValidationResult] = None

class HierarchicalForecaster:
    """
    Hierarchical forecasting with MinT reconciliation
    Supports SKU × Location × Channel structure
    """
    
    def __init__(self, 
                 covariance_method: CovarianceMethod = CovarianceMethod.SHRINKAGE_LW,
                 reconciliation_method: ReconciliationMethod = ReconciliationMethod.MINT):
        self.hierarchy_structure = {}
        self.aggregation_matrix = None
        self.reconciliation_weights = None
        self.scaler = StandardScaler()
        self.covariance_method = covariance_method
        self.reconciliation_method = reconciliation_method
        self.historical_errors = {}  # Store historical forecast errors for covariance estimation
        
    def build_hierarchy(self, data: pd.DataFrame) -> Dict[str, HierarchyNode]:
        """
        Build hierarchical structure from data
        Expected columns: sku, location, channel, date, demand
        """
        hierarchy = {}
        
        # Get unique values for each dimension
        skus = data['sku'].unique()
        locations = data['location'].unique()
        channels = data['channel'].unique()
        
        # Build hierarchy nodes
        # Level 0: Total
        hierarchy['total'] = HierarchyNode(
            level=HierarchyLevel.TOTAL,
            key='total',
            parent=None,
            children=[],
            aggregation_weights={}
        )
        
        # Level 1: Regions (aggregate locations)
        regions = self._group_locations_to_regions(locations)
        for region, region_locations in regions.items():
            hierarchy[region] = HierarchyNode(
                level=HierarchyLevel.REGION,
                key=region,
                parent='total',
                children=[],
                aggregation_weights={}
            )
            hierarchy['total'].children.append(region)
        
        # Level 2: Locations
        for location in locations:
            region = self._get_region_for_location(location, regions)
            hierarchy[location] = HierarchyNode(
                level=HierarchyLevel.LOCATION,
                key=location,
                parent=region,
                children=[],
                aggregation_weights={}
            )
            hierarchy[region].children.append(location)
        
        # Level 3: Categories (aggregate SKUs)
        categories = self._group_skus_to_categories(skus)
        for category, category_skus in categories.items():
            hierarchy[category] = HierarchyNode(
                level=HierarchyLevel.CATEGORY,
                key=category,
                parent='total',
                children=[],
                aggregation_weights={}
            )
        
        # Level 4: Channels
        for channel in channels:
            hierarchy[channel] = HierarchyNode(
                level=HierarchyLevel.CHANNEL,
                key=channel,
                parent='total',
                children=[],
                aggregation_weights={}
            )
            hierarchy['total'].children.append(channel)
        
        # Level 5: SKUs
        for sku in skus:
            category = self._get_category_for_sku(sku, categories)
            hierarchy[sku] = HierarchyNode(
                level=HierarchyLevel.SKU,
                key=sku,
                parent=category,
                children=[],
                aggregation_weights={}
            )
            hierarchy[category].children.append(sku)
        
        # Level 6: SKU × Location
        for sku in skus:
            for location in locations:
                key = f"{sku}_{location}"
                if self._combination_exists(data, sku=sku, location=location):
                    hierarchy[key] = HierarchyNode(
                        level=HierarchyLevel.SKU_LOCATION,
                        key=key,
                        parent=sku,
                        children=[],
                        aggregation_weights={}
                    )
                    hierarchy[sku].children.append(key)
        
        # Level 7: SKU × Channel
        for sku in skus:
            for channel in channels:
                key = f"{sku}_{channel}"
                if self._combination_exists(data, sku=sku, channel=channel):
                    hierarchy[key] = HierarchyNode(
                        level=HierarchyLevel.SKU_CHANNEL,
                        key=key,
                        parent=sku,
                        children=[],
                        aggregation_weights={}
                    )
                    hierarchy[sku].children.append(key)
        
        # Level 8: SKU × Location × Channel (bottom level)
        for sku in skus:
            for location in locations:
                for channel in channels:
                    key = f"{sku}_{location}_{channel}"
                    if self._combination_exists(data, sku=sku, location=location, channel=channel):
                        hierarchy[key] = HierarchyNode(
                            level=HierarchyLevel.SKU_LOCATION_CHANNEL,
                            key=key,
                            parent=f"{sku}_{location}",
                            children=[],
                            aggregation_weights={}
                        )
                        if f"{sku}_{location}" in hierarchy:
                            hierarchy[f"{sku}_{location}"].children.append(key)
        
        self.hierarchy_structure = hierarchy
        return hierarchy
    
    def _group_locations_to_regions(self, locations: List[str]) -> Dict[str, List[str]]:
        """Group locations into regions based on naming patterns or geography"""
        regions = {}
        for location in locations:
            # Simple grouping by first letter or prefix
            region = f"region_{location[0].upper()}"
            if region not in regions:
                regions[region] = []
            regions[region].append(location)
        return regions
    
    def _group_skus_to_categories(self, skus: List[str]) -> Dict[str, List[str]]:
        """Group SKUs into categories based on naming patterns"""
        categories = {}
        for sku in skus:
            # Simple grouping by prefix
            category = f"category_{sku.split('-')[0] if '-' in sku else sku[:3]}"
            if category not in categories:
                categories[category] = []
            categories[category].append(sku)
        return categories
    
    def _get_region_for_location(self, location: str, regions: Dict[str, List[str]]) -> str:
        """Get region for a specific location"""
        for region, locations in regions.items():
            if location in locations:
                return region
        return f"region_{location[0].upper()}"
    
    def _get_category_for_sku(self, sku: str, categories: Dict[str, List[str]]) -> str:
        """Get category for a specific SKU"""
        for category, skus in categories.items():
            if sku in skus:
                return category
        return f"category_{sku.split('-')[0] if '-' in sku else sku[:3]}"
    
    def _combination_exists(self, data: pd.DataFrame, **kwargs) -> bool:
        """Check if a combination of dimensions exists in the data"""
        mask = pd.Series([True] * len(data))
        for key, value in kwargs.items():
            if key in data.columns:
                mask &= (data[key] == value)
        return mask.sum() > 0
    
    def generate_base_forecasts(self, data: pd.DataFrame, horizon: int = 12) -> Dict[str, pd.Series]:
        """
        Generate base forecasts for all hierarchy levels
        """
        from ..ensemble import EnsembleForecaster
        
        base_forecasts = {}
        forecaster = EnsembleForecaster()
        
        # Generate forecasts for each node in hierarchy
        for node_key, node in self.hierarchy_structure.items():
            try:
                # Aggregate data for this node
                node_data = self._aggregate_data_for_node(data, node_key, node)
                
                if len(node_data) > 0:
                    # Fit ensemble model and generate forecast
                    forecaster.fit(node_data['demand'])
                    forecast = forecaster.forecast(horizon)
                    base_forecasts[node_key] = forecast
                    
                    # Calculate forecast error (using last 12 periods if available)
                    if len(node_data) >= 24:
                        train_data = node_data['demand'][:-12]
                        test_data = node_data['demand'][-12:]
                        forecaster.fit(train_data)
                        test_forecast = forecaster.forecast(12)
                        error = np.mean(np.abs(test_data - test_forecast))
                        node.forecast_error = error
                    
                    node.forecast = forecast
                    
            except Exception as e:
                logger.warning(f"Failed to generate forecast for {node_key}: {e}")
                # Generate dummy forecast
                base_forecasts[node_key] = pd.Series([0] * horizon)
        
        return base_forecasts
    
    def _aggregate_data_for_node(self, data: pd.DataFrame, node_key: str, node: HierarchyNode) -> pd.DataFrame:
        """Aggregate data for a specific hierarchy node"""
        if node_key == 'total':
            # Total aggregation
            return data.groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.REGION:
            # Region aggregation
            region_locations = [child for child in node.children]
            mask = data['location'].isin(region_locations)
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.LOCATION:
            # Location aggregation
            mask = data['location'] == node_key
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.CATEGORY:
            # Category aggregation
            category_skus = [child for child in node.children]
            mask = data['sku'].isin(category_skus)
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.CHANNEL:
            # Channel aggregation
            mask = data['channel'] == node_key
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.SKU:
            # SKU aggregation
            mask = data['sku'] == node_key
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.SKU_LOCATION:
            # SKU × Location
            sku, location = node_key.split('_', 1)
            mask = (data['sku'] == sku) & (data['location'] == location)
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.SKU_CHANNEL:
            # SKU × Channel
            sku, channel = node_key.split('_', 1)
            mask = (data['sku'] == sku) & (data['channel'] == channel)
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        elif node.level == HierarchyLevel.SKU_LOCATION_CHANNEL:
            # SKU × Location × Channel (bottom level)
            parts = node_key.split('_')
            sku, location, channel = parts[0], parts[1], parts[2]
            mask = (data['sku'] == sku) & (data['location'] == location) & (data['channel'] == channel)
            return data[mask].groupby('date')['demand'].sum().reset_index()
        
        return pd.DataFrame()
    
    def build_aggregation_matrix(self) -> np.ndarray:
        """
        Build aggregation matrix S for hierarchical reconciliation
        S matrix maps bottom-level forecasts to all levels
        """
        # Get bottom level nodes (SKU × Location × Channel)
        bottom_nodes = [key for key, node in self.hierarchy_structure.items() 
                       if node.level == HierarchyLevel.SKU_LOCATION_CHANNEL]
        
        # Get all nodes
        all_nodes = list(self.hierarchy_structure.keys())
        
        # Initialize aggregation matrix
        S = np.zeros((len(all_nodes), len(bottom_nodes)))
        
        for i, node_key in enumerate(all_nodes):
            node = self.hierarchy_structure[node_key]
            
            if node.level == HierarchyLevel.SKU_LOCATION_CHANNEL:
                # Bottom level - identity mapping
                j = bottom_nodes.index(node_key)
                S[i, j] = 1.0
            else:
                # Upper level - sum of children
                descendants = self._get_all_descendants(node_key)
                bottom_descendants = [d for d in descendants if d in bottom_nodes]
                
                for desc in bottom_descendants:
                    j = bottom_nodes.index(desc)
                    S[i, j] = 1.0
        
        self.aggregation_matrix = S
        return S
    
    def _get_all_descendants(self, node_key: str) -> List[str]:
        """Get all descendant nodes recursively"""
        descendants = []
        node = self.hierarchy_structure[node_key]
        
        for child in node.children:
            descendants.append(child)
            descendants.extend(self._get_all_descendants(child))
        
        return descendants
    
    def mint_reconciliation(self, base_forecasts: Dict[str, pd.Series], 
                           historical_errors: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Dict[str, pd.Series], CovarianceEstimationResult]:
        """
        Enhanced MinT (Minimum Trace) reconciliation for hierarchical coherence
        
        Args:
            base_forecasts: Base forecasts for all hierarchy nodes
            historical_errors: Optional historical forecast errors for covariance estimation
            
        Returns:
            Tuple of (reconciled_forecasts, covariance_estimation_result)
        """
        if self.aggregation_matrix is None:
            self.build_aggregation_matrix()
        
        S = self.aggregation_matrix
        horizon = len(list(base_forecasts.values())[0])
        
        # Stack base forecasts
        y_hat = np.array([base_forecasts[key].values for key in self.hierarchy_structure.keys()])
        
        # Calculate enhanced forecast error covariance matrix
        covariance_result = self._calculate_error_covariance_matrix(historical_errors)
        W = covariance_result.covariance_matrix
        
        # MinT reconciliation formula: S * (S' * W^-1 * S)^-1 * S' * W^-1 * y_hat
        try:
            # Use pseudo-inverse for numerical stability
            W_inv = np.linalg.pinv(W)
            
            # Calculate middle term with regularization if needed
            middle_matrix = S.T @ W_inv @ S
            
            # Add small regularization if matrix is ill-conditioned
            if np.linalg.cond(middle_matrix) > 1e12:
                regularization = 1e-6 * np.eye(middle_matrix.shape[0])
                middle_matrix += regularization
                logger.warning("Added regularization to middle matrix for numerical stability")
            
            middle_term = np.linalg.pinv(middle_matrix)
            reconciliation_matrix = S @ middle_term @ S.T @ W_inv
            
            reconciled_forecasts = {}
            
            for t in range(horizon):
                y_t = y_hat[:, t]
                y_tilde = reconciliation_matrix @ y_t
                
                for i, node_key in enumerate(self.hierarchy_structure.keys()):
                    if node_key not in reconciled_forecasts:
                        reconciled_forecasts[node_key] = []
                    reconciled_forecasts[node_key].append(y_tilde[i])
            
            # Convert to pandas Series
            for node_key in reconciled_forecasts:
                reconciled_forecasts[node_key] = pd.Series(reconciled_forecasts[node_key])
            
            return reconciled_forecasts, covariance_result
            
        except Exception as e:
            logger.error(f"MinT reconciliation failed: {e}")
            # Return base forecasts if reconciliation fails
            return base_forecasts, covariance_result
    
    def _calculate_error_covariance_matrix(self, historical_errors: Optional[Dict[str, np.ndarray]] = None) -> CovarianceEstimationResult:
        """
        Enhanced error covariance matrix calculation with multiple estimation methods
        
        Args:
            historical_errors: Dictionary of historical forecast errors for each node
            
        Returns:
            CovarianceEstimationResult with estimated covariance matrix and metadata
        """
        n_nodes = len(self.hierarchy_structure)
        node_keys = list(self.hierarchy_structure.keys())
        
        # Use provided historical errors or stored ones
        if historical_errors is None:
            historical_errors = self.historical_errors
        
        # Check if we have sufficient historical error data
        has_historical_data = (
            len(historical_errors) > 0 and 
            all(len(errors) >= 10 for errors in historical_errors.values())
        )
        
        if not has_historical_data:
            # Fall back to diagonal method with forecast errors
            return self._calculate_diagonal_covariance_matrix()
        
        # Prepare error matrix for covariance estimation
        min_length = min(len(errors) for errors in historical_errors.values())
        error_matrix = np.zeros((min_length, n_nodes))
        
        for i, node_key in enumerate(node_keys):
            if node_key in historical_errors:
                error_matrix[:, i] = historical_errors[node_key][-min_length:]
            else:
                # Use mean error if specific node data not available
                mean_error = np.mean([np.mean(errors) for errors in historical_errors.values()])
                error_matrix[:, i] = np.random.normal(0, mean_error, min_length)
        
        # Apply selected covariance estimation method
        if self.covariance_method == CovarianceMethod.IDENTITY:
            return self._calculate_identity_covariance_matrix()
        
        elif self.covariance_method == CovarianceMethod.DIAGONAL:
            return self._calculate_diagonal_covariance_matrix()
        
        elif self.covariance_method == CovarianceMethod.SAMPLE:
            return self._calculate_sample_covariance_matrix(error_matrix)
        
        elif self.covariance_method == CovarianceMethod.SHRINKAGE_LW:
            return self._calculate_ledoit_wolf_covariance_matrix(error_matrix)
        
        elif self.covariance_method == CovarianceMethod.SHRINKAGE_OAS:
            return self._calculate_oas_covariance_matrix(error_matrix)
        
        elif self.covariance_method == CovarianceMethod.SHRINKAGE_BASIC:
            return self._calculate_basic_shrinkage_covariance_matrix(error_matrix)
        
        elif self.covariance_method == CovarianceMethod.STRUCTURED:
            return self._calculate_structured_covariance_matrix(error_matrix)
        
        else:
            # Default to Ledoit-Wolf shrinkage
            return self._calculate_ledoit_wolf_covariance_matrix(error_matrix)
    
    def _calculate_identity_covariance_matrix(self) -> CovarianceEstimationResult:
        """Calculate identity covariance matrix"""
        n_nodes = len(self.hierarchy_structure)
        W = np.eye(n_nodes)
        
        return CovarianceEstimationResult(
            covariance_matrix=W,
            method_used=CovarianceMethod.IDENTITY,
            shrinkage_parameter=None,
            condition_number=1.0,
            estimation_error=None
        )
    
    def _calculate_diagonal_covariance_matrix(self) -> CovarianceEstimationResult:
        """Calculate diagonal covariance matrix using forecast errors"""
        n_nodes = len(self.hierarchy_structure)
        
        # Use forecast errors if available
        errors = []
        for node_key, node in self.hierarchy_structure.items():
            if node.forecast_error is not None:
                errors.append(node.forecast_error)
            else:
                errors.append(1.0)  # Default error
        
        # Diagonal covariance matrix with forecast errors
        W = np.diag(np.array(errors) ** 2)
        
        return CovarianceEstimationResult(
            covariance_matrix=W,
            method_used=CovarianceMethod.DIAGONAL,
            shrinkage_parameter=None,
            condition_number=np.linalg.cond(W),
            estimation_error=None
        )
    
    def _calculate_sample_covariance_matrix(self, error_matrix: np.ndarray) -> CovarianceEstimationResult:
        """Calculate sample covariance matrix"""
        try:
            W = np.cov(error_matrix.T)
            
            # Ensure positive definite
            eigenvals = np.linalg.eigvals(W)
            if np.min(eigenvals) <= 0:
                W += np.eye(W.shape[0]) * (abs(np.min(eigenvals)) + 1e-6)
            
            return CovarianceEstimationResult(
                covariance_matrix=W,
                method_used=CovarianceMethod.SAMPLE,
                shrinkage_parameter=None,
                condition_number=np.linalg.cond(W),
                estimation_error=None
            )
        
        except Exception as e:
            logger.warning(f"Sample covariance estimation failed: {e}")
            return self._calculate_diagonal_covariance_matrix()
    
    def _calculate_ledoit_wolf_covariance_matrix(self, error_matrix: np.ndarray) -> CovarianceEstimationResult:
        """Calculate Ledoit-Wolf shrinkage covariance matrix"""
        try:
            lw = LedoitWolf()
            W = lw.fit(error_matrix).covariance_
            shrinkage = lw.shrinkage_
            
            return CovarianceEstimationResult(
                covariance_matrix=W,
                method_used=CovarianceMethod.SHRINKAGE_LW,
                shrinkage_parameter=shrinkage,
                condition_number=np.linalg.cond(W),
                estimation_error=None
            )
        
        except Exception as e:
            logger.warning(f"Ledoit-Wolf covariance estimation failed: {e}")
            return self._calculate_diagonal_covariance_matrix()
    
    def _calculate_oas_covariance_matrix(self, error_matrix: np.ndarray) -> CovarianceEstimationResult:
        """Calculate Oracle Approximating Shrinkage covariance matrix"""
        try:
            oas = OAS()
            W = oas.fit(error_matrix).covariance_
            shrinkage = oas.shrinkage_
            
            return CovarianceEstimationResult(
                covariance_matrix=W,
                method_used=CovarianceMethod.SHRINKAGE_OAS,
                shrinkage_parameter=shrinkage,
                condition_number=np.linalg.cond(W),
                estimation_error=None
            )
        
        except Exception as e:
            logger.warning(f"OAS covariance estimation failed: {e}")
            return self._calculate_diagonal_covariance_matrix()
    
    def _calculate_basic_shrinkage_covariance_matrix(self, error_matrix: np.ndarray, shrinkage: float = 0.1) -> CovarianceEstimationResult:
        """Calculate basic shrinkage covariance matrix"""
        try:
            shrunk_cov = ShrunkCovariance(shrinkage=shrinkage)
            W = shrunk_cov.fit(error_matrix).covariance_
            
            return CovarianceEstimationResult(
                covariance_matrix=W,
                method_used=CovarianceMethod.SHRINKAGE_BASIC,
                shrinkage_parameter=shrinkage,
                condition_number=np.linalg.cond(W),
                estimation_error=None
            )
        
        except Exception as e:
            logger.warning(f"Basic shrinkage covariance estimation failed: {e}")
            return self._calculate_diagonal_covariance_matrix()
    
    def _calculate_structured_covariance_matrix(self, error_matrix: np.ndarray) -> CovarianceEstimationResult:
        """
        Calculate structured covariance matrix based on hierarchy relationships
        Assumes higher correlation between nodes at similar hierarchy levels
        """
        try:
            n_nodes = len(self.hierarchy_structure)
            node_keys = list(self.hierarchy_structure.keys())
            
            # Start with sample covariance
            sample_cov = np.cov(error_matrix.T)
            
            # Create structure matrix based on hierarchy relationships
            structure_matrix = np.eye(n_nodes)
            
            for i, node_i in enumerate(node_keys):
                for j, node_j in enumerate(node_keys):
                    if i != j:
                        # Calculate hierarchy distance
                        distance = self._calculate_hierarchy_distance(node_i, node_j)
                        # Higher correlation for closer nodes
                        correlation_factor = np.exp(-distance / 2.0)
                        structure_matrix[i, j] = correlation_factor
            
            # Apply structure to sample covariance
            # Convert correlation structure to covariance structure
            std_devs = np.sqrt(np.diag(sample_cov))
            structured_cov = np.outer(std_devs, std_devs) * structure_matrix
            
            # Ensure positive definite
            eigenvals = np.linalg.eigvals(structured_cov)
            if np.min(eigenvals) <= 0:
                structured_cov += np.eye(n_nodes) * (abs(np.min(eigenvals)) + 1e-6)
            
            return CovarianceEstimationResult(
                covariance_matrix=structured_cov,
                method_used=CovarianceMethod.STRUCTURED,
                shrinkage_parameter=None,
                condition_number=np.linalg.cond(structured_cov),
                estimation_error=None
            )
        
        except Exception as e:
            logger.warning(f"Structured covariance estimation failed: {e}")
            return self._calculate_diagonal_covariance_matrix()
    
    def _calculate_hierarchy_distance(self, node_i: str, node_j: str) -> float:
        """Calculate distance between two nodes in hierarchy"""
        node_i_obj = self.hierarchy_structure[node_i]
        node_j_obj = self.hierarchy_structure[node_j]
        
        # Same level nodes have distance 1
        if node_i_obj.level == node_j_obj.level:
            return 1.0
        
        # Parent-child relationship has distance 0.5
        if node_i_obj.parent == node_j or node_j_obj.parent == node_i:
            return 0.5
        
        # Sibling nodes (same parent) have distance 0.8
        if (node_i_obj.parent is not None and 
            node_j_obj.parent is not None and 
            node_i_obj.parent == node_j_obj.parent):
            return 0.8
        
        # Different levels have distance based on level difference
        level_order = {
            HierarchyLevel.TOTAL: 0,
            HierarchyLevel.REGION: 1,
            HierarchyLevel.LOCATION: 2,
            HierarchyLevel.CATEGORY: 1,
            HierarchyLevel.CHANNEL: 1,
            HierarchyLevel.SKU: 2,
            HierarchyLevel.SKU_LOCATION: 3,
            HierarchyLevel.SKU_CHANNEL: 3,
            HierarchyLevel.SKU_LOCATION_CHANNEL: 4
        }
        
        level_diff = abs(level_order[node_i_obj.level] - level_order[node_j_obj.level])
        return 1.0 + level_diff * 0.5
    
    def store_historical_errors(self, node_key: str, errors: np.ndarray):
        """Store historical forecast errors for covariance estimation"""
        if node_key not in self.historical_errors:
            self.historical_errors[node_key] = []
        
        # Keep only recent errors (last 100 periods)
        max_history = 100
        if len(self.historical_errors[node_key]) >= max_history:
            self.historical_errors[node_key] = self.historical_errors[node_key][-(max_history-len(errors)):]
        
        self.historical_errors[node_key].extend(errors)
    
    def update_covariance_method(self, method: CovarianceMethod):
        """Update covariance estimation method"""
        self.covariance_method = method
    
    def ols_reconciliation(self, base_forecasts: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], ReconciliationValidationResult]:
        """
        OLS (Ordinary Least Squares) reconciliation
        Uses identity matrix as weights (assumes equal variance)
        """
        import time
        start_time = time.time()
        
        if self.aggregation_matrix is None:
            self.build_aggregation_matrix()
        
        S = self.aggregation_matrix
        horizon = len(list(base_forecasts.values())[0])
        
        # Stack base forecasts
        y_hat = np.array([base_forecasts[key].values for key in self.hierarchy_structure.keys()])
        
        validation_errors = []
        
        try:
            # OLS reconciliation formula: S * (S' * S)^-1 * S' * y_hat
            middle_matrix = S.T @ S
            
            # Check for numerical issues
            condition_number = np.linalg.cond(middle_matrix)
            if condition_number > 1e12:
                regularization = 1e-6 * np.eye(middle_matrix.shape[0])
                middle_matrix += regularization
                validation_errors.append(f"Added regularization due to high condition number: {condition_number}")
            
            middle_term = np.linalg.pinv(middle_matrix)
            reconciliation_matrix = S @ middle_term @ S.T
            
            reconciled_forecasts = {}
            
            for t in range(horizon):
                y_t = y_hat[:, t]
                y_tilde = reconciliation_matrix @ y_t
                
                for i, node_key in enumerate(self.hierarchy_structure.keys()):
                    if node_key not in reconciled_forecasts:
                        reconciled_forecasts[node_key] = []
                    reconciled_forecasts[node_key].append(y_tilde[i])
            
            # Convert to pandas Series
            for node_key in reconciled_forecasts:
                reconciled_forecasts[node_key] = pd.Series(reconciled_forecasts[node_key])
            
            # Validate reconciliation
            coherence_preserved = self._validate_coherence(reconciled_forecasts)
            
            computation_time = time.time() - start_time
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.OLS,
                is_mathematically_correct=True,
                coherence_preserved=coherence_preserved,
                numerical_stability=1.0 / condition_number,
                computation_time=computation_time,
                validation_errors=validation_errors
            )
            
            return reconciled_forecasts, validation_result
            
        except Exception as e:
            logger.error(f"OLS reconciliation failed: {e}")
            validation_errors.append(f"OLS reconciliation failed: {str(e)}")
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.OLS,
                is_mathematically_correct=False,
                coherence_preserved=False,
                numerical_stability=0.0,
                computation_time=time.time() - start_time,
                validation_errors=validation_errors
            )
            
            return base_forecasts, validation_result
    
    def wls_reconciliation(self, base_forecasts: Dict[str, pd.Series], 
                          weights: Optional[np.ndarray] = None) -> Tuple[Dict[str, pd.Series], ReconciliationValidationResult]:
        """
        WLS (Weighted Least Squares) reconciliation
        Uses diagonal weight matrix based on forecast accuracy or provided weights
        """
        import time
        start_time = time.time()
        
        if self.aggregation_matrix is None:
            self.build_aggregation_matrix()
        
        S = self.aggregation_matrix
        horizon = len(list(base_forecasts.values())[0])
        
        # Stack base forecasts
        y_hat = np.array([base_forecasts[key].values for key in self.hierarchy_structure.keys()])
        
        validation_errors = []
        
        try:
            # Create weight matrix
            if weights is None:
                # Use inverse of forecast errors as weights
                weights = []
                for node_key, node in self.hierarchy_structure.items():
                    if node.forecast_error is not None and node.forecast_error > 0:
                        weights.append(1.0 / node.forecast_error)
                    else:
                        weights.append(1.0)
                W = np.diag(weights)
            else:
                W = np.diag(weights) if weights.ndim == 1 else weights
            
            # WLS reconciliation formula: S * (S' * W * S)^-1 * S' * W * y_hat
            middle_matrix = S.T @ W @ S
            
            # Check for numerical issues
            condition_number = np.linalg.cond(middle_matrix)
            if condition_number > 1e12:
                regularization = 1e-6 * np.eye(middle_matrix.shape[0])
                middle_matrix += regularization
                validation_errors.append(f"Added regularization due to high condition number: {condition_number}")
            
            middle_term = np.linalg.pinv(middle_matrix)
            reconciliation_matrix = S @ middle_term @ S.T @ W
            
            reconciled_forecasts = {}
            
            for t in range(horizon):
                y_t = y_hat[:, t]
                y_tilde = reconciliation_matrix @ y_t
                
                for i, node_key in enumerate(self.hierarchy_structure.keys()):
                    if node_key not in reconciled_forecasts:
                        reconciled_forecasts[node_key] = []
                    reconciled_forecasts[node_key].append(y_tilde[i])
            
            # Convert to pandas Series
            for node_key in reconciled_forecasts:
                reconciled_forecasts[node_key] = pd.Series(reconciled_forecasts[node_key])
            
            # Validate reconciliation
            coherence_preserved = self._validate_coherence(reconciled_forecasts)
            
            computation_time = time.time() - start_time
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.WLS,
                is_mathematically_correct=True,
                coherence_preserved=coherence_preserved,
                numerical_stability=1.0 / condition_number,
                computation_time=computation_time,
                validation_errors=validation_errors
            )
            
            return reconciled_forecasts, validation_result
            
        except Exception as e:
            logger.error(f"WLS reconciliation failed: {e}")
            validation_errors.append(f"WLS reconciliation failed: {str(e)}")
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.WLS,
                is_mathematically_correct=False,
                coherence_preserved=False,
                numerical_stability=0.0,
                computation_time=time.time() - start_time,
                validation_errors=validation_errors
            )
            
            return base_forecasts, validation_result
    
    def bottom_up_reconciliation(self, base_forecasts: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], ReconciliationValidationResult]:
        """
        Bottom-up reconciliation - aggregate from bottom level
        """
        import time
        start_time = time.time()
        
        validation_errors = []
        
        try:
            # Get bottom level forecasts
            bottom_nodes = [key for key, node in self.hierarchy_structure.items() 
                           if node.level == HierarchyLevel.SKU_LOCATION_CHANNEL]
            
            if not bottom_nodes:
                validation_errors.append("No bottom level nodes found")
                raise ValueError("No bottom level nodes found")
            
            reconciled_forecasts = {}
            
            # Start with bottom level forecasts
            for node_key in bottom_nodes:
                if node_key in base_forecasts:
                    reconciled_forecasts[node_key] = base_forecasts[node_key].copy()
            
            # Aggregate upwards
            self._aggregate_upwards(reconciled_forecasts)
            
            # Validate reconciliation
            coherence_preserved = self._validate_coherence(reconciled_forecasts)
            
            computation_time = time.time() - start_time
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.BOTTOM_UP,
                is_mathematically_correct=True,
                coherence_preserved=coherence_preserved,
                numerical_stability=1.0,
                computation_time=computation_time,
                validation_errors=validation_errors
            )
            
            return reconciled_forecasts, validation_result
            
        except Exception as e:
            logger.error(f"Bottom-up reconciliation failed: {e}")
            validation_errors.append(f"Bottom-up reconciliation failed: {str(e)}")
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.BOTTOM_UP,
                is_mathematically_correct=False,
                coherence_preserved=False,
                numerical_stability=0.0,
                computation_time=time.time() - start_time,
                validation_errors=validation_errors
            )
            
            return base_forecasts, validation_result
    
    def top_down_reconciliation(self, base_forecasts: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], ReconciliationValidationResult]:
        """
        Top-down reconciliation - disaggregate from top level
        """
        import time
        start_time = time.time()
        
        validation_errors = []
        
        try:
            reconciled_forecasts = {}
            
            # Start with total forecast
            if 'total' in base_forecasts:
                reconciled_forecasts['total'] = base_forecasts['total'].copy()
            else:
                validation_errors.append("No total forecast found")
                raise ValueError("No total forecast found")
            
            # Disaggregate downwards using historical proportions
            self._disaggregate_downwards(reconciled_forecasts, base_forecasts)
            
            # Validate reconciliation
            coherence_preserved = self._validate_coherence(reconciled_forecasts)
            
            computation_time = time.time() - start_time
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.TOP_DOWN,
                is_mathematically_correct=True,
                coherence_preserved=coherence_preserved,
                numerical_stability=1.0,
                computation_time=computation_time,
                validation_errors=validation_errors
            )
            
            return reconciled_forecasts, validation_result
            
        except Exception as e:
            logger.error(f"Top-down reconciliation failed: {e}")
            validation_errors.append(f"Top-down reconciliation failed: {str(e)}")
            
            validation_result = ReconciliationValidationResult(
                method=ReconciliationMethod.TOP_DOWN,
                is_mathematically_correct=False,
                coherence_preserved=False,
                numerical_stability=0.0,
                computation_time=time.time() - start_time,
                validation_errors=validation_errors
            )
            
            return base_forecasts, validation_result
    
    def _aggregate_upwards(self, reconciled_forecasts: Dict[str, pd.Series]):
        """Aggregate forecasts from bottom to top levels"""
        # Process levels from bottom to top
        level_order = [
            HierarchyLevel.SKU_LOCATION_CHANNEL,
            HierarchyLevel.SKU_LOCATION,
            HierarchyLevel.SKU_CHANNEL,
            HierarchyLevel.SKU,
            HierarchyLevel.LOCATION,
            HierarchyLevel.CHANNEL,
            HierarchyLevel.CATEGORY,
            HierarchyLevel.REGION,
            HierarchyLevel.TOTAL
        ]
        
        for level in level_order[1:]:  # Skip bottom level
            nodes_at_level = [key for key, node in self.hierarchy_structure.items() if node.level == level]
            
            for node_key in nodes_at_level:
                node = self.hierarchy_structure[node_key]
                
                if node.children:
                    # Sum children forecasts
                    child_forecasts = []
                    for child_key in node.children:
                        if child_key in reconciled_forecasts:
                            child_forecasts.append(reconciled_forecasts[child_key])
                    
                    if child_forecasts:
                        reconciled_forecasts[node_key] = sum(child_forecasts)
    
    def _disaggregate_downwards(self, reconciled_forecasts: Dict[str, pd.Series], base_forecasts: Dict[str, pd.Series]):
        """Disaggregate forecasts from top to bottom levels using proportions"""
        # Process levels from top to bottom
        level_order = [
            HierarchyLevel.TOTAL,
            HierarchyLevel.REGION,
            HierarchyLevel.CATEGORY,
            HierarchyLevel.CHANNEL,
            HierarchyLevel.LOCATION,
            HierarchyLevel.SKU,
            HierarchyLevel.SKU_LOCATION,
            HierarchyLevel.SKU_CHANNEL,
            HierarchyLevel.SKU_LOCATION_CHANNEL
        ]
        
        for level in level_order[1:]:  # Skip total level
            nodes_at_level = [key for key, node in self.hierarchy_structure.items() if node.level == level]
            
            for node_key in nodes_at_level:
                node = self.hierarchy_structure[node_key]
                
                if node.parent and node.parent in reconciled_forecasts:
                    parent_forecast = reconciled_forecasts[node.parent]
                    
                    # Calculate proportion based on base forecasts
                    siblings = self.hierarchy_structure[node.parent].children
                    sibling_base_total = sum(base_forecasts.get(sibling, pd.Series([0] * len(parent_forecast))) 
                                           for sibling in siblings)
                    
                    if sibling_base_total.sum() > 0:
                        proportion = base_forecasts.get(node_key, pd.Series([0] * len(parent_forecast))) / sibling_base_total
                        proportion = proportion.fillna(1.0 / len(siblings))  # Equal split if no data
                    else:
                        proportion = pd.Series([1.0 / len(siblings)] * len(parent_forecast))
                    
                    reconciled_forecasts[node_key] = parent_forecast * proportion
    
    def _validate_coherence(self, reconciled_forecasts: Dict[str, pd.Series]) -> bool:
        """Validate that reconciled forecasts maintain hierarchical coherence"""
        try:
            tolerance = 1e-6
            
            for node_key, node in self.hierarchy_structure.items():
                if node.children and node_key in reconciled_forecasts:
                    parent_forecast = reconciled_forecasts[node_key]
                    
                    children_sum = pd.Series([0] * len(parent_forecast))
                    for child_key in node.children:
                        if child_key in reconciled_forecasts:
                            children_sum += reconciled_forecasts[child_key]
                    
                    # Check if parent equals sum of children (within tolerance)
                    diff = abs(parent_forecast - children_sum)
                    if (diff > tolerance).any():
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Coherence validation failed: {e}")
            return False
    
    def select_reconciliation_method(self, data: pd.DataFrame, base_forecasts: Dict[str, pd.Series]) -> ReconciliationMethod:
        """
        Automatically select the best reconciliation method based on hierarchy characteristics and data availability
        """
        n_nodes = len(self.hierarchy_structure)
        n_bottom_nodes = len([key for key, node in self.hierarchy_structure.items() 
                             if node.level == HierarchyLevel.SKU_LOCATION_CHANNEL])
        
        # Check data availability
        has_sufficient_history = len(data) >= 24
        has_error_estimates = any(node.forecast_error is not None for node in self.hierarchy_structure.values())
        
        # Calculate hierarchy complexity
        hierarchy_depth = max(self._get_node_depth(node_key) for node_key in self.hierarchy_structure.keys())
        complexity_ratio = n_nodes / n_bottom_nodes if n_bottom_nodes > 0 else 1
        
        # Decision logic
        if not has_sufficient_history or n_nodes < 10:
            # Simple hierarchy or limited data - use bottom-up
            return ReconciliationMethod.BOTTOM_UP
        
        elif complexity_ratio > 3 and has_error_estimates:
            # Complex hierarchy with error estimates - use MinT
            return ReconciliationMethod.MINT
        
        elif has_error_estimates:
            # Moderate complexity with error estimates - use WLS
            return ReconciliationMethod.WLS
        
        else:
            # Default to OLS for moderate complexity without error estimates
            return ReconciliationMethod.OLS
    
    def _get_node_depth(self, node_key: str) -> int:
        """Calculate depth of node in hierarchy"""
        node = self.hierarchy_structure[node_key]
        depth = 0
        current_key = node_key
        
        while self.hierarchy_structure[current_key].parent is not None:
            depth += 1
            current_key = self.hierarchy_structure[current_key].parent
        
        return depth
    
    def validate_reconciliation_method(self, method: ReconciliationMethod, 
                                     base_forecasts: Dict[str, pd.Series]) -> ReconciliationValidationResult:
        """
        Validate a specific reconciliation method with test data
        """
        try:
            if method == ReconciliationMethod.MINT:
                _, validation_result = self.mint_reconciliation(base_forecasts)
            elif method == ReconciliationMethod.OLS:
                _, validation_result = self.ols_reconciliation(base_forecasts)
            elif method == ReconciliationMethod.WLS:
                _, validation_result = self.wls_reconciliation(base_forecasts)
            elif method == ReconciliationMethod.BOTTOM_UP:
                _, validation_result = self.bottom_up_reconciliation(base_forecasts)
            elif method == ReconciliationMethod.TOP_DOWN:
                _, validation_result = self.top_down_reconciliation(base_forecasts)
            else:
                raise ValueError(f"Unknown reconciliation method: {method}")
            
            return validation_result
            
        except Exception as e:
            return ReconciliationValidationResult(
                method=method,
                is_mathematically_correct=False,
                coherence_preserved=False,
                numerical_stability=0.0,
                computation_time=0.0,
                validation_errors=[f"Validation failed: {str(e)}"]
            )
    
    def reconcile_forecasts(self, base_forecasts: Dict[str, pd.Series], 
                           method: Optional[ReconciliationMethod] = None,
                           historical_errors: Optional[Dict[str, np.ndarray]] = None,
                           weights: Optional[np.ndarray] = None) -> Tuple[Dict[str, pd.Series], ReconciliationValidationResult]:
        """
        Unified method to reconcile forecasts using specified or automatically selected method
        
        Args:
            base_forecasts: Base forecasts for all hierarchy nodes
            method: Reconciliation method to use (if None, auto-select)
            historical_errors: Historical forecast errors for MinT covariance estimation
            weights: Custom weights for WLS method
            
        Returns:
            Tuple of (reconciled_forecasts, validation_result)
        """
        # Auto-select method if not specified
        if method is None:
            method = self.reconciliation_method
        
        # Call appropriate reconciliation method
        if method == ReconciliationMethod.MINT:
            return self.mint_reconciliation(base_forecasts, historical_errors)
        elif method == ReconciliationMethod.OLS:
            return self.ols_reconciliation(base_forecasts)
        elif method == ReconciliationMethod.WLS:
            return self.wls_reconciliation(base_forecasts, weights)
        elif method == ReconciliationMethod.BOTTOM_UP:
            return self.bottom_up_reconciliation(base_forecasts)
        elif method == ReconciliationMethod.TOP_DOWN:
            return self.top_down_reconciliation(base_forecasts)
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
    
    def calculate_coherence_score(self, reconciled_forecasts: Dict[str, pd.Series]) -> float:
        """
        Calculate coherence score (0-1) measuring how well forecasts aggregate
        """
        if self.aggregation_matrix is None:
            return 0.0
        
        S = self.aggregation_matrix
        horizon = len(list(reconciled_forecasts.values())[0])
        
        coherence_scores = []
        
        for t in range(horizon):
            # Get forecasts for time t
            y_t = np.array([reconciled_forecasts[key].iloc[t] for key in self.hierarchy_structure.keys()])
            
            # Check aggregation constraints: y_upper = S * y_bottom
            bottom_nodes = [key for key, node in self.hierarchy_structure.items() 
                           if node.level == HierarchyLevel.SKU_LOCATION_CHANNEL]
            
            y_bottom = np.array([reconciled_forecasts[key].iloc[t] for key in bottom_nodes])
            y_upper_expected = S @ y_bottom
            
            # Calculate coherence as 1 - normalized error
            error = np.mean(np.abs(y_t - y_upper_expected))
            total_demand = np.sum(np.abs(y_t))
            
            if total_demand > 0:
                coherence_scores.append(1 - (error / total_demand))
            else:
                coherence_scores.append(1.0)
        
        return np.mean(coherence_scores)
    
    def forecast_hierarchical(self, data: pd.DataFrame, horizon: int = 12, 
                             reconciliation_method: Optional[ReconciliationMethod] = None,
                             historical_errors: Optional[Dict[str, np.ndarray]] = None,
                             auto_select_method: bool = False) -> HierarchicalForecast:
        """
        Complete hierarchical forecasting with enhanced reconciliation
        
        Args:
            data: Historical demand data
            horizon: Forecast horizon
            reconciliation_method: Specific reconciliation method to use
            historical_errors: Optional historical forecast errors for covariance estimation
            auto_select_method: Whether to automatically select the best reconciliation method
        """
        # Build hierarchy structure
        self.build_hierarchy(data)
        
        # Generate base forecasts
        base_forecasts = self.generate_base_forecasts(data, horizon)
        
        # Build aggregation matrix
        self.build_aggregation_matrix()
        
        # Select reconciliation method
        if auto_select_method:
            selected_method = self.select_reconciliation_method(data, base_forecasts)
        elif reconciliation_method is not None:
            selected_method = reconciliation_method
        else:
            selected_method = self.reconciliation_method
        
        # Perform reconciliation
        reconciled_forecasts, validation_result = self.reconcile_forecasts(
            base_forecasts, selected_method, historical_errors
        )
        
        # Calculate coherence score
        coherence_score = self.calculate_coherence_score(reconciled_forecasts)
        
        # Calculate forecast errors
        forecast_errors = {}
        for node_key, node in self.hierarchy_structure.items():
            forecast_errors[node_key] = node.forecast_error or 0.0
        
        # Get covariance estimation result if available
        covariance_result = None
        if selected_method == ReconciliationMethod.MINT and historical_errors:
            covariance_result = self._calculate_error_covariance_matrix(historical_errors)
        
        return HierarchicalForecast(
            base_forecasts=base_forecasts,
            reconciled_forecasts=reconciled_forecasts,
            reconciliation_method=selected_method,
            coherence_score=coherence_score,
            hierarchy_structure=self.hierarchy_structure,
            forecast_errors=forecast_errors,
            aggregation_matrix=self.aggregation_matrix,
            covariance_estimation=covariance_result,
            validation_result=validation_result
        )