"""
Customer Analytics Engine for Ensemble Cyberpunk Integration
Implements comprehensive customer analytics including LTV, cohort analysis, churn prediction, and segmentation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CustomerLTV:
    """Customer Lifetime Value metrics"""
    customer_id: str
    historical_ltv: float
    predicted_ltv: float
    combined_ltv: float
    avg_order_value: float
    purchase_frequency: float
    customer_lifespan_days: int
    ltv_percentile: float
    risk_score: float
    segment: str
    last_updated: datetime

@dataclass
class CohortAnalysis:
    """Cohort analysis results"""
    cohort_month: str
    cohort_size: int
    retention_rates: Dict[int, float]  # month -> retention rate
    revenue_per_cohort: Dict[int, float]  # month -> total revenue
    avg_ltv: float
    churn_rate: float
    months_tracked: int
    analysis_date: datetime

@dataclass
class ChurnRisk:
    """Customer churn risk assessment"""
    customer_id: str
    churn_probability: float
    risk_level: str  # 'Low', 'Medium', 'High', 'Critical'
    key_risk_factors: List[str]
    days_since_last_purchase: int
    purchase_trend: str  # 'Increasing', 'Stable', 'Declining'
    recommended_actions: List[str]
    confidence_score: float
    prediction_date: datetime

@dataclass
class CustomerSegment:
    """Customer segment definition and metrics"""
    segment_id: str
    segment_name: str
    customer_count: int
    avg_ltv: float
    avg_retention_rate: float
    avg_purchase_frequency: float
    avg_order_value: float
    characteristics: List[str]
    growth_rate: float
    revenue_contribution: float
    segment_health_score: float

@dataclass
class CustomerAnalytics:
    """Comprehensive customer analytics results"""
    total_customers: int
    active_customers: int
    new_customers_this_month: int
    churned_customers: int
    overall_ltv: float
    overall_retention_rate: float
    customer_ltvs: List[CustomerLTV]
    cohort_analyses: List[CohortAnalysis]
    churn_risks: List[ChurnRisk]
    customer_segments: List[CustomerSegment]
    key_insights: List[str]
    recommendations: List[str]
    analysis_date: datetime

class CustomerLTVCalculator:
    """Advanced Customer Lifetime Value calculator"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
    
    def calculate_customer_features(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer-level features from sales data"""
        try:
            # Ensure date column is datetime
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            current_date = sales_data['date'].max()
            
            # Group by customer (assuming customer can be identified by region + category combination)
            # In real scenario, there would be a customer_id column
            customer_features = []
            
            # Create synthetic customer IDs based on region and category patterns
            sales_data['customer_id'] = (
                sales_data['region'].astype(str) + '_' + 
                sales_data['product_category'].astype(str) + '_' +
                (sales_data.index % 100).astype(str)  # Simulate multiple customers per region/category
            )
            
            for customer_id, customer_data in sales_data.groupby('customer_id'):
                customer_data = customer_data.sort_values('date')
                
                # Basic metrics
                first_purchase = customer_data['date'].min()
                last_purchase = customer_data['date'].max()
                total_revenue = customer_data['sales_amount'].sum()
                total_transactions = len(customer_data)
                
                # Time-based features
                customer_lifespan_days = (last_purchase - first_purchase).days + 1
                days_since_last_purchase = (current_date - last_purchase).days
                
                # Purchase behavior
                avg_order_value = total_revenue / total_transactions
                purchase_frequency = total_transactions / max(customer_lifespan_days / 30, 1)  # per month
                
                # Trend analysis
                if len(customer_data) >= 3:
                    recent_purchases = customer_data.tail(3)['sales_amount'].mean()
                    early_purchases = customer_data.head(3)['sales_amount'].mean()
                    purchase_trend_score = (recent_purchases - early_purchases) / early_purchases if early_purchases > 0 else 0
                else:
                    purchase_trend_score = 0
                
                # Seasonality and consistency
                monthly_sales = customer_data.groupby(customer_data['date'].dt.to_period('M'))['sales_amount'].sum()
                sales_consistency = 1 / (monthly_sales.std() / monthly_sales.mean() + 1) if len(monthly_sales) > 1 else 1
                
                # Category diversity
                categories_purchased = customer_data['product_category'].nunique()
                
                customer_features.append({
                    'customer_id': customer_id,
                    'first_purchase_date': first_purchase,
                    'last_purchase_date': last_purchase,
                    'total_revenue': total_revenue,
                    'total_transactions': total_transactions,
                    'customer_lifespan_days': customer_lifespan_days,
                    'days_since_last_purchase': days_since_last_purchase,
                    'avg_order_value': avg_order_value,
                    'purchase_frequency': purchase_frequency,
                    'purchase_trend_score': purchase_trend_score,
                    'sales_consistency': sales_consistency,
                    'categories_purchased': categories_purchased,
                    'region': customer_data['region'].iloc[0],
                    'primary_category': customer_data['product_category'].mode().iloc[0]
                })
            
            return pd.DataFrame(customer_features)
            
        except Exception as e:
            logger.error(f"Failed to calculate customer features: {e}")
            return pd.DataFrame()
    
    def calculate_ltv_metrics(self, customer_features: pd.DataFrame) -> List[CustomerLTV]:
        """Calculate LTV metrics for all customers"""
        try:
            ltv_results = []
            
            for _, customer in customer_features.iterrows():
                customer_id = customer['customer_id']
                
                # Historical LTV (actual revenue)
                historical_ltv = customer['total_revenue']
                
                # Predictive LTV calculation
                avg_order_value = customer['avg_order_value']
                purchase_frequency = customer['purchase_frequency']
                customer_lifespan_months = max(customer['customer_lifespan_days'] / 30, 1)
                
                # Estimate future lifespan based on current behavior
                if customer['days_since_last_purchase'] <= 30:
                    estimated_future_months = customer_lifespan_months * 1.5  # Active customer
                elif customer['days_since_last_purchase'] <= 90:
                    estimated_future_months = customer_lifespan_months * 0.8  # Declining
                else:
                    estimated_future_months = customer_lifespan_months * 0.3  # At risk
                
                # Apply trend adjustment
                trend_multiplier = 1 + (customer['purchase_trend_score'] * 0.2)
                trend_multiplier = max(0.5, min(2.0, trend_multiplier))  # Cap between 0.5 and 2.0
                
                # Calculate predicted LTV
                predicted_ltv = (
                    avg_order_value * 
                    purchase_frequency * 
                    estimated_future_months * 
                    trend_multiplier *
                    customer['sales_consistency']
                )
                
                # Combined LTV (weighted average of historical and predicted)
                weight_historical = min(customer_lifespan_months / 12, 0.7)  # More weight to historical if longer history
                weight_predicted = 1 - weight_historical
                combined_ltv = (historical_ltv * weight_historical) + (predicted_ltv * weight_predicted)
                
                # Calculate risk score (higher = more risk)
                risk_score = (
                    (customer['days_since_last_purchase'] / 365) * 0.4 +  # Recency risk
                    (1 / max(customer['purchase_frequency'], 0.1)) * 0.3 +  # Frequency risk
                    (1 / max(customer['sales_consistency'], 0.1)) * 0.3  # Consistency risk
                )
                risk_score = min(risk_score, 1.0)  # Cap at 1.0
                
                ltv_results.append(CustomerLTV(
                    customer_id=customer_id,
                    historical_ltv=historical_ltv,
                    predicted_ltv=predicted_ltv,
                    combined_ltv=combined_ltv,
                    avg_order_value=avg_order_value,
                    purchase_frequency=purchase_frequency,
                    customer_lifespan_days=int(customer['customer_lifespan_days']),
                    ltv_percentile=0.0,  # Will be calculated after all LTVs
                    risk_score=risk_score,
                    segment='',  # Will be assigned by segmentation
                    last_updated=datetime.now()
                ))
            
            # Calculate LTV percentiles
            if ltv_results:
                combined_ltvs = [ltv.combined_ltv for ltv in ltv_results]
                for ltv in ltv_results:
                    ltv.ltv_percentile = (
                        sum(1 for x in combined_ltvs if x <= ltv.combined_ltv) / len(combined_ltvs) * 100
                    )
            
            return ltv_results
            
        except Exception as e:
            logger.error(f"Failed to calculate LTV metrics: {e}")
            return []

class CohortAnalyzer:
    """Advanced cohort analysis for customer retention"""
    
    def analyze_cohorts(self, sales_data: pd.DataFrame) -> List[CohortAnalysis]:
        """Perform comprehensive cohort analysis"""
        try:
            sales_data = sales_data.copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Create customer IDs (same logic as LTV calculator)
            sales_data['customer_id'] = (
                sales_data['region'].astype(str) + '_' + 
                sales_data['product_category'].astype(str) + '_' +
                (sales_data.index % 100).astype(str)
            )
            
            # Get first purchase date for each customer (cohort assignment)
            customer_cohorts = sales_data.groupby('customer_id')['date'].min().reset_index()
            customer_cohorts['cohort_month'] = customer_cohorts['date'].dt.to_period('M')
            
            # Merge cohort info back to sales data
            sales_data = sales_data.merge(
                customer_cohorts[['customer_id', 'cohort_month']], 
                on='customer_id'
            )
            
            # Calculate period number (months since first purchase)
            sales_data['transaction_period'] = sales_data['date'].dt.to_period('M')
            sales_data['period_number'] = (
                sales_data['transaction_period'] - sales_data['cohort_month']
            ).apply(lambda x: x.n)
            
            cohort_analyses = []
            
            for cohort_month, cohort_data in sales_data.groupby('cohort_month'):
                # Get unique customers in this cohort
                cohort_customers = cohort_data['customer_id'].unique()
                cohort_size = len(cohort_customers)
                
                if cohort_size == 0:
                    continue
                
                # Calculate retention rates for each period
                retention_rates = {}
                revenue_per_period = {}
                
                max_period = cohort_data['period_number'].max()
                
                for period in range(max_period + 1):
                    period_data = cohort_data[cohort_data['period_number'] == period]
                    active_customers = period_data['customer_id'].nunique()
                    retention_rate = active_customers / cohort_size
                    total_revenue = period_data['sales_amount'].sum()
                    
                    retention_rates[period] = retention_rate
                    revenue_per_period[period] = total_revenue
                
                # Calculate average LTV for this cohort
                cohort_revenue = cohort_data.groupby('customer_id')['sales_amount'].sum()
                avg_ltv = cohort_revenue.mean()
                
                # Calculate churn rate (customers who haven't purchased in last 3 months)
                latest_period = max(retention_rates.keys())
                if latest_period >= 3:
                    recent_retention = retention_rates.get(latest_period, 0)
                    churn_rate = 1 - recent_retention
                else:
                    churn_rate = 0.0
                
                cohort_analyses.append(CohortAnalysis(
                    cohort_month=str(cohort_month),
                    cohort_size=cohort_size,
                    retention_rates=retention_rates,
                    revenue_per_cohort=revenue_per_period,
                    avg_ltv=avg_ltv,
                    churn_rate=churn_rate,
                    months_tracked=max_period + 1,
                    analysis_date=datetime.now()
                ))
            
            return sorted(cohort_analyses, key=lambda x: x.cohort_month)
            
        except Exception as e:
            logger.error(f"Failed to perform cohort analysis: {e}")
            return []

class ChurnPredictor:
    """Advanced churn prediction model"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        self.feature_columns = []
    
    def prepare_churn_features(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for churn prediction"""
        try:
            features = customer_features.copy()
            
            # Create churn labels (customers who haven't purchased in 90+ days)
            features['is_churned'] = (features['days_since_last_purchase'] > 90).astype(int)
            
            # Additional behavioral features
            features['recency_score'] = np.where(
                features['days_since_last_purchase'] <= 30, 5,
                np.where(features['days_since_last_purchase'] <= 60, 4,
                np.where(features['days_since_last_purchase'] <= 90, 3,
                np.where(features['days_since_last_purchase'] <= 180, 2, 1)))
            )
            
            try:
                features['frequency_score'] = pd.qcut(
                    features['purchase_frequency'], 
                    q=5, 
                    labels=[1, 2, 3, 4, 5], 
                    duplicates='drop'
                ).astype(float)
            except ValueError:
                # Fallback to simple ranking if qcut fails
                features['frequency_score'] = features['purchase_frequency'].rank(pct=True) * 5
            
            try:
                features['monetary_score'] = pd.qcut(
                    features['total_revenue'], 
                    q=5, 
                    labels=[1, 2, 3, 4, 5], 
                    duplicates='drop'
                ).astype(float)
            except ValueError:
                # Fallback to simple ranking if qcut fails
                features['monetary_score'] = features['total_revenue'].rank(pct=True) * 5
            
            # Interaction features
            features['rfm_score'] = (
                features['recency_score'] * 0.4 + 
                features['frequency_score'] * 0.3 + 
                features['monetary_score'] * 0.3
            )
            
            # Fill missing values
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features[numeric_columns] = features[numeric_columns].fillna(0)
            
            categorical_columns = features.select_dtypes(include=['object']).columns
            features[categorical_columns] = features[categorical_columns].fillna('Unknown')
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to prepare churn features: {e}")
            return customer_features
    
    def train_churn_model(self, customer_features: pd.DataFrame) -> Dict[str, Any]:
        """Train the churn prediction model"""
        try:
            # Prepare features
            features_df = self.prepare_churn_features(customer_features)
            
            # Select feature columns
            exclude_columns = [
                'customer_id', 'first_purchase_date', 'last_purchase_date', 
                'is_churned', 'region', 'primary_category'
            ]
            
            feature_columns = [col for col in features_df.columns if col not in exclude_columns]
            
            X = features_df[feature_columns].copy()
            y = features_df['is_churned']
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            if len(np.unique(y)) > 1:  # Ensure we have both classes
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            self.feature_columns = feature_columns
            self.is_trained = True
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1] if len(np.unique(y)) > 1 else np.zeros(len(y_test))
            
            auc_score = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
            
            return {
                'auc_score': auc_score,
                'feature_importance': self.feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Failed to train churn model: {e}")
            return {'error': str(e)}
    
    def predict_churn_risks(self, customer_features: pd.DataFrame) -> List[ChurnRisk]:
        """Predict churn risks for all customers"""
        try:
            if not self.is_trained:
                # Train model if not already trained
                self.train_churn_model(customer_features)
            
            features_df = self.prepare_churn_features(customer_features)
            churn_risks = []
            
            for _, customer in features_df.iterrows():
                customer_id = customer['customer_id']
                
                # Prepare features for prediction
                X = customer[self.feature_columns].values.reshape(1, -1)
                
                # Handle categorical encoding
                X_processed = X.copy()
                
                # Make prediction
                if self.is_trained and len(self.feature_columns) > 0:
                    try:
                        churn_probability = self.model.predict_proba(X_processed)[0, 1]
                    except:
                        churn_probability = customer['days_since_last_purchase'] / 365  # Fallback
                else:
                    churn_probability = customer['days_since_last_purchase'] / 365  # Fallback
                
                # Determine risk level
                if churn_probability >= 0.8:
                    risk_level = 'Critical'
                elif churn_probability >= 0.6:
                    risk_level = 'High'
                elif churn_probability >= 0.4:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                # Identify key risk factors
                risk_factors = []
                if customer['days_since_last_purchase'] > 60:
                    risk_factors.append('Long time since last purchase')
                if customer['purchase_frequency'] < 0.5:
                    risk_factors.append('Low purchase frequency')
                if customer['purchase_trend_score'] < -0.2:
                    risk_factors.append('Declining purchase trend')
                if customer['sales_consistency'] < 0.5:
                    risk_factors.append('Inconsistent purchase behavior')
                
                # Determine purchase trend
                if customer['purchase_trend_score'] > 0.1:
                    purchase_trend = 'Increasing'
                elif customer['purchase_trend_score'] < -0.1:
                    purchase_trend = 'Declining'
                else:
                    purchase_trend = 'Stable'
                
                # Generate recommendations
                recommendations = self._generate_churn_recommendations(risk_level, risk_factors)
                
                churn_risks.append(ChurnRisk(
                    customer_id=customer_id,
                    churn_probability=churn_probability,
                    risk_level=risk_level,
                    key_risk_factors=risk_factors,
                    days_since_last_purchase=int(customer['days_since_last_purchase']),
                    purchase_trend=purchase_trend,
                    recommended_actions=recommendations,
                    confidence_score=max(churn_probability, 1 - churn_probability),
                    prediction_date=datetime.now()
                ))
            
            return churn_risks
            
        except Exception as e:
            logger.error(f"Failed to predict churn risks: {e}")
            return []
    
    def _generate_churn_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate churn prevention recommendations"""
        recommendations = []
        
        if risk_level == 'Critical':
            recommendations.extend([
                'Immediate personal outreach required',
                'Offer significant discount or loyalty rewards',
                'Schedule executive-level customer call'
            ])
        elif risk_level == 'High':
            recommendations.extend([
                'Proactive retention campaign',
                'Personalized product recommendations',
                'Customer success check-in call'
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                'Targeted email marketing campaign',
                'Monitor behavior closely',
                'Consider loyalty program enrollment'
            ])
        else:
            recommendations.extend([
                'Continue regular engagement',
                'Explore upselling opportunities'
            ])
        
        # Factor-specific recommendations
        if 'Long time since last purchase' in risk_factors:
            recommendations.append('Send re-engagement campaign with new product highlights')
        if 'Low purchase frequency' in risk_factors:
            recommendations.append('Analyze purchase barriers and offer convenience improvements')
        if 'Declining purchase trend' in risk_factors:
            recommendations.append('Investigate satisfaction issues and address concerns')
        
        return recommendations

class CustomerSegmenter:
    """Advanced customer segmentation using RFM and behavioral analysis"""
    
    def __init__(self):
        self.kmeans_model = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def segment_customers(self, customer_features: pd.DataFrame, ltv_data: List[CustomerLTV]) -> List[CustomerSegment]:
        """Perform advanced customer segmentation"""
        try:
            # Create LTV lookup
            ltv_dict = {ltv.customer_id: ltv for ltv in ltv_data}
            
            # Prepare segmentation features
            segmentation_data = []
            
            for _, customer in customer_features.iterrows():
                customer_id = customer['customer_id']
                ltv_info = ltv_dict.get(customer_id)
                
                if ltv_info:
                    segmentation_data.append({
                        'customer_id': customer_id,
                        'recency': customer['days_since_last_purchase'],
                        'frequency': customer['purchase_frequency'],
                        'monetary': customer['total_revenue'],
                        'ltv': ltv_info.combined_ltv,
                        'consistency': customer['sales_consistency'],
                        'trend': customer['purchase_trend_score'],
                        'categories': customer['categories_purchased']
                    })
            
            if not segmentation_data:
                return []
            
            seg_df = pd.DataFrame(segmentation_data)
            
            # Create RFM scores with fallback
            try:
                seg_df['recency_score'] = pd.qcut(
                    seg_df['recency'], 
                    q=5, 
                    labels=[5, 4, 3, 2, 1], 
                    duplicates='drop'
                ).astype(float)
            except ValueError:
                seg_df['recency_score'] = 6 - (seg_df['recency'].rank(pct=True) * 5)
            
            try:
                seg_df['frequency_score'] = pd.qcut(
                    seg_df['frequency'], 
                    q=5, 
                    labels=[1, 2, 3, 4, 5], 
                    duplicates='drop'
                ).astype(float)
            except ValueError:
                seg_df['frequency_score'] = seg_df['frequency'].rank(pct=True) * 5
            
            try:
                seg_df['monetary_score'] = pd.qcut(
                    seg_df['monetary'], 
                    q=5, 
                    labels=[1, 2, 3, 4, 5], 
                    duplicates='drop'
                ).astype(float)
            except ValueError:
                seg_df['monetary_score'] = seg_df['monetary'].rank(pct=True) * 5
            
            # Perform K-means clustering
            feature_cols = ['recency_score', 'frequency_score', 'monetary_score', 'ltv', 'consistency', 'trend']
            X = seg_df[feature_cols].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            cluster_labels = self.kmeans_model.fit_predict(X_scaled)
            seg_df['cluster'] = cluster_labels
            self.is_fitted = True
            
            # Analyze clusters and create segments
            segments = []
            total_revenue = seg_df['monetary'].sum()
            
            for cluster_id in range(self.kmeans_model.n_clusters):
                cluster_data = seg_df[seg_df['cluster'] == cluster_id]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate segment metrics
                avg_ltv = cluster_data['ltv'].mean()
                avg_recency = cluster_data['recency'].mean()
                avg_frequency = cluster_data['frequency'].mean()
                avg_monetary = cluster_data['monetary'].mean()
                
                # Determine segment name based on characteristics
                segment_name = self._determine_segment_name(
                    avg_recency, avg_frequency, avg_monetary, avg_ltv
                )
                
                # Calculate retention rate (customers with recency <= 90 days)
                active_customers = len(cluster_data[cluster_data['recency'] <= 90])
                retention_rate = active_customers / len(cluster_data)
                
                # Calculate revenue contribution
                revenue_contribution = cluster_data['monetary'].sum() / total_revenue * 100
                
                # Calculate growth rate (based on trend scores)
                avg_trend = cluster_data['trend'].mean()
                growth_rate = avg_trend * 100  # Convert to percentage
                
                # Calculate health score
                health_score = (
                    (5 - min(avg_recency / 30, 5)) * 0.3 +  # Recency component
                    min(avg_frequency * 2, 5) * 0.3 +       # Frequency component
                    min(avg_ltv / 1000, 5) * 0.2 +          # LTV component
                    retention_rate * 5 * 0.2                # Retention component
                )
                
                # Generate characteristics
                characteristics = self._generate_segment_characteristics(
                    segment_name, avg_recency, avg_frequency, avg_monetary, avg_ltv
                )
                
                segments.append(CustomerSegment(
                    segment_id=f"segment_{cluster_id}",
                    segment_name=segment_name,
                    customer_count=len(cluster_data),
                    avg_ltv=avg_ltv,
                    avg_retention_rate=retention_rate,
                    avg_purchase_frequency=avg_frequency,
                    avg_order_value=avg_monetary / max(cluster_data['frequency'].mean(), 1),
                    characteristics=characteristics,
                    growth_rate=growth_rate,
                    revenue_contribution=revenue_contribution,
                    segment_health_score=health_score
                ))
            
            # Update LTV data with segment assignments
            for ltv in ltv_data:
                customer_segment = seg_df[seg_df['customer_id'] == ltv.customer_id]
                if not customer_segment.empty:
                    cluster_id = customer_segment['cluster'].iloc[0]
                    segment = next((s for s in segments if s.segment_id == f"segment_{cluster_id}"), None)
                    if segment:
                        ltv.segment = segment.segment_name
            
            return sorted(segments, key=lambda x: x.avg_ltv, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to segment customers: {e}")
            return []
    
    def _determine_segment_name(self, recency: float, frequency: float, monetary: float, ltv: float) -> str:
        """Determine segment name based on RFM and LTV characteristics"""
        
        # High value segments
        if ltv > 5000 and recency <= 30:
            return "VIP Champions"
        elif ltv > 3000 and frequency > 2:
            return "Loyal High-Value"
        elif ltv > 2000 and recency <= 60:
            return "Potential Champions"
        
        # Medium value segments
        elif ltv > 1000 and recency <= 90:
            return "Promising Customers"
        elif frequency > 1 and recency <= 120:
            return "Regular Customers"
        
        # At-risk segments
        elif ltv > 2000 and recency > 90:
            return "At-Risk High-Value"
        elif recency > 180:
            return "Hibernating"
        elif recency > 120:
            return "Need Attention"
        
        # New or low-value segments
        elif recency <= 30 and frequency < 1:
            return "New Customers"
        else:
            return "Standard Customers"
    
    def _generate_segment_characteristics(self, segment_name: str, recency: float, 
                                       frequency: float, monetary: float, ltv: float) -> List[str]:
        """Generate characteristics for each segment"""
        
        characteristics = []
        
        # Recency characteristics
        if recency <= 30:
            characteristics.append("Recently active")
        elif recency <= 90:
            characteristics.append("Moderately active")
        else:
            characteristics.append("Low recent activity")
        
        # Frequency characteristics
        if frequency > 2:
            characteristics.append("High purchase frequency")
        elif frequency > 1:
            characteristics.append("Regular purchaser")
        else:
            characteristics.append("Infrequent purchaser")
        
        # Value characteristics
        if ltv > 3000:
            characteristics.append("High lifetime value")
        elif ltv > 1000:
            characteristics.append("Medium lifetime value")
        else:
            characteristics.append("Lower lifetime value")
        
        # Segment-specific characteristics
        segment_specific = {
            "VIP Champions": ["Top tier customers", "Highest engagement", "Premium treatment required"],
            "Loyal High-Value": ["Consistent high spenders", "Strong loyalty", "Upselling opportunities"],
            "At-Risk High-Value": ["Valuable but declining", "Immediate attention needed", "Retention priority"],
            "New Customers": ["Recent acquisitions", "Unknown potential", "Onboarding focus"],
            "Hibernating": ["Inactive customers", "Win-back candidates", "Re-engagement needed"]
        }
        
        if segment_name in segment_specific:
            characteristics.extend(segment_specific[segment_name])
        
        return characteristics

class CustomerAnalyticsEngine:
    """Main customer analytics engine orchestrating all components"""
    
    def __init__(self):
        self.ltv_calculator = CustomerLTVCalculator()
        self.cohort_analyzer = CohortAnalyzer()
        self.churn_predictor = ChurnPredictor()
        self.customer_segmenter = CustomerSegmenter()
    
    def analyze_customers(self, sales_data: pd.DataFrame) -> CustomerAnalytics:
        """Perform comprehensive customer analytics"""
        try:
            logger.info("Starting comprehensive customer analytics...")
            
            # Calculate customer features
            customer_features = self.ltv_calculator.calculate_customer_features(sales_data)
            
            if customer_features.empty:
                logger.warning("No customer features calculated")
                return self._create_empty_analytics()
            
            # Calculate LTV metrics
            customer_ltvs = self.ltv_calculator.calculate_ltv_metrics(customer_features)
            
            # Perform cohort analysis
            cohort_analyses = self.cohort_analyzer.analyze_cohorts(sales_data)
            
            # Predict churn risks
            churn_risks = self.churn_predictor.predict_churn_risks(customer_features)
            
            # Segment customers
            customer_segments = self.customer_segmenter.segment_customers(customer_features, customer_ltvs)
            
            # Calculate overall metrics
            total_customers = len(customer_features)
            active_customers = len(customer_features[customer_features['days_since_last_purchase'] <= 90])
            new_customers = len(customer_features[customer_features['customer_lifespan_days'] <= 30])
            churned_customers = len([risk for risk in churn_risks if risk.churn_probability > 0.5])
            
            overall_ltv = np.mean([ltv.combined_ltv for ltv in customer_ltvs]) if customer_ltvs else 0
            overall_retention_rate = active_customers / total_customers if total_customers > 0 else 0
            
            # Generate insights and recommendations
            key_insights = self._generate_key_insights(
                customer_ltvs, cohort_analyses, churn_risks, customer_segments
            )
            recommendations = self._generate_recommendations(
                churn_risks, customer_segments, overall_retention_rate
            )
            
            return CustomerAnalytics(
                total_customers=total_customers,
                active_customers=active_customers,
                new_customers_this_month=new_customers,
                churned_customers=churned_customers,
                overall_ltv=overall_ltv,
                overall_retention_rate=overall_retention_rate,
                customer_ltvs=customer_ltvs,
                cohort_analyses=cohort_analyses,
                churn_risks=churn_risks,
                customer_segments=customer_segments,
                key_insights=key_insights,
                recommendations=recommendations,
                analysis_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Customer analytics failed: {e}")
            return self._create_empty_analytics()
    
    def _create_empty_analytics(self) -> CustomerAnalytics:
        """Create empty analytics result for error cases"""
        return CustomerAnalytics(
            total_customers=0,
            active_customers=0,
            new_customers_this_month=0,
            churned_customers=0,
            overall_ltv=0.0,
            overall_retention_rate=0.0,
            customer_ltvs=[],
            cohort_analyses=[],
            churn_risks=[],
            customer_segments=[],
            key_insights=["Unable to perform customer analytics - insufficient data"],
            recommendations=["Please ensure sufficient sales data is available"],
            analysis_date=datetime.now()
        )
    
    def _generate_key_insights(self, customer_ltvs: List[CustomerLTV], 
                             cohort_analyses: List[CohortAnalysis],
                             churn_risks: List[ChurnRisk], 
                             customer_segments: List[CustomerSegment]) -> List[str]:
        """Generate key business insights from analytics"""
        insights = []
        
        # LTV insights
        if customer_ltvs:
            high_ltv_customers = len([ltv for ltv in customer_ltvs if ltv.ltv_percentile >= 80])
            avg_ltv = np.mean([ltv.combined_ltv for ltv in customer_ltvs])
            insights.append(f"Average customer LTV is ${avg_ltv:.2f} with {high_ltv_customers} high-value customers (top 20%)")
        
        # Churn insights
        if churn_risks:
            high_risk_count = len([risk for risk in churn_risks if risk.risk_level in ['High', 'Critical']])
            total_risk_customers = len(churn_risks)
            risk_percentage = (high_risk_count / total_risk_customers * 100) if total_risk_customers > 0 else 0
            insights.append(f"{high_risk_count} customers ({risk_percentage:.1f}%) are at high risk of churning")
        
        # Cohort insights
        if cohort_analyses:
            latest_cohort = max(cohort_analyses, key=lambda x: x.cohort_month)
            insights.append(f"Latest cohort ({latest_cohort.cohort_month}) has {latest_cohort.cohort_size} customers with ${latest_cohort.avg_ltv:.2f} average LTV")
        
        # Segment insights
        if customer_segments:
            top_segment = max(customer_segments, key=lambda x: x.revenue_contribution)
            insights.append(f"'{top_segment.segment_name}' segment contributes {top_segment.revenue_contribution:.1f}% of total revenue")
        
        return insights
    
    def _generate_recommendations(self, churn_risks: List[ChurnRisk], 
                                customer_segments: List[CustomerSegment],
                                retention_rate: float) -> List[str]:
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Churn-based recommendations
        critical_risk_count = len([risk for risk in churn_risks if risk.risk_level == 'Critical'])
        if critical_risk_count > 0:
            recommendations.append(f"Immediate intervention needed for {critical_risk_count} critical-risk customers")
        
        high_risk_count = len([risk for risk in churn_risks if risk.risk_level == 'High'])
        if high_risk_count > 5:
            recommendations.append("Implement proactive retention campaign for high-risk customers")
        
        # Retention rate recommendations
        if retention_rate < 0.7:
            recommendations.append("Overall retention rate is below 70% - review customer experience and engagement strategies")
        elif retention_rate > 0.9:
            recommendations.append("Excellent retention rate - focus on customer expansion and referral programs")
        
        # Segment-based recommendations
        if customer_segments:
            vip_segment = next((seg for seg in customer_segments if 'VIP' in seg.segment_name or 'Champion' in seg.segment_name), None)
            if vip_segment and vip_segment.customer_count > 0:
                recommendations.append(f"Develop exclusive VIP program for {vip_segment.customer_count} top-tier customers")
            
            at_risk_segment = next((seg for seg in customer_segments if 'At-Risk' in seg.segment_name), None)
            if at_risk_segment and at_risk_segment.customer_count > 0:
                recommendations.append(f"Create targeted win-back campaign for {at_risk_segment.customer_count} at-risk high-value customers")
        
        # General recommendations
        recommendations.extend([
            "Monitor customer analytics monthly and adjust retention strategies",
            "Implement personalized marketing based on customer segments",
            "Track LTV trends to optimize customer acquisition costs"
        ])
        
        return recommendations
    
    def get_customer_analytics_summary(self, analytics: CustomerAnalytics) -> Dict[str, Any]:
        """Get a summary of customer analytics for API responses"""
        return {
            'overview': {
                'total_customers': analytics.total_customers,
                'active_customers': analytics.active_customers,
                'retention_rate': analytics.overall_retention_rate,
                'average_ltv': analytics.overall_ltv,
                'churn_risk_customers': analytics.churned_customers
            },
            'segments': [
                {
                    'name': segment.segment_name,
                    'customer_count': segment.customer_count,
                    'avg_ltv': segment.avg_ltv,
                    'revenue_contribution': segment.revenue_contribution,
                    'health_score': segment.segment_health_score
                }
                for segment in analytics.customer_segments
            ],
            'top_insights': analytics.key_insights[:5],
            'priority_recommendations': analytics.recommendations[:5],
            'analysis_date': analytics.analysis_date.isoformat()
        }