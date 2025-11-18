"""
Customer Retention Analytics Engine for Cyberpunk AI Dashboard
Implements churn prediction, cohort analysis, and customer lifetime value calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import logging
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    KaplanMeierFitter = None
    CoxPHFitter = None

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChurnPrediction:
    """Churn prediction result for a customer"""
    customer_id: str
    churn_probability: float
    risk_level: str  # 'Low', 'Medium', 'High'
    key_risk_factors: List[str]
    recommended_actions: List[str]
    confidence_score: float
    prediction_date: datetime

@dataclass
class CohortMetrics:
    """Cohort analysis metrics"""
    cohort_period: str
    cohort_size: int
    retention_rates: Dict[int, float]  # period -> retention rate
    revenue_per_cohort: Dict[int, float]
    avg_customer_lifespan: float
    ltv_estimate: float

@dataclass
class CustomerSegment:
    """Customer segment definition"""
    segment_id: str
    segment_name: str
    criteria: Dict[str, Any]
    customer_count: int
    avg_ltv: float
    avg_retention_rate: float
    characteristics: List[str]

@dataclass
class RetentionInsights:
    """Comprehensive retention analysis results"""
    total_customers: int
    overall_retention_rate: float
    churn_predictions: List[ChurnPrediction]
    cohort_analysis: List[CohortMetrics]
    customer_segments: List[CustomerSegment]
    key_insights: List[str]
    recommendations: List[str]
    analysis_date: datetime

class ChurnPredictionModel:
    """Machine learning model for churn prediction"""
    
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
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for churn prediction"""
        features = df.copy()
        
        # Calculate recency, frequency, monetary features
        if 'last_transaction_date' in features.columns:
            features['days_since_last_transaction'] = (
                datetime.now() - pd.to_datetime(features['last_transaction_date'])
            ).dt.days
        
        if 'first_transaction_date' in features.columns:
            features['customer_age_days'] = (
                datetime.now() - pd.to_datetime(features['first_transaction_date'])
            ).dt.days
        
        # Transaction-based features
        if 'total_transactions' in features.columns:
            features['transaction_frequency'] = features['total_transactions'] / features.get('customer_age_days', 1)
        
        if 'total_revenue' in features.columns and 'total_transactions' in features.columns:
            features['avg_order_value'] = features['total_revenue'] / features['total_transactions'].replace(0, 1)
        
        # Engagement features
        if 'email_opens' in features.columns and 'emails_sent' in features.columns:
            features['email_open_rate'] = features['email_opens'] / features['emails_sent'].replace(0, 1)
        
        if 'website_visits' in features.columns:
            features['avg_session_duration'] = features.get('total_session_time', 0) / features['website_visits'].replace(0, 1)
        
        # Fill missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)
        
        categorical_columns = features.select_dtypes(include=['object']).columns
        features[categorical_columns] = features[categorical_columns].fillna('Unknown')
        
        return features
    
    def train(self, df: pd.DataFrame, target_column: str = 'churned') -> Dict[str, Any]:
        """Train the churn prediction model"""
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Separate features and target
            if target_column not in features_df.columns:
                # Create synthetic churn labels for demo
                np.random.seed(42)
                features_df[target_column] = np.random.choice([0, 1], size=len(features_df), p=[0.8, 0.2])
            
            # Select feature columns (exclude target and ID columns)
            feature_columns = [col for col in features_df.columns 
                             if col not in [target_column, 'customer_id', 'first_transaction_date', 'last_transaction_date']]
            
            X = features_df[feature_columns].copy()
            y = features_df[target_column]
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            
            # Scale numerical features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate feature importance
            self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            
            # Store feature columns for prediction
            self.feature_columns = feature_columns
            self.is_trained = True
            
            # Return training results
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            return {
                'auc_score': auc_score,
                'feature_importance': self.feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'error': str(e)}
    
    def predict_churn(self, customer_features: Dict[str, Any]) -> ChurnPrediction:
        """Predict churn probability for a single customer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_features])
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Select and encode features
            X = features_df[self.feature_columns].copy()
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                else:
                    X[col] = 0  # Unknown category
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            churn_probability = self.model.predict_proba(X_scaled)[0, 1]
            
            # Determine risk level
            if churn_probability >= 0.7:
                risk_level = 'High'
            elif churn_probability >= 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Identify key risk factors
            feature_contributions = X.iloc[0] * list(self.feature_importance.values())
            top_risk_factors = feature_contributions.nlargest(3).index.tolist()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, top_risk_factors, customer_features)
            
            return ChurnPrediction(
                customer_id=customer_features.get('customer_id', 'unknown'),
                churn_probability=churn_probability,
                risk_level=risk_level,
                key_risk_factors=top_risk_factors,
                recommended_actions=recommendations,
                confidence_score=max(churn_probability, 1 - churn_probability),
                prediction_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Churn prediction failed: {e}")
            return ChurnPrediction(
                customer_id=customer_features.get('customer_id', 'unknown'),
                churn_probability=0.5,
                risk_level='Unknown',
                key_risk_factors=[],
                recommended_actions=['Unable to generate prediction'],
                confidence_score=0.0,
                prediction_date=datetime.now()
            )
    
    def _generate_recommendations(self, risk_level: str, risk_factors: List[str], 
                                customer_features: Dict[str, Any]) -> List[str]:
        """Generate personalized retention recommendations"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append('Immediate intervention required - contact customer within 24 hours')
            recommendations.append('Offer personalized discount or loyalty program enrollment')
            recommendations.append('Schedule customer success call to understand concerns')
        
        elif risk_level == 'Medium':
            recommendations.append('Proactive engagement - send targeted marketing campaign')
            recommendations.append('Monitor customer behavior closely for next 30 days')
            recommendations.append('Consider offering product training or support')
        
        else:
            recommendations.append('Continue regular engagement and monitoring')
            recommendations.append('Consider upselling or cross-selling opportunities')
        
        # Factor-specific recommendations
        if 'days_since_last_transaction' in risk_factors:
            recommendations.append('Re-engagement campaign focusing on recent product updates')
        
        if 'transaction_frequency' in risk_factors:
            recommendations.append('Analyze purchase patterns and suggest relevant products')
        
        if 'email_open_rate' in risk_factors:
            recommendations.append('Review email content and frequency preferences')
        
        return recommendations

class CohortAnalyzer:
    """Cohort analysis for customer retention"""
    
    def analyze_cohorts(self, df: pd.DataFrame, 
                       customer_id_col: str = 'customer_id',
                       date_col: str = 'transaction_date',
                       revenue_col: str = 'revenue') -> List[CohortMetrics]:
        """Perform cohort analysis"""
        try:
            # Prepare data
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Create cohort groups based on first purchase month
            df['order_period'] = df[date_col].dt.to_period('M')
            df['cohort_group'] = df.groupby(customer_id_col)[date_col].transform('min').dt.to_period('M')
            
            # Calculate period number
            df['period_number'] = (df['order_period'] - df['cohort_group']).apply(attrgetter('n'))
            
            # Create cohort table
            cohort_data = df.groupby(['cohort_group', 'period_number'])[customer_id_col].nunique().reset_index()
            cohort_counts = cohort_data.pivot(index='cohort_group', 
                                            columns='period_number', 
                                            values=customer_id_col)
            
            # Calculate cohort sizes
            cohort_sizes = df.groupby('cohort_group')[customer_id_col].nunique()
            
            # Calculate retention rates
            cohort_table = cohort_counts.divide(cohort_sizes, axis=0)
            
            # Calculate revenue per cohort
            revenue_data = df.groupby(['cohort_group', 'period_number'])[revenue_col].sum().reset_index()
            cohort_revenue = revenue_data.pivot(index='cohort_group', 
                                              columns='period_number', 
                                              values=revenue_col)
            
            # Generate cohort metrics
            cohort_metrics = []
            for cohort_group in cohort_table.index:
                retention_rates = cohort_table.loc[cohort_group].dropna().to_dict()
                revenue_per_period = cohort_revenue.loc[cohort_group].dropna().to_dict()
                
                # Calculate average customer lifespan
                retention_values = list(retention_rates.values())
                avg_lifespan = sum(i * rate for i, rate in enumerate(retention_values)) if retention_values else 0
                
                # Estimate LTV
                total_revenue = sum(revenue_per_period.values())
                cohort_size = cohort_sizes[cohort_group]
                ltv_estimate = total_revenue / cohort_size if cohort_size > 0 else 0
                
                cohort_metrics.append(CohortMetrics(
                    cohort_period=str(cohort_group),
                    cohort_size=cohort_size,
                    retention_rates=retention_rates,
                    revenue_per_cohort=revenue_per_period,
                    avg_customer_lifespan=avg_lifespan,
                    ltv_estimate=ltv_estimate
                ))
            
            return cohort_metrics
            
        except Exception as e:
            logger.error(f"Cohort analysis failed: {e}")
            return []

class LTVCalculator:
    """Customer Lifetime Value calculator"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def calculate_ltv(self, customer_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate customer lifetime value"""
        try:
            ltv_results = {}
            
            for _, customer in customer_data.iterrows():
                customer_id = customer.get('customer_id', 'unknown')
                
                # Historical LTV (actual)
                total_revenue = customer.get('total_revenue', 0)
                customer_age_days = customer.get('customer_age_days', 1)
                
                # Predictive LTV (estimated future value)
                avg_order_value = customer.get('avg_order_value', 0)
                purchase_frequency = customer.get('transaction_frequency', 0)
                
                # Simple LTV calculation: AOV * Purchase Frequency * Customer Lifespan
                estimated_lifespan_years = max(customer_age_days / 365, 0.5)  # At least 6 months
                predicted_ltv = avg_order_value * purchase_frequency * 365 * estimated_lifespan_years
                
                # Combine historical and predictive
                combined_ltv = (total_revenue + predicted_ltv) / 2
                
                ltv_results[customer_id] = {
                    'historical_ltv': total_revenue,
                    'predicted_ltv': predicted_ltv,
                    'combined_ltv': combined_ltv,
                    'avg_order_value': avg_order_value,
                    'purchase_frequency': purchase_frequency,
                    'estimated_lifespan_years': estimated_lifespan_years
                }
            
            return ltv_results
            
        except Exception as e:
            logger.error(f"LTV calculation failed: {e}")
            return {}

class SegmentAnalyzer:
    """Customer segmentation analyzer"""
    
    def segment_customers(self, df: pd.DataFrame) -> List[CustomerSegment]:
        """Segment customers based on RFM analysis"""
        try:
            # Calculate RFM metrics
            current_date = datetime.now()
            
            rfm = df.groupby('customer_id').agg({
                'transaction_date': lambda x: (current_date - pd.to_datetime(x.max())).days,  # Recency
                'customer_id': 'count',  # Frequency
                'total_amount': 'sum'  # Monetary
            }).rename(columns={
                'transaction_date': 'recency',
                'customer_id': 'frequency',
                'total_amount': 'monetary'
            })
            
            # Create RFM scores
            rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
            rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
            
            # Combine scores
            rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
            
            # Define segments
            segment_map = {
                'Champions': ['555', '554', '544', '545', '454', '455', '445'],
                'Loyal Customers': ['543', '444', '435', '355', '354', '345', '344', '335'],
                'Potential Loyalists': ['512', '511', '422', '421', '412', '411', '311'],
                'New Customers': ['512', '511', '422', '421', '412', '411', '311'],
                'Promising': ['512', '511', '422', '421', '412', '411', '311'],
                'Need Attention': ['155', '154', '144', '214', '215', '115', '114'],
                'About to Sleep': ['155', '154', '144', '214', '215', '115', '114'],
                'At Risk': ['155', '154', '144', '214', '215', '115', '114'],
                'Cannot Lose Them': ['155', '154', '144', '214', '215', '115', '114'],
                'Hibernating': ['155', '154', '144', '214', '215', '115', '114'],
                'Lost': ['111', '112', '121', '131', '141', '151']
            }
            
            # Assign segments
            def assign_segment(rfm_score):
                for segment, scores in segment_map.items():
                    if rfm_score in scores:
                        return segment
                return 'Other'
            
            rfm['segment'] = rfm['rfm_score'].apply(assign_segment)
            
            # Create segment summaries
            segments = []
            for segment_name in rfm['segment'].unique():
                segment_data = rfm[rfm['segment'] == segment_name]
                
                segments.append(CustomerSegment(
                    segment_id=segment_name.lower().replace(' ', '_'),
                    segment_name=segment_name,
                    criteria={
                        'avg_recency': segment_data['recency'].mean(),
                        'avg_frequency': segment_data['frequency'].mean(),
                        'avg_monetary': segment_data['monetary'].mean()
                    },
                    customer_count=len(segment_data),
                    avg_ltv=segment_data['monetary'].mean(),
                    avg_retention_rate=0.8,  # Placeholder - would calculate from actual data
                    characteristics=self._get_segment_characteristics(segment_name)
                ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Customer segmentation failed: {e}")
            return []
    
    def _get_segment_characteristics(self, segment_name: str) -> List[str]:
        """Get characteristics for each segment"""
        characteristics_map = {
            'Champions': ['Highest value customers', 'Recent purchasers', 'High frequency'],
            'Loyal Customers': ['Consistent purchasers', 'Good monetary value', 'Regular engagement'],
            'Potential Loyalists': ['Recent customers', 'Good potential', 'Need nurturing'],
            'New Customers': ['Recent first purchase', 'Unknown potential', 'Require onboarding'],
            'At Risk': ['Declining engagement', 'High value historically', 'Need immediate attention'],
            'Lost': ['No recent activity', 'Low engagement', 'Difficult to recover']
        }
        return characteristics_map.get(segment_name, ['Standard customer profile'])

class RetentionAnalyzer:
    """Main retention analyzer orchestrating all components"""
    
    def __init__(self):
        self.churn_model = ChurnPredictionModel()
        self.cohort_analyzer = CohortAnalyzer()
        self.ltv_calculator = LTVCalculator()
        self.segment_analyzer = SegmentAnalyzer()
    
    def analyze_customer_retention(self, customer_data: pd.DataFrame, 
                                 transaction_data: pd.DataFrame) -> RetentionInsights:
        """Comprehensive retention analysis"""
        try:
            logger.info("Starting comprehensive retention analysis...")
            
            # Train churn model
            training_result = self.churn_model.train(customer_data)
            logger.info(f"Churn model trained with AUC: {training_result.get('auc_score', 'N/A')}")
            
            # Generate churn predictions
            churn_predictions = []
            for _, customer in customer_data.iterrows():
                prediction = self.churn_model.predict_churn(customer.to_dict())
                churn_predictions.append(prediction)
            
            # Perform cohort analysis
            cohort_metrics = self.cohort_analyzer.analyze_cohorts(transaction_data)
            
            # Calculate customer segments
            customer_segments = self.segment_analyzer.segment_customers(transaction_data)
            
            # Calculate overall retention rate
            total_customers = len(customer_data)
            churned_customers = sum(1 for pred in churn_predictions if pred.churn_probability > 0.5)
            overall_retention_rate = (total_customers - churned_customers) / total_customers if total_customers > 0 else 0
            
            # Generate insights
            key_insights = self._generate_key_insights(churn_predictions, cohort_metrics, customer_segments)
            recommendations = self._generate_recommendations(churn_predictions, customer_segments)
            
            return RetentionInsights(
                total_customers=total_customers,
                overall_retention_rate=overall_retention_rate,
                churn_predictions=churn_predictions,
                cohort_analysis=cohort_metrics,
                customer_segments=customer_segments,
                key_insights=key_insights,
                recommendations=recommendations,
                analysis_date=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Retention analysis failed: {e}")
            return RetentionInsights(
                total_customers=0,
                overall_retention_rate=0,
                churn_predictions=[],
                cohort_analysis=[],
                customer_segments=[],
                key_insights=[f"Analysis failed: {str(e)}"],
                recommendations=["Please check data quality and try again"],
                analysis_date=datetime.now()
            )
    
    def _generate_key_insights(self, churn_predictions: List[ChurnPrediction], 
                             cohort_metrics: List[CohortMetrics], 
                             customer_segments: List[CustomerSegment]) -> List[str]:
        """Generate key business insights"""
        insights = []
        
        # Churn insights
        high_risk_customers = sum(1 for pred in churn_predictions if pred.risk_level == 'High')
        if high_risk_customers > 0:
            insights.append(f"{high_risk_customers} customers are at high risk of churning")
        
        # Cohort insights
        if cohort_metrics:
            latest_cohort = max(cohort_metrics, key=lambda x: x.cohort_period)
            insights.append(f"Latest cohort has {latest_cohort.cohort_size} customers with estimated LTV of ${latest_cohort.ltv_estimate:.2f}")
        
        # Segment insights
        if customer_segments:
            largest_segment = max(customer_segments, key=lambda x: x.customer_count)
            insights.append(f"Largest customer segment is '{largest_segment.segment_name}' with {largest_segment.customer_count} customers")
        
        return insights
    
    def _generate_recommendations(self, churn_predictions: List[ChurnPrediction], 
                                customer_segments: List[CustomerSegment]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Churn-based recommendations
        high_risk_count = sum(1 for pred in churn_predictions if pred.risk_level == 'High')
        if high_risk_count > 10:
            recommendations.append("Implement immediate retention campaign for high-risk customers")
        
        # Segment-based recommendations
        if customer_segments:
            champions = next((seg for seg in customer_segments if seg.segment_name == 'Champions'), None)
            if champions and champions.customer_count > 0:
                recommendations.append("Develop VIP program for Champions segment to maintain loyalty")
            
            at_risk = next((seg for seg in customer_segments if seg.segment_name == 'At Risk'), None)
            if at_risk and at_risk.customer_count > 0:
                recommendations.append("Create win-back campaign for At Risk customers")
        
        recommendations.append("Monitor retention metrics weekly and adjust strategies accordingly")
        
        return recommendations

# Helper function for cohort analysis
from operator import attrgetter