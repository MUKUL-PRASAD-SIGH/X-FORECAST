"""
Smart Data Processing Pipeline with Auto-Parameter Detection
Enhanced data processing with real-time analysis, auto-mapping, and quality scoring
"""

import pandas as pd
import numpy as np
import io
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Enhanced data type classification"""
    DATE = "date"
    SALES_AMOUNT = "sales_amount"
    PRODUCT_CATEGORY = "product_category"
    REGION = "region"
    UNITS_SOLD = "units_sold"
    CUSTOMER_ID = "customer_id"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"

@dataclass
class DetectedColumn:
    """Enhanced column detection with confidence scoring"""
    name: str
    type: DataType
    sample_values: List[str]
    confidence: float
    null_percentage: float
    unique_count: int
    data_patterns: Dict[str, Any]
    quality_score: float
    recommendations: List[str]

@dataclass
class ColumnMapping:
    """Auto-mapping result for required fields"""
    required_field: str
    detected_column: Optional[str]
    confidence: float
    status: str  # 'mapped', 'unmapped', 'uncertain', 'conflict'
    alternatives: List[Tuple[str, float]]  # Alternative mappings with confidence
    mapping_reason: str

@dataclass
class DataQuality:
    """Comprehensive data quality assessment"""
    overall_score: float
    completeness: float
    consistency: float
    validity: float
    accuracy: float
    uniqueness: float
    timeliness: float
    issues: List[str]
    recommendations: List[str]
    quality_breakdown: Dict[str, float]

@dataclass
class ProcessingResult:
    """Complete processing result with all analysis"""
    detected_columns: List[DetectedColumn]
    column_mappings: List[ColumnMapping]
    data_quality: DataQuality
    preview_data: Dict[str, Any]
    processing_stats: Dict[str, Any]
    standardized_data: Optional[pd.DataFrame] = None

class SmartDataProcessor:
    """Enhanced data processor with intelligent analysis and auto-mapping"""
    
    def __init__(self):
        self.required_fields = {
            'date': ['date', 'time', 'timestamp', 'period', 'day', 'month', 'year'],
            'sales_amount': ['sales', 'amount', 'revenue', 'value', 'price', 'total', 'sum'],
            'product_category': ['product', 'category', 'sku', 'item', 'type', 'class'],
            'region': ['region', 'location', 'area', 'territory', 'zone', 'state', 'country']
        }
        
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        self.currency_patterns = [
            r'^\$[\d,]+\.?\d*$',  # $1,234.56
            r'^[\d,]+\.?\d*\s*USD$',  # 1234.56 USD
            r'^€[\d,]+\.?\d*$',  # €1,234.56
        ]

    async def process_file_intelligent(self, file_content: bytes, filename: str, 
                                     file_format: str) -> ProcessingResult:
        """Main processing pipeline with intelligent analysis"""
        
        try:
            # Parse file into DataFrame
            df = await self._parse_file_content(file_content, file_format)
            
            # Perform intelligent column detection
            detected_columns = await self._detect_columns_intelligent(df)
            
            # Auto-map columns to required fields
            column_mappings = await self._auto_map_columns(detected_columns)
            
            # Assess data quality comprehensively
            data_quality = await self._assess_data_quality_comprehensive(df, detected_columns)
            
            # Create preview data
            preview_data = await self._create_enhanced_preview(df, detected_columns)
            
            # Generate processing statistics
            processing_stats = await self._generate_processing_stats(df, detected_columns)
            
            # Standardize data if mapping is successful
            standardized_data = None
            if self._is_mapping_successful(column_mappings):
                standardized_data = await self._standardize_data_format(df, column_mappings)
            
            return ProcessingResult(
                detected_columns=detected_columns,
                column_mappings=column_mappings,
                data_quality=data_quality,
                preview_data=preview_data,
                processing_stats=processing_stats,
                standardized_data=standardized_data
            )
            
        except Exception as e:
            logger.error(f"Smart processing failed for {filename}: {e}")
            raise

    async def _parse_file_content(self, content: bytes, file_format: str) -> pd.DataFrame:
        """Enhanced file parsing with error handling and encoding detection"""
        
        try:
            if file_format == 'csv':
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        text_content = content.decode(encoding)
                        # Try different delimiters
                        for delimiter in [',', ';', '\t', '|']:
                            try:
                                df = pd.read_csv(io.StringIO(text_content), delimiter=delimiter)
                                if len(df.columns) > 1:  # Valid CSV should have multiple columns
                                    return df
                            except:
                                continue
                    except UnicodeDecodeError:
                        continue
                
                # Fallback to default
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(content))
                
            elif file_format == 'json':
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
                    
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(r'[^\w\s]', '', regex=True)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to parse file: {str(e)}")

    async def _detect_columns_intelligent(self, df: pd.DataFrame) -> List[DetectedColumn]:
        """Enhanced column detection with pattern analysis and confidence scoring"""
        
        detected_columns = []
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Basic statistics
            null_percentage = (df[col].isnull().sum() / len(df)) * 100
            unique_count = col_data.nunique()
            sample_values = col_data.head(3).astype(str).tolist()
            
            # Detect data type and patterns
            data_type, confidence, patterns = await self._analyze_column_patterns(col, col_data)
            
            # Calculate quality score
            quality_score = await self._calculate_column_quality(col_data, data_type, patterns)
            
            # Generate recommendations
            recommendations = await self._generate_column_recommendations(
                col, col_data, data_type, quality_score, null_percentage
            )
            
            detected_columns.append(DetectedColumn(
                name=col,
                type=data_type,
                sample_values=sample_values,
                confidence=confidence,
                null_percentage=null_percentage,
                unique_count=unique_count,
                data_patterns=patterns,
                quality_score=quality_score,
                recommendations=recommendations
            ))
        
        return detected_columns

    async def _analyze_column_patterns(self, col_name: str, col_data: pd.Series) -> Tuple[DataType, float, Dict[str, Any]]:
        """Advanced pattern analysis for column type detection"""
        
        col_name_lower = col_name.lower()
        patterns = {}
        
        # Date detection with pattern analysis
        if any(keyword in col_name_lower for keyword in self.required_fields['date']):
            date_score, date_patterns = await self._analyze_date_patterns(col_data)
            if date_score > 0.7:
                patterns.update(date_patterns)
                return DataType.DATE, date_score, patterns
        
        # Numeric analysis
        if pd.api.types.is_numeric_dtype(col_data):
            # Sales amount detection
            if any(keyword in col_name_lower for keyword in self.required_fields['sales_amount']):
                currency_score = await self._analyze_currency_patterns(col_data)
                patterns['currency_format'] = currency_score > 0.5
                patterns['negative_values'] = (col_data < 0).any()
                patterns['decimal_places'] = self._get_decimal_places(col_data)
                return DataType.SALES_AMOUNT, 0.9, patterns
            
            # Units sold detection
            elif 'unit' in col_name_lower or 'quantity' in col_name_lower or 'count' in col_name_lower:
                patterns['integer_values'] = col_data.dtype in ['int64', 'int32']
                patterns['positive_only'] = (col_data >= 0).all()
                return DataType.UNITS_SOLD, 0.8, patterns
            
            else:
                patterns['data_range'] = {'min': float(col_data.min()), 'max': float(col_data.max())}
                return DataType.NUMERIC, 0.6, patterns
        
        # Categorical analysis
        elif col_data.dtype == 'object':
            unique_ratio = col_data.nunique() / len(col_data)
            
            # Product category detection
            if any(keyword in col_name_lower for keyword in self.required_fields['product_category']):
                patterns['unique_ratio'] = unique_ratio
                patterns['avg_length'] = col_data.str.len().mean()
                return DataType.PRODUCT_CATEGORY, 0.85, patterns
            
            # Region detection
            elif any(keyword in col_name_lower for keyword in self.required_fields['region']):
                patterns['unique_ratio'] = unique_ratio
                patterns['geographic_indicators'] = self._detect_geographic_patterns(col_data)
                return DataType.REGION, 0.85, patterns
            
            # General categorical
            elif unique_ratio < 0.1:
                patterns['unique_ratio'] = unique_ratio
                patterns['category_count'] = col_data.nunique()
                return DataType.CATEGORICAL, 0.7, patterns
            
            # Text data
            else:
                patterns['avg_length'] = col_data.str.len().mean()
                patterns['contains_numbers'] = col_data.str.contains(r'\d').any()
                return DataType.TEXT, 0.6, patterns
        
        # Boolean detection
        unique_values = set(col_data.unique())
        if unique_values.issubset({True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0', 'yes', 'no', 'Y', 'N'}):
            patterns['boolean_format'] = list(unique_values)
            return DataType.BOOLEAN, 0.9, patterns
        
        return DataType.UNKNOWN, 0.3, patterns

    async def _analyze_date_patterns(self, col_data: pd.Series) -> Tuple[float, Dict[str, Any]]:
        """Analyze date patterns and formats"""
        
        patterns = {}
        sample_size = min(20, len(col_data))
        sample_data = col_data.head(sample_size)
        
        # Try to parse as dates
        try:
            parsed_dates = pd.to_datetime(sample_data, errors='coerce')
            valid_dates = parsed_dates.dropna()
            
            if len(valid_dates) / len(sample_data) > 0.8:
                patterns['date_format'] = 'auto_detected'
                patterns['date_range'] = {
                    'start': valid_dates.min().isoformat() if len(valid_dates) > 0 else None,
                    'end': valid_dates.max().isoformat() if len(valid_dates) > 0 else None
                }
                patterns['has_time'] = any(parsed_dates.dt.time != pd.Timestamp('00:00:00').time())
                return 0.9, patterns
        except:
            pass
        
        # Pattern matching for date strings
        date_pattern_matches = 0
        for pattern in self.date_patterns:
            matches = sample_data.astype(str).str.match(pattern).sum()
            date_pattern_matches = max(date_pattern_matches, matches)
        
        if date_pattern_matches > 0:
            confidence = date_pattern_matches / len(sample_data)
            patterns['pattern_matches'] = date_pattern_matches
            return confidence, patterns
        
        return 0.0, patterns

    async def _analyze_currency_patterns(self, col_data: pd.Series) -> float:
        """Analyze currency patterns in numeric data"""
        
        # Check for currency symbols or formatting
        sample_strings = col_data.astype(str).head(10)
        
        currency_matches = 0
        for pattern in self.currency_patterns:
            matches = sample_strings.str.match(pattern).sum()
            currency_matches += matches
        
        return min(currency_matches / len(sample_strings), 1.0)

    def _get_decimal_places(self, col_data: pd.Series) -> int:
        """Get average number of decimal places in numeric data"""
        
        try:
            decimal_counts = []
            for value in col_data.head(20):
                if pd.notna(value):
                    str_val = str(float(value))
                    if '.' in str_val:
                        decimal_counts.append(len(str_val.split('.')[1]))
                    else:
                        decimal_counts.append(0)
            
            return int(np.mean(decimal_counts)) if decimal_counts else 0
        except:
            return 0

    def _detect_geographic_patterns(self, col_data: pd.Series) -> Dict[str, Any]:
        """Detect geographic patterns in text data"""
        
        geographic_indicators = {
            'has_state_codes': False,
            'has_country_codes': False,
            'has_zip_codes': False,
            'avg_length': 0
        }
        
        sample_data = col_data.head(20).astype(str)
        
        # State code pattern (2 letters)
        state_pattern = r'^[A-Z]{2}$'
        geographic_indicators['has_state_codes'] = sample_data.str.match(state_pattern).any()
        
        # Country code pattern (2-3 letters)
        country_pattern = r'^[A-Z]{2,3}$'
        geographic_indicators['has_country_codes'] = sample_data.str.match(country_pattern).any()
        
        # ZIP code pattern
        zip_pattern = r'^\d{5}(-\d{4})?$'
        geographic_indicators['has_zip_codes'] = sample_data.str.match(zip_pattern).any()
        
        geographic_indicators['avg_length'] = sample_data.str.len().mean()
        
        return geographic_indicators

    async def _calculate_column_quality(self, col_data: pd.Series, data_type: DataType, 
                                      patterns: Dict[str, Any]) -> float:
        """Calculate comprehensive quality score for a column"""
        
        quality_factors = []
        
        # Completeness (non-null percentage)
        completeness = (len(col_data) / (len(col_data) + col_data.isnull().sum())) if len(col_data) > 0 else 0
        quality_factors.append(completeness * 0.3)
        
        # Consistency (pattern matching)
        consistency = patterns.get('pattern_matches', 0) / len(col_data) if len(col_data) > 0 else 0
        quality_factors.append(min(consistency, 1.0) * 0.25)
        
        # Validity (data type appropriateness)
        validity = 0.8  # Base validity score
        if data_type == DataType.DATE and 'date_format' in patterns:
            validity = 0.9
        elif data_type == DataType.SALES_AMOUNT and patterns.get('currency_format', False):
            validity = 0.9
        quality_factors.append(validity * 0.25)
        
        # Uniqueness (appropriate for data type)
        unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
        if data_type in [DataType.CATEGORICAL, DataType.PRODUCT_CATEGORY, DataType.REGION]:
            uniqueness = 1.0 - unique_ratio if unique_ratio < 0.5 else 0.5  # Lower is better for categories
        else:
            uniqueness = unique_ratio  # Higher is better for other types
        quality_factors.append(uniqueness * 0.2)
        
        return sum(quality_factors)

    async def _generate_column_recommendations(self, col_name: str, col_data: pd.Series, 
                                            data_type: DataType, quality_score: float, 
                                            null_percentage: float) -> List[str]:
        """Generate actionable recommendations for column improvement"""
        
        recommendations = []
        
        # Null value recommendations
        if null_percentage > 20:
            recommendations.append(f"High null percentage ({null_percentage:.1f}%) - consider data imputation or validation")
        elif null_percentage > 5:
            recommendations.append(f"Some missing values ({null_percentage:.1f}%) detected")
        
        # Data type specific recommendations
        if data_type == DataType.DATE:
            if quality_score < 0.7:
                recommendations.append("Date format inconsistencies detected - standardize date format")
        
        elif data_type == DataType.SALES_AMOUNT:
            if (col_data < 0).any():
                recommendations.append("Negative sales amounts detected - verify data accuracy")
            if col_data.std() / col_data.mean() > 2:  # High coefficient of variation
                recommendations.append("High sales variance detected - check for outliers")
        
        elif data_type == DataType.CATEGORICAL:
            unique_ratio = col_data.nunique() / len(col_data)
            if unique_ratio > 0.5:
                recommendations.append("High category diversity - consider grouping similar categories")
        
        # Quality score recommendations
        if quality_score < 0.5:
            recommendations.append("Low data quality score - review data collection process")
        elif quality_score < 0.7:
            recommendations.append("Moderate data quality - minor improvements recommended")
        
        return recommendations

    async def _auto_map_columns(self, detected_columns: List[DetectedColumn]) -> List[ColumnMapping]:
        """Intelligent auto-mapping of detected columns to required fields"""
        
        mappings = []
        
        for required_field in self.required_fields.keys():
            mapping = await self._find_best_column_match(required_field, detected_columns)
            mappings.append(mapping)
        
        return mappings

    async def _find_best_column_match(self, required_field: str, 
                                    detected_columns: List[DetectedColumn]) -> ColumnMapping:
        """Find the best matching column for a required field"""
        
        candidates = []
        
        # Get expected data type for required field
        expected_type = self._get_expected_data_type(required_field)
        
        for col in detected_columns:
            score = 0.0
            reasons = []
            
            # Name similarity score
            name_score = self._calculate_name_similarity(required_field, col.name)
            score += name_score * 0.4
            if name_score > 0.7:
                reasons.append(f"Name similarity: {name_score:.2f}")
            
            # Data type match score
            type_score = 1.0 if col.type == expected_type else 0.3
            score += type_score * 0.4
            if type_score > 0.5:
                reasons.append(f"Data type match: {col.type.value}")
            
            # Confidence score
            score += col.confidence * 0.2
            if col.confidence > 0.8:
                reasons.append(f"High detection confidence: {col.confidence:.2f}")
            
            if score > 0.3:  # Minimum threshold
                candidates.append((col.name, score, " | ".join(reasons)))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_match = candidates[0]
            alternatives = candidates[1:3]  # Top 2 alternatives
            
            # Determine status
            if best_match[1] > 0.8:
                status = 'mapped'
            elif best_match[1] > 0.6:
                status = 'uncertain'
            else:
                status = 'mapped'  # Still map but with lower confidence
            
            return ColumnMapping(
                required_field=required_field,
                detected_column=best_match[0],
                confidence=best_match[1],
                status=status,
                alternatives=[(alt[0], alt[1]) for alt in alternatives],
                mapping_reason=best_match[2]
            )
        else:
            return ColumnMapping(
                required_field=required_field,
                detected_column=None,
                confidence=0.0,
                status='unmapped',
                alternatives=[],
                mapping_reason='No suitable column found'
            )

    def _get_expected_data_type(self, required_field: str) -> DataType:
        """Get expected data type for required field"""
        
        type_mapping = {
            'date': DataType.DATE,
            'sales_amount': DataType.SALES_AMOUNT,
            'product_category': DataType.PRODUCT_CATEGORY,
            'region': DataType.REGION
        }
        
        return type_mapping.get(required_field, DataType.UNKNOWN)

    def _calculate_name_similarity(self, required_field: str, column_name: str) -> float:
        """Calculate similarity between required field and column name"""
        
        required_keywords = self.required_fields.get(required_field, [])
        column_name_lower = column_name.lower()
        
        # Direct keyword match
        for keyword in required_keywords:
            if keyword in column_name_lower:
                return 0.9
        
        # Partial match
        for keyword in required_keywords:
            if any(part in column_name_lower for part in keyword.split('_')):
                return 0.6
        
        # Fuzzy matching (simple implementation)
        max_similarity = 0.0
        for keyword in required_keywords:
            # Simple character overlap ratio
            overlap = len(set(keyword) & set(column_name_lower))
            similarity = overlap / max(len(keyword), len(column_name_lower))
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity

    async def _assess_data_quality_comprehensive(self, df: pd.DataFrame, 
                                               detected_columns: List[DetectedColumn]) -> DataQuality:
        """Comprehensive data quality assessment"""
        
        # Calculate individual quality metrics
        completeness = await self._calculate_completeness(df)
        consistency = await self._calculate_consistency(detected_columns)
        validity = await self._calculate_validity(df, detected_columns)
        accuracy = await self._calculate_accuracy(df, detected_columns)
        uniqueness = await self._calculate_uniqueness(df)
        timeliness = await self._calculate_timeliness(df, detected_columns)
        
        # Overall score (weighted average)
        overall_score = (
            completeness * 0.25 +
            consistency * 0.20 +
            validity * 0.20 +
            accuracy * 0.15 +
            uniqueness * 0.10 +
            timeliness * 0.10
        )
        
        # Generate issues and recommendations
        issues, recommendations = await self._generate_quality_issues_and_recommendations(
            df, detected_columns, completeness, consistency, validity, accuracy, uniqueness, timeliness
        )
        
        quality_breakdown = {
            'completeness': completeness,
            'consistency': consistency,
            'validity': validity,
            'accuracy': accuracy,
            'uniqueness': uniqueness,
            'timeliness': timeliness
        }
        
        return DataQuality(
            overall_score=overall_score,
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            accuracy=accuracy,
            uniqueness=uniqueness,
            timeliness=timeliness,
            issues=issues,
            recommendations=recommendations,
            quality_breakdown=quality_breakdown
        )

    async def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        
        total_cells = df.size
        non_null_cells = df.count().sum()
        
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    async def _calculate_consistency(self, detected_columns: List[DetectedColumn]) -> float:
        """Calculate data consistency score based on column detection confidence"""
        
        if not detected_columns:
            return 0.0
        
        confidence_scores = [col.confidence for col in detected_columns]
        return sum(confidence_scores) / len(confidence_scores)

    async def _calculate_validity(self, df: pd.DataFrame, detected_columns: List[DetectedColumn]) -> float:
        """Calculate data validity score"""
        
        validity_scores = []
        
        for col in detected_columns:
            col_data = df[col.name].dropna()
            
            if col.type == DataType.DATE:
                try:
                    pd.to_datetime(col_data)
                    validity_scores.append(1.0)
                except:
                    validity_scores.append(0.5)
            
            elif col.type == DataType.SALES_AMOUNT:
                # Check for reasonable sales values
                if pd.api.types.is_numeric_dtype(col_data):
                    # Most sales should be positive and within reasonable range
                    positive_ratio = (col_data > 0).sum() / len(col_data) if len(col_data) > 0 else 0
                    validity_scores.append(positive_ratio)
                else:
                    validity_scores.append(0.3)
            
            else:
                validity_scores.append(0.8)  # Default validity for other types
        
        return sum(validity_scores) / len(validity_scores) if validity_scores else 0.0

    async def _calculate_accuracy(self, df: pd.DataFrame, detected_columns: List[DetectedColumn]) -> float:
        """Calculate data accuracy score (placeholder implementation)"""
        
        # This would typically require external validation or business rules
        # For now, use a heuristic based on data patterns
        
        accuracy_indicators = []
        
        for col in detected_columns:
            col_data = df[col.name].dropna()
            
            # Check for obvious data entry errors
            if col.type == DataType.SALES_AMOUNT and pd.api.types.is_numeric_dtype(col_data):
                # Check for unreasonable values (e.g., negative sales, extremely high values)
                reasonable_values = col_data[(col_data >= 0) & (col_data <= col_data.quantile(0.99))]
                accuracy_indicators.append(len(reasonable_values) / len(col_data) if len(col_data) > 0 else 0)
            
            elif col.type == DataType.REGION:
                # Check for consistent formatting
                avg_length = col_data.str.len().mean()
                std_length = col_data.str.len().std()
                consistency_score = 1.0 - min(std_length / avg_length, 1.0) if avg_length > 0 else 0
                accuracy_indicators.append(consistency_score)
            
            else:
                accuracy_indicators.append(0.8)  # Default accuracy
        
        return sum(accuracy_indicators) / len(accuracy_indicators) if accuracy_indicators else 0.8

    async def _calculate_uniqueness(self, df: pd.DataFrame) -> float:
        """Calculate data uniqueness score"""
        
        # Check for duplicate rows
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        
        return unique_rows / total_rows if total_rows > 0 else 0.0

    async def _calculate_timeliness(self, df: pd.DataFrame, detected_columns: List[DetectedColumn]) -> float:
        """Calculate data timeliness score"""
        
        # Look for date columns to assess timeliness
        date_columns = [col for col in detected_columns if col.type == DataType.DATE]
        
        if not date_columns:
            return 0.7  # Default score if no date columns
        
        timeliness_scores = []
        current_date = datetime.now()
        
        for date_col in date_columns:
            try:
                col_data = pd.to_datetime(df[date_col.name], errors='coerce').dropna()
                if len(col_data) > 0:
                    latest_date = col_data.max()
                    days_old = (current_date - latest_date).days
                    
                    # Score based on recency (higher score for more recent data)
                    if days_old <= 30:
                        timeliness_scores.append(1.0)
                    elif days_old <= 90:
                        timeliness_scores.append(0.8)
                    elif days_old <= 365:
                        timeliness_scores.append(0.6)
                    else:
                        timeliness_scores.append(0.4)
            except:
                timeliness_scores.append(0.5)
        
        return sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.7

    async def _generate_quality_issues_and_recommendations(self, df: pd.DataFrame, 
                                                         detected_columns: List[DetectedColumn],
                                                         completeness: float, consistency: float,
                                                         validity: float, accuracy: float,
                                                         uniqueness: float, timeliness: float) -> Tuple[List[str], List[str]]:
        """Generate quality issues and actionable recommendations"""
        
        issues = []
        recommendations = []
        
        # Completeness issues
        if completeness < 0.8:
            missing_percentage = (1 - completeness) * 100
            issues.append(f"High missing data rate: {missing_percentage:.1f}% of cells are empty")
            recommendations.append("Implement data validation at collection point to reduce missing values")
        
        # Consistency issues
        if consistency < 0.7:
            issues.append("Low column detection confidence indicates inconsistent data formats")
            recommendations.append("Standardize data entry formats and implement validation rules")
        
        # Validity issues
        if validity < 0.7:
            issues.append("Data validity concerns detected in multiple columns")
            recommendations.append("Review data validation rules and implement format checks")
        
        # Accuracy issues
        if accuracy < 0.7:
            issues.append("Potential data accuracy problems detected")
            recommendations.append("Implement data quality checks and outlier detection")
        
        # Uniqueness issues
        if uniqueness < 0.9:
            duplicate_percentage = (1 - uniqueness) * 100
            issues.append(f"Duplicate records detected: {duplicate_percentage:.1f}% of data")
            recommendations.append("Implement deduplication process and unique constraints")
        
        # Timeliness issues
        if timeliness < 0.6:
            issues.append("Data appears to be outdated")
            recommendations.append("Establish regular data refresh schedule and automated updates")
        
        # Column-specific issues
        for col in detected_columns:
            if col.quality_score < 0.5:
                issues.append(f"Column '{col.name}' has low quality score ({col.quality_score:.2f})")
                recommendations.extend(col.recommendations)
        
        return issues, recommendations

    async def _create_enhanced_preview(self, df: pd.DataFrame, 
                                     detected_columns: List[DetectedColumn]) -> Dict[str, Any]:
        """Create enhanced preview with statistics and insights"""
        
        preview_data = {
            'basic_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'file_size_estimate': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            },
            'column_summary': [],
            'sample_data': df.head(5).to_dict('records') if not df.empty else [],
            'data_types_distribution': {},
            'quality_summary': {}
        }
        
        # Column summary with enhanced information
        for col in detected_columns:
            col_info = {
                'name': col.name,
                'type': col.type.value,
                'confidence': col.confidence,
                'quality_score': col.quality_score,
                'null_percentage': col.null_percentage,
                'unique_count': col.unique_count,
                'sample_values': col.sample_values,
                'recommendations_count': len(col.recommendations)
            }
            preview_data['column_summary'].append(col_info)
        
        # Data types distribution
        type_counts = {}
        for col in detected_columns:
            type_name = col.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        preview_data['data_types_distribution'] = type_counts
        
        # Quality summary
        avg_quality = sum(col.quality_score for col in detected_columns) / len(detected_columns) if detected_columns else 0
        preview_data['quality_summary'] = {
            'average_column_quality': avg_quality,
            'high_quality_columns': sum(1 for col in detected_columns if col.quality_score > 0.8),
            'low_quality_columns': sum(1 for col in detected_columns if col.quality_score < 0.5),
            'total_recommendations': sum(len(col.recommendations) for col in detected_columns)
        }
        
        return preview_data

    async def _generate_processing_stats(self, df: pd.DataFrame, 
                                       detected_columns: List[DetectedColumn]) -> Dict[str, Any]:
        """Generate comprehensive processing statistics"""
        
        return {
            'processing_time': datetime.now().isoformat(),
            'data_shape': {'rows': len(df), 'columns': len(df.columns)},
            'detection_summary': {
                'total_columns': len(detected_columns),
                'high_confidence_detections': sum(1 for col in detected_columns if col.confidence > 0.8),
                'low_confidence_detections': sum(1 for col in detected_columns if col.confidence < 0.5),
                'average_confidence': sum(col.confidence for col in detected_columns) / len(detected_columns) if detected_columns else 0
            },
            'data_quality_summary': {
                'average_quality_score': sum(col.quality_score for col in detected_columns) / len(detected_columns) if detected_columns else 0,
                'columns_needing_attention': sum(1 for col in detected_columns if col.quality_score < 0.6),
                'total_quality_issues': sum(len(col.recommendations) for col in detected_columns)
            }
        }

    def _is_mapping_successful(self, column_mappings: List[ColumnMapping]) -> bool:
        """Check if column mapping is successful enough for standardization"""
        
        mapped_count = sum(1 for mapping in column_mappings if mapping.detected_column is not None)
        return mapped_count >= len(self.required_fields) * 0.75  # At least 75% mapped

    async def process_file_basic(self, file_content: bytes, filename: str, 
                               file_format: str) -> ProcessingResult:
        """
        Fallback processing mode with basic analysis
        Used when intelligent processing fails or circuit breaker is open
        """
        try:
            logger.info(f"Starting basic processing for {filename}")
            
            # Basic file parsing
            df = await self._parse_file_basic(file_content, file_format)
            
            # Basic column detection
            detected_columns = await self._detect_columns_basic(df)
            
            # Simple column mapping
            column_mappings = await self._map_columns_basic(detected_columns)
            
            # Basic data quality assessment
            data_quality = await self._assess_data_quality_basic(df)
            
            # Simple preview
            preview_data = await self._create_basic_preview(df)
            
            # Basic stats
            processing_stats = {
                'processing_mode': 'basic_fallback',
                'processing_time': 0.0,
                'rows_processed': len(df),
                'columns_detected': len(detected_columns),
                'columns_mapped': sum(1 for m in column_mappings if m.detected_column),
                'quality_score': data_quality.overall_score
            }
            
            logger.info(f"Basic processing completed for {filename}")
            
            return ProcessingResult(
                detected_columns=detected_columns,
                column_mappings=column_mappings,
                data_quality=data_quality,
                preview_data=preview_data,
                processing_stats=processing_stats,
                standardized_data=None  # Not available in basic mode
            )
            
        except Exception as e:
            logger.error(f"Basic processing failed for {filename}: {e}")
            raise ValueError(f"Basic processing failed: {str(e)}")
    
    async def _parse_file_basic(self, content: bytes, file_format: str) -> pd.DataFrame:
        """Basic file parsing with minimal error handling"""
        try:
            if file_format.lower() == 'csv':
                # Try common encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content_str = content.decode(encoding)
                        df = pd.read_csv(io.StringIO(content_str))
                        return df
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue
                raise ValueError("Could not decode CSV file")
            
            elif file_format.lower() in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(content))
                return df
            
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            raise ValueError(f"Failed to parse file: {str(e)}")
    
    async def _detect_columns_basic(self, df: pd.DataFrame) -> List[DetectedColumn]:
        """Basic column detection with simple heuristics"""
        detected_columns = []
        
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Basic type detection
            data_type = DataType.UNKNOWN
            confidence = 0.5
            
            # Simple pattern matching
            col_name_lower = col_name.lower()
            
            if any(keyword in col_name_lower for keyword in ['date', 'time']):
                data_type = DataType.DATE
                confidence = 0.7
            elif any(keyword in col_name_lower for keyword in ['sales', 'amount', 'revenue', 'price']):
                data_type = DataType.SALES_AMOUNT
                confidence = 0.7
            elif any(keyword in col_name_lower for keyword in ['product', 'category', 'sku']):
                data_type = DataType.PRODUCT_CATEGORY
                confidence = 0.7
            elif any(keyword in col_name_lower for keyword in ['region', 'location', 'area']):
                data_type = DataType.REGION
                confidence = 0.7
            elif pd.api.types.is_numeric_dtype(col_data):
                data_type = DataType.NUMERIC
                confidence = 0.6
            else:
                data_type = DataType.CATEGORICAL
                confidence = 0.4
            
            # Basic statistics
            null_percentage = col_data.isnull().sum() / len(col_data) * 100
            unique_count = col_data.nunique()
            sample_values = col_data.dropna().astype(str).head(3).tolist()
            
            detected_columns.append(DetectedColumn(
                name=col_name,
                type=data_type,
                sample_values=sample_values,
                confidence=confidence,
                null_percentage=null_percentage,
                unique_count=unique_count,
                data_patterns={},
                quality_score=max(0.0, 1.0 - null_percentage / 100),
                recommendations=[]
            ))
        
        return detected_columns
    
    async def _map_columns_basic(self, detected_columns: List[DetectedColumn]) -> List[ColumnMapping]:
        """Basic column mapping with simple matching"""
        mappings = []
        
        for required_field in self.required_fields.keys():
            best_match = None
            best_confidence = 0.0
            
            for col in detected_columns:
                # Simple name matching
                col_name_lower = col.name.lower()
                field_keywords = self.required_fields[required_field]
                
                for keyword in field_keywords:
                    if keyword in col_name_lower:
                        confidence = 0.8 if keyword == col_name_lower else 0.6
                        if confidence > best_confidence:
                            best_match = col.name
                            best_confidence = confidence
                        break
            
            status = 'mapped' if best_match else 'unmapped'
            
            mappings.append(ColumnMapping(
                required_field=required_field,
                detected_column=best_match,
                confidence=best_confidence,
                status=status,
                alternatives=[],
                mapping_reason=f"Basic name matching for {required_field}"
            ))
        
        return mappings
    
    async def _assess_data_quality_basic(self, df: pd.DataFrame) -> DataQuality:
        """Basic data quality assessment"""
        # Simple completeness calculation
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        # Basic consistency (assume reasonable if data loaded)
        consistency = 0.7
        
        # Basic validity (assume reasonable if no obvious issues)
        validity = 0.6
        
        # Overall score
        overall_score = (completeness + consistency + validity) / 3
        
        return DataQuality(
            overall_score=overall_score,
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            accuracy=0.6,  # Default assumption
            uniqueness=0.7,  # Default assumption
            timeliness=0.7,  # Default assumption
            issues=[],
            recommendations=["Consider using intelligent processing for detailed analysis"],
            quality_breakdown={
                'completeness': completeness,
                'consistency': consistency,
                'validity': validity
            }
        )
    
    async def _create_basic_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create basic preview data"""
        return {
            'basic_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'sample_data': df.head(5).to_dict('records'),
            'column_info': [
                {
                    'name': col,
                    'type': str(df[col].dtype),
                    'non_null': df[col].count(),
                    'null_count': df[col].isnull().sum()
                }
                for col in df.columns
            ]
        }

    async def _standardize_data_format(self, df: pd.DataFrame, 
                                     column_mappings: List[ColumnMapping]) -> pd.DataFrame:
        """Standardize data format based on successful mappings"""
        
        standardized_df = df.copy()
        
        for mapping in column_mappings:
            if mapping.detected_column and mapping.confidence > 0.5:
                try:
                    if mapping.required_field == 'date':
                        standardized_df[mapping.detected_column] = pd.to_datetime(
                            standardized_df[mapping.detected_column], errors='coerce'
                        )
                    
                    elif mapping.required_field == 'sales_amount':
                        # Clean currency formatting
                        col_data = standardized_df[mapping.detected_column]
                        if col_data.dtype == 'object':
                            # Remove currency symbols and convert to numeric
                            cleaned = col_data.str.replace(r'[$,€£]', '', regex=True)
                            standardized_df[mapping.detected_column] = pd.to_numeric(cleaned, errors='coerce')
                    
                    elif mapping.required_field in ['product_category', 'region']:
                        # Standardize text formatting
                        standardized_df[mapping.detected_column] = (
                            standardized_df[mapping.detected_column]
                            .astype(str)
                            .str.strip()
                            .str.title()
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to standardize column {mapping.detected_column}: {e}")
        
        return standardized_df

# Export the main processor class
__all__ = ['SmartDataProcessor', 'ProcessingResult', 'DetectedColumn', 'ColumnMapping', 'DataQuality']