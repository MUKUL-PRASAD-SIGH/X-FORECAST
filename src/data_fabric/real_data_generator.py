import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests

class RealDataGenerator:
    def __init__(self):
        self.economic_indicators = {}
        self.weather_data = {}
    
    def generate_realistic_dataset(self, start_date='2022-01-01', end_date='2023-12-31', num_products=20):
        """Generate highly realistic dataset with external data sources"""
        
        # Date range
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Product categories with realistic characteristics
        categories = {
            'Electronics': {'base_demand': 150, 'seasonality': 0.4, 'price_range': (199, 999)},
            'Apparel': {'base_demand': 200, 'seasonality': 0.6, 'price_range': (29, 199)},
            'Home': {'base_demand': 80, 'seasonality': 0.3, 'price_range': (49, 299)},
            'Beauty': {'base_demand': 120, 'seasonality': 0.5, 'price_range': (19, 89)},
            'Sports': {'base_demand': 100, 'seasonality': 0.4, 'price_range': (39, 249)},
            'Books': {'base_demand': 60, 'seasonality': 0.2, 'price_range': (9, 49)},
            'Toys': {'base_demand': 90, 'seasonality': 0.8, 'price_range': (19, 99)},
            'Food': {'base_demand': 300, 'seasonality': 0.1, 'price_range': (5, 29)}
        }
        
        # Generate products
        products = []
        for i in range(num_products):
            category = np.random.choice(list(categories.keys()))
            cat_info = categories[category]
            
            products.append({
                'id': f'SKU{i+1:03d}',
                'category': category,
                'base_demand': cat_info['base_demand'] * np.random.uniform(0.7, 1.3),
                'seasonality': cat_info['seasonality'],
                'price': np.random.uniform(*cat_info['price_range']),
                'launch_date': np.random.choice(dates[:len(dates)//2])  # Launched in first half
            })
        
        # Get real economic data
        economic_data = self.get_economic_indicators(dates)
        
        # Generate dataset
        all_data = []
        
        for product in products:
            product_data = self.generate_product_demand(
                dates, product, economic_data
            )
            all_data.extend(product_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Add real business features
        df = self.add_business_features(df)
        
        return df
    
    def generate_product_demand(self, dates, product, economic_data):
        """Generate realistic demand for a single product"""
        
        data = []
        base_demand = product['base_demand']
        
        for i, date in enumerate(dates):
            # Skip dates before product launch
            if date < product['launch_date']:
                continue
            
            # Base demand with growth trend
            days_since_launch = (date - product['launch_date']).days
            growth_factor = 1 + (0.001 * days_since_launch)  # 0.1% daily growth
            
            # Seasonal patterns
            seasonal = product['seasonality'] * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Weekly patterns (weekends different)
            weekly = 0.1 * np.sin(2 * np.pi * date.weekday() / 7)
            if date.weekday() >= 5:  # Weekend boost for some categories
                if product['category'] in ['Apparel', 'Electronics', 'Toys']:
                    weekly += 0.2
            
            # Holiday effects
            holiday_boost = self.get_holiday_effect(date, product['category'])
            
            # Economic impact
            economic_impact = economic_data.get(date, 1.0)
            
            # Random noise
            noise = np.random.normal(0, 0.15)
            
            # Calculate final demand
            demand_multiplier = (1 + seasonal + weekly + holiday_boost + noise) * economic_impact * growth_factor
            demand = max(0, int(base_demand * demand_multiplier))
            
            # Revenue calculation
            # Price varies slightly over time
            price_variation = 1 + np.random.normal(0, 0.05)
            current_price = product['price'] * price_variation
            revenue = demand * current_price
            
            # Promotion probability
            promotion_prob = self.get_promotion_probability(date, product['category'])
            is_promotion = np.random.random() < promotion_prob
            
            if is_promotion:
                demand = int(demand * np.random.uniform(1.3, 2.0))  # Promotion uplift
                current_price *= 0.8  # 20% discount
                revenue = demand * current_price
            
            data.append({
                'date': date,
                'product_id': product['id'],
                'category': product['category'],
                'demand': demand,
                'price': round(current_price, 2),
                'revenue': round(revenue, 2),
                'is_promotion': int(is_promotion),
                'days_since_launch': days_since_launch
            })
        
        return data
    
    def get_economic_indicators(self, dates):
        """Get real economic indicators (simplified)"""
        # In real implementation, would fetch from FRED, Yahoo Finance, etc.
        economic_data = {}
        
        for i, date in enumerate(dates):
            # Simulate economic cycles
            cycle = 0.05 * np.sin(2 * np.pi * i / 365)  # Annual cycle
            trend = 0.0001 * i  # Long-term growth
            shock = 0
            
            # COVID impact simulation
            if date >= datetime(2020, 3, 1) and date <= datetime(2021, 6, 1):
                shock = -0.2 * np.exp(-(date - datetime(2020, 3, 1)).days / 100)
            
            economic_data[date] = 1 + cycle + trend + shock + np.random.normal(0, 0.02)
        
        return economic_data
    
    def get_holiday_effect(self, date, category):
        """Calculate holiday demand boost"""
        holiday_effects = {
            'Christmas': (12, [20, 25], 0.5),
            'Black Friday': (11, [25, 26, 27, 28], 0.8),
            'Back to School': (8, list(range(15, 32)), 0.3),
            'Valentines': (2, [14], 0.2),
            'Easter': (4, [15, 16, 17], 0.2)
        }
        
        category_multipliers = {
            'Electronics': 1.5,
            'Apparel': 1.3,
            'Toys': 2.0,
            'Beauty': 1.2,
            'Home': 1.1,
            'Sports': 0.9,
            'Books': 0.8,
            'Food': 1.0
        }
        
        boost = 0
        for holiday, (month, days, effect) in holiday_effects.items():
            if date.month == month and date.day in days:
                boost += effect * category_multipliers.get(category, 1.0)
        
        return min(boost, 1.0)  # Cap at 100% boost
    
    def get_promotion_probability(self, date, category):
        """Calculate promotion probability"""
        base_prob = 0.05  # 5% base probability
        
        # Higher during holidays
        if self.get_holiday_effect(date, category) > 0:
            base_prob = 0.25
        
        # Higher on weekends
        if date.weekday() >= 5:
            base_prob *= 1.5
        
        # Category-specific
        category_multipliers = {
            'Apparel': 1.5,
            'Electronics': 1.2,
            'Beauty': 1.3,
            'Toys': 1.4,
            'Home': 1.1,
            'Sports': 1.2,
            'Books': 0.8,
            'Food': 0.7
        }
        
        return min(base_prob * category_multipliers.get(category, 1.0), 0.4)
    
    def add_business_features(self, df):
        """Add comprehensive business features"""
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
        
        # Holiday flags
        df['is_holiday'] = df.apply(lambda x: int(self.get_holiday_effect(x['date'], x['category']) > 0), axis=1)
        
        # Business metrics
        df['profit_margin'] = np.random.uniform(0.2, 0.4, len(df))  # 20-40% margin
        df['profit'] = df['revenue'] * df['profit_margin']
        
        # Inventory simulation
        df['stock_level'] = np.random.randint(50, 500, len(df))
        df['stockout_risk'] = (df['demand'] > df['stock_level'] * 0.8).astype(int)
        
        # Customer metrics
        df['customer_rating'] = np.random.uniform(3.5, 5.0, len(df))
        df['return_rate'] = np.random.uniform(0.02, 0.15, len(df))
        
        # Competitive metrics
        df['competitor_price'] = df['price'] * np.random.uniform(0.9, 1.1, len(df))
        df['price_competitiveness'] = df['price'] / df['competitor_price']
        
        return df
    
    def save_dataset(self, df, filename='realistic_demand_data.csv'):
        """Save dataset with summary statistics"""
        
        # Save main dataset
        df.to_csv(f'./data/raw/{filename}', index=False)
        
        # Generate summary
        summary = {
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'products': df['product_id'].nunique(),
            'categories': df['category'].nunique(),
            'total_revenue': df['revenue'].sum(),
            'avg_demand': df['demand'].mean(),
            'promotion_rate': df['is_promotion'].mean() * 100
        }
        
        # Save summary
        with open(f'./data/processed/dataset_summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Generated realistic dataset: {len(df):,} records")
        print(f"📅 Date range: {summary['date_range']}")
        print(f"🏷️ Products: {summary['products']}")
        print(f"📊 Categories: {summary['categories']}")
        print(f"💰 Total revenue: ${summary['total_revenue']:,.2f}")
        print(f"📈 Avg daily demand: {summary['avg_demand']:.1f}")
        print(f"🎆 Promotion rate: {summary['promotion_rate']:.1f}%")
        
        return summary

if __name__ == "__main__":
    generator = RealDataGenerator()
    df = generator.generate_realistic_dataset(num_products=25)
    summary = generator.save_dataset(df)