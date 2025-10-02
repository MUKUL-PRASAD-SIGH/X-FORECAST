import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_comprehensive_dataset():
    """Generate comprehensive dummy dataset with multiple products and realistic patterns"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Date range: 2 years of daily data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Product catalog
    products = [
        {'id': 'SKU001', 'category': 'Electronics', 'price': 299.99, 'seasonality': 'high'},
        {'id': 'SKU002', 'category': 'Apparel', 'price': 49.99, 'seasonality': 'medium'},
        {'id': 'SKU003', 'category': 'Home', 'price': 89.99, 'seasonality': 'low'},
        {'id': 'SKU004', 'category': 'Beauty', 'price': 24.99, 'seasonality': 'high'},
        {'id': 'SKU005', 'category': 'Electronics', 'price': 599.99, 'seasonality': 'medium'},
        {'id': 'SKU006', 'category': 'Apparel', 'price': 79.99, 'seasonality': 'high'},
        {'id': 'SKU007', 'category': 'Home', 'price': 149.99, 'seasonality': 'low'},
        {'id': 'SKU008', 'category': 'Beauty', 'price': 39.99, 'seasonality': 'medium'},
        {'id': 'SKU009', 'category': 'Electronics', 'price': 199.99, 'seasonality': 'high'},
        {'id': 'SKU010', 'category': 'Sports', 'price': 129.99, 'seasonality': 'medium'}
    ]
    
    all_data = []
    
    for product in products:
        product_data = generate_product_data(dates, product)
        all_data.extend(product_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Add additional business features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = generate_holiday_flags(df['date'])
    df['promotion_flag'] = generate_promotion_flags(df)
    df['stockout_flag'] = generate_stockout_flags(df)
    df['weather_impact'] = generate_weather_impact(df)
    df['economic_index'] = generate_economic_index(df['date'])
    
    return df

def generate_product_data(dates, product):
    """Generate realistic demand data for a single product"""
    
    base_demand = {
        'Electronics': 150,
        'Apparel': 200,
        'Home': 80,
        'Beauty': 120,
        'Sports': 100
    }[product['category']]
    
    seasonality_strength = {
        'high': 0.4,
        'medium': 0.2,
        'low': 0.1
    }[product['seasonality']]
    
    data = []
    
    for i, date in enumerate(dates):
        # Base demand with trend
        trend = 0.02 * i / 365  # 2% annual growth
        
        # Seasonal patterns
        seasonal = seasonality_strength * np.sin(2 * np.pi * date.dayofyear / 365)
        weekly_seasonal = 0.1 * np.sin(2 * np.pi * date.weekday() / 7)
        
        # Random noise
        noise = np.random.normal(0, 0.15)
        
        # Calculate demand
        demand = base_demand * (1 + trend + seasonal + weekly_seasonal + noise)
        demand = max(0, int(demand))  # Ensure non-negative integer
        
        # Revenue calculation
        revenue = demand * product['price']
        
        data.append({
            'date': date,
            'product_id': product['id'],
            'category': product['category'],
            'demand': demand,
            'price': product['price'],
            'revenue': revenue
        })
    
    return data

def generate_holiday_flags(dates):
    """Generate holiday flags for major holidays"""
    holidays = []
    for date in dates:
        is_holiday = 0
        # Major holidays (simplified)
        if (date.month == 12 and date.day in [24, 25, 31]) or \
           (date.month == 1 and date.day == 1) or \
           (date.month == 7 and date.day == 4) or \
           (date.month == 11 and date.day in [22, 23, 24, 25]):  # Thanksgiving week
            is_holiday = 1
        holidays.append(is_holiday)
    return holidays

def generate_promotion_flags(df):
    """Generate promotion flags with realistic patterns"""
    promotions = []
    for _, row in df.iterrows():
        # Higher promotion probability during holidays and weekends
        base_prob = 0.05
        if row.get('is_holiday', 0):
            base_prob = 0.3
        elif row['date'].weekday() in [4, 5, 6]:  # Fri, Sat, Sun
            base_prob = 0.15
        
        promotion = 1 if np.random.random() < base_prob else 0
        promotions.append(promotion)
    
    return promotions

def generate_stockout_flags(df):
    """Generate stockout flags (rare events)"""
    return [1 if np.random.random() < 0.02 else 0 for _ in range(len(df))]

def generate_weather_impact(df):
    """Generate weather impact scores"""
    return [np.random.normal(1.0, 0.1) for _ in range(len(df))]

def generate_economic_index(dates):
    """Generate economic index with trend"""
    base_index = 100
    return [base_index + 0.01 * i + np.random.normal(0, 2) for i in range(len(dates))]

if __name__ == "__main__":
    print("🚀 Generating comprehensive dummy dataset...")
    
    # Generate dataset
    df = generate_comprehensive_dataset()
    
    # Save to CSV
    df.to_csv('./data/raw/comprehensive_demand_data.csv', index=False)
    
    print(f"✅ Generated dataset with {len(df):,} records")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏷️ Products: {df['product_id'].nunique()}")
    print(f"📊 Categories: {df['category'].nunique()}")
    print(f"💰 Total revenue: ${df['revenue'].sum():,.2f}")
    print(f"📈 Avg daily demand per product: {df['demand'].mean():.1f}")
    
    # Display sample
    print("\n📋 Sample data:")
    print(df.head(10))
    
    # Save summary stats
    summary = df.groupby('product_id').agg({
        'demand': ['mean', 'std', 'sum'],
        'revenue': 'sum',
        'promotion_flag': 'sum'
    }).round(2)
    
    summary.to_csv('./data/processed/product_summary.csv')
    print("\n✅ Summary saved to ./data/processed/product_summary.csv")