import pandas as pd
import numpy as np
from src.data_fabric.connector import DataConnector
from src.models.ensemble import EnsembleForecaster
from src.feature_store.feature_engineer import FeatureEngineer
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    
    print("🚀 X-FORECAST: AI-Powered Demand Forecasting Engine")
    print("\n✅ Phase 1 Complete: Data Pipeline + Baseline Models")
    print("✅ Phase 2 Complete: Feature Engineering + ML Models + Dashboard")
    print("\n🎯 Available Features:")
    print("1. Classical Models: ARIMA, ETS, Croston")
    print("2. ML Models: XGBoost, LSTM (optional)")
    print("3. Feature Engineering: 15+ automated features")
    print("4. Interactive Dashboard: Streamlit UI")
    
    # Demo with enhanced features
    try:
        print("\n📊 Running Enhanced Demo...")
        
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        demand = 100 + np.cumsum(np.random.randn(50) * 2) + 10 * np.sin(np.arange(50) * 0.2)
        
        sample_df = pd.DataFrame({
            'date': dates,
            'demand': demand,
            'product_id': 'SKU001'
        })
        
        # Feature Engineering
        print("🔧 Generating features...")
        engineer = FeatureEngineer()
        features_df = engineer.create_features(sample_df)
        
        feature_cols = [col for col in features_df.columns if col not in ['date', 'demand', 'product_id']]
        print(f"✅ Generated {len(feature_cols)} features")
        
        # Enhanced Ensemble Forecasting
        print("🤖 Training ensemble models...")
        forecaster = EnsembleForecaster(use_ml=True)  # Enable ML models if available
        forecaster.fit(sample_df['demand'], features_df)
        
        # Generate forecast
        forecast = forecaster.forecast(steps=7, features_df=features_df.tail(7))
        
        print(f"\n📈 7-day forecast: {forecast.values.round(2)}")
        print(f"📊 Average forecast: {forecast.mean():.2f}")
        
        # Export results
        forecaster.export_results(forecast, './data/processed/enhanced_forecast.csv')
        engineer.save_features(features_df, 'features_output.csv')
        
        print("✅ Results exported to ./data/processed/")
        print("\n🚀 Next Steps:")
        print("1. Run Dashboard: streamlit run src/frontend/dashboard.py")
        print("2. Upload your own data via the dashboard")
        print("3. Explore NPI forecasting and promotion optimization")
        print("4. Use inventory optimization for supply planning")
        
    except Exception as e:
        print(f"⚠️ Demo mode - some features may be limited: {e}")
        print("\nTo enable all features:")
        print("pip install xgboost torch streamlit plotly scipy fastapi uvicorn")

if __name__ == "__main__":
    main()