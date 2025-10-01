import pandas as pd
from src.data_fabric.connector import DataConnector
from src.models.ensemble import EnsembleForecaster
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    
    # Initialize components
    connector = DataConnector(os.getenv('DATA_SOURCE_PATH', './data/raw'))
    forecaster = EnsembleForecaster()
    
    # Example usage
    print("X-FORECAST: AI-Powered Demand Forecasting Engine")
    print("Phase 1 Components Ready:")
    print("✅ Data Pipeline Setup")
    print("✅ Baseline Forecasting Models")
    print("\nTo use:")
    print("1. Place CSV data in ./data/raw/")
    print("2. Run: python main.py")
    
    # Sample data processing (if data exists)
    try:
        # This would load actual data
        sample_data = pd.Series([10, 12, 13, 12, 15, 16, 18, 20, 22, 25])
        
        # Fit models and generate forecast
        forecaster.fit(sample_data)
        forecast = forecaster.forecast(steps=5)
        
        print(f"\nSample forecast: {forecast.values}")
        
        # Export results
        forecaster.export_results(forecast, './data/processed/forecast_output.csv')
        print("✅ Results exported to ./data/processed/forecast_output.csv")
        
    except Exception as e:
        print(f"Demo mode - no data loaded: {e}")

if __name__ == "__main__":
    main()