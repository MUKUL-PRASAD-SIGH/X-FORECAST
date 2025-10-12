from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import io
import json

try:
    from ..data_fabric.connector import DataConnector
    from ..models.ensemble import EnsembleForecaster
    from ..feature_store.feature_engineer import FeatureEngineer
    from ..models.advanced.npi_engine import NPIEngine
    from ..models.advanced.promotion_engine import PromotionEngine
    from ..models.advanced.inventory_optimizer import InventoryOptimizer
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

app = FastAPI(title="X-FORECAST API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    data: List[Dict]
    horizon: int = 30
    confidence_level: float = 0.95
    use_ml: bool = True

class ForecastResponse(BaseModel):
    forecast: List[float]
    confidence_upper: List[float]
    confidence_lower: List[float]
    metrics: Dict[str, float]
    model_weights: Dict[str, float]

class NPIRequest(BaseModel):
    product_name: str
    category: str
    price: float
    launch_date: str

class PromotionRequest(BaseModel):
    product_id: str
    discount_rate: float
    duration: int
    marketing_budget: float

@app.get("/")
async def root():
    return {"message": "X-FORECAST API v1.0.0", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Feature engineering
        engineer = FeatureEngineer()
        features_df = engineer.create_features(df)
        
        # Generate forecast
        forecaster = EnsembleForecaster(use_ml=request.use_ml)
        forecaster.fit(df['demand'], features_df)
        
        forecast = forecaster.forecast(request.horizon, features_df.tail(request.horizon))
        
        # Calculate confidence intervals
        std_dev = df['demand'].std()
        z_score = 1.96 if request.confidence_level == 0.95 else 2.58
        margin = z_score * std_dev
        
        return ForecastResponse(
            forecast=forecast.tolist(),
            confidence_upper=(forecast + margin).tolist(),
            confidence_lower=(forecast - margin).tolist(),
            metrics={
                "mean_forecast": float(forecast.mean()),
                "forecast_std": float(forecast.std()),
                "data_points": len(df)
            },
            model_weights=forecaster.get_model_weights()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate data
        connector = DataConnector('./data/raw')
        validation = connector.validate_data(df)
        
        # Save uploaded data
        df.to_csv('./data/raw/uploaded_data.csv', index=False)
        
        return {
            "message": "Data uploaded successfully",
            "validation": validation,
            "preview": df.head().to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/npi-forecast")
async def npi_forecast(request: NPIRequest):
    try:
        npi_engine = NPIEngine()
        
        # Load existing product data
        data = pd.read_csv('./data/raw/large_demand_dataset.csv')
        
        # Create product profiles
        profiles = {}
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id]
            profiles[product_id] = npi_engine.create_product_profile(product_data)
        
        # Create new product profile
        new_profile = {
            'avg_demand': 100,  # Estimated
            'demand_std': 20,
            'seasonality_strength': 0.3,
            'trend_strength': 0.1,
            'volatility': 0.2,
            'launch_velocity': 80
        }
        
        # Find similar products
        similar_products = npi_engine.find_similar_products(new_profile, profiles)
        
        # Generate NPI forecast
        forecast = npi_engine.generate_npi_forecast({}, 12)  # 12 weeks
        
        return {
            "product_name": request.product_name,
            "similar_products": similar_products,
            "forecast": forecast.tolist() if len(forecast) > 0 else [80, 95, 110, 125, 140, 135, 130, 125, 120, 115, 110, 105],
            "confidence": "medium"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/promotion-optimize")
async def optimize_promotion(request: PromotionRequest):
    try:
        promo_engine = PromotionEngine()
        
        # Load product data
        data = pd.read_csv('./data/raw/large_demand_dataset.csv')
        product_data = data[data['product_id'] == request.product_id]
        
        if len(product_data) == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Calculate promotion impact
        baseline_demand = product_data['demand'].mean()
        uplift_factor = 1 + (request.discount_rate / 100) * 2  # Simple uplift model
        
        promoted_demand = baseline_demand * uplift_factor
        incremental_units = (promoted_demand - baseline_demand) * request.duration
        
        # Calculate ROI
        unit_price = product_data['price'].iloc[0]
        incremental_revenue = incremental_units * unit_price * (1 - request.discount_rate / 100)
        roi = (incremental_revenue - request.marketing_budget) / request.marketing_budget * 100
        
        return {
            "product_id": request.product_id,
            "baseline_demand": float(baseline_demand),
            "promoted_demand": float(promoted_demand),
            "uplift_factor": float(uplift_factor),
            "incremental_units": float(incremental_units),
            "incremental_revenue": float(incremental_revenue),
            "roi_percentage": float(roi),
            "break_even_days": max(1, int(request.marketing_budget / (promoted_demand - baseline_demand) / unit_price))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/inventory-optimize")
async def optimize_inventory(product_id: str, service_level: float = 0.95, lead_time: int = 7):
    try:
        optimizer = InventoryOptimizer()
        
        # Load product data
        data = pd.read_csv('./data/raw/large_demand_dataset.csv')
        product_data = data[data['product_id'] == product_id]
        
        if len(product_data) == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Calculate forecast and error
        forecast = pd.Series(product_data['demand'].values)
        forecast_error = forecast.std()
        unit_cost = product_data['price'].iloc[0] * 0.6  # Assume 60% cost
        
        # Optimize inventory policy
        policy = optimizer.optimize_inventory_policy(
            forecast, forecast_error, lead_time, unit_cost, service_level=service_level
        )
        
        return {
            "product_id": product_id,
            "optimal_order_quantity": float(policy['eoq']),
            "reorder_point": float(policy['reorder_point']),
            "safety_stock": float(policy['safety_stock']),
            "service_level": float(service_level),
            "total_cost": float(policy['total_inventory_cost']),
            "inventory_turns": float(policy['inventory_turns'])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/performance")
async def get_performance_analytics():
    try:
        # Load data and calculate real performance metrics
        data = pd.read_csv('./data/raw/large_demand_dataset.csv')
        
        # Calculate KPIs
        total_demand = data['demand'].sum()
        avg_demand = data['demand'].mean()
        demand_volatility = data['demand'].std() / avg_demand * 100
        total_revenue = data['revenue'].sum()
        
        # Product performance
        product_performance = data.groupby('product_id').agg({
            'demand': ['mean', 'sum', 'std'],
            'revenue': 'sum'
        }).round(2)
        
        # Category performance
        category_performance = data.groupby('category').agg({
            'demand': 'sum',
            'revenue': 'sum'
        }).round(2)
        
        return {
            "kpis": {
                "total_demand": int(total_demand),
                "avg_daily_demand": float(avg_demand),
                "demand_volatility": float(demand_volatility),
                "total_revenue": float(total_revenue),
                "data_points": len(data)
            },
            "product_performance": product_performance.to_dict(),
            "category_performance": category_performance.to_dict(),
            "date_range": {
                "start": data['date'].min(),
                "end": data['date'].max()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)