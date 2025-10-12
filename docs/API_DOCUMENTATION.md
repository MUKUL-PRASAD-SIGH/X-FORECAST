# X-FORECAST API Documentation

## Overview
The X-FORECAST API provides comprehensive demand forecasting capabilities through RESTful endpoints. Built with FastAPI, it offers real-time forecasting, NPI analysis, promotion optimization, and inventory management.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API is open for development. Production deployment should implement proper authentication.

## Endpoints

### Health Check
```http
GET /health
```
Returns API health status and version information.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Generate Forecast
```http
POST /forecast
```
Generate demand forecasts using ensemble models.

**Request Body:**
```json
{
  "data": [
    {"date": "2023-01-01", "demand": 100, "product_id": "SKU001"},
    {"date": "2023-01-02", "demand": 105, "product_id": "SKU001"}
  ],
  "horizon": 30,
  "confidence_level": 0.95,
  "use_ml": true
}
```

**Response:**
```json
{
  "forecast": [110.5, 112.3, 108.7, ...],
  "confidence_upper": [125.2, 127.1, 123.4, ...],
  "confidence_lower": [95.8, 97.5, 94.0, ...],
  "metrics": {
    "mean_forecast": 110.2,
    "forecast_std": 8.5,
    "data_points": 100
  },
  "model_weights": {
    "arima": 0.4,
    "ets": 0.4,
    "croston": 0.2
  }
}
```

### Upload Data
```http
POST /upload-data
```
Upload CSV data for analysis.

**Request:** Multipart form data with CSV file

**Response:**
```json
{
  "message": "Data uploaded successfully",
  "validation": {
    "rows": 1000,
    "columns": ["date", "demand", "product_id"],
    "missing_values": {"date": 0, "demand": 5}
  },
  "preview": [
    {"date": "2023-01-01", "demand": 100, "product_id": "SKU001"}
  ]
}
```

### NPI Forecast
```http
POST /npi-forecast
```
Generate forecasts for new product introductions.

**Request Body:**
```json
{
  "product_name": "New Widget 2024",
  "category": "Electronics",
  "price": 299.99,
  "launch_date": "2024-03-01"
}
```

**Response:**
```json
{
  "product_name": "New Widget 2024",
  "similar_products": [
    ["SKU001", 0.94],
    ["SKU005", 0.87]
  ],
  "forecast": [80, 95, 110, 125, 140, 135, 130, 125, 120, 115, 110, 105],
  "confidence": "medium"
}
```

### Promotion Optimization
```http
POST /promotion-optimize
```
Optimize promotion strategies for maximum ROI.

**Request Body:**
```json
{
  "product_id": "SKU001",
  "discount_rate": 20.0,
  "duration": 7,
  "marketing_budget": 5000.0
}
```

**Response:**
```json
{
  "product_id": "SKU001",
  "baseline_demand": 150.5,
  "promoted_demand": 210.7,
  "uplift_factor": 1.4,
  "incremental_units": 421.4,
  "incremental_revenue": 126420.0,
  "roi_percentage": 2428.4,
  "break_even_days": 2
}
```

### Inventory Optimization
```http
POST /inventory-optimize
```
Optimize inventory policies for cost minimization.

**Query Parameters:**
- `product_id` (required): Product identifier
- `service_level` (optional): Target service level (default: 0.95)
- `lead_time` (optional): Lead time in days (default: 7)

**Response:**
```json
{
  "product_id": "SKU001",
  "optimal_order_quantity": 2450.0,
  "reorder_point": 850.0,
  "safety_stock": 320.0,
  "service_level": 0.95,
  "total_cost": 45200.0,
  "inventory_turns": 8.7
}
```

### Performance Analytics
```http
GET /analytics/performance
```
Get comprehensive performance analytics.

**Response:**
```json
{
  "kpis": {
    "total_demand": 125000,
    "avg_daily_demand": 150.5,
    "demand_volatility": 18.5,
    "total_revenue": 2450000.0,
    "data_points": 1000
  },
  "product_performance": {
    "SKU001": {
      "demand": {"mean": 150.5, "sum": 54782, "std": 25.3},
      "revenue": {"sum": 164234.5}
    }
  },
  "category_performance": {
    "Electronics": {
      "demand": {"sum": 75000},
      "revenue": {"sum": 1200000.0}
    }
  },
  "date_range": {
    "start": "2022-01-01",
    "end": "2023-12-31"
  }
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error

Error responses include detailed messages:
```json
{
  "detail": "Product not found"
}
```

## Rate Limiting
Currently no rate limiting is implemented. Production deployment should include appropriate rate limiting.

## Data Formats

### Date Format
All dates should be in ISO format: `YYYY-MM-DD`

### CSV Upload Format
```csv
date,product_id,category,demand,price,revenue
2023-01-01,SKU001,Electronics,150,299.99,44998.50
2023-01-02,SKU001,Electronics,145,299.99,43498.55
```

## Usage Examples

### Python Client
```python
import requests
import json

# Generate forecast
data = {
    "data": [
        {"date": "2023-01-01", "demand": 100, "product_id": "SKU001"},
        {"date": "2023-01-02", "demand": 105, "product_id": "SKU001"}
    ],
    "horizon": 7,
    "use_ml": True
}

response = requests.post("http://localhost:8000/forecast", json=data)
forecast = response.json()
print(f"7-day forecast: {forecast['forecast']}")
```

### cURL Examples
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Generate forecast
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"data": [{"date": "2023-01-01", "demand": 100}], "horizon": 7}'

# Upload data
curl -X POST "http://localhost:8000/upload-data" \
  -F "file=@data.csv"
```

## Deployment

### Local Development
```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
uvicorn src.api.forecast_api:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn src.api.forecast_api:app -w 4 -k uvicorn.workers.UvicornWorker

# Using Docker
docker build -t x-forecast-api .
docker run -p 8000:8000 x-forecast-api
```

## Performance Considerations

- Use batch endpoints for multiple products
- Cache frequently requested forecasts
- Implement connection pooling for database access
- Consider async processing for long-running tasks

## Support

For API support and questions:
- GitHub Issues: [Repository Issues](https://github.com/MUKUL-PRASAD-SIGH/X-FORECAST/issues)
- Documentation: [Full Documentation](./README.md)