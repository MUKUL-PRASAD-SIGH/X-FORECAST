# 🚀 Company Sales Forecasting System - Quick Start Guide

## ✅ System Status: READY TO USE!

Your company sales forecasting system is fully implemented and ready to use. Here's how to start it:

## 🎯 Quick Start (2 Steps)

### Step 1: Start the API Server
Open a terminal and run:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Start the Cyberpunk Dashboard (Optional)
Open another terminal and run:
```bash
cd frontend && npm start
```

## 🌐 Access Points

- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Cyberpunk Dashboard**: http://localhost:3001

## 📊 What You Can Do

### 1. Register Your Company
```bash
curl -X POST http://localhost:8000/api/company-sales/register \
     -H "Authorization: Bearer my_company_123" \
     -H "Content-Type: application/json" \
     -d '{"company_name": "My Company", "industry": "Retail"}'
```

### 2. Get Data Requirements
```bash
curl -X GET http://localhost:8000/api/company-sales/data-requirements \
     -H "Authorization: Bearer my_company_123"
```

### 3. Upload Sales Data
Use the dashboard or API to upload CSV/Excel files with:
- **Required**: date, sales_amount, product_category, region
- **Optional**: units_sold, customer_count, marketing_spend, promotions

### 4. Generate Forecasts
```bash
curl -X POST http://localhost:8000/api/company-sales/forecast \
     -H "Authorization: Bearer my_company_123" \
     -H "Content-Type: application/json" \
     -d '{"horizon_months": 6}'
```

## 🎯 Key Features Working

✅ **Multi-Company Support**: Each company gets isolated data and models
✅ **Data Upload & Validation**: CSV, Excel, JSON support with quality checks
✅ **Adaptive Ensemble**: 5 models (ARIMA, ETS, XGBoost, LSTM, Croston)
✅ **Pattern Detection**: Automatic seasonal, trending, intermittent pattern detection
✅ **Dynamic Weights**: Models weights update based on performance
✅ **Confidence Intervals**: P10, P50, P90 uncertainty quantification
✅ **Business Insights**: AI-generated recommendations
✅ **Real-time Analytics**: Performance monitoring and model evolution
✅ **Interactive Dashboard**: React cyberpunk web interface
✅ **RESTful API**: Complete programmatic access

## 📋 Sample Data Format

Create a CSV file with this structure:
```csv
date,sales_amount,product_category,region,units_sold,customer_count,marketing_spend,promotions
2024-01-01,15000,Electronics,North,150,75,1200,0
2024-01-02,12000,Electronics,South,120,60,800,0
2024-01-03,18000,Clothing,North,200,100,1500,1
2024-01-04,14000,Clothing,East,140,70,900,0
```

**Requirements:**
- Minimum 6 months of data for forecasting
- Daily or monthly frequency
- Consistent date format (YYYY-MM-DD)
- Positive sales amounts

## 🔧 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   API Server    │    │  Data Storage   │
│  (React)        │◄──►│   (FastAPI)     │◄──►│   (Files)       │
│  Port: 3001     │    │  Port: 8000     │    │  company_data/  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Forecasting     │
                    │ Engine          │
                    │ (5 Models +     │
                    │ Adaptive Logic) │
                    └─────────────────┘
```

## 🎬 Demo Data Available

The system comes with 3 pre-registered demo companies:
- **TechCorp Solutions** (tech_corp_001)
- **RetailPlus Inc** (retail_plus_002) 
- **EcomWorld Ltd** (ecom_world_003)

Each has sample sales data already uploaded for testing.

## 🔍 Troubleshooting

### "Module not found" errors
- Make sure you're in the project root directory
- All required packages are installed (they should be)

### "Insufficient data" errors
- Upload at least 6 months of historical data
- Ensure data has consistent date formatting

### API not responding
- Check if port 8000 is available
- Try a different port: `--port 8001`

### Dashboard not loading
- Check if port 3001 is available
- Make sure Node.js and npm are installed
- Try: `cd frontend && npm install && npm start`

## 📈 How It Works

1. **Upload Data**: Company uploads monthly sales data
2. **Pattern Detection**: System detects seasonal/trending/intermittent patterns
3. **Model Initialization**: 5 forecasting models get initial weights based on pattern
4. **Forecasting**: Ensemble generates forecasts with confidence intervals
5. **Performance Tracking**: System monitors model accuracy over time
6. **Weight Updates**: Model weights automatically adjust based on performance
7. **Business Insights**: AI generates recommendations based on forecasts

## 🎯 Production Ready Features

- **Data Validation**: Comprehensive quality checks
- **Error Handling**: Graceful failure and recovery
- **Logging**: Detailed operation logs
- **Configuration**: Customizable per company
- **Security**: Company-specific data isolation
- **Scalability**: Supports unlimited companies
- **Monitoring**: Real-time performance tracking
- **API Documentation**: Auto-generated OpenAPI docs

## 🚀 Ready to Go!

Your system is production-ready. Just run the two commands above and start uploading your sales data to get intelligent forecasts with adaptive learning!

**Need help?** Check the API docs at http://localhost:8000/docs for detailed endpoint information.