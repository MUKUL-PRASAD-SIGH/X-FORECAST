# Company Sales Forecasting System 📈

A comprehensive, company-specific sales forecasting system with adaptive ensemble learning, automatic model weight updates, and real-time business insights.

## 🌟 Features

### 🏢 Multi-Company Support
- **Company Registration**: Each company gets isolated data and models
- **Custom Requirements**: Configurable data requirements per company
- **Secure Access**: Company-specific authentication and data isolation
- **Industry-Specific**: Tailored configurations for different industries

### 📊 Intelligent Data Management
- **Flexible Upload**: Support for CSV, Excel, and JSON formats
- **Data Validation**: Automatic validation against company requirements
- **Quality Checks**: Data quality scoring and issue detection
- **Version Control**: Track all data uploads with metadata

### 🤖 Adaptive Ensemble Forecasting
- **Pattern Detection**: Automatic detection of seasonal, trending, intermittent, and stationary patterns
- **Model Selection**: 5 forecasting models (ARIMA, ETS, XGBoost, LSTM, Croston)
- **Dynamic Weights**: Automatic weight updates based on model performance
- **Confidence Intervals**: P10, P50, P90 confidence intervals using bootstrap methods

### 📈 Real-Time Analytics
- **Performance Monitoring**: Track model accuracy over time
- **Weight Evolution**: Monitor how model weights change with new data
- **Business Insights**: AI-generated recommendations and insights
- **Interactive Dashboard**: React-based cyberpunk dashboard for visualization

### 🔗 API Integration
- **RESTful API**: Complete API for programmatic access
- **Real-time Updates**: WebSocket support for live updates
- **Comprehensive Docs**: Auto-generated API documentation
- **Easy Integration**: Simple authentication and data formats

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd x-forecast

# Install dependencies
pip install fastapi uvicorn pandas numpy plotly scikit-learn

# Start the complete system
python start_company_sales_system.py
```

### 2. Access the System

- **Cyberpunk Dashboard**: http://localhost:3001
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 3. First Steps

1. **Register Your Company**
   - Open the dashboard
   - Click "Register New Company"
   - Fill in company details and requirements

2. **Upload Sales Data**
   - Go to "Upload Data" tab
   - Download the CSV template
   - Upload your monthly sales data

3. **Generate Forecasts**
   - Navigate to "Forecasting" tab
   - Set forecast horizon (1-12 months)
   - Click "Generate Forecast"

4. **Monitor Performance**
   - Check "Analytics" tab for model performance
   - View weight evolution over time
   - Review business recommendations

## 📋 Data Requirements

### Required Columns
- **date**: Date of sales (YYYY-MM-DD format)
- **sales_amount**: Total sales amount (numeric, positive)
- **product_category**: Product category or SKU
- **region**: Sales region or location

### Optional Columns (Improve Accuracy)
- **units_sold**: Number of units sold
- **customer_count**: Number of customers
- **avg_order_value**: Average order value
- **marketing_spend**: Marketing expenditure
- **promotions**: Promotional activities (0/1)
- **seasonality_factor**: Custom seasonality indicator
- **external_factors**: External factors (weather, events, etc.)

### Data Format Example

```csv
date,sales_amount,product_category,region,units_sold,customer_count
2024-01-01,15000,Electronics,North,150,75
2024-01-01,8000,Clothing,North,200,120
2024-01-02,12000,Electronics,South,120,60
```

## 🔧 API Usage

### Authentication
Use your company ID as a Bearer token:

```bash
curl -H "Authorization: Bearer your_company_id" \
     http://localhost:8000/api/company-sales/profile
```

### Key Endpoints

#### 1. Register Company
```bash
POST /api/company-sales/register
Content-Type: application/json
Authorization: Bearer your_company_id

{
  "company_name": "My Company",
  "industry": "Retail",
  "custom_requirements": {
    "min_months": 6,
    "max_file_size_mb": 100
  }
}
```

#### 2. Upload Data
```bash
POST /api/company-sales/upload-data
Authorization: Bearer your_company_id
Content-Type: multipart/form-data

# Upload CSV, Excel, or JSON file
```

#### 3. Generate Forecast
```bash
POST /api/company-sales/forecast
Content-Type: application/json
Authorization: Bearer your_company_id

{
  "horizon_months": 6
}
```

#### 4. Get Model Status
```bash
GET /api/company-sales/model-status
Authorization: Bearer your_company_id
```

## 🧠 How It Works

### 1. Pattern Detection
The system automatically analyzes your sales data to detect:
- **Seasonal Patterns**: Weekly, monthly, or yearly cycles
- **Trending Patterns**: Upward or downward trends
- **Intermittent Patterns**: Sparse demand with many zeros
- **Stationary Patterns**: Stable, mean-reverting data

### 2. Model Ensemble
Five forecasting models work together:
- **ARIMA**: Good for trending and stationary data
- **ETS**: Excellent for seasonal patterns
- **XGBoost**: Captures complex non-linear patterns
- **LSTM**: Handles long-term dependencies
- **Croston**: Specialized for intermittent demand

### 3. Adaptive Learning
- **Initial Weights**: Set based on detected pattern type
- **Performance Tracking**: Monitor MAE, MAPE, RMSE for each model
- **Weight Updates**: Automatically adjust weights based on performance
- **Pattern Changes**: Detect and adapt to changing data patterns

### 4. Business Intelligence
- **Confidence Intervals**: Quantify forecast uncertainty
- **Recommendations**: AI-generated business insights
- **Performance Alerts**: Notifications for significant changes
- **Trend Analysis**: Identify growth opportunities and risks

## 📊 Dashboard Features

### Overview Tab
- Key performance metrics
- Sales trend visualization
- Category and region breakdowns
- Growth rate calculations

### Upload Data Tab
- Data requirements and templates
- File upload with validation
- Data preview and quality checks
- Upload history tracking

### Forecasting Tab
- Interactive forecast generation
- Confidence interval visualization
- Model weight displays
- Business recommendations

### Analytics Tab
- Performance analysis over time
- Pattern detection results
- Model evolution tracking
- Seasonality analysis

### Settings Tab
- Company profile management
- Adaptive configuration options
- Data export capabilities
- Model performance history

## ⚙️ Configuration Options

### Adaptive Learning Settings
- **Learning Window**: How many months of data to use for learning
- **Update Frequency**: How often to update model weights
- **Minimum Weight**: Minimum weight any model can have
- **Weight Change Limit**: Maximum weight change per update

### Performance Monitoring
- **Tracking Enabled**: Enable/disable performance tracking
- **Alert Thresholds**: Set thresholds for performance alerts
- **History Limit**: How many performance records to keep

### Pattern Detection
- **Detection Enabled**: Enable/disable automatic pattern detection
- **Confidence Threshold**: Minimum confidence for pattern detection
- **Redetection Frequency**: How often to check for pattern changes

## 🔍 Monitoring & Troubleshooting

### Model Performance
Monitor these key metrics:
- **MAE (Mean Absolute Error)**: Average prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **R-squared**: Explained variance (higher is better)

### Common Issues

#### Poor Forecast Accuracy
- **Solution**: Upload more historical data (minimum 6 months recommended)
- **Check**: Data quality and consistency
- **Consider**: Adding optional columns for better context

#### Model Weights Not Updating
- **Solution**: Ensure you have enough data for performance evaluation
- **Check**: Adaptive learning is enabled in settings
- **Verify**: Regular data uploads for continuous learning

#### Pattern Detection Issues
- **Solution**: Ensure consistent data formatting and frequency
- **Check**: Minimum data requirements are met
- **Consider**: Manual pattern specification if automatic detection fails

## 🚀 Advanced Usage

### Batch Processing
```python
from src.company_sales.company_forecasting_engine import CompanyForecastingEngine
from src.company_sales.company_data_manager import CompanyDataManager

# Initialize
data_manager = CompanyDataManager()
engine = CompanyForecastingEngine(data_manager)

# Generate forecasts for multiple companies
companies = ['company_1', 'company_2', 'company_3']
for company_id in companies:
    forecast = engine.generate_forecast(company_id, horizon_months=6)
    print(f"Forecast for {company_id}: {forecast.point_forecast.sum()}")
```

### Custom Model Integration
```python
# Add custom model to ensemble
class CustomModel:
    def fit(self, data):
        # Your model training logic
        pass
    
    def forecast(self, horizon):
        # Your forecasting logic
        pass

# Register with engine
engine.available_models.append('custom_model')
```

### Webhook Integration
```python
# Set up webhooks for forecast updates
@app.post("/webhook/forecast-complete")
async def forecast_webhook(data: dict):
    company_id = data['company_id']
    forecast_result = data['forecast_result']
    
    # Your custom logic here
    await notify_stakeholders(company_id, forecast_result)
```

## 📈 Performance Optimization

### Data Upload Optimization
- **Batch Uploads**: Upload multiple months at once
- **Consistent Format**: Use the same column names and formats
- **Clean Data**: Remove duplicates and fix missing values
- **Regular Updates**: Upload new data monthly for best results

### Forecasting Performance
- **Appropriate Horizon**: Don't forecast too far into the future
- **Sufficient History**: Provide at least 12 months of historical data
- **Quality Data**: Include optional columns when available
- **Regular Retraining**: Let the system update weights automatically

## 🔒 Security & Privacy

### Data Isolation
- Each company's data is completely isolated
- No cross-company data sharing or leakage
- Secure file storage with company-specific directories

### Authentication
- Company-specific tokens for API access
- Session-based authentication for dashboard
- Configurable access controls

### Data Protection
- Local data storage (no external data sharing)
- Encrypted data transmission
- Audit trails for all data operations

## 🤝 Support & Contributing

### Getting Help
1. Check the API documentation at `/docs`
2. Review the dashboard help sections
3. Check logs in the `logs/` directory
4. Run the demo script for examples

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Reporting Issues
- Include system information and error logs
- Provide sample data (anonymized) if possible
- Describe expected vs actual behavior
- Include steps to reproduce

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of proven forecasting libraries
- Inspired by industry best practices
- Designed for real-world business needs
- Community-driven development

---

**Ready to transform your sales forecasting? Start with `python start_company_sales_system.py` and experience the future of adaptive ensemble forecasting!** 🚀