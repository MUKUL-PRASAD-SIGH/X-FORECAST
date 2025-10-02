# X-FORECAST: AI-Powered Demand Forecasting Engine 🚀

## ⚡ Quick Start
```bash
# Install dependencies
pip install pandas numpy statsmodels scikit-learn pydantic python-dotenv

# Install package
pip install -e .

# Run demo
python main.py

# Launch Dashboard
streamlit run src/frontend/dashboard.py
```

X-FORECAST is an advanced demand forecasting platform that combines classical statistical methods, machine learning, and deep learning approaches to deliver accurate, scalable, and actionable forecasts.

## 📊 Project Overview

X-FORECAST revolutionizes demand planning by providing:
- **Multi-source data integration** with automated validation
- **Ensemble forecasting** combining ARIMA, ETS, and Croston methods
- **Model evaluation framework** with MAE, RMSE, and MAPE metrics
- **Automated feature engineering** (Phase 2)
- **Interactive visualization dashboard** (Phase 2)
- **Special handling for NPIs and promotions** ✅
- **Inventory optimization insights** ✅

## 🗂 Repository Structure

```
X_FORECAST/
├── .github/              # CI/CD workflows (future)
├── data/
│   ├── raw/              # Raw CSV data from source systems
│   └── processed/        # Cleaned data + forecast outputs
├── docs/                 # Documentation (future)
├── notebooks/            # Jupyter notebooks for EDA (future)
├── src/
│   ├── data_fabric/      # ✅ Data ingestion & validation
│   │   ├── __init__.py
│   │   └── connector.py  # DataConnector class
│   ├── feature_store/    # Feature engineering (Phase 2)
│   ├── models/           # ✅ Forecasting model zoo
│   │   ├── __init__.py
│   │   ├── classical/    # ✅ Statistical models
│   │   │   ├── arima.py  # ARIMA implementation
│   │   │   └── ets.py    # ETS implementation
│   │   ├── intermittent/ # ✅ Sparse demand models
│   │   │   └── croston.py # Croston method
│   │   ├── ml_deep/      # ML/DL models (Phase 2)
│   │   └── ensemble.py   # ✅ Model combination
│   ├── api/              # FastAPI backend (Phase 2)
│   └── frontend/         # Web dashboard (Phase 2)
├── tests/                # ✅ Unit tests
│   └── test_models.py    # Model testing suite
├── main.py               # ✅ Main application entry point
├── requirements.txt      # ✅ Python dependencies
└── .env.example          # ✅ Environment configuration
```

## 🔧 Technical Architecture

### Lightweight Local Setup (Prototype/Hackathon Ready)

**Data Storage Strategy:**
- **Local Files**: Raw + processed data stored in `data/` folders
- **No Cloud Dependencies**: Zero AWS/GCS costs, faster prototyping
- **SQLite Database**: Optional lightweight SQL queries (no PostgreSQL setup needed)
- **CSV/Parquet Features**: Direct file storage for feature engineering

### Core Components (Phase 1 ✅)

**DataConnector** (`src/data_fabric/connector.py`)
- Local CSV data loading with validation
- Automatic data cleaning and type conversion
- Missing value detection and reporting
- SQLite integration for SQL-style queries (optional)

**Forecasting Models**
- **ARIMAForecaster**: Auto-regressive integrated moving average
- **ETSForecaster**: Exponential smoothing with trend/seasonality
- **CrostonForecaster**: Intermittent demand specialized model
- **EnsembleForecaster**: Weighted combination of all models

**Model Evaluation**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE) 
- Mean Absolute Percentage Error (MAPE)

### Dependencies
```
pandas>=1.5.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
statsmodels>=0.14.0    # Statistical models
scikit-learn>=1.3.0    # ML utilities
pydantic>=2.0.0        # Data validation
python-dotenv>=1.0.0   # Environment management
# Optional: sqlite3 (built-in Python)
```

## 🛣 Development Roadmap

### 🚀 Phase 1: Foundations (Days 1-15)

#### 1.1 Data Pipeline Setup
**Objective:** Establish robust data ingestion and processing pipeline
- **Tasks:**
  - Create data connectors for source systems
  - Implement data validation rules
  - Set up automated cleaning pipeline
- **Deliverables:** 
  - Local CSV data ingestion
  - Clean historical dataset in `data/processed/`
- **Storage:** 
  - Local file system (no cloud costs)
  - Optional SQLite for SQL queries
- **Status:** ✅ Complete

#### 1.2 Baseline Forecasting
**Objective:** Implement foundation forecasting models
- **Tasks:**
  - Develop ARIMA pipeline
  - Implement ETS models
  - Add Croston for intermittent demand
- **Deliverables:**
  - Baseline P50 forecasts
  - Model evaluation framework
- **External Resources:**
  - statsmodels
  - scikit-learn
- **Status:** ✅ Complete

#### ✅ Implemented Features:
- **DataConnector**: CSV ingestion with validation
- **ARIMA Model**: Classical time series forecasting
- **ETS Model**: Exponential smoothing with seasonality
- **Croston Model**: Intermittent demand forecasting
- **Ensemble Pipeline**: Weighted model combination
- **Evaluation Framework**: MAE, RMSE, MAPE metrics
- **CSV Export**: Forecast results output
- **Unit Tests**: Model validation suite

### 🟡 Phase 2: Intelligence & Visualization (Days 16-45)

#### 2.1 Feature Engineering
**Objective:** Build comprehensive feature store
- **Tasks:**
  - Calculate rolling statistics
  - Extract seasonality features
  - Implement promotional flags
  - Add stockout detection
- **Deliverables:**
  - Feature store with 15+ features
  - Feature importance analysis
- **Storage:**
  - Local CSV/Parquet files in `data/processed/`
  - Simple pandas-based feature pipeline
- **Status:** ✅ Complete

#### 2.2 Advanced Models
**Objective:** Implement ML/DL models
- **Tasks:**
  - XGBoost implementation
  - LSTM model development
  - Temporal Fusion Transformer setup
- **Deliverables:**
  - Ensemble model pipeline
  - Probabilistic forecasts (P10/P50/P90)
- **External Resources:**
  - XGBoost
  - PyTorch
  - LightGBM
- **Status:** ✅ Complete

#### 2.3 Visualization
**Objective:** Create interactive dashboard
- **Tasks:**
  - Build forecast visualization
  - Add exception management
  - Implement FVA tracking
- **Deliverables:**
  - Web dashboard
  - Exception workflow
  - FVA reports
- **External Resources:**
  - React
  - Plotly
  - Streamlit
- **Status:** ✅ Complete

#### ✅ Implemented Features:
- **Feature Engineering**: Rolling stats, seasonality, lag features
- **XGBoost Model**: ML forecasting with feature importance
- **LSTM Model**: Deep learning time series prediction
- **Interactive Dashboard**: Streamlit UI with comprehensive features

### 🟠 Phase 3: Advanced Features & Business Logic (Days 46-75)

#### 3.1 NPI & Promotions
**Objective:** Handle special business cases
- **Tasks:**
  - Build NPI similarity engine
  - Implement Bayesian pooling
  - Create promotion uplift models
- **Deliverables:**
  - NPI forecast module
  - Promotion impact calculator
- **External Resources:**
  - SciPy
  - Prophet
- **Status:** ✅ Complete

#### 3.2 Inventory Optimization
**Objective:** Add inventory planning capabilities
- **Tasks:**
  - Develop ROP calculator
  - Implement safety stock optimization
  - Create scenario simulator
- **Deliverables:**
  - ROP/SS API endpoint
  - Scenario analysis dashboard
- **External Resources:**
  - SimPy
  - OR-Tools
- **Status:** ✅ Complete

#### ✅ Implemented Features:
- **NPI Engine**: Product similarity matching for new launches
- **Promotion Optimizer**: Uplift modeling and ROI calculation
- **Inventory Optimizer**: Safety stock and EOQ optimization
- **Advanced Analytics**: Performance monitoring and insights

### 🔵 Phase 4: Scaling, Governance & Handover (Days 76-90)

#### 4.1 Platform Scaling
**Objective:** Optimize for production scale
- **Tasks:**
  - Performance optimization
  - Batch processing implementation
  - Cache strategy setup
- **Deliverables:**
  - Scaled production system
  - Performance benchmarks
- **External Resources:**
  - Redis
  - Kubernetes
- **Status:** ⛔ Not Started

#### 4.2 Governance & Documentation
**Objective:** Ensure maintainable handover
- **Tasks:**
  - Create technical documentation
  - Write user guides
  - Record training videos
- **Deliverables:**
  - Complete documentation
  - Training materials
  - Governance framework
- **External Resources:**
  - Confluence
  - GitHub Wiki
- **Status:** ⛔ Not Started

#### ⛔ Future Enhancements:
- Performance optimization
- Kubernetes deployment
- Comprehensive documentation
- User training materials

## 🎯 Current Project Status

### ✅ What's Working Now:
- **Data Pipeline**: CSV ingestion, validation, cleaning
- **Forecasting Models**: ARIMA, ETS, Croston implementations
- **Ensemble Method**: Weighted model combination
- **Evaluation Suite**: Comprehensive accuracy metrics
- **Export Functionality**: CSV output generation
- **Testing Framework**: Unit tests for all models

### 🎯 Current Capabilities:
1. **Load CSV data** with automatic validation
2. **Generate forecasts** using 3 different methods
3. **Evaluate accuracy** with multiple metrics
4. **Export results** to CSV format
5. **Run tests** to validate model performance

### ✅ Phase 3 Complete:
1. **NPI Forecasting**: New product launch predictions
2. **Promotion Intelligence**: Campaign optimization and ROI
3. **Inventory Optimization**: Safety stock and replenishment
4. **Advanced Analytics**: Performance monitoring dashboard
5. **Pro Dashboard**: Hot, sexy UI with comprehensive features

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone repository
git clone <repository-url>
cd X_FORECAST

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Set up environment
cp .env.example .env
```

### Alternative Installation (Recommended)
```bash
# Install dependencies directly
pip install pandas numpy statsmodels scikit-learn pydantic python-dotenv

# Then install package
pip install -e .
```

### Quick Start
```bash
# Run demo with sample data
python main.py

# Run tests
python -m pytest tests/ -v

# Or run specific test
python tests/test_models.py
```

### Usage Example
```python
from src.data_fabric.connector import DataConnector
from src.models.ensemble import EnsembleForecaster

# Load and validate data
connector = DataConnector('./data/raw')
data = connector.load_csv('your_data.csv')
validation = connector.validate_data(data)
clean_data = connector.clean_data(data)

# Generate forecasts
forecaster = EnsembleForecaster()
forecaster.fit(clean_data['demand'])
forecast = forecaster.forecast(steps=12)

# Export results
forecaster.export_results(forecast, './data/processed/output.csv')
```

### Data Format
Expected CSV structure:
```csv
date,demand,product_id
2023-01-01,100,SKU001
2023-01-02,95,SKU001
2023-01-03,110,SKU001
```

## 📊 Model Performance

| Model | Use Case | Strengths | Limitations |
|-------|----------|-----------|-------------|
| ARIMA | Trending data | Handles trends/seasonality | Requires stationary data |
| ETS | Seasonal patterns | Automatic seasonality detection | Limited to exponential patterns |
| Croston | Intermittent demand | Specialized for sparse data | Not suitable for regular demand |
| Ensemble | General forecasting | Combines model strengths | May smooth out extremes |

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific model
python -c "from tests.test_models import TestModels; TestModels().test_arima_forecast()"
```

## 📈 Roadmap Progress

- [x] **Phase 1**: Data pipeline + baseline models (Days 1-15)
- [ ] **Phase 2**: Feature engineering + ML models (Days 16-45)
- [ ] **Phase 3**: NPI handling + inventory optimization (Days 46-75)
- [ ] **Phase 4**: Scaling + governance (Days 76-90)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new forecasting model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create Pull Request

## 📜 License

MIT License - see LICENSE file for details

# Or run specific test
python tests/test_models.py
```

### Usage Example
```python
from src.data_fabric.connector import DataConnector
from src.models.ensemble import EnsembleForecaster

# Load and validate data
connector = DataConnector('./data/raw')
data = connector.load_csv('your_data.csv')
validation = connector.validate_data(data)
clean_data = connector.clean_data(data)

# Generate forecasts
forecaster = EnsembleForecaster()
forecaster.fit(clean_data['demand'])
forecast = forecaster.forecast(steps=12)

# Export results
forecaster.export_results(forecast, './data/processed/output.csv')
```

### Data Format
Expected CSV structure:
```csv
date,demand,product_id
2023-01-01,100,SKU001
2023-01-02,95,SKU001
2023-01-03,110,SKU001
```

## 📊 Model Performance

| Model | Use Case | Strengths | Limitations |
|-------|----------|-----------|-------------|
| ARIMA | Trending data | Handles trends/seasonality | Requires stationary data |
| ETS | Seasonal patterns | Automatic seasonality detection | Limited to exponential patterns |
| Croston | Intermittent demand | Specialized for sparse data | Not suitable for regular demand |
| Ensemble | General forecasting | Combines model strengths | May smooth out extremes |

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific model
python -c "from tests.test_models import TestModels; TestModels().test_arima_forecast()"
```

## 📈 Roadmap Progress

- [x] **Phase 1**: Data pipeline + baseline models (Days 1-15)
- [ ] **Phase 2**: Feature engineering + ML models (Days 16-45)
- [ ] **Phase 3**: NPI handling + inventory optimization (Days 46-75)
- [ ] **Phase 4**: Scaling + governance (Days 76-90)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new forecasting model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create Pull Request

## 📜 License

MIT License - see LICENSE file for details