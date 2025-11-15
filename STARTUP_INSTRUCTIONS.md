# 🚀 X-FORECAST System Startup Instructions

## Quick Start Guide

Follow these steps to get the complete X-FORECAST Cyberpunk AI Dashboard system running.

---

## 📋 Prerequisites

### Install Required Software:

**🐍 Python 3.10+**
```bash
# Windows
winget install Python.Python.3.12

# macOS
brew install python@3.12

# Ubuntu/Debian
sudo apt update && sudo apt install python3.12 python3-pip

# Verify installation
python --version
pip --version
```

**📦 Node.js 18+**
```bash
# Windows
winget install OpenJS.NodeJS

# macOS
brew install node

# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

**🔧 Git**
```bash
# Windows
winget install Git.Git

# macOS
brew install git

# Ubuntu/Debian
sudo apt install git

# Verify installation
git --version
```

---

## 🎯 Option 1: Automated Startup (Recommended)

### Step 1: Run the Automated Startup Script
```bash
python start_company_sales_system.py
```

This script will:
- ✅ Check all dependencies
- ✅ Set up required directories
- ✅ Start the API server on port 8000
- ✅ Start the React dashboard on port 3001
- ✅ Open your browser automatically

### Step 2: Follow the Prompts
- When asked "Run demo first? (y/n):", type `n` and press Enter
- The script will handle the rest automatically

---

## 🛠️ Option 2: Manual Startup

### Step 1: Install Dependencies

**Python Dependencies:**
```bash
pip install -r requirements.txt
pip install python-multipart
```

**Frontend Dependencies:**
```bash
cd frontend
npm install --legacy-peer-deps
cd ..
```

### Step 2: Start Backend API Server

**In Terminal 1:**
```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Start Frontend Dashboard

**In Terminal 2:**
```bash
cd frontend
npm start
```

---

## 🌐 Access Your Services

Once both services are running:

### 🎨 **Cyberpunk AI Dashboard**
- **URL**: http://localhost:3001
- **Features**: 
  - Interactive forecasting dashboard
  - Real-time data visualization
  - Cyberpunk-themed UI with neon effects
  - Performance monitoring
  - Scenario planning tools

### 🔗 **API Server**
- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Features**:
  - RESTful API endpoints
  - Company sales forecasting
  - Data upload and processing
  - Model management

---

## 🎮 Using the System

### 1. **Company Registration**
- Open the dashboard at http://localhost:3001
- Register a new company or login to existing
- Set up your data requirements

### 2. **Data Upload**
- Upload monthly sales data (CSV, Excel, JSON)
- System automatically detects patterns
- Initializes forecasting models

### 3. **Generate Forecasts**
- Use the interactive dashboard controls
- Adjust forecast horizon (1-12 months)
- Toggle between 2D and 3D visualizations
- View confidence intervals and model weights

### 4. **Monitor Performance**
- Real-time model performance tracking
- Business insights and recommendations
- Scenario planning and comparison tools

---

## 🔧 API Usage Examples

### Authentication
Use company ID as Bearer token:
```bash
curl -H "Authorization: Bearer your-company-id" http://localhost:8000/api/company-sales/forecast
```

### Key Endpoints
- **Upload Data**: `POST /api/company-sales/upload-data`
- **Generate Forecast**: `POST /api/company-sales/forecast`
- **Model Status**: `GET /api/company-sales/model-status`
- **Business Insights**: `GET /api/company-sales/insights`

---

## 🚨 Troubleshooting

### Common Issues

**1. "python-multipart" Error**
```bash
pip install python-multipart
```

**2. Frontend TypeScript Errors**
```bash
cd frontend
npm install --legacy-peer-deps
```

**3. Port Already in Use**
- Backend: Change port in startup command: `--port 8001`
- Frontend: Set `PORT=3002` environment variable

**4. Python Version Issues**
Ensure you're using Python 3.10+:
```bash
python --version
```

### Service Status Check

**Check if services are running:**
```bash
# Backend API
curl http://localhost:8000/docs

# Frontend Dashboard  
curl http://localhost:3001
```

---

## 🎯 Development Features

### Performance Tests
Run the comprehensive test suite:
```bash
cd frontend
npm test -- --testPathPattern="performance"
```

### Code Quality
- TypeScript strict mode enabled
- ESLint and Prettier configured
- Comprehensive error handling
- Real-time WebSocket connections

---

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   AI Models     │
│   React App     │◄──►│   FastAPI       │◄──►│   Ensemble      │
│   Port: 3001    │    │   Port: 8000    │    │   Forecasting   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components
- **Frontend**: React with TypeScript, Styled Components, Framer Motion
- **Backend**: FastAPI with async/await, WebSocket support
- **AI Engine**: Ensemble forecasting with ARIMA, ETS, XGBoost, LSTM, Croston
- **Database**: File-based storage with JSON/CSV support
- **Real-time**: WebSocket connections for live updates

---

## 🎉 Success Indicators

When everything is working correctly, you should see:

✅ **Frontend Console**: "Compiled successfully!"  
✅ **Backend Console**: "Uvicorn running on http://0.0.0.0:8000"  
✅ **Browser**: Cyberpunk dashboard loads at http://localhost:3001  
✅ **API Docs**: Swagger UI available at http://localhost:8000/docs  

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure ports 3001 and 8000 are available
4. Check the console logs for specific error messages

---

**🎮 Ready to forecast the future with cyberpunk style! 🚀**