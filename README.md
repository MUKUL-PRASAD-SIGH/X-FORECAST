# 🚀 SuperX AI Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> **Enterprise-Grade AI-Powered Demand Forecasting & Analytics Platform**

## 🌟 Overview

SuperX is a comprehensive AI-powered forecasting platform that combines advanced machine learning, hierarchical forecasting, and beautiful user interfaces to deliver enterprise-grade demand forecasting and analytics solutions.

## ✨ Key Features

### 🎯 **Advanced Forecasting Engine**
- **Hierarchical Forecasting** with MinT, OLS, WLS reconciliation methods
- **Enhanced Error Covariance Matrix** with shrinkage estimators
- **Multiple Reconciliation Methods** with automatic selection
- **Cross-Category Effects** analysis with ML-based detection
- **Long-Tail Optimization** for sparse item forecasting

### 📊 **Governance & Quality**
- **FVA (Forecast Value Added) Tracking** with user-level analysis
- **FQI (Forecast Quality Index)** with real-time monitoring
- **Automated Workflow Engine** with exception detection
- **OTIF Service Level Management** with root cause analysis

### 🔐 **Authentication & Data Management**
- **Secure Login System** with role-based access control
- **CSV Data Upload** with intelligent RAG (Retrieval-Augmented Generation)
- **Persistent Knowledge Base** that learns from uploaded data
- **Dynamic Suggestions** based on user's data

### 🎨 **Beautiful User Interface**
- **Enhanced Response Formatting** with ASCII art and emojis
- **Cyberpunk-themed Dashboard** with 3D visualizations
- **Interactive Chat Interface** with contextual suggestions
- **Real-time Analytics** with professional styling

## 🚀 Quick Start & Running Instructions

### Prerequisites
```bash
Python 3.8+ (Required)
Node.js 16+ (Required for frontend)
Git (Required)
```

### 🔧 Complete Setup & Installation

#### 1. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/superx-ai-forecasting.git
cd superx-ai-forecasting

# Run automated setup (Windows)
python setup_dev_environment.py

# Or manual setup:
pip install -r requirements.txt
```

#### 2. Frontend Setup
```bash
cd frontend
npm install --legacy-peer-deps
cd ..
```

### 🚀 Running the Complete System

#### Option 1: Full Stack Development (Recommended)

**Terminal 1 - Backend API:**
```bash
# Start FastAPI backend server
py -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Backend will be available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

**Terminal 2 - Frontend React App:**
```bash
cd frontend
npm start

# Frontend will be available at: http://localhost:3000
# Automatically opens in browser
```

**Terminal 3 - AI Chat System (Optional):**
```bash
# Interactive terminal-based system
python superx_final_system.py

# Or simple version
python superx_simple_complete.py
```

#### Option 2: Quick Demo/Testing
```bash
# All-in-one system with terminal interface
python superx_final_system.py

# Simple chatbot demo
python chatbot_demo.py

# Basic system test
python main.py
```

#### Option 3: Windows PowerShell Script
```powershell
# Run the automated startup script
.\start-dev.ps1
```

### 🌐 Access Points After Starting

| Component | URL | Description |
|-----------|-----|-------------|
| **Frontend Dashboard** | http://localhost:3000 | Main cyberpunk UI with 3D visualizations |
| **Backend API** | http://localhost:8000 | REST API endpoints |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs (Swagger) |
| **WebSocket** | ws://localhost:8000/ws | Real-time data streaming |
| **SuperX Direct Chat** | http://localhost:8000/api/v1/superx/chat | Direct AI chat endpoint |

### 🔧 Development Commands

```bash
# Backend Development
py -m uvicorn src.api.main:app --reload --port 8000  # Auto-reload on changes
python -m pytest tests/                              # Run tests
python -m pytest tests/ -v                          # Verbose testing

# Frontend Development  
cd frontend
npm start                    # Development server with hot reload
npm run build               # Production build
npm test                    # Run frontend tests

# Data & AI Components
python generate_dummy_data.py          # Generate sample datasets
python create_test_user.py            # Create test users
python test_rag_personalization.py    # Test AI personalization
```

### 🛠️ System Requirements & Troubleshooting

#### Minimum System Requirements
- **OS:** Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python:** 3.8 or higher (3.11 recommended)
- **Node.js:** 16.0 or higher (18.0 recommended)
- **RAM:** 8GB minimum (16GB recommended for AI features)
- **Storage:** 5GB free space
- **Network:** Internet connection for AI model downloads

#### Common Issues & Solutions

**🔴 Backend Won't Start:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Install missing dependencies
pip install -r requirements.txt

# Check port availability
netstat -an | findstr :8000  # Windows
lsof -i :8000                # macOS/Linux

# Alternative port
py -m uvicorn src.api.main:app --reload --port 8001
```

**🔴 Frontend Won't Start:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
cd frontend
rmdir /s node_modules  # Windows
rm -rf node_modules    # macOS/Linux
npm install --legacy-peer-deps

# Check Node.js version
node --version  # Should be 16+
```

**🔴 AI Features Not Working:**
```bash
# Install AI dependencies
pip install torch transformers sentence-transformers

# Check GPU availability (optional)
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU-only mode if needed
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**🔴 Database/Data Issues:**
```bash
# Reset user database
del users.db  # Windows
rm users.db   # macOS/Linux

# Clear data cache
rmdir /s data\processed  # Windows
rm -rf data/processed    # macOS/Linux

# Regenerate sample data
python generate_dummy_data.py
```

#### Performance Optimization
```bash
# For better performance, install optional dependencies:
pip install uvloop          # Faster event loop (Linux/macOS)
pip install orjson          # Faster JSON processing
pip install python-multipart # File upload optimization

# Enable production mode
export NODE_ENV=production   # Linux/macOS
set NODE_ENV=production      # Windows
```

## 👥 Demo Accounts

The system comes with pre-configured demo accounts:

| Username | Password | Role | Upload Limit | Access Level |
|----------|----------|------|--------------|--------------|
| `admin` | `admin123` | Admin | 1GB | Full system access |
| `manager` | `manager123` | Manager | 500MB | Management features |
| `analyst` | `analyst123` | Analyst | 200MB | Analytics features |

## 📁 Project Structure

```
superx-ai-forecasting/
├── 🎯 Core System
│   ├── src/ai_chatbot/           # Enhanced conversational AI
│   ├── src/auth/                 # Authentication system
│   ├── src/data_upload/          # Data upload & processing
│   ├── src/rag/                  # RAG knowledge base
│   └── src/dashboard/            # Dynamic dashboard
│
├── 🧠 Advanced Models
│   ├── src/models/hierarchical/  # Hierarchical forecasting
│   ├── src/models/governance/    # FVA, FQI, workflows
│   ├── src/models/advanced/      # OTIF, optimization
│   └── src/models/ml_deep/       # ML & deep learning
│
├── 🎨 Frontend (React)
│   ├── frontend/src/components/  # UI components
│   ├── frontend/src/theme/       # Cyberpunk theme
│   └── frontend/public/          # Static assets
│
├── 📊 Data & APIs
│   ├── src/api/                  # REST & GraphQL APIs
│   ├── src/data_fabric/          # Data pipeline
│   └── src/monitoring/           # System monitoring
│
└── 📚 Documentation
    ├── docs/                     # Technical docs
    ├── .kiro/specs/             # Feature specifications
    └── README.md                # This file
```

## 🎯 Usage Examples

### 1. Login & Data Upload
```python
from superx_final_system import SuperXSystem

# Initialize system
system = SuperXSystem()

# Login
system.login("admin", "admin123")

# Upload CSV data
system.upload_csv("sales_data.csv")

# Chat with AI
response = system.chat("What's the forecast for next quarter?")
```

### 2. Advanced Forecasting
```python
from src.models.hierarchical.hierarchical_forecaster import HierarchicalForecaster

forecaster = HierarchicalForecaster()
forecast = forecaster.forecast_hierarchical(data, horizon=12)
print(f"Coherence Score: {forecast.coherence_score}")
```

### 3. RAG-Enhanced Chat
```python
from src.rag.csv_knowledge_base import CSVKnowledgeBase

kb = CSVKnowledgeBase()
kb.add_csv("user_data.csv")
suggestions = kb.get_smart_suggestions("sales trends")
```

## 🔧 Configuration

### Environment Variables
```bash
# Create .env file
SUPERX_SECRET_KEY=your-secret-key
SUPERX_DB_URL=sqlite:///superx.db
SUPERX_UPLOAD_DIR=data/uploads
SUPERX_LOG_LEVEL=INFO
```

### System Settings
```python
# Modify src/config/settings.py
FORECAST_HORIZON = 12
MAX_UPLOAD_SIZE_MB = 100
ENABLE_REAL_TIME_PROCESSING = True
```

## 📊 Features Deep Dive

### Advanced Forecasting Engine
- **MinT Reconciliation**: Minimum trace reconciliation with enhanced covariance
- **Method Selection**: Automatic selection based on data characteristics
- **Cross-Category Effects**: ML-based relationship discovery
- **Long-Tail Optimization**: Specialized handling for sparse items

### Governance & Quality
- **FVA Tracking**: Measures forecast value added by human overrides
- **FQI Monitoring**: Real-time forecast quality assessment
- **Workflow Automation**: Exception detection and approval routing
- **OTIF Management**: On-time in-full service level optimization

### Data Intelligence
- **RAG System**: Learns from uploaded CSV files
- **Smart Suggestions**: Context-aware question recommendations
- **Persistent Learning**: Knowledge base survives system restarts
- **Multi-format Support**: CSV, Excel, JSON data ingestion

## 🎨 UI/UX Features

### Enhanced Formatting
- Beautiful ASCII art borders and frames
- Rich emoji integration for visual engagement
- Professional typography with consistent styling
- Structured information display

### Interactive Elements
- Real-time chat interface
- Dynamic dashboard updates
- Contextual help and suggestions
- Progress indicators and status updates

## 🔒 Security Features

- **JWT Authentication** with role-based access control
- **Secure File Upload** with validation and sanitization
- **Data Encryption** for sensitive information
- **Audit Logging** for all user actions
- **Rate Limiting** to prevent abuse

## 📈 Performance & Scalability

- **Async Processing** for non-blocking operations
- **Caching Layer** for frequently accessed data
- **Database Optimization** with proper indexing
- **Memory Management** for large datasets
- **Horizontal Scaling** support

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```

## 📚 Documentation

- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with advanced machine learning and AI technologies
- Inspired by enterprise forecasting best practices
- Designed for scalability and production use

## 📞 Support

- 📧 Email: support@superx-ai.com
- 💬 Discord: [SuperX Community](https://discord.gg/superx)
- 📖 Wiki: [GitHub Wiki](https://github.com/yourusername/superx-ai-forecasting/wiki)

---

<div align="center">

**🚀 Built with ❤️ for the future of AI-powered forecasting**

[⭐ Star this repo](https://github.com/yourusername/superx-ai-forecasting) | [🐛 Report Bug](https://github.com/yourusername/superx-ai-forecasting/issues) | [💡 Request Feature](https://github.com/yourusername/superx-ai-forecasting/issues)

</div>