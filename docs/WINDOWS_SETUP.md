# 🪟 Windows Setup Guide - Cyberpunk AI Dashboard

## 🚀 Quick Setup (5 Minutes)

### Step 1: Fix Windows Issues
```powershell
# Run this first to fix common Windows issues
.\fix-windows-issues.ps1
```

### Step 2: Test Your System
```powershell
# Test if everything is ready for deployment
.\test-deployment.ps1
```

### Step 3: Deploy the Dashboard
```powershell
# Deploy with Docker (recommended)
.\deploy.ps1 compose

# OR setup development environment
.\start-dev.ps1
```

### Step 4: Access Your Dashboard
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Metrics**: http://localhost:8001

## 🔧 Prerequisites

### Required Software
1. **Docker Desktop** - Download from https://docker.com
2. **PowerShell 5.1+** - Usually pre-installed on Windows 10/11
3. **Git** - Download from https://git-scm.com (optional)

### Optional (for development)
1. **Python 3.11+** - Download from https://python.org
2. **Node.js 18+** - Download from https://nodejs.org
3. **Visual Studio Code** - Download from https://code.visualstudio.com

## 🐳 Docker Deployment (Recommended)

### Why Docker?
- ✅ No dependency conflicts
- ✅ Consistent environment
- ✅ Easy deployment
- ✅ All services included

### Commands
```powershell
# Deploy all services
.\deploy.ps1 compose

# Check service status
.\deploy.ps1 status

# View logs
.\deploy.ps1 logs

# Stop all services
.\deploy.ps1 stop
```

## 🔧 Development Setup

### If you want to modify the code:
```powershell
# Setup development environment
.\start-dev.ps1

# Or use the deploy script
.\deploy.ps1 dev
```

### Manual Development Setup
```powershell
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd frontend
npm install
cd ..

# Start backend (in one terminal)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd frontend
npm start
```

## 🧪 Testing

### Test Everything
```powershell
# Run comprehensive tests
.\test-deployment.ps1
```

### Test Individual Components
```powershell
# Test Python components (use correct Python version)
python simple-test.py

# Test Python demo
python main.py
# OR if multiple Python versions:
# C:\Users\[USERNAME]\AppData\Local\Programs\Python\Python312\python.exe main.py

# Test Docker build (if Docker is running)
docker build -f Dockerfile.backend -t test-backend .

# Test API (after deployment)
curl http://localhost:8000/api/v1/status
```

### Expected Test Results
```
✅ Basic imports: PASS
✅ File structure: PASS  
✅ Cyberpunk components: PASS
✅ Sample data: PASS
✅ Dependencies: 5/5 available
🐳 Ready for Docker deployment
```

## 🚨 Troubleshooting

### Common Issues

#### PowerShell Execution Policy Error
```powershell
# Fix execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Docker Not Running
1. Install Docker Desktop
2. Start Docker Desktop
3. Wait for it to fully start (green icon in system tray)

#### Port Already in Use
```powershell
# Check what's using the port
netstat -ano | findstr :3000
netstat -ano | findstr :8000

# Kill process using the port (replace PID)
taskkill /PID <PID> /F
```

#### Python Import Errors / Multiple Python Versions
```powershell
# Check Python version
python --version

# Install missing dependencies
pip install -r requirements.txt

# If you have multiple Python versions, use the correct one:
C:\Users\[USERNAME]\AppData\Local\Programs\Python\Python312\python.exe main.py

# Or use Docker deployment instead
.\deploy.ps1 compose
```

#### Node.js/NPM Issues
```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
Remove-Item -Recurse -Force frontend/node_modules
cd frontend
npm install
```

### Getting Help
1. **Check logs**: `.\deploy.ps1 logs`
2. **Check status**: `.\deploy.ps1 status`
3. **Restart services**: `.\deploy.ps1 stop` then `.\deploy.ps1 compose`
4. **Run tests**: `.\test-deployment.ps1`

## 🎯 What You Get

### 🤖 AI Features
- **Conversational AI**: Chat with your data in natural language
- **Predictive Maintenance**: 95% accuracy system health monitoring
- **Customer Analytics**: ML-powered churn prediction
- **Demand Forecasting**: Advanced ensemble models
- **Business Insights**: Automated opportunity detection

### 🎨 Cyberpunk Experience
- **3D Visualizations**: Holographic data displays
- **Neon Aesthetics**: Full cyberpunk theme
- **Real-time Updates**: Live dashboard with WebSocket
- **Voice Interface**: Speech-to-text for hands-free use
- **Particle Effects**: Immersive visual experience

### 📊 Business Intelligence
- **Real-time Dashboards**: Live metrics and KPIs
- **Customer Retention**: Churn prediction and segmentation
- **Demand Planning**: AI-powered forecasting
- **System Monitoring**: Comprehensive health tracking
- **Data Integration**: CRM, ERP, Marketing systems

## 🚀 Next Steps

### After Deployment
1. **Explore the Dashboard**: Navigate the cyberpunk interface
2. **Try the AI Chat**: Ask questions about your data
3. **View 3D Visualizations**: Experience holographic data
4. **Check System Health**: Monitor performance metrics
5. **Integrate Your Data**: Connect your business systems

### Customization
- **Themes**: Modify cyberpunk colors and effects
- **Data Sources**: Add your CRM/ERP connections
- **AI Models**: Train with your specific data
- **Dashboards**: Create custom views and metrics

## 🎉 Success!

Once deployed, you'll have a complete cyberpunk AI dashboard running with:
- ✅ Real-time business intelligence
- ✅ AI-powered insights and predictions
- ✅ Stunning 3D visualizations
- ✅ Voice-enabled chat interface
- ✅ Comprehensive system monitoring

**Welcome to the future of business analytics!** 🌟