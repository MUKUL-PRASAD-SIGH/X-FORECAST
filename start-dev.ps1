# X-FORECAST System Startup Script for Windows
# Complete automated startup with API server and dashboard

Write-Host "🚀 X-FORECAST Cyberpunk AI Dashboard System Startup" -ForegroundColor Magenta
Write-Host "=" * 60 -ForegroundColor Magenta
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "1. Check dependencies" -ForegroundColor Yellow
Write-Host "2. Set up directories" -ForegroundColor Yellow
Write-Host "3. Install packages" -ForegroundColor Yellow
Write-Host "4. Start API server" -ForegroundColor Yellow
Write-Host "5. Start React cyberpunk dashboard" -ForegroundColor Yellow
Write-Host "6. Open browser" -ForegroundColor Yellow
Write-Host ""

# Check dependencies
Write-Host "🔍 Checking dependencies..." -ForegroundColor Cyan

# Check Python
try {
    $pythonVersion = py --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Check required Python packages
Write-Host "🔍 Checking Python packages..." -ForegroundColor Cyan
$requiredPackages = @("fastapi", "uvicorn", "pandas", "numpy", "plotly", "scikit-learn")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        py -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $package" -ForegroundColor Green
        } else {
            Write-Host "❌ $package" -ForegroundColor Red
            $missingPackages += $package
        }
    } catch {
        Write-Host "❌ $package" -ForegroundColor Red
        $missingPackages += $package
    }
}

# Setup directories
Write-Host ""
Write-Host "📁 Setting up directories..." -ForegroundColor Cyan

$directories = @("company_data", "logs", "config", "data/users", "data/raw", "data/processed")

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "✅ $dir" -ForegroundColor Green
}

# Install Python dependencies
Write-Host ""
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install Node.js dependencies
Write-Host ""
Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Cyan
Set-Location frontend
npm install --legacy-peer-deps
Set-Location ..

Write-Host ""
Write-Host "✅ Dependencies installed!" -ForegroundColor Green

# Ask about demo
Write-Host ""
$runDemo = Read-Host "🎬 Run demo first? (y/n)"

if ($runDemo -eq "y" -or $runDemo -eq "yes") {
    Write-Host ""
    Write-Host "🎬 Running demo..." -ForegroundColor Cyan
    try {
        py demo_company_sales.py
        Write-Host "✅ Demo completed successfully" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Demo had issues, but continuing with startup..." -ForegroundColor Yellow
    }
}

# Start API server
Write-Host ""
Write-Host "🚀 Starting API server..." -ForegroundColor Cyan

$apiJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    py -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
}

Write-Host "✅ API server starting on http://localhost:8000" -ForegroundColor Green
Write-Host "   API docs available at http://localhost:8000/docs" -ForegroundColor Gray

# Wait for API to start
Start-Sleep -Seconds 3

# Start dashboard
Write-Host ""
Write-Host "📊 Starting cyberpunk React dashboard..." -ForegroundColor Cyan

if (Test-Path "frontend") {
    $dashboardJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD/frontend
        npm start
    }
    
    Write-Host "✅ Cyberpunk dashboard starting on http://localhost:3001" -ForegroundColor Green
    
    # Wait for dashboard to start
    Start-Sleep -Seconds 5
} else {
    Write-Host "❌ Frontend directory not found. Please ensure the React app is set up." -ForegroundColor Red
}

# Open browser
Write-Host ""
Write-Host "🌐 Opening browser..." -ForegroundColor Cyan

try {
    Start-Process "http://localhost:3001"
    Write-Host "✅ Browser opened to cyberpunk dashboard" -ForegroundColor Green
    
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:8000/docs"
    Write-Host "✅ Browser opened to API docs" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not open browser automatically" -ForegroundColor Yellow
    Write-Host "   Manually navigate to:" -ForegroundColor Gray
    Write-Host "   - Dashboard: http://localhost:3001" -ForegroundColor Gray
    Write-Host "   - API docs: http://localhost:8000/docs" -ForegroundColor Gray
}

# Show success message
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "🎉 System Started Successfully!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "📊 Cyberpunk Dashboard: http://localhost:3001" -ForegroundColor Cyan
Write-Host "🔗 API Server: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "🏢 Company Sales Features:" -ForegroundColor Yellow
Write-Host "• Register companies with custom data requirements" -ForegroundColor Gray
Write-Host "• Upload monthly sales data (CSV, Excel, JSON)" -ForegroundColor Gray
Write-Host "• Automatic pattern detection and model initialization" -ForegroundColor Gray
Write-Host "• Adaptive ensemble forecasting with weight updates" -ForegroundColor Gray
Write-Host "• Confidence intervals and business insights" -ForegroundColor Gray
Write-Host "• Real-time model performance monitoring" -ForegroundColor Gray
Write-Host "• Interactive dashboard for data visualization" -ForegroundColor Gray
Write-Host ""
Write-Host "📋 Quick Start:" -ForegroundColor Yellow
Write-Host "1. Open the cyberpunk dashboard at http://localhost:3001" -ForegroundColor Gray
Write-Host "2. Register a new company or login to existing" -ForegroundColor Gray
Write-Host "3. Upload your monthly sales data" -ForegroundColor Gray
Write-Host "4. Generate forecasts and view insights" -ForegroundColor Gray
Write-Host "5. Monitor model performance over time" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 API Usage:" -ForegroundColor Yellow
Write-Host "• Use company ID as Bearer token for authentication" -ForegroundColor Gray
Write-Host "• Upload data: POST /api/company-sales/upload-data" -ForegroundColor Gray
Write-Host "• Generate forecast: POST /api/company-sales/forecast" -ForegroundColor Gray
Write-Host "• Get model status: GET /api/company-sales/model-status" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Red

# Keep services running
try {
    Write-Host ""
    Write-Host "🔄 Services running... Press Ctrl+C to stop" -ForegroundColor Yellow
    
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if jobs are still running
        if ($apiJob.State -eq "Completed" -or $apiJob.State -eq "Failed") {
            Write-Host ""
            Write-Host "⚠️  API server stopped" -ForegroundColor Yellow
            break
        }
        
        if ($dashboardJob -and ($dashboardJob.State -eq "Completed" -or $dashboardJob.State -eq "Failed")) {
            Write-Host ""
            Write-Host "⚠️  Dashboard stopped" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host ""
    Write-Host ""
    Write-Host "🛑 Shutting down..." -ForegroundColor Red
    
    # Stop jobs
    if ($apiJob) {
        Stop-Job $apiJob
        Remove-Job $apiJob
        Write-Host "✅ API server stopped" -ForegroundColor Green
    }
    
    if ($dashboardJob) {
        Stop-Job $dashboardJob
        Remove-Job $dashboardJob
        Write-Host "✅ Dashboard stopped" -ForegroundColor Green
    }
    
    Write-Host "👋 Goodbye!" -ForegroundColor Cyan
}