# Development startup script for Windows

Write-Host "Starting Cyberpunk AI Dashboard Development Environment..." -ForegroundColor Magenta

# Check Python
try {
    $pythonVersion = py --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install Node.js dependencies
Write-Host "Installing Node.js dependencies..." -ForegroundColor Cyan
Set-Location frontend
npm install --legacy-peer-deps
Set-Location ..

Write-Host "Dependencies installed!" -ForegroundColor Green

# Create data directories
New-Item -ItemType Directory -Force -Path "data/raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "Created data directories" -ForegroundColor Green

Write-Host ""
Write-Host "Development environment ready!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the services:" -ForegroundColor Yellow
Write-Host "Backend:  py -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Cyan
Write-Host "Frontend: cd frontend" -ForegroundColor Cyan
Write-Host "          npm start" -ForegroundColor Cyan