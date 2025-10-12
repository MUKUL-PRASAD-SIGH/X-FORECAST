#!/usr/bin/env python3
"""
Development Environment Setup Script for Cyberpunk AI Dashboard
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_python_environment():
    """Set up Python environment and dependencies"""
    print("🐍 Setting up Python environment...")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return False
    
    return True

def setup_frontend_environment():
    """Set up React frontend environment"""
    print("⚛️ Setting up React frontend environment...")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Change to frontend directory and install dependencies
    original_dir = os.getcwd()
    try:
        os.chdir(frontend_path)
        if not run_command("npm install", "Installing Node.js dependencies"):
            return False
    finally:
        os.chdir(original_dir)
    
    return True

def setup_data_directories():
    """Create necessary data directories"""
    print("📁 Setting up data directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "data/cache",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_environment_files():
    """Set up environment configuration files"""
    print("🔧 Setting up environment files...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Cyberpunk AI Dashboard Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/cyberpunk_ai
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# AI/ML Configuration
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")
    
    return True

def check_system_requirements():
    """Check if system requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Check if Node.js is available
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        print(f"✅ Node.js {result.stdout.strip()} detected")
    except FileNotFoundError:
        print("❌ Node.js is required but not found")
        return False
    
    # Check if npm is available
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        print(f"✅ npm {result.stdout.strip()} detected")
    except FileNotFoundError:
        print("❌ npm is required but not found")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Cyberpunk AI Dashboard - Development Environment Setup")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please install required software.")
        sys.exit(1)
    
    # Setup steps
    setup_steps = [
        setup_data_directories,
        setup_environment_files,
        setup_python_environment,
        setup_frontend_environment,
    ]
    
    for step in setup_steps:
        if not step():
            print(f"❌ Setup failed at step: {step.__name__}")
            sys.exit(1)
    
    print("\n🎉 Development environment setup completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Update .env file with your API keys")
    print("2. Start the development servers:")
    print("   - Backend: py -m uvicorn src.api.main:app --reload --port 8000")
    print("   - Frontend: cd frontend && npm start")
    print("\n💫 Ready to build the cyberpunk AI dashboard!")

if __name__ == "__main__":
    main()