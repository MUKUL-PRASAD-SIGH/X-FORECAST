#!/usr/bin/env python3
"""
Simple test script for Cyberpunk AI Dashboard
Tests basic functionality without heavy dependencies
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import json
        import datetime
        import time
        print("✅ Standard library imports work")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")
        return False
    
    return True

def test_optional_imports():
    """Test optional dependencies"""
    print("\n🧪 Testing optional dependencies...")
    
    dependencies = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'fastapi': 'Web API framework',
        'pydantic': 'Data validation',
        'sklearn': 'Machine learning'
    }
    
    # Special handling for sklearn (scikit-learn)
    sklearn_available = False
    try:
        import sklearn
        sklearn_available = True
    except ImportError:
        pass
    
    available = {}
    for dep, description in dependencies.items():
        try:
            if dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep}: {description}")
            available[dep] = True
        except ImportError:
            print(f"⚠️  {dep}: {description} (not installed)")
            available[dep] = False
    
    return available

def test_file_structure():
    """Test project file structure"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        'requirements.txt',
        'src',
        'frontend',
        'main.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_cyberpunk_components():
    """Test cyberpunk AI components"""
    print("\n🧪 Testing Cyberpunk AI components...")
    
    components = {
        'src/api/main.py': 'FastAPI backend',
        'src/ai_chatbot/conversational_ai.py': 'AI Chatbot',
        'src/customer_analytics/retention_analyzer.py': 'Customer Analytics',
        'src/predictive_maintenance/maintenance_engine.py': 'Predictive Maintenance',
        'frontend/src/components/MainDashboard.tsx': 'Main Dashboard',
        'frontend/src/theme/cyberpunkTheme.ts': 'Cyberpunk Theme'
    }
    
    all_exist = True
    for file_path, description in components.items():
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} (missing)")
            all_exist = False
    
    return all_exist

def create_sample_data():
    """Create sample data for testing"""
    print("\n🧪 Creating sample data...")
    
    try:
        # Create data directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Create simple CSV data without pandas
        sample_data = """date,demand,product_id
2023-01-01,100,SKU001
2023-01-02,95,SKU001
2023-01-03,110,SKU001
2023-01-04,105,SKU001
2023-01-05,120,SKU001"""
        
        with open('data/raw/sample_data.csv', 'w') as f:
            f.write(sample_data)
        
        print("✅ Sample data created in data/raw/sample_data.csv")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Cyberpunk AI Dashboard - Simple Test Suite")
    print("=" * 50)
    
    # Run tests
    basic_ok = test_basic_imports()
    deps_available = test_optional_imports()
    files_ok = test_file_structure()
    components_ok = test_cyberpunk_components()
    data_ok = create_sample_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Basic imports: {'PASS' if basic_ok else 'FAIL'}")
    print(f"✅ File structure: {'PASS' if files_ok else 'FAIL'}")
    print(f"✅ Cyberpunk components: {'PASS' if components_ok else 'FAIL'}")
    print(f"✅ Sample data: {'PASS' if data_ok else 'FAIL'}")
    
    # Dependency summary
    available_count = sum(deps_available.values())
    total_count = len(deps_available)
    print(f"✅ Dependencies: {available_count}/{total_count} available")
    
    # Recommendations
    print("\n🎯 Recommendations:")
    if not all(deps_available.values()):
        print("📦 Install missing dependencies: pip install -r requirements.txt")
    
    if basic_ok and files_ok and components_ok:
        print("🚀 Ready to run: py main.py")
        print("🔧 Start backend: py -m uvicorn src.api.main:app --reload --port 8000")
        print("🎨 Start frontend: cd frontend && npm start")
    
    print("\n🎉 Test completed!")
    
    # Return overall success
    return basic_ok and files_ok and components_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)