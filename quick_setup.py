#!/usr/bin/env python3
"""
Quick Setup Script for X-FORECAST
Installs dependencies and creates test user
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run command and show result"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {description} completed")
            return True
        else:
            print(f"[ERROR] {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] {description} failed: {e}")
        return False

def main():
    print("X-FORECAST Quick Setup")
    print("=" * 40)
    
    # Install core dependencies
    core_deps = "fastapi uvicorn pydantic pandas numpy scikit-learn PyJWT passlib bcrypt"
    
    if run_command(f"py -m pip install {core_deps}", "Installing core dependencies"):
        print("[OK] Core dependencies installed")
    else:
        print("[ERROR] Failed to install dependencies")
        return
    
    # Create test user
    if os.path.exists("create_test_user.py"):
        if run_command("py create_test_user.py", "Creating test user"):
            print("[OK] Test user created")
    
    # Setup complete
    print("\n" + "=" * 40)
    print("Setup Complete!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Start backend: py -m uvicorn src.api.main:app --reload --port 8000")
    print("2. Start frontend: cd frontend && npm start")
    print("3. Login with: admin@superx.com / admin123")
    print("4. Visit: http://localhost:3000")

if __name__ == "__main__":
    main()