#!/usr/bin/env python3
"""
Test Login - Simple HTTP requests to test authentication
"""

import urllib.request
import urllib.parse
import json

BASE_URL = "http://localhost:8000/api/v1/auth"

def test_register():
    """Test user registration"""
    data = {
        "email": "test@superx.com",
        "password": "test123",
        "company_name": "SuperX Test",
        "business_type": "retail",
        "industry": "retail"
    }
    
    req_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/register",
        data=req_data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"[OK] Registration: {result}")
            return True
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return False

def test_login():
    """Test user login"""
    data = {
        "email": "test@superx.com",
        "password": "test123"
    }
    
    req_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/login",
        data=req_data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"[OK] Login: {result}")
            return result.get('token')
    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        return None

def main():
    print("Testing Authentication System...")
    print("=" * 40)
    
    # Test registration
    print("\n1. Testing Registration...")
    test_register()
    
    # Test login
    print("\n2. Testing Login...")
    token = test_login()
    
    if token:
        print(f"\n[SUCCESS] Got token: {token[:20]}...")
    else:
        print("\n[FAILED] No token received")

if __name__ == "__main__":
    main()