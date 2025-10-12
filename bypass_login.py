#!/usr/bin/env python3
"""
Bypass Login - Direct SuperX Access
"""

import requests
import json

def test_direct_access():
    """Test direct SuperX access without login"""
    
    print("Testing Direct SuperX Access (No Login Required)")
    print("=" * 50)
    
    # Test direct chat
    print("\n1. Testing Direct Chat...")
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/superx/chat",
            json={"message": "What products do you have?"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Chat Response: {result['response_text'][:100]}...")
            print(f"Company: {result.get('company_name', 'Unknown')}")
        else:
            print(f"[ERROR] Chat failed: {response.status_code}")
            
    except Exception as e:
        print(f"[ERROR] Chat request failed: {e}")
    
    # Test status
    print("\n2. Testing API Status...")
    try:
        response = requests.get("http://localhost:8000/api/v1/status", timeout=5)
        if response.status_code == 200:
            print("[OK] Backend is running")
        else:
            print(f"[ERROR] Status check failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Backend not responding: {e}")
        print("\nTo start backend:")
        print("py -m uvicorn src.api.main:app --reload --port 8000")

if __name__ == "__main__":
    test_direct_access()