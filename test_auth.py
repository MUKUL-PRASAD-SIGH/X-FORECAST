#!/usr/bin/env python3
"""
Test Authentication System
"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"
AUTH_URL = f"{BASE_URL}/api/v1/auth"

def test_auth_system():
    """Test the authentication system"""
    
    print("🔐 Testing X-FORECAST Authentication System")
    print("=" * 50)
    
    # Test 1: Check if auth endpoints are available
    print("\n1. Testing auth endpoint availability...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Backend server is running")
            data = response.json()
            if "auth" in str(data.get("endpoints", {})):
                print("✅ Auth endpoints are registered")
            else:
                print("⚠️  Auth endpoints may not be fully registered")
        else:
            print("❌ Backend server not responding")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 2: Test user registration
    print("\n2. Testing user registration...")
    test_user = {
        "email": "test@superx.com",
        "password": "testpass123",
        "company_name": "SuperX Test Corp",
        "business_type": "retail",
        "industry": "retail"
    }
    
    try:
        response = requests.post(f"{AUTH_URL}/register", json=test_user)
        print(f"Registration response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ User registration successful")
            result = response.json()
            print(f"   User ID: {result.get('user_id', 'N/A')}")
        elif response.status_code == 400:
            print("⚠️  User may already exist or validation error")
            print(f"   Response: {response.text}")
        elif response.status_code == 503:
            print("⚠️  Authentication service temporarily unavailable")
            print("   This is expected if dependencies are missing")
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Registration error: {e}")
    
    # Test 3: Test user login
    print("\n3. Testing user login...")
    login_data = {
        "email": test_user["email"],
        "password": test_user["password"]
    }
    
    try:
        response = requests.post(f"{AUTH_URL}/login", json=login_data)
        print(f"Login response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ User login successful")
            result = response.json()
            token = result.get("token")
            if token:
                print(f"   Token received: {token[:20]}...")
                return token
            else:
                print("⚠️  No token in response")
        elif response.status_code == 401:
            print("⚠️  Invalid credentials or user doesn't exist")
        elif response.status_code == 503:
            print("⚠️  Authentication service temporarily unavailable")
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Login error: {e}")
    
    return None

def test_protected_endpoint(token):
    """Test accessing protected endpoints with token"""
    if not token:
        print("\n4. Skipping protected endpoint test (no token)")
        return
    
    print("\n4. Testing protected endpoint access...")
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{AUTH_URL}/profile", headers=headers)
        print(f"Profile access status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Protected endpoint access successful")
            profile = response.json()
            print(f"   Company: {profile.get('company_name', 'N/A')}")
        elif response.status_code == 401:
            print("❌ Token validation failed")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Protected endpoint error: {e}")

def main():
    """Main test function"""
    print("🚀 X-FORECAST Authentication Test Suite")
    print("Testing authentication system functionality...\n")
    
    # Run authentication tests
    token = test_auth_system()
    
    # Test protected endpoints if we have a token
    test_protected_endpoint(token)
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("- If you see ✅ marks, authentication is working")
    print("- If you see ⚠️  marks, there may be configuration issues")
    print("- If you see ❌ marks, there are errors to fix")
    print("\n💡 Next steps:")
    print("1. If auth is working: You can use the frontend login")
    print("2. If auth has issues: Check backend logs for details")
    print("3. For development: Use the demo mode without auth")

if __name__ == "__main__":
    main()