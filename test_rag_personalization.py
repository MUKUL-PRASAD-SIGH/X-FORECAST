#!/usr/bin/env python3
"""
Test RAG Personalization - Verify different users get different responses
"""

import urllib.request
import urllib.parse
import json

BASE_URL = "http://localhost:8000/api/v1/auth"

def login_user(email, password):
    """Login and get token"""
    data = {"email": email, "password": password}
    req_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/login",
        data=req_data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get('token'), result.get('user', {}).get('company_name')
    except Exception as e:
        print(f"[ERROR] Login failed for {email}: {e}")
        return None, None

def test_chat(token, company, message):
    """Test chat with token"""
    data = {"message": message}
    req_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=req_data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get('response_text', 'No response')
    except Exception as e:
        return f"[ERROR] Chat failed: {e}"

def main():
    print("Testing RAG Personalization")
    print("=" * 50)
    
    # Test users
    users = [
        ("admin@superx.com", "admin123", "SuperX Corporation"),
        ("john@techcorp.com", "john123", "TechCorp Industries"), 
        ("sarah@healthplus.com", "sarah123", "HealthPlus Medical")
    ]
    
    # Test questions
    questions = [
        "What products do you recommend?",
        "Show me sales forecast",
        "What's my top performing category?"
    ]
    
    for email, password, expected_company in users:
        print(f"\n--- Testing {expected_company} ---")
        
        # Login
        token, company = login_user(email, password)
        if not token:
            print(f"[FAILED] Could not login {email}")
            continue
            
        print(f"[OK] Logged in as {company}")
        
        # Test each question
        for question in questions:
            print(f"\nQ: {question}")
            response = test_chat(token, company, question)
            print(f"A: {response[:100]}...")
            
            # Check if response mentions company name
            if company.lower() in response.lower():
                print(f"[PERSONALIZED] Response mentions {company}")
            else:
                print(f"[GENERIC] Response doesn't mention {company}")
    
    print("\n" + "=" * 50)
    print("Personalization Test Complete!")
    print("\nTo verify personalization:")
    print("1. Each user should get responses mentioning their company")
    print("2. Responses should be different for different users")
    print("3. Upload different CSV data for each user to see more personalization")

if __name__ == "__main__":
    main()