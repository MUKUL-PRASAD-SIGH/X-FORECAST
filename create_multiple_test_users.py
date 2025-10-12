#!/usr/bin/env python3
"""
Create Multiple Test Users for RAG Personalization Testing
"""

import os

def create_multiple_users():
    """Create multiple test users with different data"""
    
    auth_file = "src/api/auth_endpoints_dev.py"
    
    if not os.path.exists(auth_file):
        print("[ERROR] Auth file not found")
        return
    
    with open(auth_file, 'r') as f:
        content = f.read()
    
    # Test users with different profiles
    test_users = '''
# Multiple test users for personalization testing
dev_users["admin@superx.com"] = {
    "user_id": "user_1",
    "email": "admin@superx.com", 
    "password": "admin123",
    "company_name": "SuperX Corporation",
    "business_type": "retail",
    "industry": "retail",
    "created_at": "2024-01-01T00:00:00"
}

dev_users["john@techcorp.com"] = {
    "user_id": "user_2",
    "email": "john@techcorp.com",
    "password": "john123", 
    "company_name": "TechCorp Industries",
    "business_type": "manufacturing",
    "industry": "technology",
    "created_at": "2024-01-02T00:00:00"
}

dev_users["sarah@healthplus.com"] = {
    "user_id": "user_3",
    "email": "sarah@healthplus.com",
    "password": "sarah123",
    "company_name": "HealthPlus Medical",
    "business_type": "healthcare", 
    "industry": "healthcare",
    "created_at": "2024-01-03T00:00:00"
}
'''
    
    # Replace the existing dev_users section
    if "dev_users = {}" in content:
        updated_content = content.replace("dev_users = {}", f"dev_users = {{}}{test_users}")
    else:
        # Find and replace existing users
        import re
        pattern = r'dev_users\["admin@superx\.com"\].*?}'
        if re.search(pattern, content, re.DOTALL):
            # Replace existing with new users
            updated_content = re.sub(
                r'dev_users\[.*?\n}', 
                test_users.strip(),
                content, 
                flags=re.DOTALL
            )
        else:
            updated_content = content + test_users
    
    with open(auth_file, 'w') as f:
        f.write(updated_content)
    
    print("[OK] Multiple test users created!")
    print("\n=== TEST USERS FOR PERSONALIZATION ===")
    print("1. RETAIL COMPANY:")
    print("   Email: admin@superx.com")
    print("   Password: admin123")
    print("   Company: SuperX Corporation (Retail)")
    print()
    print("2. TECH COMPANY:")
    print("   Email: john@techcorp.com") 
    print("   Password: john123")
    print("   Company: TechCorp Industries (Manufacturing)")
    print()
    print("3. HEALTHCARE COMPANY:")
    print("   Email: sarah@healthplus.com")
    print("   Password: sarah123") 
    print("   Company: HealthPlus Medical (Healthcare)")
    print("\n=== TESTING PERSONALIZATION ===")
    print("1. Login with different users")
    print("2. Upload different CSV data for each")
    print("3. Ask same questions to RAG bot")
    print("4. Verify responses are personalized per user")

if __name__ == "__main__":
    create_multiple_users()