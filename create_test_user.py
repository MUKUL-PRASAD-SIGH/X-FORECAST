#!/usr/bin/env python3
"""
Create Test User for Login
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_user():
    """Create a test user in the system"""
    
    try:
        from src.auth.user_management import user_manager
        
        # Create test user
        result = user_manager.register_user(
            email="admin@superx.com",
            password="admin123",
            company_name="SuperX Corporation",
            business_type="retail",
            industry="retail"
        )
        
        if result["success"]:
            print("[OK] Test user created successfully!")
            print(f"   Email: admin@superx.com")
            print(f"   Password: admin123")
            print(f"   User ID: {result['user_id']}")
        else:
            print(f"[WARN] {result['message']}")
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("\nCreating development user in auth endpoints...")
        
        # Add to development auth system
        auth_file = "src/api/auth_endpoints_dev.py"
        if os.path.exists(auth_file):
            with open(auth_file, 'r') as f:
                content = f.read()
            
            # Add test user to dev_users
            test_user_code = '''
# Pre-created test user
dev_users["admin@superx.com"] = {
    "user_id": "test_user_1",
    "email": "admin@superx.com", 
    "password": "admin123",
    "company_name": "SuperX Corporation",
    "business_type": "retail",
    "industry": "retail",
    "created_at": "2024-01-01T00:00:00"
}
'''
            
            # Insert after dev_users = {}
            updated_content = content.replace(
                "dev_users = {}",
                f"dev_users = {{}}{test_user_code}"
            )
            
            with open(auth_file, 'w') as f:
                f.write(updated_content)
            
            print("[OK] Test user added to development system!")
            print("   Email: admin@superx.com")
            print("   Password: admin123")

if __name__ == "__main__":
    create_test_user()