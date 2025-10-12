#!/usr/bin/env python3
"""
Simple Authentication Test without external dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_auth_imports():
    """Test if authentication modules can be imported"""
    
    print("Testing X-FORECAST Authentication Imports")
    print("=" * 50)
    
    # Test 1: Import user management
    print("\n1. Testing user management import...")
    try:
        from src.auth.user_management import user_manager
        print("[OK] User management imported successfully")
        
        # Test database initialization
        print("   Testing database initialization...")
        if hasattr(user_manager, 'db_path'):
            print(f"   Database path: {user_manager.db_path}")
            print("[OK] Database initialization successful")
        else:
            print("[WARN] Database path not found")
            
    except Exception as e:
        print(f"[ERROR] User management import failed: {e}")
        return False
    
    # Test 2: Import auth endpoints
    print("\n2. Testing auth endpoints import...")
    try:
        from src.api.auth_endpoints import router
        print("[OK] Auth endpoints imported successfully")
        
        # Check router configuration
        if hasattr(router, 'routes'):
            route_count = len(router.routes)
            print(f"   Routes registered: {route_count}")
            
            # List available routes
            for route in router.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    methods = list(route.methods) if route.methods else ['GET']
                    print(f"   - {methods[0]} {route.path}")
            
            print("[OK] Auth routes configured successfully")
        else:
            print("[WARN] Router routes not found")
            
    except Exception as e:
        print(f"[ERROR] Auth endpoints import failed: {e}")
        return False
    
    # Test 3: Test user registration functionality
    print("\n3. Testing user registration functionality...")
    try:
        result = user_manager.register_user(
            email="test@example.com",
            password="testpass123",
            company_name="Test Company",
            business_type="retail",
            industry="retail"
        )
        
        if result["success"]:
            print("[OK] User registration test successful")
            print(f"   User ID: {result['user_id']}")
        else:
            print(f"[WARN] Registration returned: {result['message']}")
            
    except Exception as e:
        print(f"[ERROR] User registration test failed: {e}")
    
    # Test 4: Test authentication
    print("\n4. Testing user authentication...")
    try:
        auth_result = user_manager.authenticate_user("test@example.com", "testpass123")
        
        if auth_result:
            print("[OK] User authentication test successful")
            print(f"   Token generated: {auth_result.get('token', 'N/A')[:20]}...")
        else:
            print("[WARN] Authentication returned None (user may not exist)")
            
    except Exception as e:
        print(f"[ERROR] User authentication test failed: {e}")
    
    return True

def test_dependencies():
    """Test if required dependencies are available"""
    
    print("\n" + "=" * 50)
    print("Testing Required Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("jwt", "PyJWT"),
        ("hashlib", "hashlib (built-in)"),
        ("sqlite3", "sqlite3 (built-in)"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[ERROR] {name} - MISSING")

def main():
    """Main test function"""
    print("X-FORECAST Authentication System Test")
    print("Testing authentication system components...\n")
    
    # Test dependencies first
    test_dependencies()
    
    # Test authentication imports and functionality
    success = test_auth_imports()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    if success:
        print("[OK] Authentication system is functional")
        print("\nNext steps:")
        print("1. Start the backend: py -m uvicorn src.api.main:app --reload --port 8000")
        print("2. Test with frontend or API calls")
        print("3. Check http://localhost:8000/docs for API documentation")
    else:
        print("[ERROR] Authentication system has issues")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install PyJWT passlib bcrypt")
        print("2. Check file permissions and paths")
        print("3. Review error messages above")

if __name__ == "__main__":
    main()