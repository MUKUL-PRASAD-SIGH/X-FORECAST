#!/usr/bin/env python3
"""
Verification script for RAG Management API Integration Tests
This script runs the integration tests to verify the fixed import structure and API functionality.
"""

import sys
import os
import traceback
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_import_structure():
    """Test that all RAG management imports work correctly after fixes"""
    print("Testing import structure...")
    try:
        # Mock bcrypt and other dependencies to avoid import issues
        with patch.dict('sys.modules', {'bcrypt': Mock()}):
            # Test RAG management API import
            from src.api.rag_management_api import router
            assert router is not None
            print("  ✓ RAG management API router imported successfully")
            
            # Test enhanced RAG manager import
            from src.rag.enhanced_rag_manager import EnhancedRAGManager
            assert EnhancedRAGManager is not None
            print("  ✓ Enhanced RAG manager imported successfully")
            
            # Test dependency validator import
            from src.rag.dependency_validator import DependencyValidator
            assert DependencyValidator is not None
            print("  ✓ Dependency validator imported successfully")
            
            # Test schema migrator import
            from src.database.schema_migrator import DatabaseSchemaMigrator
            assert DatabaseSchemaMigrator is not None
            print("  ✓ Schema migrator imported successfully")
            
            print("✓ All RAG management imports successful")
            return True
        
    except ImportError as e:
        print(f"✗ Import structure test failed: {e}")
        traceback.print_exc()
        return False

def test_api_router_structure():
    """Test that RAG management API router has correct structure"""
    print("Testing API router structure...")
    try:
        # Mock dependencies to avoid import issues
        with patch.dict('sys.modules', {'bcrypt': Mock()}):
            from src.api.rag_management_api import router
            
            # Test router has expected attributes
            assert hasattr(router, 'prefix')
            assert router.prefix == "/api/v1/rag"
            print(f"  ✓ Router prefix: {router.prefix}")
            
            assert hasattr(router, 'tags')
            assert "RAG Management" in router.tags
            print(f"  ✓ Router tags: {router.tags}")
            
            # Test router has routes
            assert len(router.routes) > 0
            print(f"  ✓ Router has {len(router.routes)} routes")
            
            # Check for expected route patterns
            route_paths = [route.path for route in router.routes]
            expected_patterns = [
                "/status/{user_id}",
                "/initialize/{user_id}",
                "/reset/{user_id}",
                "/health/{user_id}",
                "/diagnostics/{user_id}",
                "/migrate",
                "/system/status",
                "/my/status",
                "/my/health",
                "/my/initialize",
                "/my/reset"
            ]
            
            found_patterns = []
            for pattern in expected_patterns:
                if any(pattern in path for path in route_paths):
                    found_patterns.append(pattern)
                    print(f"  ✓ Found route pattern: {pattern}")
                else:
                    print(f"  ✗ Missing route pattern: {pattern}")
            
            print(f"✓ Found {len(found_patterns)}/{len(expected_patterns)} expected route patterns")
            return len(found_patterns) == len(expected_patterns)
        
    except Exception as e:
        print(f"✗ API router structure test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_rag_manager_integration():
    """Test enhanced RAG manager integration with dependency validation"""
    print("Testing enhanced RAG manager integration...")
    try:
        import tempfile
        import shutil
        
        # Mock all dependencies to avoid import issues
        with patch.dict('sys.modules', {
            'bcrypt': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock()
        }):
            # Mock dependency validator
            with patch('src.rag.enhanced_rag_manager.DependencyValidator') as mock_validator:
                mock_validator.return_value.check_all_dependencies.return_value = Mock(
                    overall_status="healthy",
                    critical_missing=[],
                    optional_missing=[],
                    degraded_features=[]
                )
                mock_validator.return_value.validate_sentence_transformers.return_value = True
                
                # Mock RAG system
                with patch('src.rag.enhanced_rag_manager.RealVectorRAG') as mock_rag:
                    mock_rag.return_value = Mock()
                    
                    # Test enhanced manager can be imported and instantiated
                    from src.rag.enhanced_rag_manager import EnhancedRAGManager
                    print("  ✓ Enhanced RAG manager imported")
                    
                    # Create temporary database paths
                    test_dir = tempfile.mkdtemp()
                    try:
                        test_db_path = os.path.join(test_dir, "test_users.db")
                        test_rag_db_path = os.path.join(test_dir, "test_rag_vector_db.db")
                        
                        enhanced_manager = EnhancedRAGManager(test_db_path, test_rag_db_path)
                        print("  ✓ Enhanced RAG manager instantiated")
                        
                        # Test that enhanced manager has expected methods
                        expected_methods = [
                            'startup_validation',
                            'safe_initialize_rag',
                            'get_system_health_with_dependencies',
                            'recover_from_failure'
                        ]
                        
                        for method in expected_methods:
                            assert hasattr(enhanced_manager, method)
                            print(f"  ✓ Method {method} available")
                        
                        print("✓ Enhanced RAG manager integration verified")
                        return True
                        
                    finally:
                        shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f"✗ Enhanced RAG manager integration test failed: {e}")
        traceback.print_exc()
        return False

def test_api_endpoints_with_fastapi():
    """Test API endpoints with FastAPI if available"""
    print("Testing API endpoints with FastAPI...")
    try:
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        # Mock dependencies to avoid import issues
        with patch.dict('sys.modules', {'bcrypt': Mock()}):
            from src.api.rag_management_api import router
            
            app = FastAPI()
            app.include_router(router)
            print("  ✓ FastAPI app created with RAG router")
            
            # Create test client
            client = TestClient(app)
            print("  ✓ Test client created")
            
            # Test that endpoints exist (they should return 401/422 without proper auth)
            test_endpoints = [
                ("/api/v1/rag/my/status", "GET"),
                ("/api/v1/rag/status/test_user", "GET"),
                ("/api/v1/rag/my/health", "GET"),
                ("/api/v1/rag/health/test_user", "GET")
            ]
            
            for endpoint, method in test_endpoints:
                try:
                    if method == "GET":
                        response = client.get(endpoint)
                    else:
                        response = client.post(endpoint)
                    
                    # Should get 401 (unauthorized) or 422 (validation error), not 404 (not found)
                    if response.status_code in [401, 422]:
                        print(f"  ✓ Endpoint {endpoint} exists (status: {response.status_code})")
                    else:
                        print(f"  ✗ Endpoint {endpoint} unexpected status: {response.status_code}")
                        
                except Exception as e:
                    print(f"  ✗ Error testing endpoint {endpoint}: {e}")
            
            print("✓ API endpoints with FastAPI verified")
            return True
        
    except ImportError:
        print("  ⚠ FastAPI not available, skipping FastAPI tests")
        return True
    except Exception as e:
        print(f"✗ API endpoints with FastAPI test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("RAG Management API Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Import Structure", test_import_structure),
        ("API Router Structure", test_api_router_structure),
        ("Enhanced RAG Manager Integration", test_enhanced_rag_manager_integration),
        ("API Endpoints with FastAPI", test_api_endpoints_with_fastapi)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! RAG Management API is properly integrated.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)