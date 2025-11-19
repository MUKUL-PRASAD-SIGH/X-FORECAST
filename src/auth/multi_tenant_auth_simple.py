"""
Simple test version of multi-tenant auth
"""

class MultiTenantAuthManager:
    def __init__(self):
        self.test = "working"
    
    def get_test(self):
        return self.test

# Global instance
multi_tenant_auth = MultiTenantAuthManager()