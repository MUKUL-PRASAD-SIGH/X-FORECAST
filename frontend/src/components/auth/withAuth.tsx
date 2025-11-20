import React, { useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface WithAuthProps {
  onAuthError?: () => void;
}

export function withAuth<P extends WithAuthProps>(
  WrappedComponent: React.ComponentType<P>
) {
  const AuthenticatedComponent: React.FC<Omit<P, 'onAuthError'>> = (props) => {
    const { authToken, isAuthenticated, logout, isTokenExpired } = useAuth();

    useEffect(() => {
      // Check token expiration on component mount and periodically
      const checkTokenExpiration = () => {
        if (isAuthenticated && isTokenExpired()) {
          console.warn('Token expired, logging out user');
          logout();
        }
      };

      checkTokenExpiration();
      
      // Check every minute
      const interval = setInterval(checkTokenExpiration, 60000);
      
      return () => clearInterval(interval);
    }, [isAuthenticated, isTokenExpired, logout]);

    // Don't render if not authenticated or no token
    if (!isAuthenticated || !authToken) {
      return null;
    }

    // Pass the logout handler to the wrapped component
    return (
      <WrappedComponent
        {...(props as P)}
        onAuthError={logout}
      />
    );
  };

  AuthenticatedComponent.displayName = `withAuth(${WrappedComponent.displayName || WrappedComponent.name})`;

  return AuthenticatedComponent;
}