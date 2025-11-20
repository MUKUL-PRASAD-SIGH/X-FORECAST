import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  user_id: string;
  email: string;
  company_name: string;
  company_id: string;
  business_type: string;
  rag_initialized: boolean;
}

interface AuthContextType {
  user: User | null;
  authToken: string | null;
  isAuthenticated: boolean;
  login: (token: string, userData: User) => void;
  logout: () => void;
  refreshToken: () => Promise<boolean>;
  isTokenExpired: () => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check for existing authentication on mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    
    if (token && userData) {
      try {
        const parsedUser = JSON.parse(userData);
        
        // Check if token is expired before setting authentication
        if (!isTokenExpiredCheck(token)) {
          setAuthToken(token);
          setUser(parsedUser);
          setIsAuthenticated(true);
        } else {
          // Token expired, clear storage
          handleLogout();
        }
      } catch (error) {
        console.error('Error parsing stored user data:', error);
        handleLogout();
      }
    }
  }, []);

  // Set up automatic token refresh
  useEffect(() => {
    if (isAuthenticated && authToken) {
      const refreshInterval = setInterval(() => {
        if (isTokenExpiredCheck(authToken)) {
          refreshToken().catch(() => {
            // If refresh fails, logout user
            handleLogout();
          });
        }
      }, 5 * 60 * 1000); // Check every 5 minutes

      return () => clearInterval(refreshInterval);
    }
  }, [isAuthenticated, authToken]);

  const isTokenExpiredCheck = (token: string): boolean => {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      return payload.exp < currentTime;
    } catch (error) {
      console.error('Error checking token expiration:', error);
      return true; // Assume expired if we can't parse
    }
  };

  const login = (token: string, userData: User) => {
    localStorage.setItem('auth_token', token);
    localStorage.setItem('user_data', JSON.stringify(userData));
    setAuthToken(token);
    setUser(userData);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    setAuthToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  const refreshToken = async (): Promise<boolean> => {
    try {
      if (!authToken) return false;

      const response = await fetch('http://localhost:8000/api/v1/auth/refresh', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        const newToken = data.token;
        
        // Update stored token
        localStorage.setItem('auth_token', newToken);
        setAuthToken(newToken);
        
        return true;
      } else {
        // Refresh failed, logout user
        handleLogout();
        return false;
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
      handleLogout();
      return false;
    }
  };

  const isTokenExpired = (): boolean => {
    if (!authToken) return true;
    return isTokenExpiredCheck(authToken);
  };

  const contextValue: AuthContextType = {
    user,
    authToken,
    isAuthenticated,
    login,
    logout: handleLogout,
    refreshToken,
    isTokenExpired,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};