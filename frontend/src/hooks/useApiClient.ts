import { useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { apiClient, ApiResponse } from '../utils/apiClient';

export const useApiClient = () => {
  const { authToken, logout } = useAuth();

  // Update API client with current auth token and error handler
  useEffect(() => {
    apiClient.setAuthToken(authToken);
    apiClient.setAuthErrorHandler(logout);
  }, [authToken, logout]);

  return {
    apiClient,
    // Convenience methods that automatically handle auth
    get: <T>(endpoint: string, params?: Record<string, any>) => 
      apiClient.get<T>(endpoint, params),
    
    post: <T>(endpoint: string, data?: any) => 
      apiClient.post<T>(endpoint, data),
    
    put: <T>(endpoint: string, data?: any) => 
      apiClient.put<T>(endpoint, data),
    
    delete: <T>(endpoint: string) => 
      apiClient.delete<T>(endpoint),
    
    uploadFile: <T>(endpoint: string, file: File, additionalData?: Record<string, any>) => 
      apiClient.uploadFile<T>(endpoint, file, additionalData),
  };
};