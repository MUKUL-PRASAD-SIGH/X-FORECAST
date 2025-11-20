/**
 * Enhanced API Client Hook with Bulletproof Error Handling
 * Provides smart retry logic, circuit breaker pattern, and comprehensive error recovery
 */

import { useEffect, useCallback, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { 
  enhancedApiClient, 
  ApiResponse, 
  ErrorDetails, 
  ErrorCategory, 
  RecoveryAction,
  CircuitBreakerState 
} from '../utils/enhancedApiClient';

interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
  speed?: number;
  estimatedTimeRemaining?: number;
}

interface ServiceHealth {
  healthy: boolean;
  circuitBreakerState: CircuitBreakerState;
  lastError?: string;
  lastSuccessTime?: string;
}

interface UseEnhancedApiClientReturn {
  // Core API methods
  get: <T>(endpoint: string, params?: Record<string, any>) => Promise<ApiResponse<T>>;
  post: <T>(endpoint: string, data?: any) => Promise<ApiResponse<T>>;
  put: <T>(endpoint: string, data?: any) => Promise<ApiResponse<T>>;
  delete: <T>(endpoint: string) => Promise<ApiResponse<T>>;
  uploadFile: <T>(
    endpoint: string, 
    file: File, 
    additionalData?: Record<string, any>,
    onProgress?: (progress: UploadProgress) => void
  ) => Promise<ApiResponse<T>>;
  
  // Enhanced upload methods with bulletproof error handling
  uploadWithRetry: <T>(
    endpoint: string,
    file: File,
    options?: {
      maxRetries?: number;
      onProgress?: (progress: UploadProgress) => void;
      onRetry?: (attempt: number, error: ErrorDetails) => void;
      onFallback?: () => void;
    }
  ) => Promise<ApiResponse<T>>;
  
  // Health monitoring
  serviceHealth: Record<string, ServiceHealth>;
  checkHealth: () => Promise<void>;
  
  // Error recovery
  handleError: (error: ErrorDetails) => Promise<void>;
  retryLastOperation: () => Promise<ApiResponse<any> | null>;
  
  // Circuit breaker management
  circuitBreakerStatus: Record<string, any>;
  resetCircuitBreaker: (serviceName: string) => void;
  
  // Fallback mode
  fallbackMode: boolean;
  enableFallbackMode: () => void;
  disableFallbackMode: () => void;
}

export const useEnhancedApiClient = (): UseEnhancedApiClientReturn => {
  const { authToken, logout, refreshToken } = useAuth();
  const [serviceHealth, setServiceHealth] = useState<Record<string, ServiceHealth>>({});
  const [circuitBreakerStatus, setCircuitBreakerStatus] = useState<Record<string, any>>({});
  const [fallbackMode, setFallbackMode] = useState(false);
  const [lastOperation, setLastOperation] = useState<{
    method: string;
    endpoint: string;
    data?: any;
  } | null>(null);

  // Update API client with current auth token and error handler
  useEffect(() => {
    enhancedApiClient.setAuthToken(authToken);
    enhancedApiClient.setAuthErrorHandler(logout);
  }, [authToken, logout]);

  // Periodic health checks
  useEffect(() => {
    const healthCheckInterval = setInterval(async () => {
      await checkHealth();
    }, 30000); // Check every 30 seconds

    // Initial health check
    checkHealth();

    return () => clearInterval(healthCheckInterval);
  }, []);

  const checkHealth = useCallback(async () => {
    try {
      const health = await enhancedApiClient.checkServiceHealth();
      const cbStatus = enhancedApiClient.getCircuitBreakerStatus();
      const healthStatus = enhancedApiClient.getHealthStatus();

      // Update service health status
      const updatedHealth: Record<string, ServiceHealth> = {};
      Object.entries(healthStatus).forEach(([service, healthy]) => {
        updatedHealth[service] = {
          healthy: healthy as boolean,
          circuitBreakerState: cbStatus[service]?.state || CircuitBreakerState.CLOSED,
          lastSuccessTime: healthy ? new Date().toISOString() : undefined
        };
      });

      setServiceHealth(updatedHealth);
      setCircuitBreakerStatus(cbStatus);
    } catch (error) {
      console.warn('Health check failed:', error);
    }
  }, []);

  const handleError = useCallback(async (errorDetails: ErrorDetails): Promise<void> => {
    console.log('Handling error:', errorDetails);

    // Handle different recovery actions
    for (const action of errorDetails.recoveryActions) {
      switch (action) {
        case RecoveryAction.REFRESH_AUTH:
          try {
            await refreshToken();
            console.log('Auth token refreshed successfully');
          } catch (refreshError) {
            console.error('Token refresh failed:', refreshError);
            // Fall through to redirect login
            logout();
          }
          break;

        case RecoveryAction.REDIRECT_LOGIN:
          logout();
          break;

        case RecoveryAction.FALLBACK_MODE:
          setFallbackMode(true);
          console.log('Fallback mode enabled');
          break;

        case RecoveryAction.CIRCUIT_BREAKER:
          // Circuit breaker is handled automatically by the API client
          console.log('Circuit breaker activated');
          break;

        case RecoveryAction.USER_INTERVENTION:
          // This should be handled by the UI component
          console.log('User intervention required');
          break;
      }
    }
  }, [refreshToken, logout]);

  const retryLastOperation = useCallback(async (): Promise<ApiResponse<any> | null> => {
    if (!lastOperation) {
      return null;
    }

    const { method, endpoint, data } = lastOperation;

    try {
      switch (method.toLowerCase()) {
        case 'get':
          return await enhancedApiClient.get(endpoint, data);
        case 'post':
          return await enhancedApiClient.post(endpoint, data);
        case 'put':
          return await enhancedApiClient.put(endpoint, data);
        case 'delete':
          return await enhancedApiClient.delete(endpoint);
        default:
          throw new Error(`Unsupported method: ${method}`);
      }
    } catch (error) {
      console.error('Retry operation failed:', error);
      return {
        success: false,
        error: 'Retry operation failed'
      };
    }
  }, [lastOperation]);

  const resetCircuitBreaker = useCallback((serviceName: string) => {
    // This would need to be implemented in the API client
    console.log(`Resetting circuit breaker for ${serviceName}`);
  }, []);

  const enableFallbackMode = useCallback(() => {
    setFallbackMode(true);
    console.log('Fallback mode enabled manually');
  }, []);

  const disableFallbackMode = useCallback(() => {
    setFallbackMode(false);
    console.log('Fallback mode disabled');
  }, []);

  // Enhanced wrapper methods that track operations and handle errors
  const get = useCallback(async <T>(endpoint: string, params?: Record<string, any>): Promise<ApiResponse<T>> => {
    setLastOperation({ method: 'GET', endpoint, data: params });
    
    const response = await enhancedApiClient.get<T>(endpoint, params);
    
    if (!response.success && response.errorDetails) {
      await handleError(response.errorDetails);
    }
    
    return response;
  }, [handleError]);

  const post = useCallback(async <T>(endpoint: string, data?: any): Promise<ApiResponse<T>> => {
    setLastOperation({ method: 'POST', endpoint, data });
    
    const response = await enhancedApiClient.post<T>(endpoint, data);
    
    if (!response.success && response.errorDetails) {
      await handleError(response.errorDetails);
    }
    
    return response;
  }, [handleError]);

  const put = useCallback(async <T>(endpoint: string, data?: any): Promise<ApiResponse<T>> => {
    setLastOperation({ method: 'PUT', endpoint, data });
    
    const response = await enhancedApiClient.put<T>(endpoint, data);
    
    if (!response.success && response.errorDetails) {
      await handleError(response.errorDetails);
    }
    
    return response;
  }, [handleError]);

  const deleteMethod = useCallback(async <T>(endpoint: string): Promise<ApiResponse<T>> => {
    setLastOperation({ method: 'DELETE', endpoint });
    
    const response = await enhancedApiClient.delete<T>(endpoint);
    
    if (!response.success && response.errorDetails) {
      await handleError(response.errorDetails);
    }
    
    return response;
  }, [handleError]);

  const uploadFile = useCallback(async <T>(
    endpoint: string,
    file: File,
    additionalData?: Record<string, any>,
    onProgress?: (progress: UploadProgress) => void
  ): Promise<ApiResponse<T>> => {
    setLastOperation({ method: 'POST', endpoint, data: { file, additionalData } });

    // Enhanced progress tracking
    const enhancedProgressCallback = onProgress ? (percentage: number) => {
      const progress: UploadProgress = {
        loaded: (percentage / 100) * file.size,
        total: file.size,
        percentage
      };
      onProgress(progress);
    } : undefined;

    const response = await enhancedApiClient.uploadFile<T>(
      endpoint, 
      file, 
      additionalData, 
      enhancedProgressCallback
    );
    
    if (!response.success && response.errorDetails) {
      await handleError(response.errorDetails);
    }
    
    return response;
  }, [handleError]);

  const uploadWithRetry = useCallback(async <T>(
    endpoint: string,
    file: File,
    options?: {
      maxRetries?: number;
      onProgress?: (progress: UploadProgress) => void;
      onRetry?: (attempt: number, error: ErrorDetails) => void;
      onFallback?: () => void;
    }
  ): Promise<ApiResponse<T>> => {
    const maxRetries = options?.maxRetries || 3;
    let attempt = 0;
    let lastError: ErrorDetails | null = null;

    while (attempt < maxRetries) {
      try {
        const response = await uploadFile<T>(endpoint, file, undefined, options?.onProgress);
        
        if (response.success) {
          return response;
        }

        // Handle upload failure
        if (response.errorDetails) {
          lastError = response.errorDetails;
          
          // Check if error is retryable
          if (!response.errorDetails.retryable) {
            break;
          }

          attempt++;
          
          if (options?.onRetry) {
            options.onRetry(attempt, response.errorDetails);
          }

          // Wait before retry with exponential backoff
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          break;
        }
      } catch (error) {
        attempt++;
        console.error(`Upload attempt ${attempt} failed:`, error);
        
        if (attempt >= maxRetries) {
          break;
        }
        
        // Wait before retry
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    // All retries exhausted, try fallback mode
    if (options?.onFallback) {
      options.onFallback();
      setFallbackMode(true);
    }

    return {
      success: false,
      error: lastError?.category === ErrorCategory.NETWORK 
        ? 'Upload failed due to network issues. Please check your connection and try again.'
        : 'Upload failed after multiple attempts. Please try again later.',
      errorDetails: lastError || undefined
    };
  }, [uploadFile]);

  return {
    // Core API methods
    get,
    post,
    put,
    delete: deleteMethod,
    uploadFile,
    
    // Enhanced upload methods
    uploadWithRetry,
    
    // Health monitoring
    serviceHealth,
    checkHealth,
    
    // Error recovery
    handleError,
    retryLastOperation,
    
    // Circuit breaker management
    circuitBreakerStatus,
    resetCircuitBreaker,
    
    // Fallback mode
    fallbackMode,
    enableFallbackMode,
    disableFallbackMode
  };
};