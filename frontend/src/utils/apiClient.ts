interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

interface ApiClientConfig {
  baseURL?: string;
  timeout?: number;
}

class ApiClient {
  private baseURL: string;
  private timeout: number;
  private authToken: string | null = null;
  private onAuthError?: () => void;

  constructor(config: ApiClientConfig = {}) {
    this.baseURL = config.baseURL || 'http://localhost:8000';
    this.timeout = config.timeout || 10000;
  }

  setAuthToken(token: string | null) {
    this.authToken = token;
  }

  setAuthErrorHandler(handler: () => void) {
    this.onAuthError = handler;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    // Set up headers
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    // Add authorization header if token is available
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    // Set up request options
    const requestOptions: RequestInit = {
      ...options,
      headers,
    };

    try {
      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...requestOptions,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle authentication errors
      if (response.status === 401) {
        if (this.onAuthError) {
          this.onAuthError();
        }
        return {
          success: false,
          error: 'Authentication failed. Please log in again.',
        };
      }

      // Parse response
      let data: any;
      const contentType = response.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      if (response.ok) {
        return {
          success: true,
          data,
        };
      } else {
        return {
          success: false,
          error: data.message || data.detail || `HTTP ${response.status}: ${response.statusText}`,
          data,
        };
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        return {
          success: false,
          error: 'Request timeout. Please try again.',
        };
      }

      return {
        success: false,
        error: error.message || 'Network error occurred',
      };
    }
  }

  // HTTP Methods
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<ApiResponse<T>> {
    let url = endpoint;
    
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      
      if (searchParams.toString()) {
        url += `?${searchParams.toString()}`;
      }
    }

    return this.makeRequest<T>(url, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    const options: RequestInit = { method: 'POST' };
    
    if (data) {
      if (data instanceof FormData) {
        // Don't set Content-Type for FormData, let browser set it with boundary
        options.body = data;
        options.headers = {};
      } else {
        options.body = JSON.stringify(data);
      }
    }

    return this.makeRequest<T>(endpoint, options);
  }

  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.makeRequest<T>(endpoint, { method: 'DELETE' });
  }

  // File upload helper
  async uploadFile<T>(endpoint: string, file: File, additionalData?: Record<string, any>): Promise<ApiResponse<T>> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, String(value));
      });
    }

    return this.post<T>(endpoint, formData);
  }
}

// Create singleton instance
export const apiClient = new ApiClient();

// Export types for use in components
export type { ApiResponse };