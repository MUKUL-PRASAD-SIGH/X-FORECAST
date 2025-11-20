/**
 * Enhanced API Client with Bulletproof Error Handling
 * Implements smart retry logic, circuit breaker pattern, and comprehensive error classification
 */

interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  errorDetails?: ErrorDetails;
}

interface ErrorDetails {
  category: string;
  severity: string;
  retryable: boolean;
  recoveryActions: string[];
  attemptCount: number;
  context: Record<string, any>;
  timestamp: string;
}

interface RetryConfig {
  maxAttempts: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  jitter: number;
  retryOnStatusCodes: number[];
}

interface CircuitBreakerConfig {
  failureThreshold: number;
  recoveryTimeout: number;
  name: string;
}

enum CircuitBreakerState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open'
}

interface CircuitBreakerStats {
  failureCount: number;
  successCount: number;
  lastFailureTime?: Date;
  state: CircuitBreakerState;
  nextAttemptTime?: Date;
}

enum ErrorCategory {
  NETWORK = 'network',
  AUTHENTICATION = 'authentication',
  VALIDATION = 'validation',
  SERVICE_UNAVAILABLE = 'service_unavailable',
  TIMEOUT = 'timeout',
  RATE_LIMIT = 'rate_limit',
  FILE_FORMAT = 'file_format',
  DATA_QUALITY = 'data_quality',
  INTERNAL_SERVER = 'internal_server',
  UNKNOWN = 'unknown'
}

enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

enum RecoveryAction {
  RETRY = 'retry',
  REFRESH_AUTH = 'refresh_auth',
  REDIRECT_LOGIN = 'redirect_login',
  FALLBACK_MODE = 'fallback_mode',
  CIRCUIT_BREAKER = 'circuit_breaker',
  USER_INTERVENTION = 'user_intervention'
}

interface ErrorClassification {
  category: ErrorCategory;
  severity: ErrorSeverity;
  retryable: boolean;
  userMessage: string;
  technicalMessage: string;
  recoveryActions: RecoveryAction[];
  retryDelay: number;
  maxRetries: number;
  context: Record<string, any>;
}

class ErrorClassifier {
  private classificationRules: Map<string, ErrorClassification>;

  constructor() {
    this.classificationRules = this.initializeClassificationRules();
  }

  private initializeClassificationRules(): Map<string, ErrorClassification> {
    const rules = new Map<string, ErrorClassification>();

    // Network errors
    rules.set('connection_error', {
      category: ErrorCategory.NETWORK,
      severity: ErrorSeverity.MEDIUM,
      retryable: true,
      userMessage: 'Upload failed due to network issues. Please check your connection and try again.',
      technicalMessage: 'Network connection error',
      recoveryActions: [RecoveryAction.RETRY],
      retryDelay: 2000,
      maxRetries: 3,
      context: {}
    });

    rules.set('upload_network_error', {
      category: ErrorCategory.NETWORK,
      severity: ErrorSeverity.HIGH,
      retryable: true,
      userMessage: 'Upload failed due to network issues. Please check your connection and try again.',
      technicalMessage: 'Upload network error',
      recoveryActions: [RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
      retryDelay: 3000,
      maxRetries: 3,
      context: {}
    });

    rules.set('timeout_error', {
      category: ErrorCategory.TIMEOUT,
      severity: ErrorSeverity.MEDIUM,
      retryable: true,
      userMessage: 'Request timed out. The server may be busy. Please try again.',
      technicalMessage: 'Request timeout',
      recoveryActions: [RecoveryAction.RETRY],
      retryDelay: 5000,
      maxRetries: 2,
      context: {}
    });

    // Authentication errors
    rules.set('auth_token_expired', {
      category: ErrorCategory.AUTHENTICATION,
      severity: ErrorSeverity.MEDIUM,
      retryable: true,
      userMessage: 'Your session has expired. Please log in again.',
      technicalMessage: 'Authentication token expired',
      recoveryActions: [RecoveryAction.REFRESH_AUTH, RecoveryAction.REDIRECT_LOGIN],
      retryDelay: 500,
      maxRetries: 1,
      context: {}
    });

    rules.set('auth_invalid_credentials', {
      category: ErrorCategory.AUTHENTICATION,
      severity: ErrorSeverity.HIGH,
      retryable: false,
      userMessage: 'Invalid credentials. Please check your login information.',
      technicalMessage: 'Invalid authentication credentials',
      recoveryActions: [RecoveryAction.REDIRECT_LOGIN],
      retryDelay: 0,
      maxRetries: 0,
      context: {}
    });

    // File format errors
    rules.set('file_format_invalid', {
      category: ErrorCategory.FILE_FORMAT,
      severity: ErrorSeverity.MEDIUM,
      retryable: false,
      userMessage: 'Invalid file format. Please upload CSV, Excel, or PDF files only.',
      technicalMessage: 'Unsupported file format',
      recoveryActions: [RecoveryAction.USER_INTERVENTION],
      retryDelay: 0,
      maxRetries: 0,
      context: {}
    });

    rules.set('file_size_exceeded', {
      category: ErrorCategory.VALIDATION,
      severity: ErrorSeverity.MEDIUM,
      retryable: false,
      userMessage: 'File size exceeds the 50MB limit. Please upload a smaller file.',
      technicalMessage: 'File size validation failed',
      recoveryActions: [RecoveryAction.USER_INTERVENTION],
      retryDelay: 0,
      maxRetries: 0,
      context: {}
    });

    // Service errors
    rules.set('service_unavailable', {
      category: ErrorCategory.SERVICE_UNAVAILABLE,
      severity: ErrorSeverity.HIGH,
      retryable: true,
      userMessage: 'Service is temporarily unavailable. Please try again in a few moments.',
      technicalMessage: 'Service unavailable',
      recoveryActions: [RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAKER],
      retryDelay: 10000,
      maxRetries: 2,
      context: {}
    });

    rules.set('rate_limit_exceeded', {
      category: ErrorCategory.RATE_LIMIT,
      severity: ErrorSeverity.MEDIUM,
      retryable: true,
      userMessage: 'Too many requests. Please wait a moment before trying again.',
      technicalMessage: 'Rate limit exceeded',
      recoveryActions: [RecoveryAction.RETRY],
      retryDelay: 30000,
      maxRetries: 2,
      context: {}
    });

    rules.set('internal_server_error', {
      category: ErrorCategory.INTERNAL_SERVER,
      severity: ErrorSeverity.HIGH,
      retryable: true,
      userMessage: 'An internal server error occurred. Our team has been notified.',
      technicalMessage: 'Internal server error',
      recoveryActions: [RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
      retryDelay: 5000,
      maxRetries: 2,
      context: {}
    });

    // Default fallback
    rules.set('unknown_error', {
      category: ErrorCategory.UNKNOWN,
      severity: ErrorSeverity.MEDIUM,
      retryable: true,
      userMessage: 'An unexpected error occurred. Please try again or contact support if the problem persists.',
      technicalMessage: 'Unknown error',
      recoveryActions: [RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
      retryDelay: 3000,
      maxRetries: 2,
      context: {}
    });

    return rules;
  }

  classifyError(error: Error | string, statusCode?: number, context?: Record<string, any>): ErrorClassification {
    const errorMessage = typeof error === 'string' ? error : error.message;
    const errorLower = errorMessage.toLowerCase();

    let classification: ErrorClassification;

    // Network-related errors
    if (errorLower.includes('failed to fetch') || errorLower.includes('network error') || 
        errorLower.includes('connection') || errorLower.includes('unreachable')) {
      if (errorLower.includes('timeout')) {
        classification = this.classificationRules.get('timeout_error')!;
      } else if (context?.endpoint?.includes('upload') || errorLower.includes('upload')) {
        classification = this.classificationRules.get('upload_network_error')!;
      } else {
        classification = this.classificationRules.get('connection_error')!;
      }
    }
    // Authentication errors
    else if (statusCode === 401 || errorLower.includes('unauthorized') || errorLower.includes('authentication')) {
      if (errorLower.includes('expired') || errorLower.includes('token')) {
        classification = this.classificationRules.get('auth_token_expired')!;
      } else {
        classification = this.classificationRules.get('auth_invalid_credentials')!;
      }
    }
    // File format and validation errors
    else if (errorLower.includes('file format') || errorLower.includes('invalid format') || errorLower.includes('unsupported')) {
      classification = this.classificationRules.get('file_format_invalid')!;
    }
    else if (errorLower.includes('file size') || errorLower.includes('too large') || errorLower.includes('exceeds limit')) {
      classification = this.classificationRules.get('file_size_exceeded')!;
    }
    // Service errors
    else if (statusCode === 503 || errorLower.includes('service unavailable') || errorLower.includes('server unavailable')) {
      classification = this.classificationRules.get('service_unavailable')!;
    }
    else if (statusCode === 429 || errorLower.includes('rate limit') || errorLower.includes('too many requests')) {
      classification = this.classificationRules.get('rate_limit_exceeded')!;
    }
    else if (statusCode && statusCode >= 500 || errorLower.includes('internal server')) {
      classification = this.classificationRules.get('internal_server_error')!;
    }
    else if (errorLower.includes('timeout')) {
      classification = this.classificationRules.get('timeout_error')!;
    }
    // Default fallback
    else {
      classification = this.classificationRules.get('unknown_error')!;
    }

    // Add context information
    const enhancedClassification = { ...classification };
    if (context) {
      enhancedClassification.context = { ...enhancedClassification.context, ...context };
    }

    enhancedClassification.context = {
      ...enhancedClassification.context,
      originalError: errorMessage,
      statusCode,
      timestamp: new Date().toISOString()
    };

    return enhancedClassification;
  }

  isRetryable(error: Error | string, statusCode?: number): boolean {
    const classification = this.classifyError(error, statusCode);
    return classification.retryable;
  }

  getUserMessage(error: Error | string, statusCode?: number): string {
    const classification = this.classifyError(error, statusCode);
    return classification.userMessage;
  }

  getRecoveryActions(error: Error | string, statusCode?: number): RecoveryAction[] {
    const classification = this.classifyError(error, statusCode);
    return classification.recoveryActions;
  }
}

class CircuitBreaker {
  private stats: CircuitBreakerStats;
  private config: CircuitBreakerConfig;

  constructor(config: CircuitBreakerConfig) {
    this.config = config;
    this.stats = {
      failureCount: 0,
      successCount: 0,
      state: CircuitBreakerState.CLOSED
    };
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    // Check circuit breaker state
    if (this.stats.state === CircuitBreakerState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.stats.state = CircuitBreakerState.HALF_OPEN;
        console.log(`Circuit breaker ${this.config.name} transitioning to HALF_OPEN`);
      } else {
        throw new Error(`Circuit breaker ${this.config.name} is OPEN. Service temporarily unavailable.`);
      }
    }

    try {
      const result = await operation();
      this.recordSuccess();
      return result;
    } catch (error) {
      this.recordFailure();
      throw error;
    }
  }

  private recordSuccess(): void {
    this.stats.successCount++;
    
    if (this.stats.state === CircuitBreakerState.HALF_OPEN) {
      // Reset circuit breaker on successful half-open attempt
      this.stats.state = CircuitBreakerState.CLOSED;
      this.stats.failureCount = 0;
      console.log(`Circuit breaker ${this.config.name} reset to CLOSED`);
    }
  }

  private recordFailure(): void {
    this.stats.failureCount++;
    this.stats.lastFailureTime = new Date();
    
    if (this.stats.failureCount >= this.config.failureThreshold) {
      this.stats.state = CircuitBreakerState.OPEN;
      this.stats.nextAttemptTime = new Date(Date.now() + this.config.recoveryTimeout * 1000);
      console.warn(`Circuit breaker ${this.config.name} opened due to ${this.stats.failureCount} failures`);
    }
  }

  private shouldAttemptReset(): boolean {
    if (!this.stats.nextAttemptTime) {
      return true;
    }
    return new Date() >= this.stats.nextAttemptTime;
  }

  getState(): CircuitBreakerState {
    return this.stats.state;
  }

  getStats(): CircuitBreakerStats {
    return { ...this.stats };
  }
}

class RetryManager {
  private config: RetryConfig;
  private errorClassifier: ErrorClassifier;

  constructor(config?: Partial<RetryConfig>) {
    this.config = {
      maxAttempts: 3,
      baseDelay: 1000,
      maxDelay: 60000,
      backoffMultiplier: 2,
      jitter: 0.1,
      retryOnStatusCodes: [500, 502, 503, 504, 408, 429],
      ...config
    };
    this.errorClassifier = new ErrorClassifier();
  }

  async executeWithRetry<T>(
    operation: () => Promise<T>,
    context?: Record<string, any>
  ): Promise<T> {
    let attempt = 0;
    let lastError: Error;

    while (attempt < this.config.maxAttempts) {
      try {
        const result = await operation();
        return result;
      } catch (error) {
        attempt++;
        lastError = error as Error;

        // Classify the error
        const classification = this.errorClassifier.classifyError(error as Error, undefined, context);

        console.warn(
          `Operation failed (attempt ${attempt}/${this.config.maxAttempts}): ${classification.technicalMessage}`,
          {
            category: classification.category,
            severity: classification.severity,
            retryable: classification.retryable,
            context: context || {}
          }
        );

        // Check if we should retry
        if (!classification.retryable || attempt >= this.config.maxAttempts) {
          // Enhance error with classification info
          const enhancedError = new EnhancedError(
            error as Error,
            classification,
            attempt,
            context || {}
          );
          throw enhancedError;
        }

        // Calculate delay with exponential backoff and jitter
        const delay = Math.min(
          classification.retryDelay * Math.pow(this.config.backoffMultiplier, attempt - 1),
          this.config.maxDelay
        );

        // Add jitter to prevent thundering herd
        const jitter = delay * this.config.jitter * (Math.random() - 0.5);
        const totalDelay = delay + jitter;

        console.log(`Retrying in ${totalDelay}ms...`);
        await this.sleep(totalDelay);
      }
    }

    // All retries exhausted
    const classification = this.errorClassifier.classifyError(lastError!, undefined, context);
    const enhancedError = new EnhancedError(
      lastError!,
      classification,
      attempt,
      context || {}
    );
    throw enhancedError;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class EnhancedError extends Error {
  public readonly originalError: Error;
  public readonly classification: ErrorClassification;
  public readonly attemptCount: number;
  public readonly context: Record<string, any>;

  constructor(
    originalError: Error,
    classification: ErrorClassification,
    attemptCount: number,
    context: Record<string, any>
  ) {
    super(classification.userMessage);
    this.name = 'EnhancedError';
    this.originalError = originalError;
    this.classification = classification;
    this.attemptCount = attemptCount;
    this.context = context;
  }

  toJSON(): ErrorDetails {
    return {
      category: this.classification.category,
      severity: this.classification.severity,
      retryable: this.classification.retryable,
      recoveryActions: this.classification.recoveryActions,
      attemptCount: this.attemptCount,
      context: this.context,
      timestamp: new Date().toISOString()
    };
  }
}

interface EnhancedApiClientConfig {
  baseURL?: string;
  timeout?: number;
  retryConfig?: Partial<RetryConfig>;
  circuitBreakerConfig?: Partial<CircuitBreakerConfig>;
}

class EnhancedApiClient {
  private baseURL: string;
  private timeout: number;
  private authToken: string | null = null;
  private onAuthError?: () => void;
  private retryManager: RetryManager;
  private circuitBreakers: Map<string, CircuitBreaker>;
  private errorClassifier: ErrorClassifier;
  private healthStatus: Map<string, boolean>;

  constructor(config: EnhancedApiClientConfig = {}) {
    this.baseURL = config.baseURL || 'http://localhost:8000';
    this.timeout = config.timeout || 30000; // Increased timeout
    this.retryManager = new RetryManager(config.retryConfig);
    this.circuitBreakers = new Map();
    this.errorClassifier = new ErrorClassifier();
    this.healthStatus = new Map();

    // Initialize circuit breakers for critical endpoints
    this.initializeCircuitBreakers(config.circuitBreakerConfig);
  }

  private initializeCircuitBreakers(config?: Partial<CircuitBreakerConfig>): void {
    const defaultConfig = {
      failureThreshold: 5,
      recoveryTimeout: 60,
      ...config
    };

    // Create circuit breakers for critical services
    const criticalEndpoints = [
      'parameter-detection',
      'ensemble-initialization',
      'data-processing',
      'upload'
    ];

    criticalEndpoints.forEach(endpoint => {
      this.circuitBreakers.set(endpoint, new CircuitBreaker({
        ...defaultConfig,
        name: endpoint
      }));
    });
  }

  setAuthToken(token: string | null): void {
    this.authToken = token;
  }

  setAuthErrorHandler(handler: () => void): void {
    this.onAuthError = handler;
  }

  private getCircuitBreaker(endpoint: string): CircuitBreaker | null {
    // Determine which circuit breaker to use based on endpoint
    if (endpoint.includes('parameter') || endpoint.includes('detect')) {
      return this.circuitBreakers.get('parameter-detection') || null;
    }
    if (endpoint.includes('ensemble') || endpoint.includes('init')) {
      return this.circuitBreakers.get('ensemble-initialization') || null;
    }
    if (endpoint.includes('data-processing') || endpoint.includes('analyze')) {
      return this.circuitBreakers.get('data-processing') || null;
    }
    if (endpoint.includes('upload')) {
      return this.circuitBreakers.get('upload') || null;
    }
    return null;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    const circuitBreaker = this.getCircuitBreaker(endpoint);
    
    const executeRequest = async (): Promise<ApiResponse<T>> => {
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

      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
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
          throw new Error('Authentication failed. Please log in again.');
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
          // Update service health status
          this.healthStatus.set(endpoint, true);
          
          return {
            success: true,
            data,
          };
        } else {
          // Update service health status
          this.healthStatus.set(endpoint, false);
          
          const errorMessage = data.message || data.detail || `HTTP ${response.status}: ${response.statusText}`;
          const error = new Error(errorMessage);
          (error as any).statusCode = response.status;
          throw error;
        }
      } catch (error: any) {
        clearTimeout(timeoutId);
        
        // Update service health status
        this.healthStatus.set(endpoint, false);
        
        if (error.name === 'AbortError') {
          throw new Error('Request timeout. Please try again.');
        }

        // Enhance network errors
        if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
          throw new Error('Network connection failed. Please check your internet connection.');
        }

        throw error;
      }
    };

    // Execute with circuit breaker if available
    if (circuitBreaker) {
      try {
        return await circuitBreaker.execute(executeRequest);
      } catch (error) {
        // If circuit breaker is open, provide fallback response
        if (error instanceof Error && error.message.includes('Circuit breaker') && error.message.includes('OPEN')) {
          return {
            success: false,
            error: 'Service is temporarily unavailable. Please try again later.',
            errorDetails: {
              category: ErrorCategory.SERVICE_UNAVAILABLE,
              severity: ErrorSeverity.HIGH,
              retryable: true,
              recoveryActions: [RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
              attemptCount: 0,
              context: { circuitBreakerOpen: true, endpoint },
              timestamp: new Date().toISOString()
            }
          };
        }
        throw error;
      }
    }

    // Execute with retry logic
    return await this.retryManager.executeWithRetry(executeRequest, { endpoint, url });
  }

  // HTTP Methods with enhanced error handling
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

    try {
      return await this.makeRequest<T>(url, { method: 'GET' });
    } catch (error) {
      return this.handleError<T>(error as Error, { endpoint, method: 'GET' });
    }
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

    try {
      return await this.makeRequest<T>(endpoint, options);
    } catch (error) {
      return this.handleError<T>(error as Error, { endpoint, method: 'POST', data });
    }
  }

  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    try {
      return await this.makeRequest<T>(endpoint, {
        method: 'PUT',
        body: data ? JSON.stringify(data) : undefined,
      });
    } catch (error) {
      return this.handleError<T>(error as Error, { endpoint, method: 'PUT', data });
    }
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    try {
      return await this.makeRequest<T>(endpoint, { method: 'DELETE' });
    } catch (error) {
      return this.handleError<T>(error as Error, { endpoint, method: 'DELETE' });
    }
  }

  // Enhanced file upload with progress tracking
  async uploadFile<T>(
    endpoint: string, 
    file: File, 
    additionalData?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<T>> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, String(value));
      });
    }

    try {
      // Use XMLHttpRequest for progress tracking
      if (onProgress) {
        return await this.uploadWithProgress<T>(endpoint, formData, onProgress);
      }

      return await this.post<T>(endpoint, formData);
    } catch (error) {
      return this.handleError<T>(error as Error, { 
        endpoint, 
        method: 'POST', 
        fileSize: file.size, 
        fileName: file.name 
      });
    }
  }

  private async uploadWithProgress<T>(
    endpoint: string,
    formData: FormData,
    onProgress: (progress: number) => void
  ): Promise<ApiResponse<T>> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const url = `${this.baseURL}${endpoint}`;

      // Set up progress tracking
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          onProgress(progress);
        }
      });

      // Set up response handling
      xhr.addEventListener('load', () => {
        try {
          if (xhr.status >= 200 && xhr.status < 300) {
            const data = JSON.parse(xhr.responseText);
            resolve({
              success: true,
              data
            });
          } else {
            const errorData = JSON.parse(xhr.responseText);
            resolve({
              success: false,
              error: errorData.message || errorData.detail || `HTTP ${xhr.status}`,
              data: errorData
            });
          }
        } catch (parseError) {
          resolve({
            success: false,
            error: `HTTP ${xhr.status}: ${xhr.statusText}`
          });
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error occurred during file upload'));
      });

      xhr.addEventListener('timeout', () => {
        reject(new Error('File upload timed out'));
      });

      // Set up request
      xhr.open('POST', url);
      xhr.timeout = this.timeout;

      // Add authorization header if available
      if (this.authToken) {
        xhr.setRequestHeader('Authorization', `Bearer ${this.authToken}`);
      }

      // Send request
      xhr.send(formData);
    });
  }

  private handleError<T>(error: Error, context: Record<string, any>): ApiResponse<T> {
    const statusCode = (error as any).statusCode;
    const classification = this.errorClassifier.classifyError(error, statusCode, context);

    console.error('API Error:', {
      error: error.message,
      classification,
      context
    });

    // Handle authentication errors
    if (classification.recoveryActions.includes(RecoveryAction.REDIRECT_LOGIN) && this.onAuthError) {
      this.onAuthError();
    }

    return {
      success: false,
      error: classification.userMessage,
      errorDetails: {
        category: classification.category,
        severity: classification.severity,
        retryable: classification.retryable,
        recoveryActions: classification.recoveryActions,
        attemptCount: 1,
        context: classification.context,
        timestamp: new Date().toISOString()
      }
    };
  }

  // Health monitoring methods
  async checkServiceHealth(serviceName?: string): Promise<Record<string, any>> {
    try {
      const response = await this.get('/api/v1/health');
      return response.data || {};
    } catch (error) {
      return {
        healthy: false,
        error: (error as Error).message,
        timestamp: new Date().toISOString()
      };
    }
  }

  getCircuitBreakerStatus(): Record<string, any> {
    const status: Record<string, any> = {};
    
    this.circuitBreakers.forEach((breaker, name) => {
      status[name] = breaker.getStats();
    });
    
    return status;
  }

  getHealthStatus(): Record<string, boolean> {
    return Object.fromEntries(this.healthStatus);
  }

  // Fallback mode methods
  async enableFallbackMode(endpoint: string): Promise<void> {
    console.warn(`Enabling fallback mode for ${endpoint}`);
    // Implement fallback logic here
    // This could involve using cached data, simplified processing, etc.
  }

  async disableFallbackMode(endpoint: string): Promise<void> {
    console.log(`Disabling fallback mode for ${endpoint}`);
    // Re-enable normal processing
  }
}

// Create singleton instance
export const enhancedApiClient = new EnhancedApiClient();

// Export types and classes for use in components
export type { 
  ApiResponse, 
  ErrorDetails, 
  RetryConfig, 
  CircuitBreakerConfig 
};

export { 
  EnhancedApiClient, 
  ErrorClassifier, 
  RetryManager, 
  CircuitBreaker, 
  EnhancedError,
  ErrorCategory,
  ErrorSeverity,
  RecoveryAction,
  CircuitBreakerState
};