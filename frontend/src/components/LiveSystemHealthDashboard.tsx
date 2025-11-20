/**
 * Live System Health Dashboard
 * Real-time system health monitoring with circuit breaker status and service recovery detection
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { CyberpunkCard } from './ui/CyberpunkCard';
import { CyberpunkButton } from './ui/CyberpunkButton';
import { useEnhancedApiClient } from '../hooks/useEnhancedApiClient';

interface ServiceHealth {
  service_name: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  health_score: number;
  last_check: string;
  consecutive_failures: number;
  consecutive_successes: number;
  uptime_percentage: number;
  average_response_time: number;
  circuit_breaker_state: 'closed' | 'open' | 'half_open';
  recent_checks: Array<{
    status: string;
    response_time_ms: number;
    timestamp: string;
    message: string;
    error?: string;
  }>;
  alerts: string[];
}

interface SystemHealthSummary {
  overall_status: string;
  overall_health_score: number;
  service_counts: {
    healthy: number;
    degraded: number;
    unhealthy: number;
    unknown: number;
  };
  total_services: number;
  services: Record<string, ServiceHealth>;
  circuit_breakers: Record<string, any>;
  timestamp: string;
}

interface HealthAlert {
  type: string;
  service_name: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
}

export const LiveSystemHealthDashboard: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealthSummary | null>(null);
  const [alerts, setAlerts] = useState<HealthAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedService, setSelectedService] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const { get, post } = useEnhancedApiClient();

  // Fetch system health data
  const fetchSystemHealth = useCallback(async () => {
    try {
      setError(null);
      const response = await get<SystemHealthSummary>('/api/v1/health/dashboard');
      
      if (response.success && response.data) {
        setSystemHealth(response.data);
        
        // Extract alerts from the response
        if (response.data.services) {
          const newAlerts: HealthAlert[] = [];
          Object.values(response.data.services).forEach(service => {
            if (service.status === 'unhealthy') {
              newAlerts.push({
                type: 'service_unhealthy',
                service_name: service.service_name,
                severity: 'high',
                message: `Service ${service.service_name} is unhealthy`,
                timestamp: service.last_check
              });
            } else if (service.status === 'degraded') {
              newAlerts.push({
                type: 'service_degraded',
                service_name: service.service_name,
                severity: 'medium',
                message: `Service ${service.service_name} is degraded`,
                timestamp: service.last_check
              });
            }
            
            if (service.uptime_percentage < 95) {
              newAlerts.push({
                type: 'low_uptime',
                service_name: service.service_name,
                severity: 'medium',
                message: `Service ${service.service_name} has low uptime: ${service.uptime_percentage.toFixed(1)}%`,
                timestamp: service.last_check
              });
            }
          });
          setAlerts(newAlerts);
        }
      } else {
        setError(response.error || 'Failed to fetch system health');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  }, [get]);

  // Force health check for a specific service
  const forceHealthCheck = useCallback(async (serviceName: string) => {
    try {
      const response = await post(`/api/v1/health/services/${serviceName}/check`);
      if (response.success) {
        // Refresh the dashboard data
        await fetchSystemHealth();
      }
    } catch (err) {
      console.error('Failed to force health check:', err);
    }
  }, [post, fetchSystemHealth]);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/v1/health/live`;
      
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('Health monitoring WebSocket connected');
        setWsConnected(true);
        setError(null);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'system_health_update') {
            setSystemHealth(message.data);
          } else if (message.type === 'health_update') {
            // Update specific service health
            setSystemHealth(prev => {
              if (!prev) return prev;
              
              const updatedServices = { ...prev.services };
              if (updatedServices[message.service_name]) {
                updatedServices[message.service_name] = {
                  ...updatedServices[message.service_name],
                  status: message.data.status,
                  last_check: message.timestamp
                };
              }
              
              return {
                ...prev,
                services: updatedServices
              };
            });
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      wsRef.current.onclose = () => {
        console.log('Health monitoring WebSocket disconnected');
        setWsConnected(false);
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (autoRefresh) {
            connectWebSocket();
          }
        }, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      setWsConnected(false);
    }
  }, [autoRefresh]);

  // Setup auto-refresh and WebSocket
  useEffect(() => {
    // Initial fetch
    fetchSystemHealth();

    if (autoRefresh) {
      // Setup WebSocket for real-time updates
      connectWebSocket();
      
      // Fallback polling every 30 seconds
      refreshIntervalRef.current = setInterval(fetchSystemHealth, 30000);
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [autoRefresh, fetchSystemHealth, connectWebSocket]);

  // Get status color for UI
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'degraded': return 'text-yellow-400';
      case 'unhealthy': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  // Get circuit breaker color
  const getCircuitBreakerColor = (state: string): string => {
    switch (state) {
      case 'closed': return 'text-green-400';
      case 'half_open': return 'text-yellow-400';
      case 'open': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleString();
  };

  if (isLoading) {
    return (
      <CyberpunkCard className="w-full">
        <div className="mb-4">
          <h2 className="text-xl font-bold text-cyan-400">System Health Dashboard</h2>
        </div>
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400"></div>
          <span className="ml-2 text-cyan-400">Loading system health...</span>
        </div>
      </CyberpunkCard>
    );
  }

  if (error) {
    return (
      <CyberpunkCard className="w-full">
        <div className="mb-4">
          <h2 className="text-xl font-bold text-cyan-400">System Health Dashboard</h2>
        </div>
        <div className="text-center p-8">
          <div className="text-red-400 mb-4">⚠️ Error loading system health</div>
          <p className="text-gray-300 mb-4">{error}</p>
          <CyberpunkButton onClick={fetchSystemHealth}>
            Retry
          </CyberpunkButton>
        </div>
      </CyberpunkCard>
    );
  }

  if (!systemHealth) {
    return (
      <CyberpunkCard className="w-full">
        <div className="mb-4">
          <h2 className="text-xl font-bold text-cyan-400">System Health Dashboard</h2>
        </div>
        <div className="text-center p-8 text-gray-400">
          No health data available
        </div>
      </CyberpunkCard>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with controls */}
      <CyberpunkCard>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-cyan-400">System Health Dashboard</h2>
            <div className={`w-3 h-3 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`} 
                 title={wsConnected ? 'Real-time updates active' : 'Real-time updates disconnected'} />
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              Auto-refresh
            </label>
            <CyberpunkButton onClick={fetchSystemHealth} $size="sm">
              Refresh Now
            </CyberpunkButton>
          </div>
        </div>
      </CyberpunkCard>

      {/* System Overview */}
      <CyberpunkCard>
        <div className="mb-4">
          <h3 className="text-lg font-bold text-cyan-400">System Overview</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className={`text-2xl font-bold ${getStatusColor(systemHealth.overall_status)}`}>
              {systemHealth.overall_status.toUpperCase()}
            </div>
            <div className="text-sm text-gray-400">Overall Status</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-cyan-400">
              {(systemHealth.overall_health_score * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Health Score</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {systemHealth.service_counts.healthy}
            </div>
            <div className="text-sm text-gray-400">Healthy Services</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              {systemHealth.service_counts.unhealthy}
            </div>
            <div className="text-sm text-gray-400">Unhealthy Services</div>
          </div>
        </div>
      </CyberpunkCard>

      {/* Active Alerts */}
      {alerts.length > 0 && (
        <CyberpunkCard>
          <div className="mb-4">
            <h3 className="text-lg font-bold text-cyan-400">Active Alerts</h3>
          </div>
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <div key={index} className={`p-3 rounded border-l-4 ${
                alert.severity === 'high' ? 'border-red-400 bg-red-900/20' :
                alert.severity === 'medium' ? 'border-yellow-400 bg-yellow-900/20' :
                'border-blue-400 bg-blue-900/20'
              }`}>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{alert.message}</div>
                    <div className="text-sm text-gray-400">
                      Service: {alert.service_name} • {formatTimestamp(alert.timestamp)}
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs ${
                    alert.severity === 'high' ? 'bg-red-600' :
                    alert.severity === 'medium' ? 'bg-yellow-600' :
                    'bg-blue-600'
                  }`}>
                    {alert.severity.toUpperCase()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CyberpunkCard>
      )}

      {/* Services Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Object.entries(systemHealth.services).map(([serviceName, service]) => (
          <CyberpunkCard key={serviceName}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-cyan-400">{service.service_name}</h3>
              <div className="flex items-center gap-2">
                <div className={`px-2 py-1 rounded text-xs ${getStatusColor(service.status)} bg-gray-800`}>
                  {service.status.toUpperCase()}
                </div>
                <CyberpunkButton 
                  $size="sm" 
                  onClick={() => forceHealthCheck(serviceName)}
                >
                  Check Now
                </CyberpunkButton>
              </div>
            </div>
              <div className="space-y-4">
                {/* Health Metrics */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400">Health Score</div>
                    <div className="text-cyan-400 font-medium">
                      {(service.health_score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Uptime</div>
                    <div className="text-cyan-400 font-medium">
                      {service.uptime_percentage.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Avg Response</div>
                    <div className="text-cyan-400 font-medium">
                      {service.average_response_time.toFixed(0)}ms
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Circuit Breaker</div>
                    <div className={`font-medium ${getCircuitBreakerColor(service.circuit_breaker_state)}`}>
                      {service.circuit_breaker_state.toUpperCase()}
                    </div>
                  </div>
                </div>

                {/* Failure/Success Counters */}
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                    <span>Failures: {service.consecutive_failures}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span>Successes: {service.consecutive_successes}</span>
                  </div>
                </div>

                {/* Recent Checks */}
                {service.recent_checks.length > 0 && (
                  <div>
                    <div className="text-sm text-gray-400 mb-2">Recent Checks</div>
                    <div className="space-y-1">
                      {service.recent_checks.slice(-3).map((check, index) => (
                        <div key={index} className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${getStatusColor(check.status)}`}></div>
                            <span>{check.message || 'Health check'}</span>
                          </div>
                          <div className="text-gray-400">
                            {check.response_time_ms.toFixed(0)}ms
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Last Check Time */}
                <div className="text-xs text-gray-400">
                  Last checked: {formatTimestamp(service.last_check)}
                </div>
              </div>
          </CyberpunkCard>
        ))}
      </div>

      {/* Circuit Breaker Status */}
      {Object.keys(systemHealth.circuit_breakers).length > 0 && (
        <CyberpunkCard>
          <div className="mb-4">
            <h3 className="text-lg font-bold text-cyan-400">Circuit Breaker Status</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(systemHealth.circuit_breakers).map(([name, stats]) => (
              <div key={name} className="p-4 border border-gray-700 rounded">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-medium">{name}</div>
                  <div className={`px-2 py-1 rounded text-xs ${getCircuitBreakerColor(stats.state)}`}>
                    {stats.state?.toUpperCase() || 'UNKNOWN'}
                  </div>
                </div>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Failures:</span>
                    <span className="text-red-400">{stats.failure_count || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Successes:</span>
                    <span className="text-green-400">{stats.success_count || 0}</span>
                  </div>
                  {stats.last_failure_time && (
                    <div className="text-xs text-gray-500">
                      Last failure: {formatTimestamp(stats.last_failure_time)}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CyberpunkCard>
      )}

      {/* Footer */}
      <div className="text-center text-sm text-gray-400">
        Last updated: {formatTimestamp(systemHealth.timestamp)}
      </div>
    </div>
  );
};