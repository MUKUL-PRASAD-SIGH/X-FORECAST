import { useState, useEffect } from 'react';
import { useApiClient } from './useApiClient';
import { useAuth } from '../contexts/AuthContext';

interface CompanyMetrics {
  totalCustomers: number;
  retentionRate: number;
  forecastAccuracy: number;
  systemHealth: number;
  activeAlerts: number;
  revenueGrowth: number;
  totalDocuments: number;
  ragStatus: string;
  lastDataUpload: string | null;
}

interface SystemStatus {
  status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  uptime: string;
  lastUpdate: string;
}

export const useCompanyMetrics = () => {
  const { isAuthenticated, user } = useAuth();
  const { get } = useApiClient();
  
  const [metrics, setMetrics] = useState<CompanyMetrics>({
    totalCustomers: 0,
    retentionRate: 0,
    forecastAccuracy: 0,
    systemHealth: 0,
    activeAlerts: 0,
    revenueGrowth: 0,
    totalDocuments: 0,
    ragStatus: 'not_initialized',
    lastDataUpload: null,
  });
  
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'good',
    uptime: '99.9%',
    lastUpdate: new Date().toLocaleTimeString()
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCompanyMetrics = async () => {
    if (!isAuthenticated) return;

    setLoading(true);
    setError(null);

    try {
      // Fetch company dashboard metrics
      const metricsResponse = await get<any>('/api/v1/company/metrics');
      
      if (metricsResponse.success && metricsResponse.data) {
        const data = metricsResponse.data;
        
        setMetrics({
          totalCustomers: data.total_customers || 0,
          retentionRate: data.retention_rate || 0,
          forecastAccuracy: data.forecast_accuracy || 0,
          systemHealth: data.system_health || 0.85,
          activeAlerts: data.active_alerts || 0,
          revenueGrowth: data.revenue_growth || 0,
          totalDocuments: data.total_documents || 0,
          ragStatus: data.rag_status || 'not_initialized',
          lastDataUpload: data.last_data_upload || null,
        });
      } else {
        // Use fallback metrics based on user data if API fails
        setMetrics(prev => ({
          ...prev,
          ragStatus: user?.rag_initialized ? 'initialized' : 'not_initialized',
          systemHealth: user?.rag_initialized ? 0.85 : 0.60,
        }));
      }

      // Update system status
      setSystemStatus({
        status: 'good',
        uptime: '99.9%',
        lastUpdate: new Date().toLocaleTimeString()
      });

    } catch (err) {
      console.error('Failed to fetch company metrics:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
      
      // Use basic fallback metrics
      setMetrics(prev => ({
        ...prev,
        ragStatus: user?.rag_initialized ? 'initialized' : 'not_initialized',
        systemHealth: user?.rag_initialized ? 0.75 : 0.50,
      }));
    } finally {
      setLoading(false);
    }
  };

  // Fetch metrics on mount and when user changes
  useEffect(() => {
    if (isAuthenticated && user) {
      fetchCompanyMetrics();
      
      // Set up periodic refresh every 30 seconds
      const interval = setInterval(fetchCompanyMetrics, 30000);
      return () => clearInterval(interval);
    }
  }, [isAuthenticated, user]);

  return {
    metrics,
    systemStatus,
    loading,
    error,
    refetch: fetchCompanyMetrics,
  };
};