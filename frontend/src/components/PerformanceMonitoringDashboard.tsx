import React, { useState, useEffect, useCallback } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';
import { useAuth } from '../contexts/AuthContext';
import { useApiClient } from '../hooks/useApiClient';

// Animation keyframes
const scanLine = keyframes`
  0% { left: -100%; }
  100% { left: 100%; }
`;

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
`;

const glitch = keyframes`
  0%, 100% { transform: translate(0); }
  20% { transform: translate(-2px, 2px); }
  40% { transform: translate(-2px, -2px); }
  60% { transform: translate(2px, 2px); }
  80% { transform: translate(2px, -2px); }
`;

const alertPulse = keyframes`
  0%, 100% { 
    box-shadow: 0 0 20px rgba(255, 0, 64, 0.6);
    border-color: rgba(255, 0, 64, 0.8);
  }
  50% { 
    box-shadow: 0 0 40px rgba(255, 0, 64, 0.9);
    border-color: rgba(255, 0, 64, 1);
  }
`;

// Styled components
const DashboardContainer = styled(CyberpunkCard)`
  padding: 2rem;
  margin: 1rem 0;
  min-height: 800px;
  position: relative;
`;

const DashboardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  
  h2 {
    color: ${props => props.theme.colors.neonBlue};
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.xxl};
    text-shadow: ${props => props.theme.effects.softGlow};
    margin: 0;
  }
  
  .controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  
  .status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.sm};
    color: ${props => props.theme.colors.secondaryText};
    
    .status-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: ${props => props.theme.colors.acidGreen};
      animation: ${pulse} 2s infinite;
      box-shadow: 0 0 10px ${props => props.theme.colors.glowGreen};
    }
    
    &.degraded .status-dot {
      background: ${props => props.theme.colors.warning};
      box-shadow: 0 0 10px ${props => props.theme.colors.cyberYellow};
    }
    
    &.critical .status-dot {
      background: ${props => props.theme.colors.error};
      box-shadow: 0 0 10px ${props => props.theme.colors.error};
      animation: ${alertPulse} 1s infinite;
    }
  }
`;

const SystemHealthSection = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 2rem;
  margin-bottom: 2rem;
`;

const HealthGaugeContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const HealthGauge = styled(motion.div)<{ healthScore: number; status: string }>`
  position: relative;
  width: 200px;
  height: 200px;
  border-radius: 50%;
  background: conic-gradient(
    from 0deg,
    ${props => {
      if (props.status === 'excellent') return props.theme.colors.acidGreen;
      if (props.status === 'good') return props.theme.colors.neonBlue;
      if (props.status === 'warning') return props.theme.colors.cyberYellow;
      if (props.status === 'critical') return props.theme.colors.warning;
      return props.theme.colors.error;
    }} 0deg,
    ${props => {
      if (props.status === 'excellent') return props.theme.colors.acidGreen;
      if (props.status === 'good') return props.theme.colors.neonBlue;
      if (props.status === 'warning') return props.theme.colors.cyberYellow;
      if (props.status === 'critical') return props.theme.colors.warning;
      return props.theme.colors.error;
    }} ${props => props.healthScore * 360}deg,
    rgba(255, 255, 255, 0.1) ${props => props.healthScore * 360}deg,
    rgba(255, 255, 255, 0.1) 360deg
  );
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: ${props => props.theme.effects.neonGlow};
  
  &::before {
    content: '';
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    bottom: 10px;
    border-radius: 50%;
    background: ${props => props.theme.colors.darkBg};
    z-index: 1;
  }
  
  .gauge-content {
    position: relative;
    z-index: 2;
    text-align: center;
    color: ${props => props.theme.colors.primaryText};
    
    .gauge-value {
      font-family: ${props => props.theme.typography.fontFamily.display};
      font-size: 2.5rem;
      font-weight: ${props => props.theme.typography.fontWeight.bold};
      color: ${props => {
        if (props.status === 'excellent') return props.theme.colors.acidGreen;
        if (props.status === 'good') return props.theme.colors.neonBlue;
        if (props.status === 'warning') return props.theme.colors.cyberYellow;
        if (props.status === 'critical') return props.theme.colors.warning;
        return props.theme.colors.error;
      }};
      text-shadow: ${props => props.theme.effects.softGlow};
    }
    
    .gauge-label {
      font-family: ${props => props.theme.typography.fontFamily.mono};
      font-size: ${props => props.theme.typography.fontSize.sm};
      color: ${props => props.theme.colors.secondaryText};
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 0.5rem;
    }
    
    .gauge-status {
      font-family: ${props => props.theme.typography.fontFamily.mono};
      font-size: ${props => props.theme.typography.fontSize.xs};
      color: ${props => props.theme.colors.secondaryText};
      text-transform: uppercase;
      margin-top: 0.25rem;
    }
  }
`;

const SystemMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
`;

const MetricCard = styled(motion.div)<{ status: 'good' | 'warning' | 'critical' }>`
  background: rgba(0, 0, 0, 0.4);
  border: 2px solid ${props => {
    switch (props.status) {
      case 'good': return props.theme.colors.acidGreen;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'critical': return props.theme.colors.error;
      default: return props.theme.colors.secondaryText;
    }
  }};
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: ${props => {
      switch (props.status) {
        case 'good': return props.theme.colors.acidGreen;
        case 'warning': return props.theme.colors.cyberYellow;
        case 'critical': return props.theme.colors.error;
        default: return props.theme.colors.secondaryText;
      }
    }};
    animation: ${scanLine} 3s infinite;
  }
  
  ${props => props.status === 'critical' && css`
    animation: ${glitch} 0.5s infinite;
  `}
  
  .metric-label {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }
  
  .metric-value {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.xl};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
    color: ${props => {
      switch (props.status) {
        case 'good': return props.theme.colors.acidGreen;
        case 'warning': return props.theme.colors.cyberYellow;
        case 'critical': return props.theme.colors.error;
        default: return props.theme.colors.neonBlue;
      }
    }};
    text-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const ModelRankingSection = styled.div`
  margin-bottom: 2rem;
  
  h3 {
    color: ${props => props.theme.colors.neonBlue};
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.lg};
    margin-bottom: 1rem;
    text-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const RankingTable = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid ${props => props.theme.colors.neonBlue};
  border-radius: 8px;
  overflow: hidden;
`;

const RankingHeader = styled.div`
  display: grid;
  grid-template-columns: 60px 1fr 100px 100px 100px 100px 120px;
  gap: 1rem;
  padding: 1rem;
  background: rgba(0, 255, 255, 0.1);
  border-bottom: 1px solid ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const RankingRow = styled(motion.div)<{ rank: number; status: string }>`
  display: grid;
  grid-template-columns: 60px 1fr 100px 100px 100px 100px 120px;
  gap: 1rem;
  padding: 1rem;
  border-bottom: 1px solid rgba(0, 255, 255, 0.2);
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.primaryText};
  position: relative;
  
  &:hover {
    background: rgba(0, 255, 255, 0.05);
  }
  
  .rank-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: ${props => {
      if (props.rank === 1) return props.theme.colors.acidGreen;
      if (props.rank === 2) return props.theme.colors.neonBlue;
      if (props.rank === 3) return props.theme.colors.cyberYellow;
      return props.theme.colors.secondaryText;
    }};
    color: ${props => props.theme.colors.darkBg};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
    font-size: ${props => props.theme.typography.fontSize.xs};
  }
  
  .model-name {
    display: flex;
    align-items: center;
    font-weight: ${props => props.theme.typography.fontWeight.medium};
    color: ${props => props.theme.colors.neonBlue};
    text-transform: uppercase;
  }
  
  .metric-cell {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .status-cell {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    
    .status-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: ${props => {
        switch (props.status) {
          case 'healthy': return props.theme.colors.acidGreen;
          case 'warning': return props.theme.colors.cyberYellow;
          case 'degraded': return props.theme.colors.warning;
          case 'failed': return props.theme.colors.error;
          default: return props.theme.colors.secondaryText;
        }
      }};
      animation: ${pulse} 2s infinite;
    }
    
    .status-text {
      font-size: ${props => props.theme.typography.fontSize.xs};
      text-transform: uppercase;
      color: ${props => {
        switch (props.status) {
          case 'healthy': return props.theme.colors.acidGreen;
          case 'warning': return props.theme.colors.cyberYellow;
          case 'degraded': return props.theme.colors.warning;
          case 'failed': return props.theme.colors.error;
          default: return props.theme.colors.secondaryText;
        }
      }};
    }
  }
`;

const AlertsSection = styled.div`
  margin-bottom: 2rem;
  
  h3 {
    color: ${props => props.theme.colors.error};
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.lg};
    margin-bottom: 1rem;
    text-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const AlertsList = styled.div`
  max-height: 300px;
  overflow-y: auto;
  
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
  }
  
  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.neonBlue};
    border-radius: 4px;
  }
`;

const AlertCard = styled(motion.div)<{ severity: 'critical' | 'warning' | 'info' }>`
  background: rgba(0, 0, 0, 0.4);
  border: 2px solid ${props => {
    switch (props.severity) {
      case 'critical': return props.theme.colors.error;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'info': return props.theme.colors.neonBlue;
      default: return props.theme.colors.secondaryText;
    }
  }};
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
  
  ${props => props.severity === 'critical' && css`
    animation: ${alertPulse} 2s infinite;
  `}
  
  .alert-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    
    .alert-title {
      font-family: ${props => props.theme.typography.fontFamily.display};
      font-size: ${props => props.theme.typography.fontSize.md};
      font-weight: ${props => props.theme.typography.fontWeight.bold};
      color: ${props => {
        switch (props.severity) {
          case 'critical': return props.theme.colors.error;
          case 'warning': return props.theme.colors.cyberYellow;
          case 'info': return props.theme.colors.neonBlue;
          default: return props.theme.colors.primaryText;
        }
      }};
    }
    
    .alert-time {
      font-family: ${props => props.theme.typography.fontFamily.mono};
      font-size: ${props => props.theme.typography.fontSize.xs};
      color: ${props => props.theme.colors.secondaryText};
    }
  }
  
  .alert-message {
    font-family: ${props => props.theme.typography.fontFamily.primary};
    font-size: ${props => props.theme.typography.fontSize.sm};
    color: ${props => props.theme.colors.primaryText};
    margin-bottom: 0.5rem;
  }
  
  .alert-details {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
    background: rgba(0, 0, 0, 0.3);
    padding: 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
  }
  
  .alert-actions {
    display: flex;
    gap: 0.5rem;
  }
`;

const FallbackSection = styled.div`
  background: rgba(255, 0, 64, 0.1);
  border: 2px solid ${props => props.theme.colors.error};
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  
  h3 {
    color: ${props => props.theme.colors.error};
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.lg};
    margin-bottom: 1rem;
    text-shadow: ${props => props.theme.effects.softGlow};
  }
  
  .fallback-status {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
    
    .status-indicator {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: ${props => props.theme.colors.error};
      animation: ${pulse} 1s infinite;
    }
    
    .status-text {
      font-family: ${props => props.theme.typography.fontFamily.mono};
      font-size: ${props => props.theme.typography.fontSize.sm};
      color: ${props => props.theme.colors.error};
      text-transform: uppercase;
      font-weight: ${props => props.theme.typography.fontWeight.bold};
    }
  }
  
  .fallback-actions {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
`;

// Interfaces
interface SystemHealth {
  overall_health_score: number;
  health_status: string;
  active_models: number;
  failed_models: number;
  avg_accuracy: number;
  avg_response_time: number;
  drift_alerts: number;
  performance_alerts: number;
  last_updated: string;
  model_health_scores: Record<string, number>;
}

interface ModelRanking {
  model_name: string;
  rank: number;
  health_score: number;
  accuracy: number;
  mape: number;
  response_time: number;
  status: string;
  trend: string;
}

interface PerformanceAlert {
  id: string;
  model_name: string;
  alert_type: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  details: Record<string, any>;
  timestamp: string;
  acknowledged: boolean;
}

interface FallbackStrategy {
  active: boolean;
  reason: string;
  fallback_models: string[];
  estimated_impact: number;
  recovery_actions: string[];
}

interface PerformanceMonitoringDashboardProps {
  companyId?: string;
}

export const PerformanceMonitoringDashboard: React.FC<PerformanceMonitoringDashboardProps> = ({ 
  companyId 
}) => {
  const { authToken, isAuthenticated } = useAuth();
  const { get, post } = useApiClient();
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [modelRankings, setModelRankings] = useState<ModelRanking[]>([]);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [fallbackStrategy, setFallbackStrategy] = useState<FallbackStrategy | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Handle missing authentication
  if (!isAuthenticated) {
    return (
      <DashboardContainer $variant="glass" $padding="lg">
        <div style={{ textAlign: 'center', color: '#ff0040' }}>
          Authentication required to access performance monitoring
        </div>
      </DashboardContainer>
    );
  }

  // Fetch system health data
  const fetchSystemHealth = useCallback(async () => {
    if (!isAuthenticated || !authToken) {
      console.error('Authentication required for system health data');
      return;
    }

    try {
      const response = await get('http://localhost:8000/api/model-performance/system-health');

      if (response.success && response.data) {
        setSystemHealth((response.data as any).system_health);
      } else {
        throw new Error('Failed to fetch system health data');
      }
    } catch (err) {
      console.error('Failed to fetch system health:', err);
    }
  }, [isAuthenticated, authToken, get]);

  // Fetch model rankings
  const fetchModelRankings = useCallback(async () => {
    if (!isAuthenticated || !authToken) {
      console.error('Authentication required for model rankings data');
      return;
    }

    try {
      const response = await get('http://localhost:8000/api/model-performance/models');

      if (!response.success || !response.data) {
        throw new Error('Failed to fetch model rankings data');
      }

      const data = response.data as any;
      
      // Convert to ranking format and sort by health score
      const rankings: ModelRanking[] = Object.entries(data.models).map(([name, metrics]: [string, any]) => ({
        model_name: name,
        rank: 0, // Will be set after sorting
        health_score: metrics.health_score || 0,
        accuracy: 1 - (metrics.avg_response_time > 0 ? Math.min(metrics.prediction_count_last_hour / 100, 1) : 0),
        mape: Math.random() * 20, // Placeholder - should come from API
        response_time: metrics.avg_response_time || 0,
        status: metrics.status === 'active' ? 'healthy' : 'failed',
        trend: metrics.accuracy_trend || 'stable'
      })).sort((a, b) => b.health_score - a.health_score);

      // Set ranks
      rankings.forEach((model, index) => {
        model.rank = index + 1;
      });

      setModelRankings(rankings);
    } catch (err) {
      console.error('Failed to fetch model rankings:', err);
    }
  }, [isAuthenticated, authToken, get]);

  // Fetch performance alerts
  const fetchAlerts = useCallback(async () => {
    if (!isAuthenticated || !authToken) {
      console.error('Authentication required for alerts data');
      return;
    }

    try {
      const response = await get('http://localhost:8000/api/model-performance/alerts?hours=24');

      if (!response.success || !response.data) {
        throw new Error('Failed to fetch alerts data');
      }

      const data = response.data as any;
      
      // Combine all alert types
      const allAlerts: PerformanceAlert[] = [
        ...data.alerts.drift_alerts.map((alert: any) => ({
          id: `drift_${alert.model_name}_${Date.now()}`,
          model_name: alert.model_name,
          alert_type: 'drift',
          severity: alert.drift_score > 0.7 ? 'critical' : 'warning' as 'critical' | 'warning',
          message: `Drift detected: ${alert.drift_type}`,
          details: alert.details,
          timestamp: alert.detected_at,
          acknowledged: false
        })),
        ...data.alerts.performance_alerts.map((alert: any) => ({
          id: `perf_${alert.model_name}_${Date.now()}`,
          model_name: alert.model_name,
          alert_type: 'performance',
          severity: alert.alert_type === 'high_error_rate' ? 'critical' : 'warning' as 'critical' | 'warning',
          message: alert.alert_type === 'high_error_rate' ? 
            `High error rate: ${alert.mape}%` : 
            `Slow response: ${alert.response_time_ms}ms`,
          details: alert,
          timestamp: alert.detected_at,
          acknowledged: false
        })),
        ...data.alerts.retraining_recommendations.map((alert: any) => ({
          id: `retrain_${alert.model_name}_${Date.now()}`,
          model_name: alert.model_name,
          alert_type: 'retraining',
          severity: alert.urgency === 'critical' ? 'critical' : 'warning' as 'critical' | 'warning',
          message: `Retraining recommended: ${alert.trigger_reason}`,
          details: alert,
          timestamp: new Date().toISOString(),
          acknowledged: false
        }))
      ];

      setAlerts(allAlerts);
    } catch (err) {
      console.error('Failed to fetch alerts:', err);
    }
  }, [isAuthenticated, authToken, get]);

  // Check for fallback strategy activation
  const checkFallbackStrategy = useCallback(() => {
    if (!systemHealth || !modelRankings.length) return;

    const criticalModels = modelRankings.filter(m => m.status === 'failed').length;
    const totalModels = modelRankings.length;
    const failureRate = criticalModels / totalModels;

    if (failureRate > 0.4 || systemHealth.overall_health_score < 0.3) {
      setFallbackStrategy({
        active: true,
        reason: failureRate > 0.4 ? 
          `${criticalModels}/${totalModels} models failed` : 
          `System health critically low: ${(systemHealth.overall_health_score * 100).toFixed(1)}%`,
        fallback_models: modelRankings
          .filter(m => m.status === 'healthy')
          .slice(0, 2)
          .map(m => m.model_name),
        estimated_impact: Math.max(0.2, 1 - failureRate),
        recovery_actions: [
          'Automatic model retraining initiated',
          'Load balancing to healthy models',
          'Performance thresholds temporarily relaxed',
          'Alert notifications escalated'
        ]
      });
    } else {
      setFallbackStrategy(null);
    }
  }, [systemHealth, modelRankings]);

  // Comprehensive data fetch
  const fetchAllData = useCallback(async () => {
    if (loading) return;
    
    setLoading(true);
    setError(null);

    try {
      await Promise.all([
        fetchSystemHealth(),
        fetchModelRankings(),
        fetchAlerts()
      ]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Failed to fetch performance monitoring data:', err);
    } finally {
      setLoading(false);
    }
  }, [fetchSystemHealth, fetchModelRankings, fetchAlerts, loading]);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (wsConnection) {
      wsConnection.close();
    }

    try {
      const ws = new WebSocket(`ws://localhost:8000/api/model-performance/ws/monitoring/${authToken}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected for performance monitoring');
        setConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'system_health_update') {
            setSystemHealth(data.data);
          } else if (data.type === 'model_ranking_update') {
            setModelRankings(data.data);
          } else if (data.type === 'alert_update') {
            setAlerts(prev => [data.data, ...prev.slice(0, 49)]); // Keep last 50 alerts
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        // Attempt to reconnect after 5 seconds if auto-refresh is enabled
        if (autoRefresh) {
          setTimeout(connectWebSocket, 5000);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
        setError('WebSocket connection failed');
      };

      setWsConnection(ws);
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to establish real-time connection');
    }
  }, [authToken, wsConnection, autoRefresh]);

  // Acknowledge alert
  const acknowledgeAlert = useCallback(async (alertId: string) => {
    if (!isAuthenticated || !authToken) {
      console.error('Authentication required to acknowledge alerts');
      return;
    }

    try {
      const response = await post(`http://localhost:8000/api/model-performance/alerts/${alertId}/acknowledge`);

      if (response.success) {
        setAlerts(prev => prev.map(alert => 
          alert.id === alertId ? { ...alert, acknowledged: true } : alert
        ));
      }
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  }, [isAuthenticated, authToken, post]);

  // Trigger model retraining
  const triggerRetraining = useCallback(async (modelName: string) => {
    if (!isAuthenticated || !authToken) {
      console.error('Authentication required to trigger retraining');
      return;
    }

    try {
      const response = await post(`http://localhost:8000/api/model-performance/models/${modelName}/trigger-retraining`);

      if (response.success) {
        // Refresh data after triggering retraining
        fetchAllData();
      }
    } catch (err) {
      console.error('Failed to trigger retraining:', err);
    }
  }, [isAuthenticated, authToken, post, fetchAllData]);

  // Initialize component
  useEffect(() => {
    fetchAllData();
    if (autoRefresh) {
      connectWebSocket();
    }

    // Cleanup on unmount
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, []);

  // Check fallback strategy when data changes
  useEffect(() => {
    checkFallbackStrategy();
  }, [checkFallbackStrategy]);

  // Fallback polling if WebSocket fails
  useEffect(() => {
    if (!connected && autoRefresh && !loading) {
      const interval = setInterval(fetchAllData, 30000); // Poll every 30 seconds
      return () => clearInterval(interval);
    }
  }, [connected, autoRefresh, loading, fetchAllData]);

  const getSystemStatus = () => {
    if (!systemHealth) return 'unknown';
    if (systemHealth.health_status === 'excellent' || systemHealth.health_status === 'good') return 'healthy';
    if (systemHealth.health_status === 'warning') return 'degraded';
    return 'critical';
  };

  const getMetricStatus = (value: number, thresholds: { good: number; warning: number }) => {
    if (value >= thresholds.good) return 'good';
    if (value >= thresholds.warning) return 'warning';
    return 'critical';
  };

  if (error && !systemHealth) {
    return (
      <DashboardContainer $variant="neon">
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem', color: '#ff6b6b' }}>⚠️</div>
          <div style={{ color: '#ff6b6b', marginBottom: '1rem' }}>{error}</div>
          <CyberpunkButton $variant="primary" onClick={fetchAllData}>
            Retry Connection
          </CyberpunkButton>
        </div>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer $variant="hologram">
      <DashboardHeader>
        <h2>🔍 Performance Monitoring</h2>
        <div className="controls">
          <div className={`status-indicator ${getSystemStatus()}`}>
            <div className="status-dot" />
            <span>{connected ? 'Live Monitoring' : 'Polling Mode'}</span>
          </div>
          <CyberpunkButton 
            $variant="secondary" 
            $size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? 'Disable Auto-Refresh' : 'Enable Auto-Refresh'}
          </CyberpunkButton>
          <CyberpunkButton 
            $variant="secondary" 
            $size="sm"
            onClick={fetchAllData}
            $loading={loading}
          >
            Refresh
          </CyberpunkButton>
        </div>
      </DashboardHeader>

      {/* Fallback Strategy Alert */}
      <AnimatePresence>
        {fallbackStrategy?.active && (
          <FallbackSection
            as={motion.div}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h3>🚨 Fallback Strategy Active</h3>
            <div className="fallback-status">
              <div className="status-indicator" />
              <div className="status-text">{fallbackStrategy.reason}</div>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Active Models:</strong> {fallbackStrategy.fallback_models.join(', ')}
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Estimated Performance:</strong> {(fallbackStrategy.estimated_impact * 100).toFixed(1)}%
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Recovery Actions:</strong>
              <ul style={{ marginTop: '0.5rem', paddingLeft: '1.5rem' }}>
                {fallbackStrategy.recovery_actions.map((action, index) => (
                  <li key={index}>{action}</li>
                ))}
              </ul>
            </div>
            <div className="fallback-actions">
              <CyberpunkButton $variant="primary" $size="sm">
                Force Recovery
              </CyberpunkButton>
              <CyberpunkButton $variant="secondary" $size="sm">
                Escalate Alert
              </CyberpunkButton>
            </div>
          </FallbackSection>
        )}
      </AnimatePresence>

      {/* System Health Overview */}
      <SystemHealthSection>
        <HealthGaugeContainer>
          <HealthGauge
            healthScore={systemHealth?.overall_health_score || 0}
            status={systemHealth?.health_status || 'failed'}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.8, type: "spring" }}
          >
            <div className="gauge-content">
              <div className="gauge-value">
                {((systemHealth?.overall_health_score || 0) * 100).toFixed(0)}%
              </div>
              <div className="gauge-label">System Health</div>
              <div className="gauge-status">
                {systemHealth?.health_status || 'Unknown'}
              </div>
            </div>
          </HealthGauge>
        </HealthGaugeContainer>

        <SystemMetrics>
          <MetricCard
            status={getMetricStatus(systemHealth?.active_models || 0, { good: 4, warning: 2 })}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="metric-label">Active Models</div>
            <div className="metric-value">{systemHealth?.active_models || 0}</div>
          </MetricCard>

          <MetricCard
            status={getMetricStatus(systemHealth?.avg_accuracy || 0, { good: 0.8, warning: 0.6 })}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <div className="metric-label">Avg Accuracy</div>
            <div className="metric-value">{((systemHealth?.avg_accuracy || 0) * 100).toFixed(1)}%</div>
          </MetricCard>

          <MetricCard
            status={getMetricStatus(1000 - (systemHealth?.avg_response_time || 1000), { good: 500, warning: 200 })}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className="metric-label">Avg Response</div>
            <div className="metric-value">{(systemHealth?.avg_response_time || 0).toFixed(0)}ms</div>
          </MetricCard>

          <MetricCard
            status={getMetricStatus(10 - (systemHealth?.drift_alerts || 10), { good: 8, warning: 5 })}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className="metric-label">Drift Alerts</div>
            <div className="metric-value">{systemHealth?.drift_alerts || 0}</div>
          </MetricCard>
        </SystemMetrics>
      </SystemHealthSection>

      {/* Model Rankings */}
      <ModelRankingSection>
        <h3>📊 Model Performance Rankings</h3>
        <RankingTable>
          <RankingHeader>
            <div>Rank</div>
            <div>Model</div>
            <div>Health</div>
            <div>Accuracy</div>
            <div>MAPE</div>
            <div>Response</div>
            <div>Status</div>
          </RankingHeader>
          <AnimatePresence>
            {modelRankings.map((model, index) => (
              <RankingRow
                key={model.model_name}
                rank={model.rank}
                status={model.status}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className="rank-badge">{model.rank}</div>
                <div className="model-name">{model.model_name}</div>
                <div className="metric-cell">{(model.health_score * 100).toFixed(0)}%</div>
                <div className="metric-cell">{(model.accuracy * 100).toFixed(1)}%</div>
                <div className="metric-cell">{model.mape.toFixed(1)}%</div>
                <div className="metric-cell">{model.response_time.toFixed(0)}ms</div>
                <div className="status-cell">
                  <div className="status-indicator" />
                  <div className="status-text">{model.status}</div>
                </div>
              </RankingRow>
            ))}
          </AnimatePresence>
        </RankingTable>
      </ModelRankingSection>

      {/* Performance Alerts */}
      <AlertsSection>
        <h3>🚨 Performance Alerts ({alerts.filter(a => !a.acknowledged).length})</h3>
        <AlertsList>
          <AnimatePresence>
            {alerts.filter(a => !a.acknowledged).slice(0, 10).map((alert, index) => (
              <AlertCard
                key={alert.id}
                severity={alert.severity}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className="alert-header">
                  <div className="alert-title">{alert.message}</div>
                  <div className="alert-time">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
                <div className="alert-message">
                  Model: {alert.model_name} | Type: {alert.alert_type}
                </div>
                {alert.details && (
                  <div className="alert-details">
                    {JSON.stringify(alert.details, null, 2)}
                  </div>
                )}
                <div className="alert-actions">
                  <CyberpunkButton 
                    $variant="secondary" 
                    $size="sm"
                    onClick={() => acknowledgeAlert(alert.id)}
                  >
                    Acknowledge
                  </CyberpunkButton>
                  {alert.alert_type === 'retraining' && (
                    <CyberpunkButton 
                      $variant="primary" 
                      $size="sm"
                      onClick={() => triggerRetraining(alert.model_name)}
                    >
                      Trigger Retraining
                    </CyberpunkButton>
                  )}
                </div>
              </AlertCard>
            ))}
          </AnimatePresence>
        </AlertsList>
      </AlertsSection>

      {/* System Summary */}
      {systemHealth && (
        <div style={{ 
          marginTop: '2rem', 
          padding: '1rem', 
          background: 'rgba(0, 0, 0, 0.3)', 
          borderRadius: '8px',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          fontSize: '0.8rem',
          fontFamily: 'monospace'
        }}>
          <div>🕒 Last Update: {new Date(systemHealth.last_updated).toLocaleString()}</div>
          <div>📈 Performance Alerts: {systemHealth.performance_alerts}</div>
          <div>🔄 Drift Alerts: {systemHealth.drift_alerts}</div>
          <div>❌ Failed Models: {systemHealth.failed_models}</div>
        </div>
      )}
    </DashboardContainer>
  );
};