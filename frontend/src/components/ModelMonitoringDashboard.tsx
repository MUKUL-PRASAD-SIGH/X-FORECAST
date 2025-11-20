import React, { useState, useEffect, useCallback } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';
import { useAuth } from '../contexts/AuthContext';
import { useApiClient } from '../hooks/useApiClient';
import { ExtendedEnsembleMetrics as EnsembleMetrics } from '../types/fixes';

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

// Styled components
const DashboardContainer = styled(CyberpunkCard)`
  padding: 2rem;
  margin: 1rem 0;
  min-height: 600px;
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
  }
`;

const EnsembleGaugeContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 2rem;
`;

const EnsembleGauge = styled(motion.div)<{ accuracy: number }>`
  position: relative;
  width: 200px;
  height: 200px;
  border-radius: 50%;
  background: conic-gradient(
    from 0deg,
    ${props => props.accuracy >= 0.8 ? props.theme.colors.acidGreen : 
      props.accuracy >= 0.6 ? props.theme.colors.neonBlue : 
      props.theme.colors.error} 0deg,
    ${props => props.accuracy >= 0.8 ? props.theme.colors.acidGreen : 
      props.accuracy >= 0.6 ? props.theme.colors.neonBlue : 
      props.theme.colors.error} ${props => props.accuracy * 360}deg,
    rgba(255, 255, 255, 0.1) ${props => props.accuracy * 360}deg,
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
      color: ${props => props.accuracy >= 0.8 ? props.theme.colors.acidGreen : 
        props.accuracy >= 0.6 ? props.theme.colors.neonBlue : 
        props.theme.colors.error};
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
  }
`;

const ModelsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const ModelCard = styled(motion.div)<{ 
  status: 'healthy' | 'warning' | 'degraded' | 'failed';
  trend: 'improving' | 'stable' | 'declining';
}>`
  background: rgba(0, 0, 0, 0.4);
  border: 2px solid ${props => {
    switch (props.status) {
      case 'healthy': return props.theme.colors.acidGreen;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'degraded': return props.theme.colors.warning;
      case 'failed': return props.theme.colors.error;
      default: return props.theme.colors.secondaryText;
    }
  }};
  border-radius: 12px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 3px;
    background: ${props => {
      switch (props.status) {
        case 'healthy': return props.theme.colors.acidGreen;
        case 'warning': return props.theme.colors.cyberYellow;
        case 'degraded': return props.theme.colors.warning;
        case 'failed': return props.theme.colors.error;
        default: return props.theme.colors.secondaryText;
      }
    }};
    animation: ${scanLine} 3s infinite;
  }
  
  ${props => props.status === 'failed' && css`
    animation: ${glitch} 0.5s infinite;
  `}
`;

const ModelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  
  .model-name {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.lg};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
    color: ${props => props.theme.colors.neonBlue};
    text-transform: uppercase;
  }
  
  .model-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    text-transform: uppercase;
    letter-spacing: 1px;
    
    .status-icon {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      animation: ${pulse} 2s infinite;
    }
    
    &.healthy {
      color: ${props => props.theme.colors.acidGreen};
      .status-icon { background: ${props => props.theme.colors.acidGreen}; }
    }
    
    &.warning {
      color: ${props => props.theme.colors.cyberYellow};
      .status-icon { background: ${props => props.theme.colors.cyberYellow}; }
    }
    
    &.degraded {
      color: ${props => props.theme.colors.warning};
      .status-icon { background: ${props => props.theme.colors.warning}; }
    }
    
    &.failed {
      color: ${props => props.theme.colors.error};
      .status-icon { background: ${props => props.theme.colors.error}; }
    }
  }
`;

const ModelMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
`;

const MetricItem = styled.div`
  text-align: center;
  
  .metric-label {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem;
  }
  
  .metric-value {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.lg};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
    color: ${props => props.theme.colors.neonBlue};
    text-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const TrendIndicator = styled.div<{ trend: 'improving' | 'stable' | 'declining' }>`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => {
    switch (props.trend) {
      case 'improving': return props.theme.colors.acidGreen;
      case 'declining': return props.theme.colors.error;
      default: return props.theme.colors.secondaryText;
    }
  }};
  
  .trend-icon {
    font-size: ${props => props.theme.typography.fontSize.sm};
  }
`;

const WeightBar = styled.div<{ weight: number; color: string }>`
  width: 100%;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
  margin-top: 0.5rem;
  
  .weight-fill {
    height: 100%;
    width: ${props => props.weight * 100}%;
    background: ${props => props.color};
    border-radius: 4px;
    transition: width 0.5s ease;
    box-shadow: 0 0 10px ${props => props.color};
    position: relative;
    
    &::after {
      content: '';
      position: absolute;
      top: 0;
      right: 0;
      width: 2px;
      height: 100%;
      background: rgba(255, 255, 255, 0.8);
      animation: ${pulse} 2s infinite;
    }
  }
`;

const RefreshButton = styled(CyberpunkButton)`
  margin-left: 1rem;
`;

// Interfaces
interface ModelPerformance {
  model_name: string;
  accuracy: number;
  mape: number;
  mae: number;
  rmse: number;
  weight: number;
  last_updated: string;
  trend: 'improving' | 'stable' | 'declining';
  status: 'healthy' | 'warning' | 'degraded' | 'failed';
  prediction_count: number;
  error_rate: number;
}

// EnsembleMetrics interface is imported from types

interface ModelMonitoringDashboardProps {
  companyId?: string;
}

export const ModelMonitoringDashboard: React.FC<ModelMonitoringDashboardProps> = ({ 
  companyId 
}) => {
  const { authToken, isAuthenticated } = useAuth();
  const { get, post } = useApiClient();
  const [metrics, setMetrics] = useState<EnsembleMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);

  // Model colors for consistent styling
  const modelColors = {
    arima: '#00d4ff',
    ets: '#ff1493',
    xgboost: '#39ff14',
    lstm: '#ff6b35',
    croston: '#9d4edd'
  };

  // Handle missing authentication
  if (!isAuthenticated) {
    return (
      <DashboardContainer $variant="glass" $padding="lg">
        <div style={{ textAlign: 'center', color: '#ff0040' }}>
          Authentication required to access model monitoring
        </div>
      </DashboardContainer>
    );
  }

  // Fetch performance metrics
  const fetchMetrics = useCallback(async () => {
    if (loading) return;
    
    setLoading(true);
    setError(null);

    if (!isAuthenticated || !authToken) {
      setError('Authentication required for model monitoring data');
      setLoading(false);
      return;
    }

    try {
      const response = await get('http://localhost:8000/api/company-sales/ensemble/performance');

      if (response.success && response.data) {
        setMetrics(response.data as EnsembleMetrics);
      } else {
        throw new Error('Failed to fetch ensemble performance metrics');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Failed to fetch ensemble performance metrics:', err);
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated, authToken, get, loading]);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (wsConnection) {
      wsConnection.close();
    }

    if (!authToken) {
      console.log('No auth token, skipping WebSocket connection');
      return;
    }

    try {
      const ws = new WebSocket(`ws://localhost:8000/api/company-sales/ws/ensemble-performance/${authToken}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        setConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMetrics(data);
        } catch (err) {
          console.error('WebSocket message parse error:', err);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code);
        setConnected(false);
        
        if (event.code !== 1000 && event.code !== 1001) {
          setTimeout(connectWebSocket, 5000);
        }
      };

      ws.onerror = () => {
        console.log('WebSocket error - using polling mode');
        setConnected(false);
        setError(null);
      };

      setWsConnection(ws);
    } catch (err) {
      console.log('WebSocket creation failed - using polling mode');
      setError(null);
    }
  }, [authToken]);

  // Initialize component
  useEffect(() => {
    fetchMetrics();
    
    // Try WebSocket first, but don't block if it fails
    const wsTimeout = setTimeout(() => {
      if (!connected) {
        console.log('WebSocket connection timeout, using polling mode');
        setError('Using polling mode for updates');
      }
    }, 3000);
    
    connectWebSocket();

    // Cleanup on unmount
    return () => {
      clearTimeout(wsTimeout);
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, []);

  // Fallback polling if WebSocket fails
  useEffect(() => {
    if (!connected && !loading) {
      const interval = setInterval(() => {
        fetchMetrics();
      }, 5000); // Poll every 5 seconds when WebSocket is down
      return () => clearInterval(interval);
    }
  }, [connected, loading, fetchMetrics]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return '↗️';
      case 'declining': return '↘️';
      default: return '→';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'healthy': return 'Optimal';
      case 'warning': return 'Warning';
      case 'degraded': return 'Degraded';
      case 'failed': return 'Failed';
      default: return 'Unknown';
    }
  };

  if (error && !metrics) {
    return (
      <DashboardContainer $variant="neon">
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem', color: '#ff6b6b' }}>⚠️</div>
          <div style={{ color: '#ff6b6b', marginBottom: '1rem' }}>{error}</div>
          <CyberpunkButton $variant="primary" onClick={fetchMetrics}>
            Retry Connection
          </CyberpunkButton>
        </div>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer $variant="hologram">
      <DashboardHeader>
        <h2>🤖 Model Performance Monitor</h2>
        <div className="status-indicator">
          <div className="status-dot" style={{
            background: connected ? '#39ff14' : '#ff6b35',
            boxShadow: connected ? '0 0 10px #39ff14' : '0 0 10px #ff6b35'
          }} />
          <span>{connected ? 'Live Updates' : 'Polling Mode'}</span>
          <RefreshButton 
            $variant="secondary" 
            $size="sm"
            onClick={fetchMetrics}
            $loading={loading}
          >
            Refresh
          </RefreshButton>
        </div>
      </DashboardHeader>

      {/* Ensemble Accuracy Gauge */}
      <EnsembleGaugeContainer>
        <EnsembleGauge
          accuracy={metrics?.overall_accuracy || 0}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.8, type: "spring" }}
        >
          <div className="gauge-content">
            <div className="gauge-value">
              {((metrics?.overall_accuracy || 0) * 100).toFixed(1)}%
            </div>
            <div className="gauge-label">Ensemble Accuracy</div>
          </div>
        </EnsembleGauge>
      </EnsembleGaugeContainer>

      {/* Individual Model Performance Cards */}
      <ModelsGrid>
        <AnimatePresence>
          {metrics?.model_performances?.map((model, index) => (
            <ModelCard
              key={model.model_name}
              status={model.status}
              trend={model.trend}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <ModelHeader>
                <div className="model-name">{model.model_name}</div>
                <div className={`model-status ${model.status}`}>
                  <div className="status-icon" />
                  {getStatusText(model.status)}
                </div>
              </ModelHeader>

              <ModelMetrics>
                <MetricItem>
                  <div className="metric-label">Accuracy</div>
                  <div className="metric-value">{(model.accuracy * 100).toFixed(1)}%</div>
                </MetricItem>
                <MetricItem>
                  <div className="metric-label">MAPE</div>
                  <div className="metric-value">{model.mape.toFixed(1)}%</div>
                </MetricItem>
                <MetricItem>
                  <div className="metric-label">MAE</div>
                  <div className="metric-value">{model.mae.toFixed(0)}</div>
                </MetricItem>
                <MetricItem>
                  <div className="metric-label">Weight</div>
                  <div className="metric-value">{(model.weight * 100).toFixed(1)}%</div>
                </MetricItem>
              </ModelMetrics>

              <WeightBar 
                weight={model.weight} 
                color={modelColors[model.model_name as keyof typeof modelColors] || '#888'}
              >
                <div className="weight-fill" />
              </WeightBar>

              <TrendIndicator trend={model.trend}>
                <span className="trend-icon">{getTrendIcon(model.trend)}</span>
                <span>{model.trend.charAt(0).toUpperCase() + model.trend.slice(1)}</span>
              </TrendIndicator>
            </ModelCard>
          ))}
        </AnimatePresence>
      </ModelsGrid>

      {/* System Summary */}
      {metrics && (
        <div style={{ 
          marginTop: '2rem', 
          padding: '1rem', 
          background: 'rgba(0, 0, 0, 0.3)', 
          borderRadius: '8px',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          fontSize: '0.8rem',
          fontFamily: 'monospace'
        }}>
          <div>📊 Total Predictions: {metrics.total_predictions?.toLocaleString() || 'N/A'}</div>
          <div>🎯 Confidence Score: {((metrics.confidence_score || 0) * 100).toFixed(1)}%</div>
          <div>⚡ System Health: {((metrics.system_health || 0) * 100).toFixed(0)}%</div>
          <div>🕒 Last Update: {metrics.last_updated ? new Date(metrics.last_updated).toLocaleString() : 'N/A'}</div>
        </div>
      )}
    </DashboardContainer>
  );
};