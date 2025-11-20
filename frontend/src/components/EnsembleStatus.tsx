import React, { useState, useEffect } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';
import { useAuth } from '../contexts/AuthContext';
import { useApiClient } from '../hooks/useApiClient';

const EnsembleContainer = styled(CyberpunkCard)`
  padding: 2rem;
  margin: 1rem 0;
`;

const ModelsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
`;

const ModelCard = styled(motion.div)<{ status: 'initialized' | 'training' | 'error' | 'inactive' }>`
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid ${props => {
    switch (props.status) {
      case 'initialized': return props.theme.colors.acidGreen;
      case 'training': return props.theme.colors.neonBlue;
      case 'error': return props.theme.colors.error;
      default: return props.theme.colors.secondaryText;
    }
  }};
  border-radius: 8px;
  padding: 1rem;
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
        case 'initialized': return props.theme.colors.acidGreen;
        case 'training': return props.theme.colors.neonBlue;
        case 'error': return props.theme.colors.error;
        default: return props.theme.colors.secondaryText;
      }
    }};
    ${props => props.status === 'training' && css`
      animation: scan 2s infinite;
    `}
  }
  
  @keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
  }
`;

const ModelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  
  .model-name {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-weight: bold;
    color: ${props => props.theme.colors.neonBlue};
  }
  
  .model-status {
    font-size: 0.8rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    
    &.initialized {
      background: rgba(0, 255, 127, 0.2);
      color: ${props => props.theme.colors.acidGreen};
    }
    
    &.training {
      background: rgba(0, 212, 255, 0.2);
      color: ${props => props.theme.colors.neonBlue};
    }
    
    &.error {
      background: rgba(255, 107, 107, 0.2);
      color: ${props => props.theme.colors.error};
    }
    
    &.inactive {
      background: rgba(128, 128, 128, 0.2);
      color: ${props => props.theme.colors.secondaryText};
    }
  }
`;

const ModelDescription = styled.div`
  font-size: 0.8rem;
  color: ${props => props.theme.colors.secondaryText};
  margin-bottom: 0.5rem;
  line-height: 1.4;
`;

const ModelMetrics = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  
  .metric {
    text-align: center;
    
    .metric-label {
      color: ${props => props.theme.colors.secondaryText};
      display: block;
    }
    
    .metric-value {
      color: ${props => props.theme.colors.neonBlue};
      font-weight: bold;
      display: block;
      margin-top: 0.2rem;
    }
  }
`;

const EnsembleHealthGauge = styled.div<{ health: number }>`
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid ${props => 
    props.health >= 0.8 ? props.theme.colors.acidGreen :
    props.health >= 0.6 ? props.theme.colors.neonBlue :
    props.theme.colors.error
  };
  border-radius: 8px;
  
  .health-label {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    color: ${props => props.theme.colors.primaryText};
    margin-bottom: 0.5rem;
    text-align: center;
  }
  
  .health-bar {
    width: 100%;
    height: 12px;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
    
    .health-fill {
      height: 100%;
      background: ${props => 
        props.health >= 0.8 ? props.theme.colors.acidGreen :
        props.health >= 0.6 ? props.theme.colors.neonBlue :
        props.theme.colors.error
      };
      width: ${props => props.health * 100}%;
      transition: width 0.5s ease;
      box-shadow: 0 0 10px currentColor;
      position: relative;
      
      &::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 4px;
        height: 100%;
        background: rgba(255, 255, 255, 0.8);
        animation: pulse 2s infinite;
      }
    }
  }
  
  .health-score {
    text-align: center;
    margin-top: 0.5rem;
    font-family: ${props => props.theme.typography.fontFamily.mono};
    color: ${props => 
      props.health >= 0.8 ? props.theme.colors.acidGreen :
      props.health >= 0.6 ? props.theme.colors.neonBlue :
      props.theme.colors.error
    };
    font-weight: bold;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;

const WeightDistribution = styled.div`
  margin-top: 1rem;
  
  .weight-title {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    color: ${props => props.theme.colors.hotPink};
    margin-bottom: 0.5rem;
    text-align: center;
  }
  
  .weight-bars {
    display: flex;
    height: 20px;
    border-radius: 10px;
    overflow: hidden;
    background: rgba(0, 0, 0, 0.5);
  }
`;

const WeightBar = styled.div<{ weight: number; color: string }>`
  width: ${props => props.weight * 100}%;
  background: ${props => props.color};
  transition: width 0.5s ease;
  position: relative;
  
  &:hover::after {
    content: '${props => (props.weight * 100).toFixed(1)}%';
    position: absolute;
    top: -25px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    white-space: nowrap;
  }
`;

interface ModelConfig {
  model_type: string;
  description: string;
  best_for: string;
  parameters: any;
  status: string;
}

interface EnsembleStatusData {
  initialized: boolean;
  ensemble_health: number;
  total_models: number;
  active_models: string[];
  model_weights: Record<string, number>;
  model_configurations: Record<string, ModelConfig>;
  pattern_detected: string;
  pattern_confidence: number;
  recent_performance: Record<string, any>;
  initialized_date: string;
  last_update_date: string;
  total_forecasts: number;
  adaptive_features: Record<string, boolean>;
}

interface EnsembleStatusProps {
  authToken?: string; // Make optional since we'll use AuthContext
  companyId?: string;
}

export const EnsembleStatus: React.FC<EnsembleStatusProps> = ({ authToken: propAuthToken, companyId }) => {
  const { isAuthenticated } = useAuth();
  const { get, post } = useApiClient();
  const [status, setStatus] = useState<EnsembleStatusData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const modelColors = {
    arima: '#00d4ff',
    ets: '#ff1493',
    xgboost: '#39ff14',
    lstm: '#ff6b35',
    croston: '#9d4edd'
  };

  const fetchEnsembleStatus = async () => {
    if (!isAuthenticated) {
      setError('Authentication required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await get<EnsembleStatusData>('/api/company-sales/ensemble/status');
      
      if (response.success && response.data) {
        setStatus(response.data);
      } else {
        setError(response.error || 'Failed to fetch ensemble status');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const initializeEnsemble = async () => {
    if (!isAuthenticated) {
      setError('Authentication required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await post<{ success: boolean; status?: EnsembleStatusData; message?: string }>('/api/company-sales/ensemble/initialize');
      
      if (response.success && response.data?.success) {
        setStatus(response.data.status!);
      } else {
        setError(response.error || response.data?.message || 'Initialization failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Initialization failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (propAuthToken) {
      fetchEnsembleStatus();
      
      // Auto-refresh every 30 seconds
      const interval = setInterval(fetchEnsembleStatus, 30000);
      return () => clearInterval(interval);
    }
  }, [propAuthToken]);

  if (loading && !status) {
    return (
      <EnsembleContainer variant="neon">
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚡</div>
          <div>Loading ensemble status...</div>
        </div>
      </EnsembleContainer>
    );
  }

  if (error && !status) {
    return (
      <EnsembleContainer variant="neon">
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem', color: '#ff6b6b' }}>❌</div>
          <div style={{ color: '#ff6b6b', marginBottom: '1rem' }}>{error}</div>
          <CyberpunkButton $variant="primary" onClick={fetchEnsembleStatus}>
            Retry
          </CyberpunkButton>
        </div>
      </EnsembleContainer>
    );
  }

  if (!status?.initialized) {
    return (
      <EnsembleContainer variant="neon">
        <motion.h3 
          style={{ color: '#00d4ff', marginBottom: '1rem', textAlign: 'center' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          🤖 Ensemble Models Not Initialized
        </motion.h3>
        
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>🚀</div>
          <div style={{ marginBottom: '1rem', color: '#888' }}>
            Initialize the 5-model ensemble system (ARIMA, ETS, XGBoost, LSTM, Croston) 
            to start adaptive forecasting with real-time weight updates.
          </div>
          <CyberpunkButton 
            $variant="primary" 
            onClick={initializeEnsemble}
            $loading={loading}
          >
            Initialize Ensemble Models
          </CyberpunkButton>
        </div>
      </EnsembleContainer>
    );
  }

  return (
    <EnsembleContainer variant="neon">
      <motion.h3 
        style={{ color: '#00d4ff', marginBottom: '1rem', textAlign: 'center' }}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        🤖 Ensemble Model Status
      </motion.h3>

      {/* Ensemble Health Gauge */}
      <EnsembleHealthGauge health={status.ensemble_health}>
        <div className="health-label">Ensemble Health Score</div>
        <div className="health-bar">
          <div className="health-fill" />
        </div>
        <div className="health-score">
          {(status.ensemble_health * 100).toFixed(1)}% • Pattern: {status.pattern_detected}
        </div>
      </EnsembleHealthGauge>

      {/* Weight Distribution */}
      <WeightDistribution>
        <div className="weight-title">⚖️ Model Weight Distribution</div>
        <div className="weight-bars">
          {Object.entries(status.model_weights).map(([model, weight]) => (
            <WeightBar 
              key={model}
              weight={weight}
              color={modelColors[model as keyof typeof modelColors] || '#888'}
            />
          ))}
        </div>
      </WeightDistribution>

      {/* Individual Model Cards */}
      <ModelsGrid>
        <AnimatePresence>
          {Object.entries(status.model_configurations).map(([modelName, config]) => {
            const performance = status.recent_performance[modelName];
            const weight = status.model_weights[modelName] || 0;
            
            return (
              <ModelCard
                key={modelName}
                status="initialized"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <ModelHeader>
                  <div className="model-name">{config.model_type}</div>
                  <div className="model-status initialized">Active</div>
                </ModelHeader>
                
                <ModelDescription>
                  {config.description}
                  <br />
                  <strong>Best for:</strong> {config.best_for}
                </ModelDescription>
                
                <ModelMetrics>
                  <div className="metric">
                    <span className="metric-label">Weight</span>
                    <span className="metric-value">{(weight * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">MAPE</span>
                    <span className="metric-value">
                      {performance?.mape ? `${performance.mape.toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">MAE</span>
                    <span className="metric-value">
                      {performance?.mae ? performance.mae.toFixed(0) : 'N/A'}
                    </span>
                  </div>
                </ModelMetrics>
              </ModelCard>
            );
          })}
        </AnimatePresence>
      </ModelsGrid>

      {/* Status Summary */}
      <div style={{ 
        marginTop: '1rem', 
        padding: '1rem', 
        background: 'rgba(0, 0, 0, 0.3)', 
        borderRadius: '8px',
        fontSize: '0.8rem',
        fontFamily: 'monospace'
      }}>
        <div>📊 Total Forecasts: {status.total_forecasts}</div>
        <div>🕒 Last Update: {new Date(status.last_update_date).toLocaleString()}</div>
        <div>🎯 Pattern Confidence: {(status.pattern_confidence * 100).toFixed(1)}%</div>
        <div>⚡ Adaptive Features: Weight Updates, Pattern Detection, Performance Monitoring</div>
      </div>

      <div style={{ textAlign: 'center', marginTop: '1rem' }}>
        <CyberpunkButton 
          $variant="secondary" 
          $size="sm"
          onClick={fetchEnsembleStatus}
          $loading={loading}
        >
          Refresh Status
        </CyberpunkButton>
      </div>
    </EnsembleContainer>
  );
};