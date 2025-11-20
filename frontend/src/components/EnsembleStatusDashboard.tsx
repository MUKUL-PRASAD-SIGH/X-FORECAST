import React, { useState, useEffect, useCallback, useRef } from 'react';
import styled, { keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';

// Types
interface ModelStatus {
  model_name: string;
  status: 'pending' | 'initializing' | 'training' | 'validating' | 'completed' | 'failed' | 'swapping';
  progress: number;
  weight: number;
  performance?: {
    mae: number;
    mape: number;
    rmse: number;
    r_squared: number;
  };
}

interface EnsembleStatus {
  session_id: string;
  timestamp: string;
  total_models: number;
  active_models: number;
  failed_models: number;
  overall_health: number;
  ensemble_accuracy: number;
  pattern_type: string;
  pattern_confidence: number;
  model_weights: Record<string, number>;
  model_performances: Record<string, any>;
  system_metrics: {
    system_load: number;
    memory_usage: number;
    processing_speed: number;
  };
}

interface InitializationProgress {
  session_id: string;
  stage: string;
  overall_progress: number;
  stage_progress: number;
  current_operation: string;
  models_status: Record<string, string>;
  models_progress: Record<string, number>;
  pattern_analysis?: {
    pattern_type: string;
    seasonality_strength: number;
    trend_strength: number;
    intermittency_ratio: number;
    volatility: number;
    confidence: number;
  };
  weights: Record<string, number>;
  performance_metrics: Record<string, any>;
  errors: string[];
  warnings: string[];
}

// Animations
const pulseGlow = keyframes`
  0% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.3); }
  50% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.8), 0 0 30px rgba(255, 20, 147, 0.4); }
  100% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.3); }
`;

const dataFlow = keyframes`
  0% { transform: translateX(-100%) scaleX(0); }
  50% { transform: translateX(0%) scaleX(1); }
  100% { transform: translateX(100%) scaleX(0); }
`;

const scanLine = keyframes`
  0% { left: -100%; }
  100% { left: 100%; }
`;

const matrixRain = keyframes`
  0% { transform: translateY(-100%); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateY(100vh); opacity: 0; }
`;

// Styled Components
const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto auto;
  gap: 1.5rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
  min-height: 100vh;
  color: #00ffff;
  font-family: 'Orbitron', monospace;
`;

const StatusCard = styled(CyberpunkCard)`
  position: relative;
  overflow: hidden;
  background: rgba(0, 20, 40, 0.8);
  border: 1px solid rgba(0, 255, 255, 0.3);
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
    animation: ${scanLine} 3s infinite;
  }
  
  &.critical {
    border-color: rgba(255, 20, 147, 0.6);
    animation: ${pulseGlow} 2s infinite;
  }
`;

const ModelGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  grid-column: 1 / -1;
`;

const ModelCard = styled(motion.div)<{ status: string }>`
  background: rgba(0, 20, 40, 0.9);
  border: 1px solid ${props => {
    switch (props.status) {
      case 'completed': return 'rgba(0, 255, 0, 0.6)';
      case 'failed': return 'rgba(255, 0, 0, 0.6)';
      case 'training': return 'rgba(255, 165, 0, 0.6)';
      case 'swapping': return 'rgba(255, 20, 147, 0.6)';
      default: return 'rgba(0, 255, 255, 0.3)';
    }
  }};
  border-radius: 8px;
  padding: 1rem;
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: ${props => {
      switch (props.status) {
        case 'completed': return 'linear-gradient(90deg, #00ff00, #00ffff)';
        case 'failed': return 'linear-gradient(90deg, #ff0000, #ff6b6b)';
        case 'training': return 'linear-gradient(90deg, #ffa500, #ffff00)';
        case 'swapping': return 'linear-gradient(90deg, #ff1493, #ff69b4)';
        default: return 'linear-gradient(90deg, #00ffff, #0080ff)';
      }
    }};
    animation: ${dataFlow} 2s infinite;
  }
`;

const ProgressBar = styled.div<{ progress: number; status: string }>`
  width: 100%;
  height: 8px;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 4px;
  overflow: hidden;
  margin: 0.5rem 0;
  
  &::after {
    content: '';
    display: block;
    width: ${props => props.progress}%;
    height: 100%;
    background: ${props => {
      switch (props.status) {
        case 'completed': return 'linear-gradient(90deg, #00ff00, #00ffff)';
        case 'failed': return 'linear-gradient(90deg, #ff0000, #ff6b6b)';
        case 'training': return 'linear-gradient(90deg, #ffa500, #ffff00)';
        default: return 'linear-gradient(90deg, #00ffff, #0080ff)';
      }
    }};
    transition: width 0.3s ease;
    animation: ${pulseGlow} 1.5s infinite;
  }
`;

const MetricDisplay = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  
  .label {
    color: rgba(0, 255, 255, 0.8);
  }
  
  .value {
    color: #00ffff;
    font-weight: bold;
  }
`;

const PatternIndicator = styled.div<{ patternType: string }>`
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: bold;
  background: ${props => {
    switch (props.patternType) {
      case 'seasonal': return 'rgba(0, 255, 0, 0.2)';
      case 'trending': return 'rgba(255, 165, 0, 0.2)';
      case 'intermittent': return 'rgba(255, 20, 147, 0.2)';
      default: return 'rgba(0, 255, 255, 0.2)';
    }
  }};
  color: ${props => {
    switch (props.patternType) {
      case 'seasonal': return '#00ff00';
      case 'trending': return '#ffa500';
      case 'intermittent': return '#ff1493';
      default: return '#00ffff';
    }
  }};
  border: 1px solid currentColor;
`;

const SystemMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1rem;
`;

const MetricGauge = styled.div<{ value: number; critical?: boolean }>`
  text-align: center;
  
  .gauge-container {
    position: relative;
    width: 80px;
    height: 80px;
    margin: 0 auto 0.5rem;
  }
  
  .gauge-bg {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: conic-gradient(
      from 0deg,
      rgba(0, 255, 255, 0.1) 0deg,
      rgba(0, 255, 255, 0.1) ${props => props.value * 3.6}deg,
      rgba(0, 0, 0, 0.3) ${props => props.value * 3.6}deg,
      rgba(0, 0, 0, 0.3) 360deg
    );
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid ${props => props.critical ? '#ff1493' : '#00ffff'};
  }
  
  .gauge-value {
    font-size: 1.2rem;
    font-weight: bold;
    color: ${props => props.critical ? '#ff1493' : '#00ffff'};
  }
  
  .gauge-label {
    font-size: 0.8rem;
    color: rgba(0, 255, 255, 0.8);
  }
`;

const ErrorList = styled.div`
  max-height: 150px;
  overflow-y: auto;
  
  .error-item {
    background: rgba(255, 0, 0, 0.1);
    border: 1px solid rgba(255, 0, 0, 0.3);
    border-radius: 4px;
    padding: 0.5rem;
    margin: 0.25rem 0;
    font-size: 0.8rem;
    color: #ff6b6b;
  }
  
  .warning-item {
    background: rgba(255, 165, 0, 0.1);
    border: 1px solid rgba(255, 165, 0, 0.3);
    border-radius: 4px;
    padding: 0.5rem;
    margin: 0.25rem 0;
    font-size: 0.8rem;
    color: #ffa500;
  }
`;

const SwapControls = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-top: 1rem;
  
  select {
    background: rgba(0, 20, 40, 0.8);
    border: 1px solid rgba(0, 255, 255, 0.3);
    color: #00ffff;
    padding: 0.5rem;
    border-radius: 4px;
    font-family: inherit;
  }
`;

interface EnsembleStatusDashboardProps {
  sessionId?: string;
  onModelSwap?: (oldModel: string, newModel: string) => void;
}

const EnsembleStatusDashboard: React.FC<EnsembleStatusDashboardProps> = ({
  sessionId,
  onModelSwap
}) => {
  const [progress, setProgress] = useState<InitializationProgress | null>(null);
  const [status, setStatus] = useState<EnsembleStatus | null>(null);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedOldModel, setSelectedOldModel] = useState('');
  const [selectedNewModel, setSelectedNewModel] = useState('');
  const [availableModels] = useState(['arima', 'ets', 'xgboost', 'lstm', 'croston', 'prophet', 'catboost']);
  
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connectWebSocket = useCallback(() => {
    if (!sessionId) return;

    try {
      const wsUrl = `ws://localhost:8000/api/v1/live-ensemble/ws/${sessionId}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setWebsocket(ws);
        
        // Request initial status
        ws.send(JSON.stringify({ type: 'request_status' }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'initialization_progress') {
            setProgress(data);
          } else if (data.type === 'ensemble_status') {
            setStatus(data);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setWebsocket(null);
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }

    return () => {
      if (websocket) {
        websocket.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [sessionId, connectWebSocket]);

  const handleModelSwap = async () => {
    if (!selectedOldModel || !selectedNewModel || !onModelSwap) return;
    
    try {
      await onModelSwap(selectedOldModel, selectedNewModel);
      setSelectedOldModel('');
      setSelectedNewModel('');
    } catch (error) {
      console.error('Model swap failed:', error);
    }
  };

  const getModelCards = () => {
    if (!progress) return [];

    return Object.entries(progress.models_status).map(([modelName, modelStatus]) => {
      const modelProgress = progress.models_progress[modelName] || 0;
      const weight = progress.weights[modelName] || 0;
      const performance = progress.performance_metrics[modelName];

      return (
        <ModelCard
          key={modelName}
          status={modelStatus}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h4 style={{ margin: '0 0 0.5rem 0', textTransform: 'uppercase' }}>
            {modelName}
          </h4>
          
          <MetricDisplay>
            <span className="label">Status:</span>
            <span className="value">{modelStatus}</span>
          </MetricDisplay>
          
          <ProgressBar progress={modelProgress} status={modelStatus} />
          
          <MetricDisplay>
            <span className="label">Weight:</span>
            <span className="value">{(weight * 100).toFixed(1)}%</span>
          </MetricDisplay>
          
          {performance && (
            <>
              <MetricDisplay>
                <span className="label">MAPE:</span>
                <span className="value">{performance.mape?.toFixed(2)}%</span>
              </MetricDisplay>
              <MetricDisplay>
                <span className="label">R²:</span>
                <span className="value">{performance.r_squared?.toFixed(3)}</span>
              </MetricDisplay>
            </>
          )}
        </ModelCard>
      );
    });
  };

  if (!sessionId) {
    return (
      <DashboardContainer>
        <StatusCard>
          <h3>No Active Session</h3>
          <p>Please start an ensemble initialization to view the dashboard.</p>
        </StatusCard>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      {/* Overall Progress */}
      <StatusCard className={!isConnected ? 'critical' : ''}>
        <h3>Ensemble Initialization Progress</h3>
        {progress && (
          <>
            <MetricDisplay>
              <span className="label">Stage:</span>
              <span className="value">{progress.stage.replace('_', ' ').toUpperCase()}</span>
            </MetricDisplay>
            
            <ProgressBar 
              progress={progress.overall_progress} 
              status={progress.stage}
            />
            
            <MetricDisplay>
              <span className="label">Progress:</span>
              <span className="value">{progress.overall_progress.toFixed(1)}%</span>
            </MetricDisplay>
            
            <p style={{ fontSize: '0.9rem', margin: '0.5rem 0' }}>
              {progress.current_operation}
            </p>
            
            {progress.pattern_analysis && (
              <div style={{ marginTop: '1rem' }}>
                <PatternIndicator patternType={progress.pattern_analysis.pattern_type}>
                  {progress.pattern_analysis.pattern_type.toUpperCase()} PATTERN
                </PatternIndicator>
                <MetricDisplay>
                  <span className="label">Confidence:</span>
                  <span className="value">{(progress.pattern_analysis.confidence * 100).toFixed(1)}%</span>
                </MetricDisplay>
              </div>
            )}
          </>
        )}
        
        <MetricDisplay>
          <span className="label">Connection:</span>
          <span className="value" style={{ color: isConnected ? '#00ff00' : '#ff0000' }}>
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </MetricDisplay>
      </StatusCard>

      {/* System Health */}
      <StatusCard>
        <h3>System Health</h3>
        {status && (
          <>
            <MetricDisplay>
              <span className="label">Overall Health:</span>
              <span className="value">{(status.overall_health * 100).toFixed(1)}%</span>
            </MetricDisplay>
            
            <MetricDisplay>
              <span className="label">Ensemble Accuracy:</span>
              <span className="value">{(status.ensemble_accuracy * 100).toFixed(1)}%</span>
            </MetricDisplay>
            
            <MetricDisplay>
              <span className="label">Active Models:</span>
              <span className="value">{status.active_models}/{status.total_models}</span>
            </MetricDisplay>
            
            <SystemMetrics>
              <MetricGauge 
                value={status.system_metrics.system_load * 100}
                critical={status.system_metrics.system_load > 0.8}
              >
                <div className="gauge-container">
                  <div className="gauge-bg">
                    <div className="gauge-value">
                      {(status.system_metrics.system_load * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
                <div className="gauge-label">CPU Load</div>
              </MetricGauge>
              
              <MetricGauge 
                value={status.system_metrics.memory_usage * 100}
                critical={status.system_metrics.memory_usage > 0.8}
              >
                <div className="gauge-container">
                  <div className="gauge-bg">
                    <div className="gauge-value">
                      {(status.system_metrics.memory_usage * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
                <div className="gauge-label">Memory</div>
              </MetricGauge>
              
              <MetricGauge 
                value={status.system_metrics.processing_speed * 100}
                critical={status.system_metrics.processing_speed < 0.5}
              >
                <div className="gauge-container">
                  <div className="gauge-bg">
                    <div className="gauge-value">
                      {(status.system_metrics.processing_speed * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
                <div className="gauge-label">Speed</div>
              </MetricGauge>
            </SystemMetrics>
          </>
        )}
      </StatusCard>

      {/* Model Status Grid */}
      <ModelGrid>
        <AnimatePresence>
          {getModelCards()}
        </AnimatePresence>
      </ModelGrid>

      {/* Hot Model Swapping */}
      {progress && progress.stage === 'completed' && (
        <StatusCard>
          <h3>Hot Model Swapping</h3>
          <p>Replace models without stopping the ensemble</p>
          
          <SwapControls>
            <div>
              <label>Remove Model:</label>
              <select 
                value={selectedOldModel} 
                onChange={(e) => setSelectedOldModel(e.target.value)}
              >
                <option value="">Select model to remove</option>
                {Object.keys(progress.models_status).map(model => (
                  <option key={model} value={model}>{model.toUpperCase()}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label>Add Model:</label>
              <select 
                value={selectedNewModel} 
                onChange={(e) => setSelectedNewModel(e.target.value)}
              >
                <option value="">Select model to add</option>
                {availableModels
                  .filter(model => !Object.keys(progress.models_status).includes(model))
                  .map(model => (
                    <option key={model} value={model}>{model.toUpperCase()}</option>
                  ))}
              </select>
            </div>
            
            <CyberpunkButton
              onClick={handleModelSwap}
              disabled={!selectedOldModel || !selectedNewModel}
            >
              SWAP MODELS
            </CyberpunkButton>
          </SwapControls>
        </StatusCard>
      )}

      {/* Errors and Warnings */}
      {progress && (progress.errors.length > 0 || progress.warnings.length > 0) && (
        <StatusCard className="critical">
          <h3>System Messages</h3>
          <ErrorList>
            {progress.errors.map((error, index) => (
              <div key={`error-${index}`} className="error-item">
                ERROR: {error}
              </div>
            ))}
            {progress.warnings.map((warning, index) => (
              <div key={`warning-${index}`} className="warning-item">
                WARNING: {warning}
              </div>
            ))}
          </ErrorList>
        </StatusCard>
      )}
    </DashboardContainer>
  );
};

export default EnsembleStatusDashboard;