import React, { useState, useEffect, useCallback } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from '../ui';

// Animation keyframes
const pulseGlow = keyframes`
  0%, 100% { 
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.4);
  }
  50% { 
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
  }
`;

const slideGlow = keyframes`
  0% { left: 0%; }
  100% { left: 100%; }
`;

const weightPulse = keyframes`
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
`;

// Styled components
const WeightControlContainer = styled(CyberpunkCard)`
  padding: 2rem;
  background: linear-gradient(135deg, 
    rgba(0, 0, 0, 0.9) 0%, 
    rgba(20, 20, 40, 0.9) 50%, 
    rgba(0, 0, 0, 0.9) 100%
  );
  border: 2px solid ${props => props.theme.colors.neonBlue};
  position: relative;
  overflow: hidden;
`;

const ControlHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  
  h3 {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.lg};
    color: ${props => props.theme.colors.neonBlue};
    margin: 0;
    text-shadow: ${props => props.theme.effects.softGlow};
    text-transform: uppercase;
    letter-spacing: 2px;
  }
`;

const ModeSelector = styled.div`
  display: flex;
  gap: 0.5rem;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 6px;
  padding: 0.25rem;
  border: 1px solid rgba(0, 255, 255, 0.3);
`;

const ModeButton = styled.button<{ active: boolean }>`
  padding: 0.5rem 1rem;
  border: none;
  background: ${props => props.active ? props.theme.colors.neonBlue : 'transparent'};
  color: ${props => props.active ? props.theme.colors.darkBg : props.theme.colors.secondaryText};
  border-radius: 4px;
  cursor: pointer;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.active ? props.theme.colors.neonBlue : 'rgba(0, 255, 255, 0.2)'};
    color: ${props => props.active ? props.theme.colors.darkBg : props.theme.colors.neonBlue};
  }
`;

const WeightGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const ModelWeightCard = styled(motion.div)<{ 
  isActive: boolean;
  performance: 'excellent' | 'good' | 'average' | 'poor';
}>`
  background: rgba(0, 0, 0, 0.6);
  border: 2px solid ${props => {
    switch (props.performance) {
      case 'excellent': return props.theme.colors.acidGreen;
      case 'good': return props.theme.colors.neonBlue;
      case 'average': return props.theme.colors.cyberYellow;
      case 'poor': return props.theme.colors.error;
      default: return props.theme.colors.secondaryText;
    }
  }};
  border-radius: 8px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  
  ${props => props.isActive && css`
    animation: ${pulseGlow} 2s infinite;
  `}
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: ${props => {
      switch (props.performance) {
        case 'excellent': return props.theme.colors.acidGreen;
        case 'good': return props.theme.colors.neonBlue;
        case 'average': return props.theme.colors.cyberYellow;
        case 'poor': return props.theme.colors.error;
        default: return props.theme.colors.secondaryText;
      }
    }};
    box-shadow: 0 0 10px ${props => {
      switch (props.performance) {
        case 'excellent': return props.theme.colors.acidGreen;
        case 'good': return props.theme.colors.neonBlue;
        case 'average': return props.theme.colors.cyberYellow;
        case 'poor': return props.theme.colors.error;
        default: return props.theme.colors.secondaryText;
      }
    }};
  }
`;

const ModelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const ModelName = styled.h4`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.md};
  color: ${props => props.theme.colors.primaryText};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const PerformanceBadge = styled.div<{ performance: 'excellent' | 'good' | 'average' | 'poor' }>`
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  background: ${props => {
    switch (props.performance) {
      case 'excellent': return props.theme.colors.acidGreen;
      case 'good': return props.theme.colors.neonBlue;
      case 'average': return props.theme.colors.cyberYellow;
      case 'poor': return props.theme.colors.error;
      default: return props.theme.colors.secondaryText;
    }
  }};
  color: ${props => props.theme.colors.darkBg};
`;

const WeightSliderContainer = styled.div`
  margin: 1rem 0;
`;

const WeightLabel = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  
  .label {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.sm};
    color: ${props => props.theme.colors.secondaryText};
    text-transform: uppercase;
  }
  
  .value {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.md};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
    color: ${props => props.theme.colors.neonBlue};
    text-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const SliderTrack = styled.div`
  position: relative;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid rgba(0, 255, 255, 0.3);
`;

const SliderFill = styled.div<{ percentage: number; color: string }>`
  height: 100%;
  width: ${props => props.percentage}%;
  background: linear-gradient(90deg, 
    ${props => props.color} 0%, 
    ${props => props.color}80 100%
  );
  border-radius: 4px;
  position: relative;
  transition: width 0.3s ease;
  box-shadow: 0 0 10px ${props => props.color}40;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 4px;
    height: 100%;
    background: ${props => props.color};
    box-shadow: 0 0 8px ${props => props.color};
    animation: ${weightPulse} 2s infinite;
  }
`;

const SliderInput = styled.input`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
  
  &:disabled {
    cursor: not-allowed;
  }
`;

const ModelMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  margin-top: 1rem;
`;

const MetricItem = styled.div`
  text-align: center;
  
  .metric-label {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }
  
  .metric-value {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.sm};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
    color: ${props => props.theme.colors.primaryText};
  }
`;

const ControlActions = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(0, 255, 255, 0.3);
`;

const WeightSummary = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.secondaryText};
  
  .total-weight {
    color: ${props => props.theme.colors.neonBlue};
    font-weight: ${props => props.theme.typography.fontWeight.bold};
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 1rem;
`;

// Interfaces
export interface ModelWeight {
  modelName: string;
  weight: number;
  performance: 'excellent' | 'good' | 'average' | 'poor';
  accuracy: number;
  mape: number;
  responseTime: number;
  isActive: boolean;
}

export interface ModelWeightAdjustmentProps {
  models: ModelWeight[];
  mode: 'manual' | 'auto' | 'hybrid';
  onWeightChange: (modelName: string, weight: number) => void;
  onModeChange: (mode: 'manual' | 'auto' | 'hybrid') => void;
  onApplyWeights: (weights: Record<string, number>) => void;
  onResetWeights: () => void;
  onOptimizeWeights: () => void;
  disabled?: boolean;
}

export const ModelWeightAdjustment: React.FC<ModelWeightAdjustmentProps> = ({
  models,
  mode,
  onWeightChange,
  onModeChange,
  onApplyWeights,
  onResetWeights,
  onOptimizeWeights,
  disabled = false
}) => {
  const [localWeights, setLocalWeights] = useState<Record<string, number>>({});
  const [hasChanges, setHasChanges] = useState(false);

  // Initialize local weights
  useEffect(() => {
    const weights: Record<string, number> = {};
    models.forEach(model => {
      weights[model.modelName] = model.weight;
    });
    setLocalWeights(weights);
  }, [models]);

  // Check for changes
  useEffect(() => {
    const hasChanged = models.some(model => 
      localWeights[model.modelName] !== model.weight
    );
    setHasChanges(hasChanged);
  }, [localWeights, models]);

  const handleWeightChange = useCallback((modelName: string, newWeight: number) => {
    if (disabled || mode === 'auto') return;
    
    setLocalWeights(prev => ({
      ...prev,
      [modelName]: newWeight
    }));
    
    onWeightChange(modelName, newWeight);
  }, [disabled, mode, onWeightChange]);

  const handleApplyWeights = () => {
    onApplyWeights(localWeights);
    setHasChanges(false);
  };

  const handleResetWeights = () => {
    const originalWeights: Record<string, number> = {};
    models.forEach(model => {
      originalWeights[model.modelName] = model.weight;
    });
    setLocalWeights(originalWeights);
    setHasChanges(false);
    onResetWeights();
  };

  const getTotalWeight = () => {
    return Object.values(localWeights).reduce((sum, weight) => sum + weight, 0);
  };

  const getSliderColor = (performance: string) => {
    switch (performance) {
      case 'excellent': return '#39FF14';
      case 'good': return '#00FFFF';
      case 'average': return '#FFFF00';
      case 'poor': return '#FF0040';
      default: return '#B0B0B0';
    }
  };

  const formatPercentage = (value: number) => {
    return (value * 100).toFixed(1) + '%';
  };

  return (
    <WeightControlContainer variant="hologram">
      <ControlHeader>
        <h3>🎛️ Model Weight Control</h3>
        <ModeSelector>
          <ModeButton
            active={mode === 'manual'}
            onClick={() => onModeChange('manual')}
            disabled={disabled}
          >
            Manual
          </ModeButton>
          <ModeButton
            active={mode === 'auto'}
            onClick={() => onModeChange('auto')}
            disabled={disabled}
          >
            Auto
          </ModeButton>
          <ModeButton
            active={mode === 'hybrid'}
            onClick={() => onModeChange('hybrid')}
            disabled={disabled}
          >
            Hybrid
          </ModeButton>
        </ModeSelector>
      </ControlHeader>

      <WeightGrid>
        <AnimatePresence>
          {models.map((model, index) => (
            <ModelWeightCard
              key={model.modelName}
              isActive={model.isActive}
              performance={model.performance}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <ModelHeader>
                <ModelName>{model.modelName}</ModelName>
                <PerformanceBadge performance={model.performance}>
                  {model.performance}
                </PerformanceBadge>
              </ModelHeader>

              <WeightSliderContainer>
                <WeightLabel>
                  <span className="label">Weight</span>
                  <span className="value">
                    {formatPercentage(localWeights[model.modelName] || 0)}
                  </span>
                </WeightLabel>
                
                <SliderTrack>
                  <SliderFill
                    percentage={(localWeights[model.modelName] || 0) * 100}
                    color={getSliderColor(model.performance)}
                  />
                  <SliderInput
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={localWeights[model.modelName] || 0}
                    onChange={(e) => handleWeightChange(model.modelName, parseFloat(e.target.value))}
                    disabled={disabled || mode === 'auto'}
                  />
                </SliderTrack>
              </WeightSliderContainer>

              <ModelMetrics>
                <MetricItem>
                  <div className="metric-label">Accuracy</div>
                  <div className="metric-value">
                    {formatPercentage(model.accuracy)}
                  </div>
                </MetricItem>
                <MetricItem>
                  <div className="metric-label">MAPE</div>
                  <div className="metric-value">
                    {model.mape.toFixed(1)}%
                  </div>
                </MetricItem>
                <MetricItem>
                  <div className="metric-label">Response</div>
                  <div className="metric-value">
                    {model.responseTime.toFixed(0)}ms
                  </div>
                </MetricItem>
              </ModelMetrics>
            </ModelWeightCard>
          ))}
        </AnimatePresence>
      </WeightGrid>

      <ControlActions>
        <WeightSummary>
          Total Weight: <span className="total-weight">{formatPercentage(getTotalWeight())}</span>
          {getTotalWeight() !== 1 && (
            <div style={{ color: '#FF0040', fontSize: '0.8rem', marginTop: '0.25rem' }}>
              ⚠️ Weights should sum to 100%
            </div>
          )}
        </WeightSummary>

        <ActionButtons>
          <CyberpunkButton
            variant="secondary"
            size="sm"
            onClick={onOptimizeWeights}
            disabled={disabled || mode === 'manual'}
          >
            Auto-Optimize
          </CyberpunkButton>
          
          <CyberpunkButton
            variant="ghost"
            size="sm"
            onClick={handleResetWeights}
            disabled={disabled || !hasChanges}
          >
            Reset
          </CyberpunkButton>
          
          <CyberpunkButton
            variant="primary"
            size="sm"
            onClick={handleApplyWeights}
            disabled={disabled || !hasChanges || getTotalWeight() !== 1}
          >
            Apply Changes
          </CyberpunkButton>
        </ActionButtons>
      </ControlActions>
    </WeightControlContainer>
  );
};

export default ModelWeightAdjustment;