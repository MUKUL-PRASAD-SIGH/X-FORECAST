import React, { useState, useEffect, useRef } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard } from '../ui';

// Animation keyframes
const scanLineMove = keyframes`
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
`;

const dataFlow = keyframes`
  0% { transform: translateX(-100%); opacity: 0; }
  50% { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
`;

const matrixRain = keyframes`
  0% { transform: translateY(-100%); opacity: 1; }
  100% { transform: translateY(100vh); opacity: 0; }
`;

const pulseGlow = keyframes`
  0%, 100% { 
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
  }
  50% { 
    box-shadow: 0 0 40px rgba(0, 255, 255, 0.8);
  }
`;

// Styled components
const DashboardContainer = styled(CyberpunkCard)`
  position: relative;
  min-height: 400px;
  overflow: hidden;
  background: linear-gradient(135deg, 
    rgba(0, 0, 0, 0.9) 0%, 
    rgba(10, 10, 30, 0.9) 50%, 
    rgba(0, 0, 0, 0.9) 100%
  );
  border: 2px solid ${props => props.theme.colors.neonBlue};
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 20%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 80% 80%, rgba(255, 20, 147, 0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: 1;
  }
`;

const ScanLineOverlay = styled.div<{ active: boolean }>`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 10;
  
  ${props => props.active && css`
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, 
        transparent 0%, 
        ${props.theme.colors.neonBlue} 20%, 
        ${props.theme.colors.hotPink} 50%, 
        ${props.theme.colors.acidGreen} 80%, 
        transparent 100%
      );
      animation: ${scanLineMove} 3s linear infinite;
      box-shadow: 0 0 20px ${props.theme.colors.neonBlue};
    }
    
    &::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, 
        transparent 0%, 
        ${props.theme.colors.cyberYellow} 50%, 
        transparent 100%
      );
      animation: ${scanLineMove} 2s linear infinite reverse;
      animation-delay: 1s;
    }
  `}
`;

const MatrixRainEffect = styled.div<{ active: boolean }>`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 2;
  overflow: hidden;
  
  ${props => props.active && css`
    .matrix-column {
      position: absolute;
      top: -100%;
      font-family: ${props.theme.typography.fontFamily.mono};
      font-size: 12px;
      color: ${props.theme.colors.acidGreen};
      opacity: 0.3;
      animation: ${matrixRain} 4s linear infinite;
      text-shadow: 0 0 5px ${props.theme.colors.acidGreen};
    }
  `}
`;

const DataFlowLines = styled.div<{ active: boolean }>`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 3;
  
  ${props => props.active && css`
    .data-line {
      position: absolute;
      height: 1px;
      background: linear-gradient(90deg, 
        transparent 0%, 
        ${props.theme.colors.neonBlue} 50%, 
        transparent 100%
      );
      animation: ${dataFlow} 2s ease-in-out infinite;
      box-shadow: 0 0 10px ${props.theme.colors.neonBlue};
    }
  `}
`;

const DashboardHeader = styled.div`
  position: relative;
  z-index: 5;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid rgba(0, 255, 255, 0.3);
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(10px);
`;

const DashboardTitle = styled.h3`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  color: ${props => props.theme.colors.neonBlue};
  margin: 0;
  text-shadow: ${props => props.theme.effects.softGlow};
  text-transform: uppercase;
  letter-spacing: 2px;
`;

const StatusIndicators = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const StatusDot = styled.div<{ 
  status: 'active' | 'warning' | 'error' | 'idle';
  animated?: boolean;
}>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: ${props => {
    switch (props.status) {
      case 'active': return props.theme.colors.acidGreen;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'error': return props.theme.colors.error;
      case 'idle': return props.theme.colors.secondaryText;
      default: return props.theme.colors.secondaryText;
    }
  }};
  box-shadow: 0 0 10px ${props => {
    switch (props.status) {
      case 'active': return props.theme.colors.acidGreen;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'error': return props.theme.colors.error;
      case 'idle': return props.theme.colors.secondaryText;
      default: return props.theme.colors.secondaryText;
    }
  }};
  
  ${props => props.animated && css`
    animation: ${pulseGlow} 2s infinite;
  `}
`;

const StatusLabel = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const DashboardContent = styled.div`
  position: relative;
  z-index: 5;
  padding: 1.5rem;
  min-height: 300px;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const MetricCard = styled(motion.div)<{ status: 'good' | 'warning' | 'critical' }>`
  background: rgba(0, 0, 0, 0.6);
  border: 1px solid ${props => {
    switch (props.status) {
      case 'good': return props.theme.colors.acidGreen;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'critical': return props.theme.colors.error;
      default: return props.theme.colors.neonBlue;
    }
  }};
  border-radius: 6px;
  padding: 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(5px);
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
      transparent 0%, 
      rgba(255, 255, 255, 0.1) 50%, 
      transparent 100%
    );
    transition: left 0.5s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
  
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
    margin-bottom: 0.25rem;
  }
  
  .metric-trend {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
  }
`;

const ActivityFeed = styled.div`
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 6px;
  padding: 1rem;
  max-height: 200px;
  overflow-y: auto;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
  }
  
  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.neonBlue};
    border-radius: 3px;
  }
`;

const ActivityItem = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(0, 255, 255, 0.1);
  
  &:last-child {
    border-bottom: none;
  }
  
  .activity-time {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
    min-width: 60px;
  }
  
  .activity-message {
    font-family: ${props => props.theme.typography.fontFamily.primary};
    font-size: ${props => props.theme.typography.fontSize.sm};
    color: ${props => props.theme.colors.primaryText};
    flex: 1;
  }
`;

const ControlPanel = styled.div`
  position: absolute;
  top: 1rem;
  right: 1rem;
  display: flex;
  gap: 0.5rem;
  z-index: 15;
`;

const EffectToggle = styled.button<{ active: boolean }>`
  width: 32px;
  height: 32px;
  border: 1px solid ${props => props.active ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.5)'};
  color: ${props => props.active ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.7rem;
  transition: all 0.3s ease;
  backdrop-filter: blur(5px);
  
  &:hover {
    border-color: ${props => props.theme.colors.neonBlue};
    color: ${props => props.theme.colors.neonBlue};
    background: rgba(0, 255, 255, 0.1);
  }
`;

// Interfaces
export interface DashboardMetric {
  label: string;
  value: string | number;
  trend?: string;
  status: 'good' | 'warning' | 'critical';
}

export interface ActivityItem {
  id: string;
  timestamp: string;
  message: string;
}

export interface RealTimeDashboardProps {
  title: string;
  status: 'active' | 'warning' | 'error' | 'idle';
  metrics: DashboardMetric[];
  activities: ActivityItem[];
  onRefresh?: () => void;
  refreshInterval?: number;
  enableEffects?: boolean;
}

export const RealTimeDashboard: React.FC<RealTimeDashboardProps> = ({
  title,
  status,
  metrics,
  activities,
  onRefresh,
  refreshInterval = 5000,
  enableEffects = true
}) => {
  const [scanLineActive, setScanLineActive] = useState(enableEffects);
  const [matrixActive, setMatrixActive] = useState(enableEffects);
  const [dataFlowActive, setDataFlowActive] = useState(enableEffects);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const matrixRef = useRef<HTMLDivElement>(null);

  // Auto-refresh functionality
  useEffect(() => {
    if (onRefresh && refreshInterval > 0) {
      const interval = setInterval(() => {
        onRefresh();
        setLastUpdate(new Date());
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [onRefresh, refreshInterval]);

  // Matrix rain effect
  useEffect(() => {
    if (!matrixActive || !matrixRef.current) return;

    const container = matrixRef.current;
    const characters = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
    
    const createMatrixColumn = () => {
      const column = document.createElement('div');
      column.className = 'matrix-column';
      column.style.left = Math.random() * 100 + '%';
      column.style.animationDelay = Math.random() * 2 + 's';
      column.style.animationDuration = (Math.random() * 3 + 2) + 's';
      
      let text = '';
      for (let i = 0; i < Math.floor(Math.random() * 20) + 10; i++) {
        text += characters[Math.floor(Math.random() * characters.length)] + '\n';
      }
      column.textContent = text;
      
      container.appendChild(column);
      
      setTimeout(() => {
        if (container.contains(column)) {
          container.removeChild(column);
        }
      }, 5000);
    };

    const interval = setInterval(createMatrixColumn, 200);
    return () => clearInterval(interval);
  }, [matrixActive]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString();
  };

  return (
    <DashboardContainer variant="hologram">
      <ScanLineOverlay active={scanLineActive} />
      <MatrixRainEffect ref={matrixRef} active={matrixActive} />
      <DataFlowLines active={dataFlowActive}>
        {Array.from({ length: 5 }, (_, i) => (
          <div
            key={i}
            className="data-line"
            style={{
              top: `${20 + i * 15}%`,
              width: '100%',
              animationDelay: `${i * 0.4}s`,
              animationDuration: `${2 + i * 0.3}s`
            }}
          />
        ))}
      </DataFlowLines>

      <ControlPanel>
        <EffectToggle
          active={scanLineActive}
          onClick={() => setScanLineActive(!scanLineActive)}
          title="Toggle scan lines"
        >
          📡
        </EffectToggle>
        <EffectToggle
          active={matrixActive}
          onClick={() => setMatrixActive(!matrixActive)}
          title="Toggle matrix rain"
        >
          🌧
        </EffectToggle>
        <EffectToggle
          active={dataFlowActive}
          onClick={() => setDataFlowActive(!dataFlowActive)}
          title="Toggle data flow"
        >
          ⚡
        </EffectToggle>
      </ControlPanel>

      <DashboardHeader>
        <DashboardTitle>{title}</DashboardTitle>
        <StatusIndicators>
          <StatusDot status={status} animated={status === 'active'} />
          <StatusLabel>{status.toUpperCase()}</StatusLabel>
          <StatusLabel>|</StatusLabel>
          <StatusLabel>Last Update: {formatTime(lastUpdate)}</StatusLabel>
        </StatusIndicators>
      </DashboardHeader>

      <DashboardContent>
        <MetricsGrid>
          {metrics.map((metric, index) => (
            <MetricCard
              key={index}
              status={metric.status}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="metric-label">{metric.label}</div>
              <div className="metric-value">{metric.value}</div>
              {metric.trend && (
                <div className="metric-trend">{metric.trend}</div>
              )}
            </MetricCard>
          ))}
        </MetricsGrid>

        <ActivityFeed>
          <h4 style={{ 
            margin: '0 0 1rem 0', 
            color: '#00FFFF', 
            fontSize: '0.9rem',
            textTransform: 'uppercase',
            letterSpacing: '1px'
          }}>
            Real-time Activity
          </h4>
          <AnimatePresence>
            {activities.slice(0, 10).map((activity, index) => (
              <ActivityItem
                key={activity.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className="activity-time">
                  {new Date(activity.timestamp).toLocaleTimeString()}
                </div>
                <div className="activity-message">
                  {activity.message}
                </div>
              </ActivityItem>
            ))}
          </AnimatePresence>
        </ActivityFeed>
      </DashboardContent>
    </DashboardContainer>
  );
};

export default RealTimeDashboard;