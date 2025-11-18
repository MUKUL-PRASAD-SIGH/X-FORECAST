/**
 * Performance Monitor Component
 * Displays real-time performance statistics and optimization controls
 */

import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { usePerformanceOptimization, PerformanceStats, PerformanceSettings } from '../../hooks/usePerformanceOptimization';

interface PerformanceMonitorProps {
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  minimized?: boolean;
  showControls?: boolean;
}

const MonitorContainer = styled(motion.div)<{ position: string }>`
  position: fixed;
  ${props => {
    switch (props.position) {
      case 'top-left': return 'top: 20px; left: 20px;';
      case 'top-right': return 'top: 20px; right: 20px;';
      case 'bottom-left': return 'bottom: 20px; left: 20px;';
      case 'bottom-right': return 'bottom: 20px; right: 20px;';
      default: return 'top: 20px; right: 20px;';
    }
  }}
  z-index: 1000;
  background: rgba(0, 0, 0, 0.9);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 8px;
  padding: 15px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #00ffff;
  backdrop-filter: blur(10px);
  min-width: 200px;
  max-width: 300px;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(0, 255, 255, 0.2);
`;

const Title = styled.h3`
  margin: 0;
  font-size: 14px;
  color: #00ffff;
  text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
`;

const ToggleButton = styled.button`
  background: transparent;
  border: 1px solid rgba(0, 255, 255, 0.3);
  color: #00ffff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: #ff1493;
    color: #ff1493;
    box-shadow: 0 0 5px rgba(255, 20, 147, 0.3);
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 15px;
`;

const StatItem = styled.div<{ warning?: boolean; critical?: boolean }>`
  display: flex;
  flex-direction: column;
  padding: 6px;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 4px;
  border: 1px solid ${props => 
    props.critical ? 'rgba(255, 0, 0, 0.5)' :
    props.warning ? 'rgba(255, 255, 0, 0.5)' :
    'rgba(0, 255, 255, 0.2)'
  };
`;

const StatLabel = styled.span`
  font-size: 10px;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 2px;
`;

const StatValue = styled.span<{ warning?: boolean; critical?: boolean }>`
  font-size: 12px;
  font-weight: bold;
  color: ${props => 
    props.critical ? '#ff0000' :
    props.warning ? '#ffff00' :
    '#00ffff'
  };
  text-shadow: 0 0 3px currentColor;
`;

const PerformanceTier = styled.div<{ tier: string }>`
  text-align: center;
  padding: 8px;
  margin-bottom: 10px;
  background: ${props => 
    props.tier === 'high' ? 'rgba(0, 255, 0, 0.1)' :
    props.tier === 'medium' ? 'rgba(255, 255, 0, 0.1)' :
    'rgba(255, 0, 0, 0.1)'
  };
  border: 1px solid ${props => 
    props.tier === 'high' ? 'rgba(0, 255, 0, 0.3)' :
    props.tier === 'medium' ? 'rgba(255, 255, 0, 0.3)' :
    'rgba(255, 0, 0, 0.3)'
  };
  border-radius: 4px;
  color: ${props => 
    props.tier === 'high' ? '#00ff00' :
    props.tier === 'medium' ? '#ffff00' :
    '#ff0000'
  };
  font-weight: bold;
  text-transform: uppercase;
`;

const ControlsSection = styled.div`
  border-top: 1px solid rgba(0, 255, 255, 0.2);
  padding-top: 10px;
`;

const ControlGroup = styled.div`
  margin-bottom: 8px;
`;

const ControlLabel = styled.label`
  display: block;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 4px;
`;

const Checkbox = styled.input`
  margin-right: 6px;
  accent-color: #00ffff;
`;

const Slider = styled.input`
  width: 100%;
  margin-top: 4px;
  accent-color: #00ffff;
`;

const MinimizedView = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
`;

const FPSIndicator = styled.div<{ fps: number }>`
  font-weight: bold;
  color: ${props => 
    props.fps >= 55 ? '#00ff00' :
    props.fps >= 30 ? '#ffff00' :
    '#ff0000'
  };
  text-shadow: 0 0 3px currentColor;
`;

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  position = 'top-right',
  minimized: initialMinimized = false,
  showControls = true
}) => {
  const [minimized, setMinimized] = useState(initialMinimized);
  const [showControlsPanel, setShowControlsPanel] = useState(false);
  
  const {
    performanceStats,
    settings,
    performanceTier,
    isInitialized,
    updateSettings
  } = usePerformanceOptimization();

  if (!isInitialized) {
    return null;
  }

  const handleSettingChange = (key: keyof PerformanceSettings, value: any) => {
    updateSettings({ [key]: value });
  };

  const getPerformanceColor = (fps: number) => {
    if (fps >= 55) return '#00ff00';
    if (fps >= 30) return '#ffff00';
    return '#ff0000';
  };

  if (minimized) {
    return (
      <MonitorContainer
        position={position}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <MinimizedView>
          <FPSIndicator fps={performanceStats.fps}>
            {performanceStats.fps} FPS
          </FPSIndicator>
          <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
            {performanceTier.toUpperCase()}
          </span>
          <ToggleButton onClick={() => setMinimized(false)}>
            EXPAND
          </ToggleButton>
        </MinimizedView>
      </MonitorContainer>
    );
  }

  return (
    <MonitorContainer
      position={position}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Header>
        <Title>Performance Monitor</Title>
        <div>
          {showControls && (
            <ToggleButton 
              onClick={() => setShowControlsPanel(!showControlsPanel)}
              style={{ marginRight: '8px' }}
            >
              {showControlsPanel ? 'HIDE' : 'CTRL'}
            </ToggleButton>
          )}
          <ToggleButton onClick={() => setMinimized(true)}>
            MIN
          </ToggleButton>
        </div>
      </Header>

      <PerformanceTier tier={performanceTier}>
        {performanceTier} Performance
      </PerformanceTier>

      <StatsGrid>
        <StatItem critical={performanceStats.fps < 30} warning={performanceStats.fps < 55}>
          <StatLabel>FPS</StatLabel>
          <StatValue 
            critical={performanceStats.fps < 30} 
            warning={performanceStats.fps < 55}
          >
            {performanceStats.fps}
          </StatValue>
        </StatItem>

        <StatItem warning={performanceStats.frameTime > 20}>
          <StatLabel>Frame Time</StatLabel>
          <StatValue warning={performanceStats.frameTime > 20}>
            {performanceStats.frameTime}ms
          </StatValue>
        </StatItem>

        <StatItem warning={performanceStats.memoryUsage > 100}>
          <StatLabel>Memory</StatLabel>
          <StatValue warning={performanceStats.memoryUsage > 100}>
            {performanceStats.memoryUsage}MB
          </StatValue>
        </StatItem>

        <StatItem>
          <StatLabel>Draw Calls</StatLabel>
          <StatValue>{performanceStats.drawCalls}</StatValue>
        </StatItem>

        <StatItem>
          <StatLabel>Triangles</StatLabel>
          <StatValue>{performanceStats.triangles.toLocaleString()}</StatValue>
        </StatItem>

        <StatItem>
          <StatLabel>Particles</StatLabel>
          <StatValue>{performanceStats.particles}</StatValue>
        </StatItem>

        <StatItem>
          <StatLabel>LOD Level</StatLabel>
          <StatValue>{performanceStats.lodLevel.toFixed(1)}</StatValue>
        </StatItem>

        <StatItem>
          <StatLabel>Target FPS</StatLabel>
          <StatValue>{settings.targetFPS}</StatValue>
        </StatItem>
      </StatsGrid>

      <AnimatePresence>
        {showControlsPanel && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <ControlsSection>
              <ControlGroup>
                <ControlLabel>
                  <Checkbox
                    type="checkbox"
                    checked={settings.enableLOD}
                    onChange={(e) => handleSettingChange('enableLOD', e.target.checked)}
                  />
                  Enable LOD Optimization
                </ControlLabel>
              </ControlGroup>

              <ControlGroup>
                <ControlLabel>
                  <Checkbox
                    type="checkbox"
                    checked={settings.enableParticleOptimization}
                    onChange={(e) => handleSettingChange('enableParticleOptimization', e.target.checked)}
                  />
                  Particle Optimization
                </ControlLabel>
              </ControlGroup>

              <ControlGroup>
                <ControlLabel>
                  <Checkbox
                    type="checkbox"
                    checked={settings.enableShaderOptimization}
                    onChange={(e) => handleSettingChange('enableShaderOptimization', e.target.checked)}
                  />
                  Shader Optimization
                </ControlLabel>
              </ControlGroup>

              <ControlGroup>
                <ControlLabel>
                  <Checkbox
                    type="checkbox"
                    checked={settings.adaptiveQuality}
                    onChange={(e) => handleSettingChange('adaptiveQuality', e.target.checked)}
                  />
                  Adaptive Quality
                </ControlLabel>
              </ControlGroup>

              <ControlGroup>
                <ControlLabel>
                  Max Particles: {settings.maxParticles}
                  <Slider
                    type="range"
                    min="100"
                    max="5000"
                    step="100"
                    value={settings.maxParticles}
                    onChange={(e) => handleSettingChange('maxParticles', parseInt(e.target.value))}
                  />
                </ControlLabel>
              </ControlGroup>

              <ControlGroup>
                <ControlLabel>
                  Target FPS: {settings.targetFPS}
                  <Slider
                    type="range"
                    min="30"
                    max="120"
                    step="10"
                    value={settings.targetFPS}
                    onChange={(e) => handleSettingChange('targetFPS', parseInt(e.target.value))}
                  />
                </ControlLabel>
              </ControlGroup>
            </ControlsSection>
          </motion.div>
        )}
      </AnimatePresence>
    </MonitorContainer>
  );
};

export default PerformanceMonitor;