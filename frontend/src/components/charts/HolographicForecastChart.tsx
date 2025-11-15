import React, { useRef, useEffect, useState, useCallback } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

interface TimeSeriesPoint {
  date: string;
  value: number;
}

interface ModelForecast {
  arima: TimeSeriesPoint[];
  ets: TimeSeriesPoint[];
  xgboost: TimeSeriesPoint[];
  lstm: TimeSeriesPoint[];
  croston: TimeSeriesPoint[];
  ensemble: TimeSeriesPoint[];
}

interface ConfidenceIntervals {
  p10: TimeSeriesPoint[];
  p50: TimeSeriesPoint[];
  p90: TimeSeriesPoint[];
}

interface ForecastVisualization {
  historical: TimeSeriesPoint[];
  forecasts: ModelForecast;
  confidenceIntervals: ConfidenceIntervals;
  modelWeights: Record<string, number>;
}

interface HolographicForecastChartProps {
  data: ForecastVisualization;
  showIndividualModels?: boolean;
  showConfidenceIntervals?: boolean;
  width?: number;
  height?: number;
  realTimeAccuracy?: number;
  enableParticleEffects?: boolean;
  enable3DEffects?: boolean;
}

const ChartContainer = styled.div<{ width: number; height: number }>`
  width: ${props => props.width}px;
  height: ${props => props.height}px;
  position: relative;
  background: 
    radial-gradient(circle at 30% 30%, rgba(0, 255, 255, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 70% 70%, rgba(255, 20, 147, 0.06) 0%, transparent 50%),
    linear-gradient(135deg, rgba(57, 255, 20, 0.02) 0%, transparent 100%);
  border: 2px solid transparent;
  border-radius: 12px;
  overflow: hidden;
  backdrop-filter: blur(2px);
  
  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, 
      rgba(0, 255, 255, 0.6) 0%, 
      rgba(255, 20, 147, 0.6) 25%, 
      rgba(57, 255, 20, 0.6) 50%, 
      rgba(255, 20, 147, 0.6) 75%, 
      rgba(0, 255, 255, 0.6) 100%
    );
    border-radius: 12px;
    z-index: -1;
    animation: borderGlow 4s ease-in-out infinite;
  }
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      linear-gradient(90deg, transparent 0%, rgba(0, 255, 255, 0.15) 50%, transparent 100%),
      linear-gradient(0deg, transparent 0%, rgba(255, 20, 147, 0.1) 50%, transparent 100%);
    pointer-events: none;
    animation: hologramScan 3s ease-in-out infinite;
    z-index: 1;
  }
  
  @keyframes hologramScan {
    0%, 100% { 
      opacity: 0.3; 
      transform: translateY(0px);
    }
    50% { 
      opacity: 0.8; 
      transform: translateY(-2px);
    }
  }
  
  @keyframes borderGlow {
    0%, 100% { 
      opacity: 0.6;
      filter: blur(1px);
    }
    50% { 
      opacity: 1;
      filter: blur(0px);
    }
  }
`;

const SVGChart = styled.svg`
  width: 100%;
  height: 100%;
  position: relative;
  z-index: 2;
  filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.3));
`;

const GridLine = styled.line`
  stroke: rgba(0, 255, 255, 0.2);
  stroke-width: 1;
  stroke-dasharray: 2,2;
`;

const AxisLine = styled.line`
  stroke: rgba(0, 255, 255, 0.5);
  stroke-width: 2;
`;

const AxisLabel = styled.text`
  fill: #00ffff;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  text-anchor: middle;
`;

const ForecastLine = styled.path.withConfig({
  shouldForwardProp: (prop) => !['color', 'glowing'].includes(prop),
})<{ color: string; glowing?: boolean }>`
  fill: none;
  stroke: ${props => props.color};
  stroke-width: ${props => props.glowing ? 3 : 2};
  stroke-linecap: round;
  stroke-linejoin: round;
  filter: ${props => props.glowing ? `drop-shadow(0 0 6px ${props.color})` : 'none'};
  opacity: 0.9;
  
  &:hover {
    stroke-width: 4;
    filter: drop-shadow(0 0 10px ${props => props.color});
  }
`;

const ConfidenceArea = styled.path`
  fill: url(#confidenceGradient);
  stroke: rgba(57, 255, 20, 0.4);
  stroke-width: 2;
  opacity: 0.7;
  filter: drop-shadow(0 0 8px rgba(57, 255, 20, 0.4));
  animation: confidencePulse 2s ease-in-out infinite;
  
  @keyframes confidencePulse {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 0.9; }
  }
`;

const DataPoint = styled.circle<{ color: string }>`
  fill: ${props => props.color};
  stroke: rgba(255, 255, 255, 0.8);
  stroke-width: 2;
  filter: drop-shadow(0 0 4px ${props => props.color});
  cursor: pointer;
  
  &:hover {
    r: 6;
    filter: drop-shadow(0 0 8px ${props => props.color});
  }
`;

const Legend = styled.div`
  position: absolute;
  top: 20px;
  right: 20px;
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 4px;
  padding: 12px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  z-index: 2;
`;

const LegendItem = styled.div<{ color: string }>`
  display: flex;
  align-items: center;
  margin-bottom: 6px;
  color: ${props => props.color};
  
  &::before {
    content: '';
    width: 16px;
    height: 2px;
    background: ${props => props.color};
    margin-right: 8px;
    filter: drop-shadow(0 0 2px ${props => props.color});
  }
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const Tooltip = styled(motion.div)<{ x: number; y: number }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y}px;
  background: rgba(0, 0, 0, 0.95);
  border: 2px solid #00ffff;
  border-radius: 8px;
  padding: 12px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #00ffff;
  pointer-events: none;
  z-index: 15;
  white-space: nowrap;
  backdrop-filter: blur(4px);
  box-shadow: 
    0 0 20px rgba(0, 255, 255, 0.5),
    inset 0 0 20px rgba(0, 255, 255, 0.1);
  
  &::before {
    content: '';
    position: absolute;
    top: -8px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 8px solid transparent;
    border-right: 8px solid transparent;
    border-bottom: 8px solid #00ffff;
    filter: drop-shadow(0 0 4px rgba(0, 255, 255, 0.8));
  }
`;

const ParticleCanvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 3;
  opacity: 0.6;
`;

const AccuracyIndicator = styled(motion.div)<{ accuracy: number }>`
  position: absolute;
  top: 15px;
  left: 15px;
  background: rgba(0, 0, 0, 0.8);
  border: 2px solid ${props => 
    props.accuracy >= 0.9 ? '#39ff14' : 
    props.accuracy >= 0.7 ? '#ffff00' : '#ff6b6b'
  };
  border-radius: 8px;
  padding: 8px 12px;
  font-family: 'Courier New', monospace;
  font-size: 11px;
  color: ${props => 
    props.accuracy >= 0.9 ? '#39ff14' : 
    props.accuracy >= 0.7 ? '#ffff00' : '#ff6b6b'
  };
  z-index: 10;
  backdrop-filter: blur(4px);
  box-shadow: 0 0 15px ${props => 
    props.accuracy >= 0.9 ? 'rgba(57, 255, 20, 0.4)' : 
    props.accuracy >= 0.7 ? 'rgba(255, 255, 0, 0.4)' : 'rgba(255, 107, 107, 0.4)'
  };
  
  &::before {
    content: '🎯';
    margin-right: 6px;
  }
`;

const ModelPerformanceBar = styled.div<{ performance: number }>`
  width: 60px;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  margin-top: 4px;
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    width: ${props => props.performance * 100}%;
    height: 100%;
    background: linear-gradient(90deg, 
      ${props => props.performance >= 0.8 ? '#39ff14' : 
                 props.performance >= 0.6 ? '#ffff00' : '#ff6b6b'} 0%,
      ${props => props.performance >= 0.8 ? '#00ff88' : 
                 props.performance >= 0.6 ? '#ff8800' : '#ff0040'} 100%
    );
    border-radius: 2px;
    animation: performanceGlow 2s ease-in-out infinite;
  }
  
  @keyframes performanceGlow {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.3); }
  }
`;

// Particle system for confidence intervals
interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
}

export const HolographicForecastChart: React.FC<HolographicForecastChartProps> = ({
  data,
  showIndividualModels = true,
  showConfidenceIntervals = true,
  width = 800,
  height = 400,
  realTimeAccuracy = 0.85,
  enableParticleEffects = true,
  enable3DEffects = true
}) => {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string; model?: string } | null>(null);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([realTimeAccuracy]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  // Chart dimensions
  const margin = { top: 40, right: 160, bottom: 60, left: 80 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  
  // Combine all data points to find scales
  const allData = [
    ...data.historical,
    ...data.forecasts.ensemble,
    ...(showConfidenceIntervals ? data.confidenceIntervals.p10 : []),
    ...(showConfidenceIntervals ? data.confidenceIntervals.p90 : [])
  ];
  
  const allValues = allData.map(d => d.value);
  const minValue = Math.min(...allValues) * 0.95;
  const maxValue = Math.max(...allValues) * 1.05;
  
  // Create scales
  const xScale = (index: number) => (index / (data.historical.length + data.forecasts.ensemble.length - 1)) * chartWidth;
  const yScale = (value: number) => chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight;
  
  // Create path strings
  const createPath = (points: TimeSeriesPoint[], startIndex: number = 0) => {
    return points.map((point, index) => {
      const x = xScale(startIndex + index);
      const y = yScale(point.value);
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  };
  
  // Model colors
  const modelColors = {
    historical: '#00ffff',
    arima: '#ff6b6b',
    ets: '#4ecdc4',
    xgboost: '#45b7d1',
    lstm: '#f9ca24',
    croston: '#6c5ce7',
    ensemble: '#ff1493'
  };
  
  // Particle system functions
  const createParticle = useCallback((x: number, y: number, color: string = '#39ff14'): Particle => ({
    x,
    y,
    vx: (Math.random() - 0.5) * 2,
    vy: (Math.random() - 0.5) * 2,
    life: 60,
    maxLife: 60,
    color,
    size: Math.random() * 3 + 1
  }), []);

  const updateParticles = useCallback(() => {
    if (!enableParticleEffects) return;
    
    setParticles(prevParticles => {
      const updated = prevParticles
        .map(particle => ({
          ...particle,
          x: particle.x + particle.vx,
          y: particle.y + particle.vy,
          life: particle.life - 1,
          vx: particle.vx * 0.99,
          vy: particle.vy * 0.99
        }))
        .filter(particle => particle.life > 0);
      
      // Add new particles around confidence intervals
      if (showConfidenceIntervals && Math.random() < 0.3) {
        const newParticles = Array.from({ length: 2 }, () => 
          createParticle(
            Math.random() * width,
            Math.random() * height,
            Math.random() > 0.5 ? '#39ff14' : '#00ffff'
          )
        );
        return [...updated, ...newParticles];
      }
      
      return updated;
    });
  }, [enableParticleEffects, showConfidenceIntervals, width, height, createParticle]);

  const drawParticles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !enableParticleEffects) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, width, height);
    
    particles.forEach(particle => {
      const alpha = particle.life / particle.maxLife;
      ctx.globalAlpha = alpha * 0.6;
      ctx.fillStyle = particle.color;
      ctx.shadowBlur = 10;
      ctx.shadowColor = particle.color;
      
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.shadowBlur = 0;
    });
    
    ctx.globalAlpha = 1;
  }, [particles, enableParticleEffects, width, height]);

  // Animation loop for particles
  useEffect(() => {
    if (!enableParticleEffects) return;
    
    const animate = () => {
      updateParticles();
      drawParticles();
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [updateParticles, drawParticles, enableParticleEffects]);

  // Update accuracy history
  useEffect(() => {
    setAccuracyHistory(prev => {
      const newHistory = [...prev, realTimeAccuracy];
      return newHistory.slice(-10); // Keep last 10 values
    });
  }, [realTimeAccuracy]);

  const handleMouseMove = (event: React.MouseEvent, point: TimeSeriesPoint, model: string) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const modelPerformance = Math.random() * 0.3 + 0.7; // Mock performance data
    
    setTooltip({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top - 10,
      content: `${model.toUpperCase()}: ${point.value.toFixed(2)} (${point.date})`,
      model
    });
    
    // Create particles on hover
    if (enableParticleEffects) {
      const newParticles = Array.from({ length: 5 }, () => 
        createParticle(
          event.clientX - rect.left,
          event.clientY - rect.top,
          modelColors[model as keyof typeof modelColors] || '#ffffff'
        )
      );
      setParticles(prev => [...prev, ...newParticles]);
    }
  };
  
  const handleMouseLeave = () => {
    setTooltip(null);
  };
  
  return (
    <ChartContainer width={width} height={height}>
      {/* Particle Canvas */}
      {enableParticleEffects && (
        <ParticleCanvas
          ref={canvasRef}
          width={width}
          height={height}
        />
      )}
      
      {/* Real-time Accuracy Indicator */}
      <AccuracyIndicator
        accuracy={realTimeAccuracy}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        Accuracy: {(realTimeAccuracy * 100).toFixed(1)}%
        <div style={{ fontSize: '9px', marginTop: '2px' }}>
          Trend: {accuracyHistory.length > 1 && 
            accuracyHistory[accuracyHistory.length - 1] > accuracyHistory[accuracyHistory.length - 2] 
            ? '↗️' : '↘️'
          }
        </div>
      </AccuracyIndicator>

      <SVGChart viewBox={`0 0 ${width} ${height}`}>
        {/* Gradient Definitions */}
        <defs>
          <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(57, 255, 20, 0.3)" />
            <stop offset="50%" stopColor="rgba(57, 255, 20, 0.15)" />
            <stop offset="100%" stopColor="rgba(57, 255, 20, 0.05)" />
          </linearGradient>
          
          <linearGradient id="ensembleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ff1493" />
            <stop offset="50%" stopColor="#ff6b9d" />
            <stop offset="100%" stopColor="#ff1493" />
          </linearGradient>
          
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          
          <filter id="hologram" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="1"/>
            <feColorMatrix values="0 0 0 0 0   0 1 1 0 0   0 1 1 0 0  0 0 0 1 0"/>
          </filter>
        </defs>
        
        {/* Grid lines */}
        {Array.from({ length: 6 }, (_, i) => (
          <GridLine
            key={`grid-y-${i}`}
            x1={margin.left}
            y1={margin.top + (i * chartHeight / 5)}
            x2={margin.left + chartWidth}
            y2={margin.top + (i * chartHeight / 5)}
          />
        ))}
        
        {Array.from({ length: 8 }, (_, i) => (
          <GridLine
            key={`grid-x-${i}`}
            x1={margin.left + (i * chartWidth / 7)}
            y1={margin.top}
            x2={margin.left + (i * chartWidth / 7)}
            y2={margin.top + chartHeight}
          />
        ))}
        
        {/* Axes */}
        <AxisLine
          x1={margin.left}
          y1={margin.top + chartHeight}
          x2={margin.left + chartWidth}
          y2={margin.top + chartHeight}
        />
        <AxisLine
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={margin.top + chartHeight}
        />
        
        {/* Y-axis labels */}
        {Array.from({ length: 6 }, (_, i) => {
          const value = minValue + (i * (maxValue - minValue) / 5);
          return (
            <AxisLabel
              key={`y-label-${i}`}
              x={margin.left - 10}
              y={margin.top + chartHeight - (i * chartHeight / 5) + 4}
              textAnchor="end"
            >
              {value.toFixed(0)}
            </AxisLabel>
          );
        })}
        
        {/* Confidence intervals */}
        {showConfidenceIntervals && data.confidenceIntervals.p10.length > 0 && (
          <ConfidenceArea
            d={`
              ${createPath(data.confidenceIntervals.p10, data.historical.length)}
              ${createPath([...data.confidenceIntervals.p90].reverse(), data.historical.length).replace('M', 'L')}
              Z
            `}
            transform={`translate(${margin.left}, ${margin.top})`}
          />
        )}
        
        {/* Historical data line */}
        <ForecastLine
          color={modelColors.historical}
          glowing
          d={createPath(data.historical)}
          transform={`translate(${margin.left}, ${margin.top})`}
        />
        
        {/* Individual model forecasts */}
        {showIndividualModels && Object.entries(data.forecasts).map(([model, points]) => {
          if (model === 'ensemble') return null;
          return (
            <ForecastLine
              key={model}
              color={modelColors[model as keyof typeof modelColors]}
              d={createPath(points, data.historical.length)}
              transform={`translate(${margin.left}, ${margin.top})`}
              opacity={0.6}
            />
          );
        })}
        
        {/* Ensemble forecast line */}
        <ForecastLine
          color={modelColors.ensemble}
          glowing
          d={createPath(data.forecasts.ensemble, data.historical.length)}
          transform={`translate(${margin.left}, ${margin.top})`}
        />
        
        {/* Data points */}
        {data.historical.map((point, index) => (
          <DataPoint
            key={`hist-${index}`}
            cx={margin.left + xScale(index)}
            cy={margin.top + yScale(point.value)}
            r={3}
            color={modelColors.historical}
            onMouseMove={(e) => handleMouseMove(e, point, 'Historical')}
            onMouseLeave={handleMouseLeave}
          />
        ))}
        
        {data.forecasts.ensemble.map((point, index) => (
          <DataPoint
            key={`forecast-${index}`}
            cx={margin.left + xScale(data.historical.length + index)}
            cy={margin.top + yScale(point.value)}
            r={4}
            color={modelColors.ensemble}
            onMouseMove={(e) => handleMouseMove(e, point, 'Ensemble')}
            onMouseLeave={handleMouseLeave}
          />
        ))}
      </SVGChart>
      
      {/* Legend */}
      <Legend>
        <LegendItem color={modelColors.historical}>Historical Data</LegendItem>
        <LegendItem color={modelColors.ensemble}>Ensemble Forecast</LegendItem>
        {showIndividualModels && (
          <>
            <LegendItem color={modelColors.arima}>ARIMA</LegendItem>
            <LegendItem color={modelColors.ets}>ETS</LegendItem>
            <LegendItem color={modelColors.xgboost}>XGBoost</LegendItem>
            <LegendItem color={modelColors.lstm}>LSTM</LegendItem>
            <LegendItem color={modelColors.croston}>Croston</LegendItem>
          </>
        )}
        {showConfidenceIntervals && (
          <LegendItem color="rgba(57, 255, 20, 0.6)">Confidence Interval</LegendItem>
        )}
      </Legend>
      
      {/* Tooltip */}
      {tooltip && (
        <Tooltip
          x={tooltip.x}
          y={tooltip.y}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          transition={{ duration: 0.2 }}
        >
          {tooltip.content}
        </Tooltip>
      )}
    </ChartContainer>
  );
};