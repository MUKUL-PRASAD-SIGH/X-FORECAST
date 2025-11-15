import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

interface WeightDistributionChartProps {
  weights: Record<string, number>;
  width?: number;
  height?: number;
}

const ChartContainer = styled.div<{ width: number; height: number }>`
  width: ${props => props.width}px;
  height: ${props => props.height}px;
  position: relative;
  background: radial-gradient(circle at center, rgba(255, 20, 147, 0.05) 0%, transparent 70%);
  border: 1px solid rgba(255, 20, 147, 0.3);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const SVGChart = styled.svg`
  width: 100%;
  height: 100%;
`;

const WeightArc = styled.path.withConfig({
  shouldForwardProp: (prop) => !['color', 'isHovered'].includes(prop),
})<{ color: string; isHovered: boolean }>`
  fill: ${props => props.color};
  stroke: rgba(255, 255, 255, 0.2);
  stroke-width: 2;
  cursor: pointer;
  transition: all 0.3s ease;
  filter: ${props => props.isHovered ? `drop-shadow(0 0 10px ${props.color})` : `drop-shadow(0 0 4px ${props.color})`};
  opacity: ${props => props.isHovered ? 1 : 0.8};
  
  &:hover {
    stroke-width: 3;
    stroke: rgba(255, 255, 255, 0.6);
  }
`;

const CenterText = styled.text`
  fill: #00ffff;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  font-weight: bold;
  text-anchor: middle;
  dominant-baseline: middle;
  filter: drop-shadow(0 0 4px #00ffff);
`;

const WeightLabel = styled.text<{ color: string }>`
  fill: ${props => props.color};
  font-family: 'Courier New', monospace;
  font-size: 12px;
  font-weight: bold;
  text-anchor: middle;
  dominant-baseline: middle;
  filter: drop-shadow(0 0 2px ${props => props.color});
`;

const Tooltip = styled(motion.div)<{ x: number; y: number }>`
  position: absolute;
  left: ${props => props.x}px;
  top: ${props => props.y}px;
  background: rgba(0, 0, 0, 0.9);
  border: 1px solid #ff1493;
  border-radius: 4px;
  padding: 8px 12px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #ff1493;
  pointer-events: none;
  z-index: 10;
  white-space: nowrap;
  
  &::before {
    content: '';
    position: absolute;
    top: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 5px solid #ff1493;
  }
`;

const Legend = styled.div`
  position: absolute;
  bottom: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  font-family: 'Courier New', monospace;
  font-size: 10px;
`;

const LegendItem = styled.div<{ color: string }>`
  display: flex;
  align-items: center;
  color: ${props => props.color};
  
  &::before {
    content: '';
    width: 12px;
    height: 12px;
    background: ${props => props.color};
    margin-right: 6px;
    border-radius: 2px;
    filter: drop-shadow(0 0 2px ${props => props.color});
  }
`;

export const WeightDistributionChart: React.FC<WeightDistributionChartProps> = ({
  weights,
  width = 300,
  height = 300
}) => {
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  
  // Model colors
  const modelColors = {
    arima: '#ff6b6b',
    ets: '#4ecdc4',
    xgboost: '#45b7d1',
    lstm: '#f9ca24',
    croston: '#6c5ce7'
  };
  
  // Chart dimensions
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 2 - 40;
  const innerRadius = radius * 0.4;
  
  // Calculate angles
  let currentAngle = -Math.PI / 2; // Start at top
  const segments = Object.entries(weights).map(([model, weight]) => {
    const angle = weight * 2 * Math.PI;
    const startAngle = currentAngle;
    const endAngle = currentAngle + angle;
    currentAngle = endAngle;
    
    return {
      model,
      weight,
      startAngle,
      endAngle,
      color: modelColors[model as keyof typeof modelColors] || '#ffffff'
    };
  });
  
  // Create SVG path for donut segment
  const createArcPath = (startAngle: number, endAngle: number, outerRadius: number, innerRadius: number) => {
    const x1 = centerX + outerRadius * Math.cos(startAngle);
    const y1 = centerY + outerRadius * Math.sin(startAngle);
    const x2 = centerX + outerRadius * Math.cos(endAngle);
    const y2 = centerY + outerRadius * Math.sin(endAngle);
    const x3 = centerX + innerRadius * Math.cos(endAngle);
    const y3 = centerY + innerRadius * Math.sin(endAngle);
    const x4 = centerX + innerRadius * Math.cos(startAngle);
    const y4 = centerY + innerRadius * Math.sin(startAngle);
    
    const largeArcFlag = endAngle - startAngle > Math.PI ? 1 : 0;
    
    return `
      M ${x1} ${y1}
      A ${outerRadius} ${outerRadius} 0 ${largeArcFlag} 1 ${x2} ${y2}
      L ${x3} ${y3}
      A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${x4} ${y4}
      Z
    `;
  };
  
  // Calculate label position
  const getLabelPosition = (startAngle: number, endAngle: number, radius: number) => {
    const midAngle = (startAngle + endAngle) / 2;
    const labelRadius = radius * 0.7;
    return {
      x: centerX + labelRadius * Math.cos(midAngle),
      y: centerY + labelRadius * Math.sin(midAngle)
    };
  };
  
  const handleMouseMove = (event: React.MouseEvent, model: string, weight: number) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setTooltip({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top - 10,
      content: `${model.toUpperCase()}: ${(weight * 100).toFixed(1)}%`
    });
    setHoveredSegment(model);
  };
  
  const handleMouseLeave = () => {
    setTooltip(null);
    setHoveredSegment(null);
  };
  
  return (
    <ChartContainer width={width} height={height}>
      <SVGChart viewBox={`0 0 ${width} ${height}`}>
        {/* Donut segments */}
        {segments.map(({ model, weight, startAngle, endAngle, color }) => (
          <WeightArc
            key={model}
            d={createArcPath(startAngle, endAngle, radius, innerRadius)}
            color={color}
            isHovered={hoveredSegment === model}
            onMouseMove={(e) => handleMouseMove(e, model, weight)}
            onMouseLeave={handleMouseLeave}
          />
        ))}
        
        {/* Weight labels */}
        {segments.map(({ model, weight, startAngle, endAngle, color }) => {
          const labelPos = getLabelPosition(startAngle, endAngle, radius);
          const percentage = (weight * 100).toFixed(0);
          
          // Only show label if segment is large enough
          if (weight < 0.05) return null;
          
          return (
            <WeightLabel
              key={`label-${model}`}
              x={labelPos.x}
              y={labelPos.y}
              color={color}
            >
              {percentage}%
            </WeightLabel>
          );
        })}
        
        {/* Center text */}
        <CenterText x={centerX} y={centerY - 5}>
          MODEL
        </CenterText>
        <CenterText x={centerX} y={centerY + 10} fontSize="12">
          WEIGHTS
        </CenterText>
        
        {/* Animated center circle */}
        <circle
          cx={centerX}
          cy={centerY}
          r={innerRadius - 5}
          fill="none"
          stroke="rgba(0, 255, 255, 0.3)"
          strokeWidth="2"
          strokeDasharray="4,4"
        >
          <animateTransform
            attributeName="transform"
            type="rotate"
            values={`0 ${centerX} ${centerY}; 360 ${centerX} ${centerY}`}
            dur="20s"
            repeatCount="indefinite"
          />
        </circle>
      </SVGChart>
      
      {/* Legend */}
      <Legend>
        {segments.map(({ model, color }) => (
          <LegendItem key={model} color={color}>
            {model.toUpperCase()}
          </LegendItem>
        ))}
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