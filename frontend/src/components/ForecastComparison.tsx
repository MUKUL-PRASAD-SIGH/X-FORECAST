import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';

interface ComparisonScenario {
  name: string;
  color: string;
  data: {
    point_forecast: Record<string, number>;
    confidence_intervals: {
      p10: Record<string, number>;
      p50: Record<string, number>;
      p90: Record<string, number>;
    };
  };
  parameters: {
    seasonality_factor: number;
    trend_adjustment: number;
    volatility_modifier: number;
    external_factors: number;
    market_conditions: string;
  };
  metrics: {
    total_change: number;
    peak_value: number;
    risk_score: number;
    confidence_score: number;
  };
}

interface ForecastComparisonProps {
  scenarios: ComparisonScenario[];
  baselineData?: any;
  onScenarioSelect?: (scenario: ComparisonScenario) => void;
}

const ComparisonContainer = styled(CyberpunkCard)`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
`;

const ChartContainer = styled.div`
  position: relative;
  height: 400px;
  background: radial-gradient(circle at center, rgba(0, 255, 255, 0.05) 0%, transparent 70%);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 8px;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      linear-gradient(90deg, transparent 0%, rgba(0, 255, 255, 0.1) 50%, transparent 100%),
      linear-gradient(0deg, transparent 0%, rgba(255, 20, 147, 0.1) 50%, transparent 100%);
    pointer-events: none;
    animation: hologramScan 3s ease-in-out infinite;
  }
  
  @keyframes hologramScan {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.7; }
  }
`;

const SVGChart = styled.svg`
  width: 100%;
  height: 100%;
  position: relative;
  z-index: 1;
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

const ConfidenceArea = styled.path<{ color: string }>`
  fill: ${props => props.color}20;
  stroke: ${props => props.color}60;
  stroke-width: 1;
  opacity: 0.4;
`;

const ScenarioLegend = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.md};
  justify-content: center;
`;

const LegendItem = styled(motion.div)<{ color: string; active: boolean }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm};
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.5)'};
  border: 1px solid ${props => props.color};
  border-radius: 4px;
  cursor: pointer;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.color};
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(0, 255, 255, 0.1);
    transform: translateY(-2px);
  }
  
  &::before {
    content: '';
    width: 16px;
    height: 3px;
    background: ${props => props.color};
    filter: drop-shadow(0 0 2px ${props => props.color});
  }
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
`;

const MetricCard = styled(CyberpunkCard)<{ color: string }>`
  border-color: ${props => props.color};
  background: ${props => props.color}10;
`;

const MetricHeader = styled.div<{ color: string }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.md};
  color: ${props => props.color};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-weight: bold;
  
  &::before {
    content: '';
    width: 12px;
    height: 12px;
    background: ${props => props.color};
    border-radius: 50%;
    filter: drop-shadow(0 0 4px ${props => props.color});
  }
`;

const MetricRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.xs} 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  
  &:last-child {
    border-bottom: none;
  }
`;

const MetricLabel = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.secondaryText};
`;

const MetricValue = styled.span<{ color?: string }>`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.color || props.theme.colors.primaryText};
  font-weight: bold;
`;

const ComparisonControls = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  justify-content: center;
  flex-wrap: wrap;
`;

const CardTitle = styled.h3`
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: ${props => props.theme.typography.fontSize.xl};
  text-shadow: ${props => props.theme.effects.softGlow};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

export const ForecastComparison: React.FC<ForecastComparisonProps> = ({ 
  scenarios, 
  baselineData,
  onScenarioSelect 
}) => {
  const [visibleScenarios, setVisibleScenarios] = useState<Set<string>>(
    new Set(scenarios.map(s => s.name))
  );
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);

  // Chart dimensions
  const margin = { top: 40, right: 40, bottom: 60, left: 80 };
  const chartWidth = 800 - margin.left - margin.right;
  const chartHeight = 400 - margin.top - margin.bottom;

  // Get all data points for scaling
  const allDataPoints = scenarios.flatMap(scenario => 
    Object.values(scenario.data.point_forecast)
  );
  const minValue = Math.min(...allDataPoints) * 0.95;
  const maxValue = Math.max(...allDataPoints) * 1.05;

  // Create scales
  const getXScale = (index: number, totalPoints: number) => 
    (index / (totalPoints - 1)) * chartWidth;
  
  const getYScale = (value: number) => 
    chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight;

  // Create path string for a scenario
  const createPath = (data: Record<string, number>) => {
    const points = Object.entries(data).sort(([a], [b]) => a.localeCompare(b));
    return points.map(([date, value], index) => {
      const x = getXScale(index, points.length);
      const y = getYScale(value);
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  };

  // Create confidence area path
  const createConfidenceArea = (p10: Record<string, number>, p90: Record<string, number>) => {
    const p10Points = Object.entries(p10).sort(([a], [b]) => a.localeCompare(b));
    const p90Points = Object.entries(p90).sort(([a], [b]) => a.localeCompare(b));
    
    const topPath = p90Points.map(([date, value], index) => {
      const x = getXScale(index, p90Points.length);
      const y = getYScale(value);
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
    
    const bottomPath = p10Points.reverse().map(([date, value], index) => {
      const x = getXScale(p10Points.length - 1 - index, p10Points.length);
      const y = getYScale(value);
      return `L ${x} ${y}`;
    }).join(' ');
    
    return `${topPath} ${bottomPath} Z`;
  };

  // Toggle scenario visibility
  const toggleScenario = (scenarioName: string) => {
    setVisibleScenarios(prev => {
      const newSet = new Set(prev);
      if (newSet.has(scenarioName)) {
        newSet.delete(scenarioName);
      } else {
        newSet.add(scenarioName);
      }
      return newSet;
    });
  };

  // Select scenario for detailed view
  const selectScenario = (scenario: ComparisonScenario) => {
    setSelectedScenario(scenario.name);
    if (onScenarioSelect) {
      onScenarioSelect(scenario);
    }
  };

  // Show all scenarios
  const showAllScenarios = () => {
    setVisibleScenarios(new Set(scenarios.map(s => s.name)));
  };

  // Hide all scenarios
  const hideAllScenarios = () => {
    setVisibleScenarios(new Set());
  };

  return (
    <ComparisonContainer $variant="hologram" $padding="lg">
      <CardTitle>📊 Scenario Comparison Analysis</CardTitle>
      
      {/* Chart */}
      <ChartContainer>
        <SVGChart viewBox={`0 0 800 400`}>
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
          {showConfidenceIntervals && scenarios.map(scenario => {
            if (!visibleScenarios.has(scenario.name)) return null;
            
            return (
              <ConfidenceArea
                key={`confidence-${scenario.name}`}
                color={scenario.color}
                d={createConfidenceArea(
                  scenario.data.confidence_intervals.p10,
                  scenario.data.confidence_intervals.p90
                )}
                transform={`translate(${margin.left}, ${margin.top})`}
              />
            );
          })}
          
          {/* Forecast lines */}
          {scenarios.map(scenario => {
            if (!visibleScenarios.has(scenario.name)) return null;
            
            return (
              <ForecastLine
                key={scenario.name}
                color={scenario.color}
                glowing={selectedScenario === scenario.name}
                d={createPath(scenario.data.point_forecast)}
                transform={`translate(${margin.left}, ${margin.top})`}
                style={{ cursor: 'pointer' }}
                onClick={() => selectScenario(scenario)}
              />
            );
          })}
        </SVGChart>
      </ChartContainer>

      {/* Legend */}
      <ScenarioLegend>
        {scenarios.map(scenario => (
          <LegendItem
            key={scenario.name}
            color={scenario.color}
            active={visibleScenarios.has(scenario.name)}
            onClick={() => toggleScenario(scenario.name)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {scenario.name.replace('_', ' ').toUpperCase()}
          </LegendItem>
        ))}
      </ScenarioLegend>

      {/* Metrics Comparison */}
      <MetricsGrid>
        {scenarios.map(scenario => {
          if (!visibleScenarios.has(scenario.name)) return null;
          
          return (
            <MetricCard
              key={`metrics-${scenario.name}`}
              variant="glass"
              color={scenario.color}
              padding="md"
            >
              <MetricHeader color={scenario.color}>
                {scenario.name.replace('_', ' ').toUpperCase()}
              </MetricHeader>
              
              <MetricRow>
                <MetricLabel>Total Change</MetricLabel>
                <MetricValue color={scenario.metrics.total_change >= 0 ? '#39ff14' : '#ff0040'}>
                  {scenario.metrics.total_change >= 0 ? '+' : ''}
                  {scenario.metrics.total_change.toFixed(1)}%
                </MetricValue>
              </MetricRow>
              
              <MetricRow>
                <MetricLabel>Peak Value</MetricLabel>
                <MetricValue color={scenario.color}>
                  {scenario.metrics.peak_value.toFixed(0)}
                </MetricValue>
              </MetricRow>
              
              <MetricRow>
                <MetricLabel>Risk Score</MetricLabel>
                <MetricValue color={
                  scenario.metrics.risk_score > 0.7 ? '#ff0040' :
                  scenario.metrics.risk_score > 0.4 ? '#ffff00' : '#39ff14'
                }>
                  {(scenario.metrics.risk_score * 10).toFixed(1)}/10
                </MetricValue>
              </MetricRow>
              
              <MetricRow>
                <MetricLabel>Confidence</MetricLabel>
                <MetricValue color={scenario.color}>
                  {(scenario.metrics.confidence_score * 100).toFixed(0)}%
                </MetricValue>
              </MetricRow>
            </MetricCard>
          );
        })}
      </MetricsGrid>

      {/* Controls */}
      <ComparisonControls>
        <CyberpunkButton
          variant="primary"
          onClick={showAllScenarios}
        >
          Show All Scenarios
        </CyberpunkButton>
        
        <CyberpunkButton
          variant="secondary"
          onClick={hideAllScenarios}
        >
          Hide All Scenarios
        </CyberpunkButton>
        
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#00ffff' }}>
          <input
            type="checkbox"
            checked={showConfidenceIntervals}
            onChange={(e) => setShowConfidenceIntervals(e.target.checked)}
          />
          Show Confidence Intervals
        </label>
      </ComparisonControls>
    </ComparisonContainer>
  );
};