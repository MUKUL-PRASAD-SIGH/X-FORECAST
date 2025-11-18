import React, { useMemo } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar,
  Cell,
  PieChart,
  Pie
} from 'recharts';

interface CustomerSegment {
  segment_id: string;
  segment_name: string;
  customer_count: number;
  avg_ltv: number;
  avg_retention_rate: number;
  revenue_contribution: number;
  growth_rate: number;
  health_score: number;
  characteristics: string[];
}

interface CustomerSegmentChartProps {
  segments: CustomerSegment[];
  chartType?: 'bar' | 'radar' | 'pie';
  width?: number;
  height?: number;
  showComparison?: boolean;
}

const ChartContainer = styled(motion.div)<{ width: number; height: number }>`
  width: ${props => props.width}px;
  height: ${props => props.height}px;
  background: 
    radial-gradient(circle at 30% 30%, rgba(57, 255, 20, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 70% 70%, rgba(191, 0, 255, 0.06) 0%, transparent 50%);
  border: 2px solid transparent;
  border-radius: 12px;
  padding: 20px;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(135deg, 
      rgba(57, 255, 20, 0.5) 0%, 
      rgba(0, 255, 255, 0.5) 25%, 
      rgba(255, 20, 147, 0.5) 50%, 
      rgba(191, 0, 255, 0.5) 75%, 
      rgba(57, 255, 20, 0.5) 100%
    );
    border-radius: 12px;
    z-index: -1;
    animation: segmentGlow 4s ease-in-out infinite;
  }
  
  @keyframes segmentGlow {
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

const ChartTitle = styled.h3`
  color: ${props => props.theme.colors.acidGreen};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  text-align: center;
  margin-bottom: 20px;
  text-shadow: ${props => props.theme.effects.softGlow};
`;

const ChartTypeSelector = styled.div`
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 20px;
`;

const TypeButton = styled.button<{ active: boolean }>`
  padding: 6px 12px;
  border: 1px solid ${props => props.active ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'transparent'};
  color: ${props => props.active ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  border-radius: 4px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: ${props => props.theme.colors.neonBlue};
    color: ${props => props.theme.colors.neonBlue};
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
  }
`;

const CustomTooltip = styled.div`
  background: rgba(0, 0, 0, 0.95);
  border: 2px solid #39ff14;
  border-radius: 8px;
  padding: 12px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #39ff14;
  box-shadow: 
    0 0 20px rgba(57, 255, 20, 0.5),
    inset 0 0 20px rgba(57, 255, 20, 0.1);
  
  .tooltip-label {
    color: #ff1493;
    font-weight: bold;
    margin-bottom: 8px;
    text-transform: uppercase;
  }
  
  .tooltip-value {
    color: #00ffff;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    gap: 10px;
  }
  
  .tooltip-metric {
    color: #ffffff;
  }
`;

const Legend = styled.div`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 15px;
  margin-top: 15px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
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
    box-shadow: 0 0 8px ${props => props.color};
  }
`;

const ComparisonMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-top: 20px;
  padding: 15px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  border: 1px solid rgba(57, 255, 20, 0.3);
`;

const MetricCard = styled.div`
  text-align: center;
  padding: 10px;
  border-radius: 6px;
  background: rgba(57, 255, 20, 0.1);
  border: 1px solid rgba(57, 255, 20, 0.3);
`;

const MetricLabel = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  text-transform: uppercase;
  margin-bottom: 4px;
`;

const MetricValue = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.md};
  color: ${props => props.theme.colors.acidGreen};
  font-weight: bold;
`;

export const CustomerSegmentChart: React.FC<CustomerSegmentChartProps> = ({
  segments,
  chartType = 'bar',
  width = 700,
  height = 500,
  showComparison = true
}) => {
  const [currentChartType, setCurrentChartType] = React.useState(chartType);

  const segmentColors = [
    '#00ffff', '#ff1493', '#39ff14', '#ffff00', '#bf00ff', 
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7'
  ];

  const barChartData = useMemo(() => {
    return segments.map((segment, index) => ({
      name: segment.segment_name.length > 12 
        ? segment.segment_name.substring(0, 12) + '...' 
        : segment.segment_name,
      fullName: segment.segment_name,
      customers: segment.customer_count,
      ltv: Math.round(segment.avg_ltv),
      retention: Math.round(segment.avg_retention_rate * 100),
      revenue: Math.round(segment.revenue_contribution),
      health: Math.round(segment.health_score),
      growth: Math.round(segment.growth_rate),
      color: segmentColors[index % segmentColors.length]
    }));
  }, [segments]);

  const radarChartData = useMemo(() => {
    return segments.map((segment, index) => ({
      segment: segment.segment_name,
      LTV: Math.min(segment.avg_ltv / 100, 100), // Normalize to 0-100
      Retention: segment.avg_retention_rate * 100,
      Revenue: segment.revenue_contribution,
      Health: segment.health_score,
      Growth: Math.max(0, Math.min(segment.growth_rate + 50, 100)), // Normalize growth rate
      color: segmentColors[index % segmentColors.length]
    }));
  }, [segments]);

  const pieChartData = useMemo(() => {
    return segments.map((segment, index) => ({
      name: segment.segment_name,
      value: segment.revenue_contribution,
      customers: segment.customer_count,
      color: segmentColors[index % segmentColors.length]
    }));
  }, [segments]);

  const topPerformingSegment = useMemo(() => {
    return segments.reduce((top, current) => 
      current.health_score > top.health_score ? current : top
    , segments[0]);
  }, [segments]);

  const totalCustomers = useMemo(() => {
    return segments.reduce((sum, segment) => sum + segment.customer_count, 0);
  }, [segments]);

  const averageLTV = useMemo(() => {
    const totalLTV = segments.reduce((sum, segment) => 
      sum + (segment.avg_ltv * segment.customer_count), 0
    );
    return totalLTV / totalCustomers;
  }, [segments, totalCustomers]);

  const CustomTooltipComponent = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <CustomTooltip>
          <div className="tooltip-label">{data.fullName || label}</div>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="tooltip-value">
              <span className="tooltip-metric">{entry.name}:</span>
              <span>{entry.value}{entry.name === 'retention' ? '%' : entry.name === 'ltv' ? '$' : ''}</span>
            </div>
          ))}
        </CustomTooltip>
      );
    }
    return null;
  };

  const CustomPieTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <CustomTooltip>
          <div className="tooltip-label">{data.name}</div>
          <div className="tooltip-value">
            <span className="tooltip-metric">Revenue Share:</span>
            <span>{data.value.toFixed(1)}%</span>
          </div>
          <div className="tooltip-value">
            <span className="tooltip-metric">Customers:</span>
            <span>{data.customers}</span>
          </div>
        </CustomTooltip>
      );
    }
    return null;
  };

  const renderBarChart = () => (
    <ResponsiveContainer width="100%" height="70%">
      <BarChart data={barChartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(57, 255, 20, 0.2)" />
        <XAxis 
          dataKey="name" 
          stroke="#39ff14"
          fontSize={10}
          fontFamily="Courier New, monospace"
          angle={-45}
          textAnchor="end"
          height={80}
        />
        <YAxis 
          stroke="#39ff14"
          fontSize={10}
          fontFamily="Courier New, monospace"
        />
        <Tooltip content={<CustomTooltipComponent />} />
        
        <Bar dataKey="health" name="Health Score" radius={[2, 2, 0, 0]}>
          {barChartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );

  const renderRadarChart = () => (
    <ResponsiveContainer width="100%" height="70%">
      <RadarChart data={radarChartData[0] ? [radarChartData[0]] : []}>
        <PolarGrid stroke="rgba(57, 255, 20, 0.3)" />
        <PolarAngleAxis 
          dataKey="subject" 
          tick={{ fontSize: 12, fill: '#39ff14', fontFamily: 'Courier New' }}
        />
        <PolarRadiusAxis 
          angle={90} 
          domain={[0, 100]}
          tick={{ fontSize: 10, fill: '#39ff14' }}
        />
        {radarChartData.slice(0, 3).map((segment, index) => (
          <Radar
            key={segment.segment}
            name={segment.segment}
            dataKey={(key) => segment[key as keyof typeof segment]}
            stroke={segment.color}
            fill={segment.color}
            fillOpacity={0.2}
            strokeWidth={2}
          />
        ))}
      </RadarChart>
    </ResponsiveContainer>
  );

  const renderPieChart = () => (
    <ResponsiveContainer width="100%" height="70%">
      <PieChart>
        <Pie
          data={pieChartData}
          cx="50%"
          cy="50%"
          outerRadius={120}
          innerRadius={40}
          paddingAngle={2}
          dataKey="value"
          stroke="rgba(255, 255, 255, 0.2)"
          strokeWidth={2}
        >
          {pieChartData.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.color}
              style={{
                filter: `drop-shadow(0 0 6px ${entry.color})`,
              }}
            />
          ))}
        </Pie>
        <Tooltip content={<CustomPieTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );

  if (!segments || segments.length === 0) {
    return (
      <ChartContainer width={width} height={height}>
        <ChartTitle>No Segment Data Available</ChartTitle>
      </ChartContainer>
    );
  }

  return (
    <ChartContainer
      width={width}
      height={height}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
    >
      <ChartTitle>Customer Segment Performance</ChartTitle>
      
      <ChartTypeSelector>
        <TypeButton 
          active={currentChartType === 'bar'} 
          onClick={() => setCurrentChartType('bar')}
        >
          Bar Chart
        </TypeButton>
        <TypeButton 
          active={currentChartType === 'radar'} 
          onClick={() => setCurrentChartType('radar')}
        >
          Radar Chart
        </TypeButton>
        <TypeButton 
          active={currentChartType === 'pie'} 
          onClick={() => setCurrentChartType('pie')}
        >
          Revenue Share
        </TypeButton>
      </ChartTypeSelector>
      
      {currentChartType === 'bar' && renderBarChart()}
      {currentChartType === 'radar' && renderRadarChart()}
      {currentChartType === 'pie' && renderPieChart()}
      
      <Legend>
        {segments.slice(0, 5).map((segment, index) => (
          <LegendItem key={segment.segment_id} color={segmentColors[index]}>
            {segment.segment_name}
          </LegendItem>
        ))}
      </Legend>
      
      {showComparison && (
        <ComparisonMetrics>
          <MetricCard>
            <MetricLabel>Top Performer</MetricLabel>
            <MetricValue>{topPerformingSegment?.segment_name || 'N/A'}</MetricValue>
          </MetricCard>
          
          <MetricCard>
            <MetricLabel>Total Customers</MetricLabel>
            <MetricValue>{totalCustomers.toLocaleString()}</MetricValue>
          </MetricCard>
          
          <MetricCard>
            <MetricLabel>Avg LTV</MetricLabel>
            <MetricValue>${averageLTV.toFixed(0)}</MetricValue>
          </MetricCard>
          
          <MetricCard>
            <MetricLabel>Segments</MetricLabel>
            <MetricValue>{segments.length}</MetricValue>
          </MetricCard>
        </ComparisonMetrics>
      )}
    </ChartContainer>
  );
};