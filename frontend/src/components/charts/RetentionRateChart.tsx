import React, { useMemo } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface CohortData {
  cohort_month: string;
  cohort_size: number;
  retention_rates: Record<string, number>;
  revenue_per_period: Record<string, number>;
  avg_ltv: number;
  churn_rate: number;
  months_tracked: number;
}

interface RetentionRateChartProps {
  cohortData: CohortData[];
  width?: number;
  height?: number;
  showTrendAnalysis?: boolean;
}

const ChartContainer = styled(motion.div)<{ width: number; height: number }>`
  width: ${props => props.width}px;
  height: ${props => props.height}px;
  background: 
    radial-gradient(circle at 20% 20%, rgba(0, 255, 255, 0.05) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(255, 20, 147, 0.05) 0%, transparent 50%);
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
    background: linear-gradient(45deg, 
      rgba(0, 255, 255, 0.4) 0%, 
      rgba(255, 20, 147, 0.4) 25%, 
      rgba(57, 255, 20, 0.4) 50%, 
      rgba(255, 20, 147, 0.4) 75%, 
      rgba(0, 255, 255, 0.4) 100%
    );
    border-radius: 12px;
    z-index: -1;
    animation: borderPulse 3s ease-in-out infinite;
  }
  
  @keyframes borderPulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }
`;

const ChartTitle = styled.h3`
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  text-align: center;
  margin-bottom: 20px;
  text-shadow: ${props => props.theme.effects.softGlow};
`;

const TrendIndicator = styled.div<{ trend: 'up' | 'down' | 'stable' }>`
  position: absolute;
  top: 20px;
  right: 20px;
  padding: 8px 12px;
  border-radius: 6px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: bold;
  
  ${props => {
    switch (props.trend) {
      case 'up':
        return `
          background: rgba(57, 255, 20, 0.2);
          color: ${props.theme.colors.success};
          border: 1px solid ${props.theme.colors.success};
        `;
      case 'down':
        return `
          background: rgba(255, 0, 64, 0.2);
          color: ${props.theme.colors.error};
          border: 1px solid ${props.theme.colors.error};
        `;
      default:
        return `
          background: rgba(0, 255, 255, 0.2);
          color: ${props.theme.colors.info};
          border: 1px solid ${props.theme.colors.info};
        `;
    }
  }}
  
  &::before {
    content: '${props => 
      props.trend === 'up' ? '↗️' : 
      props.trend === 'down' ? '↘️' : '➡️'
    }';
    margin-right: 4px;
  }
`;

const CustomTooltip = styled.div`
  background: rgba(0, 0, 0, 0.95);
  border: 2px solid #00ffff;
  border-radius: 8px;
  padding: 12px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #00ffff;
  box-shadow: 
    0 0 20px rgba(0, 255, 255, 0.5),
    inset 0 0 20px rgba(0, 255, 255, 0.1);
  
  .tooltip-label {
    color: #ff1493;
    font-weight: bold;
    margin-bottom: 8px;
  }
  
  .tooltip-value {
    color: #39ff14;
    margin: 4px 0;
  }
`;

const Legend = styled.div`
  display: flex;
  justify-content: center;
  gap: 20px;
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
    width: 16px;
    height: 3px;
    background: ${props => props.color};
    margin-right: 8px;
    border-radius: 2px;
    box-shadow: 0 0 6px ${props => props.color};
  }
`;

export const RetentionRateChart: React.FC<RetentionRateChartProps> = ({
  cohortData,
  width = 600,
  height = 400,
  showTrendAnalysis = true
}) => {
  const chartData = useMemo(() => {
    if (!cohortData || cohortData.length === 0) return [];
    
    // Transform cohort data for visualization
    const maxMonths = Math.max(...cohortData.map(c => c.months_tracked));
    const months = Array.from({ length: maxMonths }, (_, i) => i);
    
    return months.map(month => {
      const dataPoint: any = { month: `Month ${month}` };
      
      cohortData.forEach((cohort, index) => {
        const retentionRate = cohort.retention_rates[month.toString()];
        if (retentionRate !== undefined) {
          dataPoint[`cohort_${index}`] = (retentionRate * 100).toFixed(1);
        }
      });
      
      // Calculate average retention for this month
      const validRates = cohortData
        .map(c => c.retention_rates[month.toString()])
        .filter(rate => rate !== undefined);
      
      if (validRates.length > 0) {
        dataPoint.average = (validRates.reduce((sum, rate) => sum + rate, 0) / validRates.length * 100).toFixed(1);
      }
      
      return dataPoint;
    });
  }, [cohortData]);

  const trendAnalysis = useMemo(() => {
    if (!showTrendAnalysis || chartData.length < 2) return 'stable';
    
    const firstMonth = parseFloat(chartData[0]?.average || '0');
    const lastMonth = parseFloat(chartData[chartData.length - 1]?.average || '0');
    
    const difference = lastMonth - firstMonth;
    
    if (difference > 2) return 'up';
    if (difference < -2) return 'down';
    return 'stable';
  }, [chartData, showTrendAnalysis]);

  const cohortColors = [
    '#00ffff', '#ff1493', '#39ff14', '#ffff00', '#bf00ff', 
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7'
  ];

  const CustomTooltipComponent = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <CustomTooltip>
          <div className="tooltip-label">{label}</div>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="tooltip-value">
              {entry.name === 'average' ? 'Average' : `Cohort ${entry.name.split('_')[1]}`}: {entry.value}%
            </div>
          ))}
        </CustomTooltip>
      );
    }
    return null;
  };

  if (!cohortData || cohortData.length === 0) {
    return (
      <ChartContainer width={width} height={height}>
        <ChartTitle>No Retention Data Available</ChartTitle>
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
      <ChartTitle>Customer Retention Rate Analysis</ChartTitle>
      
      {showTrendAnalysis && (
        <TrendIndicator trend={trendAnalysis}>
          {trendAnalysis === 'up' ? 'Improving' : 
           trendAnalysis === 'down' ? 'Declining' : 'Stable'}
        </TrendIndicator>
      )}
      
      <ResponsiveContainer width="100%" height="80%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <defs>
            <linearGradient id="averageGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ff1493" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#ff1493" stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke="rgba(0, 255, 255, 0.2)" 
            strokeWidth={1}
          />
          
          <XAxis 
            dataKey="month" 
            stroke="#00ffff"
            fontSize={10}
            fontFamily="Courier New, monospace"
          />
          
          <YAxis 
            stroke="#00ffff"
            fontSize={10}
            fontFamily="Courier New, monospace"
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
          />
          
          <Tooltip content={<CustomTooltipComponent />} />
          
          {/* Individual cohort lines */}
          {cohortData.map((_, index) => (
            <Line
              key={`cohort_${index}`}
              type="monotone"
              dataKey={`cohort_${index}`}
              stroke={cohortColors[index % cohortColors.length]}
              strokeWidth={2}
              dot={{ r: 3, fill: cohortColors[index % cohortColors.length] }}
              strokeOpacity={0.7}
              connectNulls={false}
            />
          ))}
          
          {/* Average retention line */}
          <Line
            type="monotone"
            dataKey="average"
            stroke="#ff1493"
            strokeWidth={4}
            dot={{ r: 5, fill: "#ff1493", strokeWidth: 2, stroke: "#ffffff" }}
            strokeDasharray="none"
            filter="drop-shadow(0 0 6px #ff1493)"
          />
        </LineChart>
      </ResponsiveContainer>
      
      <Legend>
        <LegendItem color="#ff1493">Average Retention</LegendItem>
        {cohortData.slice(0, 3).map((cohort, index) => (
          <LegendItem key={index} color={cohortColors[index]}>
            {cohort.cohort_month}
          </LegendItem>
        ))}
      </Legend>
    </ChartContainer>
  );
};