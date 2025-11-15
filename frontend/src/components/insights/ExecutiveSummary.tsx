import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard } from '../ui/CyberpunkCard';
import { CyberpunkButton } from '../ui/CyberpunkButton';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface BusinessInsightsResult {
  executive_summary: string;
  key_findings: string[];
  confidence_score: number;
  generated_at: string;
  performance_analysis: {
    revenue_growth_rate: number;
    revenue_trend: string;
    performance_score: number;
  };
  growth_indicators: {
    monthly_growth_rate: number;
    growth_acceleration: number;
    growth_sustainability_score: number;
  };
  risk_assessment: {
    overall_risk_score: number;
    risk_level: string;
    primary_risks: Array<{
      risk_type: string;
      description: string;
      impact: string;
      probability: string;
    }>;
  };
  opportunity_identification: {
    opportunity_score: number;
    opportunities: Array<{
      opportunity_type: string;
      description: string;
      potential_impact: string;
    }>;
  };
}

interface ExecutiveSummaryProps {
  insights: BusinessInsightsResult;
  onExport?: () => void;
  onRefresh?: () => void;
  className?: string;
}

const SummaryContainer = styled(CyberpunkCard)`
  margin-bottom: 2rem;
`;

const SummaryHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.5rem;
  gap: 1rem;
`;

const HeaderLeft = styled.div`
  flex: 1;
`;

const SummaryTitle = styled.h2`
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.xxl};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 0.5rem 0;
  text-transform: uppercase;
  letter-spacing: 2px;
  background: ${props => props.theme.effects.primaryGradient};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const SummaryMeta = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
`;

const GeneratedTime = styled.span`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-family: ${props => props.theme.typography.fontFamily.mono};
`;

const StatusIndicator = styled.div<{ status: string }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.status === 'healthy' && css`
    background: rgba(57, 255, 20, 0.2);
    color: ${props.theme.colors.acidGreen};
    border: 1px solid ${props.theme.colors.acidGreen};
  `}
  
  ${props => props.status === 'warning' && css`
    background: rgba(255, 255, 0, 0.2);
    color: ${props.theme.colors.warning};
    border: 1px solid ${props.theme.colors.warning};
  `}
  
  ${props => props.status === 'critical' && css`
    background: rgba(255, 0, 64, 0.2);
    color: ${props.theme.colors.error};
    border: 1px solid ${props.theme.colors.error};
    animation: pulse 1s infinite;
  `}
  
  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 6px currentColor;
  }
`;

const SummaryContent = styled.div`
  margin-bottom: 1.5rem;
`;

const SummaryText = styled(motion.div)`
  color: ${props => props.theme.colors.primaryText};
  font-size: ${props => props.theme.typography.fontSize.lg};
  line-height: 1.8;
  margin-bottom: 1.5rem;
  padding: 1.5rem;
  background: rgba(0, 255, 255, 0.05);
  border-left: 4px solid ${props => props.theme.colors.neonBlue};
  border-radius: 8px;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '"';
    position: absolute;
    top: -10px;
    left: 15px;
    font-size: 4rem;
    color: ${props => props.theme.colors.neonBlue};
    opacity: 0.2;
    font-family: ${props => props.theme.typography.fontFamily.display};
  }
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, 
      transparent 0%, 
      ${props => props.theme.colors.neonBlue} 50%, 
      transparent 100%
    );
    animation: scan 3s ease-in-out infinite;
  }
  
  @keyframes scan {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
  }
`;

const KeyMetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const MetricCard = styled(motion.div)<{ trend: string }>`
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  
  ${props => props.trend === 'positive' && css`
    border-color: ${props.theme.colors.acidGreen};
    box-shadow: 0 0 10px rgba(57, 255, 20, 0.2);
  `}
  
  ${props => props.trend === 'negative' && css`
    border-color: ${props.theme.colors.error};
    box-shadow: 0 0 10px rgba(255, 0, 64, 0.2);
  `}
  
  ${props => props.trend === 'neutral' && css`
    border-color: ${props.theme.colors.warning};
    box-shadow: 0 0 10px rgba(255, 255, 0, 0.2);
  `}
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: currentColor;
  }
`;

const MetricValue = styled.div<{ trend: string }>`
  font-size: ${props => props.theme.typography.fontSize.xxl};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  margin-bottom: 0.5rem;
  
  ${props => props.trend === 'positive' && css`
    color: ${props.theme.colors.acidGreen};
  `}
  
  ${props => props.trend === 'negative' && css`
    color: ${props.theme.colors.error};
  `}
  
  ${props => props.trend === 'neutral' && css`
    color: ${props.theme.colors.warning};
  `}
`;

const MetricLabel = styled.div`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 0.25rem;
`;

const MetricTrend = styled.div<{ trend: string }>`
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.trend === 'positive' && css`
    color: ${props.theme.colors.acidGreen};
    &::before { content: '↗ '; }
  `}
  
  ${props => props.trend === 'negative' && css`
    color: ${props.theme.colors.error};
    &::before { content: '↘ '; }
  `}
  
  ${props => props.trend === 'neutral' && css`
    color: ${props.theme.colors.warning};
    &::before { content: '→ '; }
  `}
`;

const KeyFindingsContainer = styled.div`
  margin-bottom: 1.5rem;
`;

const FindingsTitle = styled.h3`
  color: ${props => props.theme.colors.primaryText};
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 1rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const FindingsToggle = styled(CyberpunkButton)`
  margin-bottom: 1rem;
`;

const FindingsList = styled(motion.ul)`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const FindingItem = styled(motion.li)`
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: rgba(255, 20, 147, 0.05);
  border-left: 3px solid ${props => props.theme.colors.hotPink};
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.md};
  border-radius: 4px;
  
  &::before {
    content: '◆';
    color: ${props => props.theme.colors.hotPink};
    margin-right: 0.5rem;
    font-weight: bold;
  }
`;

const ActionsContainer = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  flex-wrap: wrap;
`;

export const ExecutiveSummary: React.FC<ExecutiveSummaryProps> = ({
  insights,
  onExport,
  onRefresh,
  className
}) => {
  const [showFindings, setShowFindings] = useState(false);
  
  const getOverallStatus = (): string => {
    const riskLevel = insights.risk_assessment.risk_level;
    const performanceScore = insights.performance_analysis.performance_score;
    
    if (riskLevel === 'critical' || performanceScore < 3) return 'critical';
    if (riskLevel === 'high' || performanceScore < 6) return 'warning';
    return 'healthy';
  };

  const getTrendDirection = (value: number): string => {
    if (value > 0.02) return 'positive';
    if (value < -0.02) return 'negative';
    return 'neutral';
  };

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatScore = (value: number): string => {
    return `${value.toFixed(1)}/10`;
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <SummaryContainer
      variant="hologram"
      className={className}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
    >
      <SummaryHeader>
        <HeaderLeft>
          <SummaryTitle>Executive Summary</SummaryTitle>
          <SummaryMeta>
            <GeneratedTime>
              Generated: {formatDate(insights.generated_at)}
            </GeneratedTime>
            <StatusIndicator status={getOverallStatus()}>
              {getOverallStatus()} Status
            </StatusIndicator>
          </SummaryMeta>
        </HeaderLeft>
        <ConfidenceIndicator 
          confidence={insights.confidence_score}
          size="lg"
        />
      </SummaryHeader>
      
      <SummaryContent>
        <SummaryText>
          {insights.executive_summary}
        </SummaryText>
        
        <KeyMetricsGrid>
          <MetricCard
            trend={getTrendDirection(insights.performance_analysis.revenue_growth_rate)}
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <MetricValue trend={getTrendDirection(insights.performance_analysis.revenue_growth_rate)}>
              {formatPercentage(insights.performance_analysis.revenue_growth_rate)}
            </MetricValue>
            <MetricLabel>Revenue Growth</MetricLabel>
            <MetricTrend trend={getTrendDirection(insights.performance_analysis.revenue_growth_rate)}>
              {insights.performance_analysis.revenue_trend}
            </MetricTrend>
          </MetricCard>
          
          <MetricCard
            trend={insights.performance_analysis.performance_score > 6 ? 'positive' : 
                   insights.performance_analysis.performance_score > 4 ? 'neutral' : 'negative'}
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <MetricValue trend={insights.performance_analysis.performance_score > 6 ? 'positive' : 
                               insights.performance_analysis.performance_score > 4 ? 'neutral' : 'negative'}>
              {formatScore(insights.performance_analysis.performance_score)}
            </MetricValue>
            <MetricLabel>Performance Score</MetricLabel>
            <MetricTrend trend={insights.performance_analysis.performance_score > 6 ? 'positive' : 
                               insights.performance_analysis.performance_score > 4 ? 'neutral' : 'negative'}>
              Overall Performance
            </MetricTrend>
          </MetricCard>
          
          <MetricCard
            trend={insights.risk_assessment.risk_level === 'low' ? 'positive' :
                   insights.risk_assessment.risk_level === 'medium' ? 'neutral' : 'negative'}
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <MetricValue trend={insights.risk_assessment.risk_level === 'low' ? 'positive' :
                               insights.risk_assessment.risk_level === 'medium' ? 'neutral' : 'negative'}>
              {formatScore(insights.risk_assessment.overall_risk_score)}
            </MetricValue>
            <MetricLabel>Risk Level</MetricLabel>
            <MetricTrend trend={insights.risk_assessment.risk_level === 'low' ? 'positive' :
                               insights.risk_assessment.risk_level === 'medium' ? 'neutral' : 'negative'}>
              {insights.risk_assessment.risk_level}
            </MetricTrend>
          </MetricCard>
          
          <MetricCard
            trend={insights.opportunity_identification.opportunity_score > 7 ? 'positive' :
                   insights.opportunity_identification.opportunity_score > 4 ? 'neutral' : 'negative'}
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <MetricValue trend={insights.opportunity_identification.opportunity_score > 7 ? 'positive' :
                               insights.opportunity_identification.opportunity_score > 4 ? 'neutral' : 'negative'}>
              {formatScore(insights.opportunity_identification.opportunity_score)}
            </MetricValue>
            <MetricLabel>Opportunity Score</MetricLabel>
            <MetricTrend trend={insights.opportunity_identification.opportunity_score > 7 ? 'positive' :
                               insights.opportunity_identification.opportunity_score > 4 ? 'neutral' : 'negative'}>
              Growth Potential
            </MetricTrend>
          </MetricCard>
        </KeyMetricsGrid>
        
        <KeyFindingsContainer>
          <FindingsTitle>Key Findings</FindingsTitle>
          <FindingsToggle
            variant="ghost"
            size="sm"
            onClick={() => setShowFindings(!showFindings)}
          >
            {showFindings ? 'Hide' : 'Show'} Detailed Findings ({insights.key_findings.length})
          </FindingsToggle>
          
          <AnimatePresence>
            {showFindings && (
              <FindingsList
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                {insights.key_findings.map((finding, index) => (
                  <FindingItem
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    {finding}
                  </FindingItem>
                ))}
              </FindingsList>
            )}
          </AnimatePresence>
        </KeyFindingsContainer>
      </SummaryContent>
      
      <ActionsContainer>
        {onRefresh && (
          <CyberpunkButton
            variant="secondary"
            size="sm"
            onClick={onRefresh}
          >
            Refresh Analysis
          </CyberpunkButton>
        )}
        {onExport && (
          <CyberpunkButton
            variant="primary"
            size="sm"
            onClick={onExport}
          >
            Export Summary
          </CyberpunkButton>
        )}
      </ActionsContainer>
    </SummaryContainer>
  );
};