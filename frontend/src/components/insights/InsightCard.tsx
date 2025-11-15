import React from 'react';
import styled, { css } from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkCard } from '../ui/CyberpunkCard';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface BusinessInsight {
  insight_type: string;
  title: string;
  description: string;
  confidence: number;
  impact_score: number;
  supporting_data: Record<string, any>;
  recommended_actions: string[];
  urgency: 'low' | 'medium' | 'high' | 'critical';
}

interface InsightCardProps {
  insight: BusinessInsight;
  onActionClick?: (action: string) => void;
  className?: string;
}

const InsightContainer = styled(CyberpunkCard)<{ urgency: string }>`
  margin-bottom: 1rem;
  position: relative;
  
  ${props => props.urgency === 'critical' && css`
    border-color: ${props.theme.colors.error};
    box-shadow: 0 0 20px rgba(255, 0, 64, 0.3);
    
    &::after {
      border-color: ${props.theme.colors.error};
    }
  `}
  
  ${props => props.urgency === 'high' && css`
    border-color: ${props.theme.colors.warning};
    box-shadow: 0 0 15px rgba(255, 255, 0, 0.2);
    
    &::after {
      border-color: ${props.theme.colors.warning};
    }
  `}
  
  ${props => props.urgency === 'medium' && css`
    border-color: ${props.theme.colors.info};
    
    &::after {
      border-color: ${props.theme.colors.info};
    }
  `}
`;

const InsightHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
`;

const InsightTitle = styled.h3`
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const InsightType = styled.span<{ type: string }>`
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.type === 'opportunity' && css`
    background: rgba(57, 255, 20, 0.2);
    color: ${props.theme.colors.acidGreen};
    border: 1px solid ${props.theme.colors.acidGreen};
  `}
  
  ${props => props.type === 'risk' && css`
    background: rgba(255, 0, 64, 0.2);
    color: ${props.theme.colors.error};
    border: 1px solid ${props.theme.colors.error};
  `}
  
  ${props => props.type === 'recommendation' && css`
    background: rgba(0, 255, 255, 0.2);
    color: ${props.theme.colors.neonBlue};
    border: 1px solid ${props.theme.colors.neonBlue};
  `}
  
  ${props => props.type === 'performance' && css`
    background: rgba(255, 20, 147, 0.2);
    color: ${props.theme.colors.hotPink};
    border: 1px solid ${props.theme.colors.hotPink};
  `}
`;

const InsightDescription = styled.p`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.md};
  line-height: 1.6;
  margin-bottom: 1rem;
`;

const MetricsContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
`;

const MetricItem = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.5rem;
  background: rgba(0, 255, 255, 0.1);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 4px;
  min-width: 80px;
`;

const MetricValue = styled.span`
  color: ${props => props.theme.colors.neonBlue};
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-family: ${props => props.theme.typography.fontFamily.mono};
`;

const MetricLabel = styled.span`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.xs};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const ActionsContainer = styled.div`
  margin-top: 1rem;
`;

const ActionsTitle = styled.h4`
  color: ${props => props.theme.colors.primaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const ActionsList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const ActionItem = styled(motion.li)`
  padding: 0.5rem;
  margin-bottom: 0.25rem;
  background: rgba(0, 255, 255, 0.05);
  border-left: 3px solid ${props => props.theme.colors.neonBlue};
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(0, 255, 255, 0.1);
    color: ${props => props.theme.colors.primaryText};
    border-left-color: ${props => props.theme.colors.hotPink};
    transform: translateX(4px);
  }
  
  &::before {
    content: '▶';
    color: ${props => props.theme.colors.neonBlue};
    margin-right: 0.5rem;
    font-size: 0.8em;
  }
`;

const UrgencyIndicator = styled.div<{ urgency: string }>`
  position: absolute;
  top: -2px;
  right: -2px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  
  ${props => props.urgency === 'critical' && css`
    background: ${props.theme.colors.error};
    box-shadow: 0 0 10px ${props.theme.colors.error};
    animation: pulse 1s infinite;
  `}
  
  ${props => props.urgency === 'high' && css`
    background: ${props.theme.colors.warning};
    box-shadow: 0 0 8px ${props.theme.colors.warning};
  `}
  
  ${props => props.urgency === 'medium' && css`
    background: ${props.theme.colors.info};
    box-shadow: 0 0 6px ${props.theme.colors.info};
  `}
  
  ${props => props.urgency === 'low' && css`
    background: ${props.theme.colors.success};
    box-shadow: 0 0 4px ${props.theme.colors.success};
  `}
`;

export const InsightCard: React.FC<InsightCardProps> = ({
  insight,
  onActionClick,
  className
}) => {
  const formatMetricValue = (value: any): string => {
    if (typeof value === 'number') {
      if (value < 1) {
        return `${(value * 100).toFixed(1)}%`;
      }
      return value.toFixed(1);
    }
    return String(value);
  };

  const handleActionClick = (action: string) => {
    if (onActionClick) {
      onActionClick(action);
    }
  };

  return (
    <InsightContainer
      variant="glass"
      urgency={insight.urgency}
      className={className}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <UrgencyIndicator urgency={insight.urgency} />
      
      <InsightHeader>
        <div>
          <InsightTitle>{insight.title}</InsightTitle>
          <InsightType type={insight.insight_type}>
            {insight.insight_type}
          </InsightType>
        </div>
        <ConfidenceIndicator 
          confidence={insight.confidence}
          size="sm"
        />
      </InsightHeader>
      
      <InsightDescription>
        {insight.description}
      </InsightDescription>
      
      <MetricsContainer>
        <MetricItem>
          <MetricValue>{formatMetricValue(insight.confidence)}</MetricValue>
          <MetricLabel>Confidence</MetricLabel>
        </MetricItem>
        <MetricItem>
          <MetricValue>{insight.impact_score.toFixed(1)}</MetricValue>
          <MetricLabel>Impact</MetricLabel>
        </MetricItem>
        {Object.entries(insight.supporting_data).slice(0, 2).map(([key, value]) => (
          <MetricItem key={key}>
            <MetricValue>{formatMetricValue(value)}</MetricValue>
            <MetricLabel>{key.replace(/_/g, ' ')}</MetricLabel>
          </MetricItem>
        ))}
      </MetricsContainer>
      
      {insight.recommended_actions.length > 0 && (
        <ActionsContainer>
          <ActionsTitle>Recommended Actions</ActionsTitle>
          <ActionsList>
            {insight.recommended_actions.map((action, index) => (
              <ActionItem
                key={index}
                onClick={() => handleActionClick(action)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {action}
              </ActionItem>
            ))}
          </ActionsList>
        </ActionsContainer>
      )}
    </InsightContainer>
  );
};