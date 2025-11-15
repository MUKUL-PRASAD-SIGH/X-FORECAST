import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard } from '../ui/CyberpunkCard';
import { CyberpunkButton } from '../ui/CyberpunkButton';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface TimelineBasedRecommendation {
  recommendation_id: string;
  title: string;
  description: string;
  recommendation_type: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  confidence_score: number;
  timeline: {
    immediate: string[];
    short_term: string[];
    medium_term: string[];
    long_term: string[];
  };
  metrics: {
    expected_impact: number;
    implementation_difficulty: number;
    resource_requirement: number;
    time_to_impact: number;
    success_probability: number;
    roi_estimate: number;
  };
  dependencies: string[];
  risks: string[];
  success_criteria: string[];
  created_at: string;
  expires_at?: string;
}

interface RecommendationCardProps {
  recommendation: TimelineBasedRecommendation;
  onImplement?: (recommendationId: string) => void;
  onViewDetails?: (recommendationId: string) => void;
  className?: string;
}

const RecommendationContainer = styled(CyberpunkCard)<{ priority: string }>`
  margin-bottom: 1.5rem;
  position: relative;
  
  ${props => props.priority === 'critical' && css`
    border-color: ${props.theme.colors.error};
    box-shadow: 0 0 25px rgba(255, 0, 64, 0.4);
    
    &::after {
      border-color: ${props.theme.colors.error};
      box-shadow: 0 0 10px ${props.theme.colors.error};
    }
  `}
  
  ${props => props.priority === 'high' && css`
    border-color: ${props.theme.colors.warning};
    box-shadow: 0 0 20px rgba(255, 255, 0, 0.3);
    
    &::after {
      border-color: ${props.theme.colors.warning};
    }
  `}
  
  ${props => props.priority === 'medium' && css`
    border-color: ${props.theme.colors.info};
    
    &::after {
      border-color: ${props.theme.colors.info};
    }
  `}
`;

const RecommendationHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
  gap: 1rem;
`;

const HeaderLeft = styled.div`
  flex: 1;
`;

const RecommendationTitle = styled.h3`
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 0.5rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const RecommendationMeta = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
`;

const PriorityBadge = styled.span<{ priority: string }>`
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${(props) => props.priority === 'critical' && css`
    background: rgba(255, 0, 64, 0.2);
    color: ${props.theme.colors.error};
    border: 1px solid ${props.theme.colors.error};
    animation: pulse 1s infinite;
  `}
  
  ${(props) => props.priority === 'high' && css`
    background: rgba(255, 255, 0, 0.2);
    color: ${props.theme.colors.warning};
    border: 1px solid ${props.theme.colors.warning};
  `}
  
  ${(props) => props.priority === 'medium' && css`
    background: rgba(0, 255, 255, 0.2);
    color: ${props.theme.colors.info};
    border: 1px solid ${props.theme.colors.info};
  `}
  
  ${(props) => props.priority === 'low' && css`
    background: rgba(57, 255, 20, 0.2);
    color: ${props.theme.colors.success};
    border: 1px solid ${props.theme.colors.success};
  `}
`;

const TypeBadge = styled.span`
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  text-transform: uppercase;
  letter-spacing: 1px;
  background: rgba(255, 20, 147, 0.2);
  color: ${props => props.theme.colors.hotPink};
  border: 1px solid ${props => props.theme.colors.hotPink};
`;

const RecommendationDescription = styled.p`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.md};
  line-height: 1.6;
  margin-bottom: 1.5rem;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const MetricCard = styled.div`
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 4px;
  padding: 0.75rem;
  text-align: center;
`;

const MetricValue = styled.div`
  color: ${props => props.theme.colors.neonBlue};
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  margin-bottom: 0.25rem;
`;

const MetricLabel = styled.div`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.xs};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const TimelineContainer = styled.div`
  margin-bottom: 1.5rem;
`;

const TimelineHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const TimelineTitle = styled.h4`
  color: ${props => props.theme.colors.primaryText};
  font-size: ${props => props.theme.typography.fontSize.md};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const TimelineToggle = styled(CyberpunkButton)`
  padding: 0.25rem 0.5rem;
  font-size: ${props => props.theme.typography.fontSize.xs};
`;

const TimelineContent = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const TimelinePhase = styled.div`
  background: rgba(255, 20, 147, 0.05);
  border: 1px solid rgba(255, 20, 147, 0.2);
  border-radius: 4px;
  padding: 1rem;
`;

const PhaseTitle = styled.h5`
  color: ${props => props.theme.colors.hotPink};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 0.5rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const PhaseActions = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const PhaseAction = styled.li`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  margin-bottom: 0.25rem;
  padding-left: 1rem;
  position: relative;
  
  &::before {
    content: '▶';
    position: absolute;
    left: 0;
    color: ${props => props.theme.colors.neonBlue};
    font-size: 0.8em;
  }
`;

const ActionsContainer = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  margin-top: 1rem;
  flex-wrap: wrap;
`;

const ExpirationWarning = styled.div<{ isExpiring: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.sm};
  margin-bottom: 1rem;
  
  ${props => props.isExpiring && css`
    background: rgba(255, 255, 0, 0.1);
    border: 1px solid ${props.theme.colors.warning};
    color: ${props.theme.colors.warning};
    
    &::before {
      content: '⚠';
      font-size: 1.2em;
    }
  `}
`;

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendation,
  onImplement,
  onViewDetails,
  className
}) => {
  const [showTimeline, setShowTimeline] = useState(false);
  
  const formatMetricValue = (key: string, value: number): string => {
    switch (key) {
      case 'time_to_impact':
        return `${value}d`;
      case 'success_probability':
      case 'confidence_score':
        return `${(value * 100).toFixed(0)}%`;
      case 'roi_estimate':
        return `${value.toFixed(1)}x`;
      default:
        return value.toFixed(1);
    }
  };

  const isExpiring = recommendation.expires_at && 
    new Date(recommendation.expires_at) < new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);

  const handleImplement = () => {
    if (onImplement) {
      onImplement(recommendation.recommendation_id);
    }
  };

  const handleViewDetails = () => {
    if (onViewDetails) {
      onViewDetails(recommendation.recommendation_id);
    }
  };

  return (
    <RecommendationContainer
      variant="glass"
      priority={recommendation.priority}
      className={className}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {isExpiring && (
        <ExpirationWarning isExpiring={true}>
          Recommendation expires soon - consider implementing immediately
        </ExpirationWarning>
      )}
      
      <RecommendationHeader>
        <HeaderLeft>
          <RecommendationTitle>{recommendation.title}</RecommendationTitle>
          <RecommendationMeta>
            <PriorityBadge priority={recommendation.priority}>
              {recommendation.priority} Priority
            </PriorityBadge>
            <TypeBadge>
              {recommendation.recommendation_type.replace(/_/g, ' ')}
            </TypeBadge>
          </RecommendationMeta>
        </HeaderLeft>
        <ConfidenceIndicator 
          confidence={recommendation.confidence_score}
          size="md"
        />
      </RecommendationHeader>
      
      <RecommendationDescription>
        {recommendation.description}
      </RecommendationDescription>
      
      <MetricsGrid>
        {Object.entries(recommendation.metrics).map(([key, value]) => (
          <MetricCard key={key}>
            <MetricValue>{formatMetricValue(key, value)}</MetricValue>
            <MetricLabel>{key.replace(/_/g, ' ')}</MetricLabel>
          </MetricCard>
        ))}
      </MetricsGrid>
      
      <TimelineContainer>
        <TimelineHeader>
          <TimelineTitle>Implementation Timeline</TimelineTitle>
          <TimelineToggle
            variant="ghost"
            size="sm"
            onClick={() => setShowTimeline(!showTimeline)}
          >
            {showTimeline ? 'Hide' : 'Show'} Timeline
          </TimelineToggle>
        </TimelineHeader>
        
        <AnimatePresence>
          {showTimeline && (
            <TimelineContent
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              {Object.entries(recommendation.timeline).map(([phase, actions]) => (
                <TimelinePhase key={phase}>
                  <PhaseTitle>{phase.replace(/_/g, ' ')}</PhaseTitle>
                  <PhaseActions>
                    {actions.map((action, index) => (
                      <PhaseAction key={index}>{action}</PhaseAction>
                    ))}
                  </PhaseActions>
                </TimelinePhase>
              ))}
            </TimelineContent>
          )}
        </AnimatePresence>
      </TimelineContainer>
      
      <ActionsContainer>
        <CyberpunkButton
          variant="secondary"
          size="sm"
          onClick={handleViewDetails}
        >
          View Details
        </CyberpunkButton>
        <CyberpunkButton
          variant="primary"
          size="sm"
          onClick={handleImplement}
        >
          Implement
        </CyberpunkButton>
      </ActionsContainer>
    </RecommendationContainer>
  );
};