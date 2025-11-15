import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard } from '../ui/CyberpunkCard';
import { CyberpunkButton } from '../ui/CyberpunkButton';
import { ConfidenceIndicator } from './ConfidenceIndicator';

interface ActionTimelineProps {
  recommendation: {
    recommendation_id: string;
    title: string;
    description: string;
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
    success_criteria: string[];
    risks: string[];
  };
  onStartImplementation?: (recommendationId: string, phase: string) => void;
  onMarkComplete?: (recommendationId: string, phase: string, actionIndex: number) => void;
  className?: string;
}

const TimelineContainer = styled(CyberpunkCard)`
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
`;

const TimelineHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 2rem;
  gap: 1rem;
`;

const HeaderLeft = styled.div`
  flex: 1;
`;

const TimelineTitle = styled.h3`
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.xl};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 0.5rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const PriorityBadge = styled.span<{ priority: string }>`
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.priority === 'critical' && css`
    background: rgba(255, 0, 64, 0.2);
    color: ${props.theme.colors.error};
    border: 1px solid ${props.theme.colors.error};
    animation: pulse 1s infinite;
  `}
  
  ${props => props.priority === 'high' && css`
    background: rgba(255, 255, 0, 0.2);
    color: ${props.theme.colors.warning};
    border: 1px solid ${props.theme.colors.warning};
  `}
  
  ${props => props.priority === 'medium' && css`
    background: rgba(0, 255, 255, 0.2);
    color: ${props.theme.colors.info};
    border: 1px solid ${props.theme.colors.info};
  `}
  
  ${props => props.priority === 'low' && css`
    background: rgba(57, 255, 20, 0.2);
    color: ${props.theme.colors.success};
    border: 1px solid ${props.theme.colors.success};
  `}
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
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

const TimelinePhases = styled.div`
  position: relative;
  margin: 2rem 0;
`;

const TimelineConnector = styled.div`
  position: absolute;
  left: 2rem;
  top: 0;
  bottom: 0;
  width: 2px;
  background: linear-gradient(
    to bottom,
    ${props => props.theme.colors.error},
    ${props => props.theme.colors.warning},
    ${props => props.theme.colors.info},
    ${props => props.theme.colors.success}
  );
  opacity: 0.5;
`;

const PhaseContainer = styled(motion.div)<{ phaseIndex: number }>`
  position: relative;
  margin-bottom: 2rem;
  padding-left: 4rem;
  
  &::before {
    content: '';
    position: absolute;
    left: 1.5rem;
    top: 1rem;
    width: 1rem;
    height: 1rem;
    border-radius: 50%;
    border: 2px solid;
    background: ${props => props.theme.colors.darkBg};
    z-index: 2;
    
    ${props => props.phaseIndex === 0 && css`
      border-color: ${props.theme.colors.error};
      box-shadow: 0 0 10px ${props.theme.colors.error};
    `}
    
    ${props => props.phaseIndex === 1 && css`
      border-color: ${props.theme.colors.warning};
      box-shadow: 0 0 8px ${props.theme.colors.warning};
    `}
    
    ${props => props.phaseIndex === 2 && css`
      border-color: ${props.theme.colors.info};
      box-shadow: 0 0 6px ${props.theme.colors.info};
    `}
    
    ${props => props.phaseIndex === 3 && css`
      border-color: ${props.theme.colors.success};
      box-shadow: 0 0 4px ${props.theme.colors.success};
    `}
  }
`;

const PhaseHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const PhaseTitle = styled.h4<{ phaseIndex: number }>`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.phaseIndex === 0 && css`
    color: ${props.theme.colors.error};
  `}
  
  ${props => props.phaseIndex === 1 && css`
    color: ${props.theme.colors.warning};
  `}
  
  ${props => props.phaseIndex === 2 && css`
    color: ${props.theme.colors.info};
  `}
  
  ${props => props.phaseIndex === 3 && css`
    color: ${props.theme.colors.success};
  `}
`;

const PhaseActions = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const ActionsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const ActionItem = styled(motion.div)<{ completed?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem;
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 4px;
  transition: all 0.3s ease;
  
  ${props => props.completed && css`
    background: rgba(57, 255, 20, 0.1);
    border-color: ${props.theme.colors.acidGreen};
    opacity: 0.7;
  `}
  
  &:hover {
    background: rgba(0, 255, 255, 0.1);
    border-color: ${props => props.theme.colors.neonBlue};
  }
`;

const ActionText = styled.span<{ completed?: boolean }>`
  color: ${props => props.theme.colors.primaryText};
  font-size: ${props => props.theme.typography.fontSize.md};
  flex: 1;
  
  ${props => props.completed && css`
    text-decoration: line-through;
    color: ${props.theme.colors.secondaryText};
  `}
`;

const ActionButton = styled(CyberpunkButton)`
  padding: 0.25rem 0.5rem;
  font-size: ${props => props.theme.typography.fontSize.xs};
`;

const RisksSection = styled.div`
  margin-top: 2rem;
  padding: 1rem;
  background: rgba(255, 0, 64, 0.05);
  border: 1px solid rgba(255, 0, 64, 0.2);
  border-radius: 4px;
`;

const RisksTitle = styled.h4`
  color: ${props => props.theme.colors.error};
  font-size: ${props => props.theme.typography.fontSize.md};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 0.5rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const RisksList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const RiskItem = styled.li`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  margin-bottom: 0.25rem;
  padding-left: 1rem;
  position: relative;
  
  &::before {
    content: '⚠';
    position: absolute;
    left: 0;
    color: ${props => props.theme.colors.error};
  }
`;

const SuccessCriteriaSection = styled.div`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(57, 255, 20, 0.05);
  border: 1px solid rgba(57, 255, 20, 0.2);
  border-radius: 4px;
`;

const SuccessTitle = styled.h4`
  color: ${props => props.theme.colors.acidGreen};
  font-size: ${props => props.theme.typography.fontSize.md};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 0.5rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const SuccessList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const SuccessItem = styled.li`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  margin-bottom: 0.25rem;
  padding-left: 1rem;
  position: relative;
  
  &::before {
    content: '✓';
    position: absolute;
    left: 0;
    color: ${props => props.theme.colors.acidGreen};
  }
`;

export const ActionTimeline: React.FC<ActionTimelineProps> = ({
  recommendation,
  onStartImplementation,
  onMarkComplete,
  className
}) => {
  const [completedActions, setCompletedActions] = useState<Set<string>>(new Set());
  const [activePhase, setActivePhase] = useState<string>('immediate');

  const phases = [
    { key: 'immediate', title: '🚨 Immediate Actions', subtitle: '0-7 days' },
    { key: 'short_term', title: '⚡ Short-term Actions', subtitle: '1-4 weeks' },
    { key: 'medium_term', title: '📋 Medium-term Actions', subtitle: '1-3 months' },
    { key: 'long_term', title: '📝 Long-term Actions', subtitle: '3+ months' }
  ];

  const handleStartPhase = (phase: string) => {
    setActivePhase(phase);
    if (onStartImplementation) {
      onStartImplementation(recommendation.recommendation_id, phase);
    }
  };

  const handleCompleteAction = (phase: string, actionIndex: number) => {
    const actionKey = `${phase}-${actionIndex}`;
    const newCompleted = new Set(completedActions);
    
    if (completedActions.has(actionKey)) {
      newCompleted.delete(actionKey);
    } else {
      newCompleted.add(actionKey);
    }
    
    setCompletedActions(newCompleted);
    
    if (onMarkComplete) {
      onMarkComplete(recommendation.recommendation_id, phase, actionIndex);
    }
  };

  const formatMetricValue = (key: string, value: number): string => {
    switch (key) {
      case 'time_to_impact':
        return `${value}d`;
      case 'success_probability':
        return `${(value * 100).toFixed(0)}%`;
      case 'roi_estimate':
        return `${value.toFixed(1)}x`;
      default:
        return value.toFixed(1);
    }
  };

  const getPhaseProgress = (phase: string): number => {
    const actions = recommendation.timeline[phase as keyof typeof recommendation.timeline] || [];
    const completed = actions.filter((_, index) => 
      completedActions.has(`${phase}-${index}`)
    ).length;
    return actions.length > 0 ? (completed / actions.length) * 100 : 0;
  };

  return (
    <TimelineContainer variant="hologram" className={className}>
      <TimelineHeader>
        <HeaderLeft>
          <TimelineTitle>{recommendation.title}</TimelineTitle>
          <PriorityBadge priority={recommendation.priority}>
            {recommendation.priority} Priority
          </PriorityBadge>
        </HeaderLeft>
        <ConfidenceIndicator 
          confidence={recommendation.confidence_score}
          size="lg"
          showLabel={true}
        />
      </TimelineHeader>

      <MetricsGrid>
        {Object.entries(recommendation.metrics).map(([key, value]) => (
          <MetricCard key={key}>
            <MetricValue>{formatMetricValue(key, value)}</MetricValue>
            <MetricLabel>{key.replace(/_/g, ' ')}</MetricLabel>
          </MetricCard>
        ))}
      </MetricsGrid>

      <TimelinePhases>
        <TimelineConnector />
        
        {phases.map((phase, phaseIndex) => {
          const actions = recommendation.timeline[phase.key as keyof typeof recommendation.timeline] || [];
          const progress = getPhaseProgress(phase.key);
          
          return (
            <PhaseContainer
              key={phase.key}
              phaseIndex={phaseIndex}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: phaseIndex * 0.1 }}
            >
              <PhaseHeader>
                <div>
                  <PhaseTitle phaseIndex={phaseIndex}>
                    {phase.title}
                  </PhaseTitle>
                  <div style={{ 
                    color: '#B0B0B0', 
                    fontSize: '0.875rem',
                    marginTop: '0.25rem'
                  }}>
                    {phase.subtitle} • {progress.toFixed(0)}% Complete
                  </div>
                </div>
                <PhaseActions>
                  <ActionButton
                    variant="secondary"
                    size="sm"
                    onClick={() => handleStartPhase(phase.key)}
                    disabled={activePhase === phase.key}
                  >
                    {activePhase === phase.key ? 'Active' : 'Start Phase'}
                  </ActionButton>
                </PhaseActions>
              </PhaseHeader>

              <ActionsList>
                <AnimatePresence>
                  {actions.map((action, actionIndex) => {
                    const actionKey = `${phase.key}-${actionIndex}`;
                    const isCompleted = completedActions.has(actionKey);
                    
                    return (
                      <ActionItem
                        key={actionIndex}
                        completed={isCompleted}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: actionIndex * 0.05 }}
                        whileHover={{ scale: 1.01 }}
                      >
                        <ActionText completed={isCompleted}>
                          {action}
                        </ActionText>
                        <ActionButton
                          variant={isCompleted ? "secondary" : "primary"}
                          size="sm"
                          onClick={() => handleCompleteAction(phase.key, actionIndex)}
                        >
                          {isCompleted ? '✓ Done' : 'Mark Done'}
                        </ActionButton>
                      </ActionItem>
                    );
                  })}
                </AnimatePresence>
              </ActionsList>
            </PhaseContainer>
          );
        })}
      </TimelinePhases>

      <RisksSection>
        <RisksTitle>⚠ Risk Factors</RisksTitle>
        <RisksList>
          {recommendation.risks.map((risk, index) => (
            <RiskItem key={index}>{risk}</RiskItem>
          ))}
        </RisksList>
      </RisksSection>

      <SuccessCriteriaSection>
        <SuccessTitle>🎯 Success Criteria</SuccessTitle>
        <SuccessList>
          {recommendation.success_criteria.map((criteria, index) => (
            <SuccessItem key={index}>{criteria}</SuccessItem>
          ))}
        </SuccessList>
      </SuccessCriteriaSection>
    </TimelineContainer>
  );
};