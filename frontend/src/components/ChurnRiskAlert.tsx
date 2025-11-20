import React, { useState, useEffect } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';

interface ChurnRisk {
  customer_id: string;
  churn_probability: number;
  risk_level: 'Low' | 'Medium' | 'High' | 'Critical';
  key_risk_factors: string[];
  days_since_last_purchase: number;
  purchase_trend: string;
  recommended_actions: string[];
  confidence_score: number;
}

interface ChurnRiskAlertProps {
  churnRisks: ChurnRisk[];
  onActionTaken?: (customerId: string, action: string) => void;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const glitchAnimation = keyframes`
  0% { transform: translate(0); }
  20% { transform: translate(-2px, 2px); }
  40% { transform: translate(-2px, -2px); }
  60% { transform: translate(2px, 2px); }
  80% { transform: translate(2px, -2px); }
  100% { transform: translate(0); }
`;

const pulseGlow = keyframes`
  0%, 100% { 
    box-shadow: 0 0 20px rgba(255, 0, 64, 0.4);
    opacity: 0.8;
  }
  50% { 
    box-shadow: 0 0 40px rgba(255, 0, 64, 0.8);
    opacity: 1;
  }
`;

const AlertContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.md};
  max-height: 600px;
  overflow-y: auto;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, #ff0040, #ff1493);
    border-radius: 4px;
    
    &:hover {
      background: linear-gradient(45deg, #ff1493, #ff0040);
    }
  }
`;

const AlertHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const AlertTitle = styled.h3`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  color: ${props => props.theme.colors.error};
  margin: 0;
  text-shadow: ${props => props.theme.effects.softGlow};
  
  &::before {
    content: '⚠️ ';
    margin-right: 8px;
  }
`;

const RiskSummary = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
`;

const RiskCount = styled.span<{ level: string }>`
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: bold;
  
  ${props => {
    switch (props.level) {
      case 'critical':
        return css`
          background: rgba(255, 0, 64, 0.2);
          color: ${props.theme.colors.error};
          border: 1px solid ${props.theme.colors.error};
        `;
      case 'high':
        return css`
          background: rgba(255, 255, 0, 0.2);
          color: ${props.theme.colors.warning};
          border: 1px solid ${props.theme.colors.warning};
        `;
      case 'medium':
        return css`
          background: rgba(0, 255, 255, 0.2);
          color: ${props.theme.colors.info};
          border: 1px solid ${props.theme.colors.info};
        `;
      default:
        return css`
          background: rgba(57, 255, 20, 0.2);
          color: ${props.theme.colors.success};
          border: 1px solid ${props.theme.colors.success};
        `;
    }
  }}
`;

const RiskCard = styled(CyberpunkCard)<{ riskLevel: string; urgent?: boolean }>`
  position: relative;
  transition: all 0.3s ease;
  
  ${props => {
    switch (props.riskLevel) {
      case 'Critical':
        return css`
          border-color: ${props.theme.colors.error};
          background: linear-gradient(135deg, 
            rgba(255, 0, 64, 0.1) 0%, 
            rgba(0, 0, 0, 0.8) 100%
          );
          ${props.urgent && css`
            animation: ${pulseGlow} 2s ease-in-out infinite;
          `}
        `;
      case 'High':
        return css`
          border-color: ${props.theme.colors.warning};
          background: linear-gradient(135deg, 
            rgba(255, 255, 0, 0.1) 0%, 
            rgba(0, 0, 0, 0.8) 100%
          );
        `;
      case 'Medium':
        return css`
          border-color: ${props.theme.colors.info};
          background: linear-gradient(135deg, 
            rgba(0, 255, 255, 0.1) 0%, 
            rgba(0, 0, 0, 0.8) 100%
          );
        `;
      default:
        return css`
          border-color: ${props.theme.colors.success};
          background: linear-gradient(135deg, 
            rgba(57, 255, 20, 0.1) 0%, 
            rgba(0, 0, 0, 0.8) 100%
          );
        `;
    }
  }}
  
  &:hover {
    transform: translateY(-2px);
    ${props => props.riskLevel === 'Critical' && css`
      animation: ${glitchAnimation} 0.3s ease-in-out;
    `}
  }
`;

const RiskHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const CustomerInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const CustomerId = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  color: ${props => props.theme.colors.primaryText};
  font-weight: bold;
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const LastPurchase = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.xs};
`;

const RiskBadge = styled.div<{ level: string; probability: number }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
`;

const RiskLevel = styled.div<{ level: string }>`
  padding: 6px 12px;
  border-radius: 6px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: bold;
  text-transform: uppercase;
  
  ${props => {
    switch (props.level) {
      case 'Critical':
        return css`
          background: rgba(255, 0, 64, 0.3);
          color: ${props.theme.colors.error};
          border: 2px solid ${props.theme.colors.error};
          box-shadow: 0 0 15px rgba(255, 0, 64, 0.5);
        `;
      case 'High':
        return css`
          background: rgba(255, 255, 0, 0.3);
          color: ${props.theme.colors.warning};
          border: 2px solid ${props.theme.colors.warning};
          box-shadow: 0 0 15px rgba(255, 255, 0, 0.3);
        `;
      case 'Medium':
        return css`
          background: rgba(0, 255, 255, 0.3);
          color: ${props.theme.colors.info};
          border: 2px solid ${props.theme.colors.info};
          box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        `;
      default:
        return css`
          background: rgba(57, 255, 20, 0.3);
          color: ${props.theme.colors.success};
          border: 2px solid ${props.theme.colors.success};
          box-shadow: 0 0 15px rgba(57, 255, 20, 0.3);
        `;
    }
  }}
`;

const ProbabilityBar = styled.div<{ probability: number }>`
  width: 60px;
  height: 8px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: ${props => props.probability}%;
    background: linear-gradient(90deg, 
      ${props => props.probability >= 80 ? '#ff0040' : 
                 props.probability >= 60 ? '#ffff00' : 
                 props.probability >= 40 ? '#00ffff' : '#39ff14'} 0%,
      ${props => props.probability >= 80 ? '#ff6b6b' : 
                 props.probability >= 60 ? '#fff700' : 
                 props.probability >= 40 ? '#4dd0e1' : '#66ff66'} 100%
    );
    border-radius: 4px;
    animation: ${props => props.probability >= 80 ? css`${pulseGlow} 1.5s ease-in-out infinite` : 'none'};
  }
`;

const ProbabilityText = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
`;

const RiskFactors = styled.div`
  margin: ${props => props.theme.spacing.sm} 0;
`;

const FactorsTitle = styled.h4`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.neonBlue};
  margin: 0 0 8px 0;
`;

const FactorsList = styled.ul`
  margin: 0;
  padding-left: ${props => props.theme.spacing.md};
  
  li {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.xs};
    color: ${props => props.theme.colors.secondaryText};
    margin-bottom: 4px;
    
    &::marker {
      color: ${props => props.theme.colors.error};
    }
  }
`;

const ActionsSection = styled.div`
  margin-top: ${props => props.theme.spacing.sm};
  padding-top: ${props => props.theme.spacing.sm};
  border-top: 1px solid rgba(0, 255, 255, 0.2);
`;

const ActionsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.xs};
  margin-top: ${props => props.theme.spacing.xs};
`;

const ActionButton = styled(CyberpunkButton)<{ actionType: string }>`
  font-size: ${props => props.theme.typography.fontSize.xs};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  
  ${props => {
    switch (props.actionType) {
      case 'immediate':
        return css`
          background: linear-gradient(135deg, ${props.theme.colors.error}, ${props.theme.colors.hotPink});
          border-color: ${props.theme.colors.error};
        `;
      case 'proactive':
        return css`
          background: linear-gradient(135deg, ${props.theme.colors.warning}, ${props.theme.colors.cyberYellow});
          border-color: ${props.theme.colors.warning};
          color: ${props.theme.colors.darkBg};
        `;
      default:
        return css`
          background: transparent;
          border-color: ${props.theme.colors.info};
          color: ${props.theme.colors.info};
        `;
    }
  }}
`;

const TrendIndicator = styled.div<{ trend: string }>`
  display: flex;
  align-items: center;
  gap: 4px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  
  ${props => {
    switch (props.trend) {
      case 'Declining':
        return css`
          color: ${props.theme.colors.error};
          &::before { content: '📉'; }
        `;
      case 'Increasing':
        return css`
          color: ${props.theme.colors.success};
          &::before { content: '📈'; }
        `;
      default:
        return css`
          color: ${props.theme.colors.info};
          &::before { content: '➡️'; }
        `;
    }
  }}
`;

export const ChurnRiskAlert: React.FC<ChurnRiskAlertProps> = ({
  churnRisks,
  onActionTaken,
  autoRefresh = true,
  refreshInterval = 30000
}) => {
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      // Trigger refresh logic here if needed
      console.log('Auto-refreshing churn risk data...');
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  const riskCounts = React.useMemo(() => {
    return churnRisks.reduce((counts, risk) => {
      const level = risk.risk_level.toLowerCase();
      counts[level] = (counts[level] || 0) + 1;
      return counts;
    }, {} as Record<string, number>);
  }, [churnRisks]);

  const sortedRisks = React.useMemo(() => {
    const riskOrder = { 'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1 };
    return [...churnRisks].sort((a, b) => {
      const orderDiff = riskOrder[b.risk_level] - riskOrder[a.risk_level];
      if (orderDiff !== 0) return orderDiff;
      return b.churn_probability - a.churn_probability;
    });
  }, [churnRisks]);

  const toggleExpanded = (customerId: string) => {
    setExpandedCards(prev => {
      const newSet = new Set(prev);
      if (newSet.has(customerId)) {
        newSet.delete(customerId);
      } else {
        newSet.add(customerId);
      }
      return newSet;
    });
  };

  const handleAction = async (customerId: string, action: string) => {
    setActionLoading(`${customerId}-${action}`);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      if (onActionTaken) {
        onActionTaken(customerId, action);
      }
      
      console.log(`Action taken for customer ${customerId}: ${action}`);
    } catch (error) {
      console.error('Failed to execute action:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const getActionType = (action: string): string => {
    if (action.toLowerCase().includes('immediate') || action.toLowerCase().includes('executive')) {
      return 'immediate';
    }
    if (action.toLowerCase().includes('proactive') || action.toLowerCase().includes('campaign')) {
      return 'proactive';
    }
    return 'standard';
  };

  if (!churnRisks || churnRisks.length === 0) {
    return (
      <CyberpunkCard $variant="glass">
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <span style={{ fontSize: '2rem' }}>✅</span>
          <h3 style={{ color: '#39ff14', margin: '1rem 0' }}>No Churn Risks Detected</h3>
          <p style={{ color: '#b0b0b0' }}>All customers are showing healthy engagement patterns.</p>
        </div>
      </CyberpunkCard>
    );
  }

  return (
    <div>
      <AlertHeader>
        <AlertTitle>Churn Risk Alerts</AlertTitle>
        <RiskSummary>
          {Object.entries(riskCounts).map(([level, count]) => (
            <RiskCount key={level} level={level}>
              {level}: {count}
            </RiskCount>
          ))}
        </RiskSummary>
      </AlertHeader>

      <AlertContainer>
        <AnimatePresence>
          {sortedRisks.map((risk, index) => (
            <motion.div
              key={risk.customer_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <RiskCard 
                riskLevel={risk.risk_level}
                urgent={risk.risk_level === 'Critical' && risk.churn_probability >= 0.9}
                variant="glass"
                as="div"
                style={{ cursor: 'pointer' }}
                onClick={() => toggleExpanded(risk.customer_id)}
              >
                <RiskHeader>
                  <CustomerInfo>
                    <CustomerId>Customer: {risk.customer_id}</CustomerId>
                    <LastPurchase>
                      Last purchase: {risk.days_since_last_purchase} days ago
                    </LastPurchase>
                    <TrendIndicator trend={risk.purchase_trend}>
                      {risk.purchase_trend} trend
                    </TrendIndicator>
                  </CustomerInfo>
                  
                  <RiskBadge level={risk.risk_level} probability={risk.churn_probability * 100}>
                    <RiskLevel level={risk.risk_level}>
                      {risk.risk_level}
                    </RiskLevel>
                    <ProbabilityBar probability={risk.churn_probability * 100} />
                    <ProbabilityText>
                      {(risk.churn_probability * 100).toFixed(1)}%
                    </ProbabilityText>
                  </RiskBadge>
                </RiskHeader>

                <AnimatePresence>
                  {expandedCards.has(risk.customer_id) && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <RiskFactors>
                        <FactorsTitle>Risk Factors:</FactorsTitle>
                        <FactorsList>
                          {risk.key_risk_factors.map((factor, idx) => (
                            <li key={idx}>{factor}</li>
                          ))}
                        </FactorsList>
                      </RiskFactors>

                      <ActionsSection>
                        <FactorsTitle>Recommended Actions:</FactorsTitle>
                        <ActionsGrid>
                          {risk.recommended_actions.slice(0, 3).map((action, idx) => (
                            <ActionButton
                              key={idx}
                              actionType={getActionType(action)}
                              $size="sm"
                              $loading={actionLoading === `${risk.customer_id}-${action}`}
                              onClick={() => handleAction(risk.customer_id, action)}
                            >
                              {action}
                            </ActionButton>
                          ))}
                        </ActionsGrid>
                      </ActionsSection>
                    </motion.div>
                  )}
                </AnimatePresence>
              </RiskCard>
            </motion.div>
          ))}
        </AnimatePresence>
      </AlertContainer>
    </div>
  );
};