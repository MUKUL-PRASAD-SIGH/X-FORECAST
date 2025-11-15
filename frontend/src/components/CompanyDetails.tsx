import React from 'react';
import styled, { keyframes, css } from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkTheme } from '../theme/cyberpunkTheme';

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(0, 255, 255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 255, 255, 0); }
`;

const Section = styled.section`
  margin-bottom: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.xl};
`;

const SectionTitle = styled.h3`
  color: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.hotPink};
  border-bottom: 1px solid ${(props: { theme: CyberpunkTheme }) => props.theme.colors.neonBlue};
  padding-bottom: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.sm};
  margin-bottom: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.md};
  font-family: ${(props: { theme: CyberpunkTheme }) => props.theme.typography.fontFamily.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.md};
  margin-bottom: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.xl};
`;

const MetricCard = styled(motion.div)<{ $highlight?: boolean }>`
  background: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.cardBg};
  border: 1px solid ${(props: { theme: CyberpunkTheme }) => props.theme.colors.neonBlue};
  border-radius: 8px;
  padding: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.md};
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
  ${props => props.$highlight && css`
    animation: ${pulse} 2s infinite;
    border-color: ${props.theme.colors.hotPink};
  `}
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.neonBlue};
    opacity: 0.7;
  }
  
  .metric-label {
    font-size: ${(props: { theme: CyberpunkTheme }) => props.theme.typography.fontSize.sm};
    color: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.secondaryText};
    margin-bottom: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.xs};
    font-family: ${(props: { theme: CyberpunkTheme }) => props.theme.typography.fontFamily.mono};
    display: flex;
    align-items: center;
    gap: 8px;
    
    .metric-icon {
      font-size: 1.2em;
      opacity: 0.8;
    }
  }
  
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.neonBlue};
    margin: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.xs} 0;
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
  }
  
  .metric-change {
    font-size: ${(props: { theme: CyberpunkTheme }) => props.theme.typography.fontSize.xs};
    display: flex;
    align-items: center;
    gap: 4px;
    
    .icon {
      font-size: 1.2em;
    }
    
    &.positive {
      color: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.acidGreen};
    }
    
    &.negative {
      color: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.error};
    }
  }
  
  .metric-description {
    font-size: ${(props: { theme: CyberpunkTheme }) => props.theme.typography.fontSize.xs};
    color: ${(props: { theme: CyberpunkTheme }) => props.theme.colors.secondaryText};
    margin-top: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.sm};
    opacity: 0.8;
  }
`;

const Divider = styled.hr`
  border: none;
  border-top: 1px dashed ${(props: { theme: CyberpunkTheme }) => props.theme.colors.neonBlue};
  margin: ${(props: { theme: CyberpunkTheme }) => props.theme.spacing.xl} 0;
  opacity: 0.3;
`;

const StatusBadge = styled.span<{ $status: 'good' | 'warning' | 'critical' }>`
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: ${props => {
    switch(props.$status) {
      case 'good': return 'rgba(0, 255, 100, 0.2)';
      case 'warning': return 'rgba(255, 200, 0, 0.2)';
      case 'critical': return 'rgba(255, 50, 50, 0.2)';
      default: return 'rgba(100, 100, 100, 0.2)';
    }
  }};
  color: ${props => {
    switch(props.$status) {
      case 'good': return '#0f0';
      case 'warning': return '#ffcc00';
      case 'critical': return '#ff3333';
      default: return '#999';
    }
  }};
`;

interface CompanyDetailsProps {
  metrics: {
    totalCustomers: number;
    retentionRate: number;
    forecastAccuracy: number;
    systemHealth: number;
    activeAlerts: number;
    revenueGrowth: number;
  };
  companyName: string;
}

export const CompanyDetails: React.FC<CompanyDetailsProps> = ({ 
  metrics, 
  companyName 
}) => {
  const formatNumber = (num: number, decimals = 1) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: decimals
    }).format(num);
  };

  // Calculate derived metrics
  const activeUsers = Math.floor(metrics.totalCustomers * 0.87);
  const churnRate = ((1 - metrics.retentionRate) * 100).toFixed(1);
  const revenueGrowthValue = (metrics.revenueGrowth * 100).toFixed(1);
  const forecastAccuracy = (metrics.forecastAccuracy * 100).toFixed(1);
  const systemHealth = (metrics.systemHealth * 100).toFixed(0);

  return (
    <div>
      {/* Business Growth Section */}
      <Section>
        <SectionTitle>🚀 Business Growth</SectionTitle>
        <MetricsGrid>
          <MetricCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="metric-label">
              <span className="metric-icon">👥</span> Total Customers
            </div>
            <div className="metric-value">{formatNumber(metrics.totalCustomers)}</div>
            <div className="metric-change positive">
              <span className="icon">↗</span> +2.3% from last month
            </div>
            <div className="metric-description">
              {activeUsers.toLocaleString()} active users ({churnRate}% monthly churn)
            </div>
          </MetricCard>

          <MetricCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="metric-label">
              <span className="metric-icon">📈</span> Revenue Growth
            </div>
            <div className="metric-value">{revenueGrowthValue}%</div>
            <div className="metric-change positive">
              <span className="icon">↗</span> +5.2% this quarter
            </div>
            <div className="metric-description">
              Outperforming market by 3.1%
            </div>
          </MetricCard>
        </MetricsGrid>
      </Section>

      <Divider />

      {/* Customer Health Section */}
      <Section>
        <SectionTitle>❤️ Customer Health</SectionTitle>
        <MetricsGrid>
          <MetricCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <div className="metric-label">
              <span className="metric-icon">🔄</span> Retention Rate
              <StatusBadge $status={metrics.retentionRate > 0.7 ? 'good' : 'warning'}>
                {metrics.retentionRate > 0.7 ? 'Excellent' : 'Needs Attention'}
              </StatusBadge>
            </div>
            <div className="metric-value">{(metrics.retentionRate * 100).toFixed(1)}%</div>
            <div className="metric-change positive">
              <span className="icon">↗</span> +1.2% from last week
            </div>
            <div className="metric-description">
              Industry average: 75%
            </div>
          </MetricCard>

          <MetricCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <div className="metric-label">
              <span className="metric-icon">🎯</span> Forecast Accuracy
              <StatusBadge $status={metrics.forecastAccuracy > 0.8 ? 'good' : 'warning'}>
                {metrics.forecastAccuracy > 0.8 ? 'High' : 'Medium'}
              </StatusBadge>
            </div>
            <div className="metric-value">{forecastAccuracy}%</div>
            <div className="metric-change positive">
              <span className="icon">↗</span> +0.8% improvement
            </div>
            <div className="metric-description">
              Based on last quarter's predictions
            </div>
          </MetricCard>
        </MetricsGrid>
      </Section>

      <Divider />

      {/* System Status Section */}
      <Section>
        <SectionTitle>⚙️ System Status</SectionTitle>
        <MetricsGrid>
          <MetricCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <div className="metric-label">
              <span className="metric-icon">🖥️</span> System Health
              <StatusBadge $status={metrics.systemHealth > 0.85 ? 'good' : 'warning'}>
                {metrics.systemHealth > 0.85 ? 'Optimal' : 'Stable'}
              </StatusBadge>
            </div>
            <div className="metric-value">{systemHealth}%</div>
            <div className="metric-change positive">
              <span className="icon">↗</span> All systems optimal
            </div>
            <div className="metric-description">
              99.9% uptime this month
            </div>
          </MetricCard>

          <MetricCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.5 }}
            $highlight={metrics.activeAlerts > 0}
          >
            <div className="metric-label">
              <span className="metric-icon">⚠️</span> Active Alerts
              <StatusBadge $status={metrics.activeAlerts === 0 ? 'good' : 'warning'}>
                {metrics.activeAlerts === 0 ? 'None' : 'Action Needed'}
              </StatusBadge>
            </div>
            <div className="metric-value">{metrics.activeAlerts}</div>
            <div className={`metric-change ${metrics.activeAlerts > 0 ? 'negative' : 'positive'}`}>
              <span className="icon">{metrics.activeAlerts > 0 ? '⚠️' : '✓'}</span>
              {metrics.activeAlerts > 0 ? 'Needs attention' : 'All clear'}
            </div>
            <div className="metric-description">
              {metrics.activeAlerts > 0 
                ? `${metrics.activeAlerts} issues require attention`
                : 'No critical issues detected'}
            </div>
          </MetricCard>
        </MetricsGrid>
      </Section>
    </div>
  );
};

export default CompanyDetails;
