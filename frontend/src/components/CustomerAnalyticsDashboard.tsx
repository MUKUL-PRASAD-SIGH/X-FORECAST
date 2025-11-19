import React, { useState, useEffect, useCallback } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';
import { RetentionRateChart } from './charts/RetentionRateChart';
import { CustomerSegmentChart } from './charts/CustomerSegmentChart';
import { ChurnRiskAlert } from './ChurnRiskAlert';

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

interface CohortData {
  cohort_month: string;
  cohort_size: number;
  avg_ltv: number;
  churn_rate: number;
  months_tracked: number;
}

interface CustomerAnalyticsData {
  company_id: string;
  analysis_date: string;
  overview: {
    total_customers: number;
    active_customers: number;
    new_customers_this_month: number;
    churned_customers: number;
    overall_ltv: number;
    overall_retention_rate: number;
  };
  customer_segments: CustomerSegment[];
  churn_analysis: {
    total_at_risk: number;
    risk_breakdown: {
      critical: number;
      high: number;
      medium: number;
      low: number;
    };
  };
  cohort_summary: CohortData[];
  key_insights: string[];
  recommendations: string[];
}

interface DetailedChurnRisk {
  customer_id: string;
  churn_probability: number;
  risk_level: 'Low' | 'Medium' | 'High' | 'Critical';
  key_risk_factors: string[];
  days_since_last_purchase: number;
  purchase_trend: string;
  recommended_actions: string[];
  confidence_score: number;
}

interface CustomerAnalyticsDashboardProps {
  authToken: string;
  companyId?: string;
}

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
  padding: ${props => props.theme.spacing.md};
`;

const HeaderSection = styled(motion.div)`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const DashboardTitle = styled.h2`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.xxl};
  background: ${props => props.theme.effects.primaryGradient};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: ${props => props.theme.effects.softGlow};
  margin: 0;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const MetricCard = styled(CyberpunkCard)<{ variant?: 'success' | 'warning' | 'danger' | 'info' }>`
  text-align: center;
  position: relative;
  
  ${props => props.variant === 'success' && css`
    border-color: ${props.theme.colors.success};
    box-shadow: 0 0 20px rgba(57, 255, 20, 0.3);
  `}
  
  ${props => props.variant === 'warning' && css`
    border-color: ${props.theme.colors.warning};
    box-shadow: 0 0 20px rgba(255, 255, 0, 0.3);
  `}
  
  ${props => props.variant === 'danger' && css`
    border-color: ${props.theme.colors.error};
    box-shadow: 0 0 20px rgba(255, 0, 64, 0.3);
  `}
  
  ${props => props.variant === 'info' && css`
    border-color: ${props.theme.colors.info};
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
  `}
`;

const MetricLabel = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const MetricValue = styled.div<{ color?: string }>`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.xl};
  color: ${props => props.color || props.theme.colors.neonBlue};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-shadow: ${props => props.theme.effects.softGlow};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const MetricChange = styled.div<{ positive?: boolean }>`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.positive ? props.theme.colors.success : props.theme.colors.error};
`;

const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.lg};
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-template-columns: 1fr;
  }
`;

const SegmentCard = styled(CyberpunkCard)`
  margin-bottom: ${props => props.theme.spacing.md};
`;

const SegmentHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const SegmentName = styled.h3`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  color: ${props => props.theme.colors.neonBlue};
  margin: 0;
  font-size: ${props => props.theme.typography.fontSize.lg};
`;

const HealthScore = styled.div<{ score: number }>`
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: 4px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  
  ${props => {
    if (props.score >= 80) {
      return css`
        background: rgba(57, 255, 20, 0.2);
        color: ${props.theme.colors.success};
        border: 1px solid ${props.theme.colors.success};
      `;
    } else if (props.score >= 60) {
      return css`
        background: rgba(255, 255, 0, 0.2);
        color: ${props.theme.colors.warning};
        border: 1px solid ${props.theme.colors.warning};
      `;
    } else {
      return css`
        background: rgba(255, 0, 64, 0.2);
        color: ${props.theme.colors.error};
        border: 1px solid ${props.theme.colors.error};
      `;
    }
  }}
`;

const SegmentMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const SegmentMetric = styled.div`
  text-align: center;
`;

const SegmentMetricLabel = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  margin-bottom: 2px;
`;

const SegmentMetricValue = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  color: ${props => props.theme.colors.primaryText};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
`;

const CharacteristicsList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.xs};
`;

const CharacteristicTag = styled.span`
  padding: 2px 8px;
  background: rgba(0, 255, 255, 0.1);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 12px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.neonBlue};
`;

const ChurnRiskCard = styled(CyberpunkCard)<{ riskLevel: string }>`
  margin-bottom: ${props => props.theme.spacing.sm};
  
  ${props => {
    switch (props.riskLevel) {
      case 'Critical':
        return css`
          border-color: ${props.theme.colors.error};
          box-shadow: 0 0 15px rgba(255, 0, 64, 0.4);
        `;
      case 'High':
        return css`
          border-color: ${props.theme.colors.warning};
          box-shadow: 0 0 15px rgba(255, 255, 0, 0.3);
        `;
      case 'Medium':
        return css`
          border-color: ${props.theme.colors.info};
          box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
        `;
      default:
        return css`
          border-color: ${props.theme.colors.success};
          box-shadow: 0 0 15px rgba(57, 255, 20, 0.2);
        `;
    }
  }}
`;

const RiskLevel = styled.div<{ level: string }>`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  
  ${props => {
    switch (props.level) {
      case 'Critical':
        return css`
          background: rgba(255, 0, 64, 0.2);
          color: ${props.theme.colors.error};
          border: 1px solid ${props.theme.colors.error};
        `;
      case 'High':
        return css`
          background: rgba(255, 255, 0, 0.2);
          color: ${props.theme.colors.warning};
          border: 1px solid ${props.theme.colors.warning};
        `;
      case 'Medium':
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

const ActionsList = styled.ul`
  margin: ${props => props.theme.spacing.sm} 0;
  padding-left: ${props => props.theme.spacing.md};
  
  li {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.sm};
    color: ${props => props.theme.colors.secondaryText};
    margin-bottom: ${props => props.theme.spacing.xs};
  }
`;

const InsightCard = styled(CyberpunkCard)`
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const InsightText = styled.p`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  color: ${props => props.theme.colors.primaryText};
  margin: 0;
  line-height: 1.6;
`;

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
`;

const ErrorContainer = styled(CyberpunkCard)`
  text-align: center;
  border-color: ${props => props.theme.colors.error};
`;

const ErrorText = styled.p`
  color: ${props => props.theme.colors.error};
  font-family: ${props => props.theme.typography.fontFamily.mono};
`;

const TabNavigation = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
  border-bottom: 2px solid rgba(0, 255, 255, 0.2);
  padding-bottom: ${props => props.theme.spacing.sm};
`;

const TabButton = styled.button<{ active: boolean }>`
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'transparent'};
  border: 2px solid ${props => props.active ? props.theme.colors.neonBlue : 'transparent'};
  color: ${props => props.active ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.3s ease;
  
  &:hover {
    color: ${props => props.theme.colors.neonBlue};
    border-color: ${props => props.theme.colors.neonBlue};
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
  }
  
  ${props => props.active && css`
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
    text-shadow: ${props.theme.effects.softGlow};
  `}
`;

const TabContent = styled(motion.div)`
  min-height: 400px;
`;

const VisualizationGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: ${props => props.theme.spacing.lg};
  
  @media (min-width: ${props => props.theme.breakpoints.desktop}) {
    grid-template-columns: 1fr 1fr;
  }
`;

const ChartSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.md};
`;

const SectionTitle = styled.h4`
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  margin: 0;
  text-shadow: ${props => props.theme.effects.softGlow};
`;

export const CustomerAnalyticsDashboard: React.FC<CustomerAnalyticsDashboardProps> = ({
  authToken,
  companyId = 'superx_retail_001'
}) => {
  const [analyticsData, setAnalyticsData] = useState<CustomerAnalyticsData | null>(null);
  const [detailedChurnRisks, setDetailedChurnRisks] = useState<DetailedChurnRisk[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'segments' | 'retention' | 'churn'>('overview');

  const fetchAnalyticsData = useCallback(async (forceRefresh = false) => {
    try {
      setLoading(!analyticsData);
      setRefreshing(forceRefresh);
      setError(null);

      // Fetch main analytics data
      const analyticsResponse = await fetch(
        `http://localhost:8000/api/customer-analytics/companies/${companyId}/analytics?force_refresh=${forceRefresh}`,
        {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (!analyticsResponse.ok) {
        throw new Error(`Failed to fetch analytics: ${analyticsResponse.statusText}`);
      }

      const fetchedAnalyticsData = await analyticsResponse.json();
      setAnalyticsData(fetchedAnalyticsData);

      // Fetch detailed churn risks
      try {
        const churnResponse = await fetch(
          `http://localhost:8000/api/customer-analytics/companies/${companyId}/churn-risks`,
          {
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },
          }
        );

        if (churnResponse.ok) {
          const churnData = await churnResponse.json();
          setDetailedChurnRisks(churnData.churn_risks || []);
        }
      } catch (churnError) {
        console.warn('Failed to fetch detailed churn risks:', churnError);
        // Continue without churn risk details
      }

    } catch (err) {
      console.error('Error fetching customer analytics:', err);
      setError(err instanceof Error ? err.message : 'Failed to load customer analytics');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [authToken, companyId, analyticsData]);

  useEffect(() => {
    fetchAnalyticsData();
  }, [fetchAnalyticsData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading && !refreshing) {
        fetchAnalyticsData();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchAnalyticsData, loading, refreshing]);

  const handleChurnAction = useCallback(async (customerId: string, action: string) => {
    try {
      console.log(`Executing action "${action}" for customer ${customerId}`);
      
      // Here you would typically make an API call to execute the action
      // For now, we'll just simulate the action
      
      // Optionally refresh the data after action
      setTimeout(() => {
        fetchAnalyticsData();
      }, 2000);
      
    } catch (error) {
      console.error('Failed to execute churn action:', error);
    }
  }, [fetchAnalyticsData]);

  if (loading) {
    return (
      <DashboardContainer>
        <LoadingContainer>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            style={{
              width: 40,
              height: 40,
              border: '3px solid rgba(0, 255, 255, 0.3)',
              borderTop: '3px solid #00ffff',
              borderRadius: '50%'
            }}
          />
        </LoadingContainer>
      </DashboardContainer>
    );
  }

  if (error) {
    return (
      <DashboardContainer>
        <ErrorContainer variant="neon">
          <ErrorText>⚠️ {error}</ErrorText>
          <CyberpunkButton
            variant="primary"
            onClick={() => fetchAnalyticsData(true)}
          >
            Retry
          </CyberpunkButton>
        </ErrorContainer>
      </DashboardContainer>
    );
  }

  if (!analyticsData) {
    return (
      <DashboardContainer>
        <ErrorContainer variant="glass">
          <ErrorText>No customer analytics data available</ErrorText>
        </ErrorContainer>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <HeaderSection
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <DashboardTitle>Customer Analytics</DashboardTitle>
        <CyberpunkButton
          variant="secondary"
          size="sm"
          loading={refreshing}
          onClick={() => fetchAnalyticsData(true)}
        >
          {refreshing ? 'Refreshing...' : 'Refresh Data'}
        </CyberpunkButton>
      </HeaderSection>

      {/* Overview Metrics */}
      <MetricsGrid>
        <MetricCard variant="info">
          <MetricLabel>Total Customers</MetricLabel>
          <MetricValue>{analyticsData.overview.total_customers.toLocaleString()}</MetricValue>
          <MetricChange positive>
            Active: {analyticsData.overview.active_customers.toLocaleString()}
          </MetricChange>
        </MetricCard>

        <MetricCard variant="success">
          <MetricLabel>Retention Rate</MetricLabel>
          <MetricValue color="#39ff14">
            {analyticsData.overview.overall_retention_rate.toFixed(1)}%
          </MetricValue>
          <MetricChange positive>
            New: {analyticsData.overview.new_customers_this_month}
          </MetricChange>
        </MetricCard>

        <MetricCard variant="warning">
          <MetricLabel>Average LTV</MetricLabel>
          <MetricValue color="#ffff00">
            ${analyticsData.overview.overall_ltv.toFixed(2)}
          </MetricValue>
          <MetricChange positive>
            Per customer value
          </MetricChange>
        </MetricCard>

        <MetricCard variant="danger">
          <MetricLabel>At Risk Customers</MetricLabel>
          <MetricValue color="#ff0040">
            {analyticsData.churn_analysis.total_at_risk}
          </MetricValue>
          <MetricChange>
            Critical: {analyticsData.churn_analysis.risk_breakdown.critical}
          </MetricChange>
        </MetricCard>
      </MetricsGrid>

      {/* Tab Navigation */}
      <TabNavigation>
        <TabButton 
          active={activeTab === 'overview'} 
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </TabButton>
        <TabButton 
          active={activeTab === 'segments'} 
          onClick={() => setActiveTab('segments')}
        >
          Segments
        </TabButton>
        <TabButton 
          active={activeTab === 'retention'} 
          onClick={() => setActiveTab('retention')}
        >
          Retention
        </TabButton>
        <TabButton 
          active={activeTab === 'churn'} 
          onClick={() => setActiveTab('churn')}
        >
          Churn Risk
        </TabButton>
      </TabNavigation>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <TabContent
            key="overview"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <ContentGrid>
              {/* Customer Segments Overview */}
              <div>
                <SectionTitle>Customer Segments</SectionTitle>
                {analyticsData.customer_segments.slice(0, 3).map((segment, index) => (
                  <motion.div
                    key={segment.segment_id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    <SegmentCard variant="glass">
                      <SegmentHeader>
                        <SegmentName>{segment.segment_name}</SegmentName>
                        <HealthScore score={segment.health_score}>
                          {segment.health_score.toFixed(0)}%
                        </HealthScore>
                      </SegmentHeader>
                      
                      <SegmentMetrics>
                        <SegmentMetric>
                          <SegmentMetricLabel>Customers</SegmentMetricLabel>
                          <SegmentMetricValue>{segment.customer_count}</SegmentMetricValue>
                        </SegmentMetric>
                        <SegmentMetric>
                          <SegmentMetricLabel>Avg LTV</SegmentMetricLabel>
                          <SegmentMetricValue>${segment.avg_ltv.toFixed(0)}</SegmentMetricValue>
                        </SegmentMetric>
                        <SegmentMetric>
                          <SegmentMetricLabel>Revenue %</SegmentMetricLabel>
                          <SegmentMetricValue>{segment.revenue_contribution.toFixed(1)}%</SegmentMetricValue>
                        </SegmentMetric>
                      </SegmentMetrics>
                    </SegmentCard>
                  </motion.div>
                ))}
              </div>

              {/* Key Insights */}
              <div>
                <SectionTitle>Key Insights & Recommendations</SectionTitle>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {analyticsData.key_insights.slice(0, 3).map((insight, index) => (
                    <InsightCard key={index} variant="glass">
                      <InsightText>{insight}</InsightText>
                    </InsightCard>
                  ))}
                  {analyticsData.recommendations.slice(0, 2).map((recommendation, index) => (
                    <InsightCard key={index} variant="neon">
                      <InsightText>{recommendation}</InsightText>
                    </InsightCard>
                  ))}
                </div>
              </div>
            </ContentGrid>
          </TabContent>
        )}

        {activeTab === 'segments' && (
          <TabContent
            key="segments"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <VisualizationGrid>
              <ChartSection>
                <SectionTitle>Segment Performance Comparison</SectionTitle>
                <CyberpunkCard $variant="glass">
                  <CustomerSegmentChart 
                    segments={analyticsData.customer_segments}
                    width={600}
                    height={400}
                    showComparison={true}
                  />
                </CyberpunkCard>
              </ChartSection>

              <ChartSection>
                <SectionTitle>Detailed Segment Analysis</SectionTitle>
                {analyticsData.customer_segments.map((segment, index) => (
                  <SegmentCard key={segment.segment_id} variant="glass">
                    <SegmentHeader>
                      <SegmentName>{segment.segment_name}</SegmentName>
                      <HealthScore score={segment.health_score}>
                        {segment.health_score.toFixed(0)}%
                      </HealthScore>
                    </SegmentHeader>
                    
                    <SegmentMetrics>
                      <SegmentMetric>
                        <SegmentMetricLabel>Customers</SegmentMetricLabel>
                        <SegmentMetricValue>{segment.customer_count}</SegmentMetricValue>
                      </SegmentMetric>
                      <SegmentMetric>
                        <SegmentMetricLabel>Avg LTV</SegmentMetricLabel>
                        <SegmentMetricValue>${segment.avg_ltv.toFixed(0)}</SegmentMetricValue>
                      </SegmentMetric>
                      <SegmentMetric>
                        <SegmentMetricLabel>Retention</SegmentMetricLabel>
                        <SegmentMetricValue>{(segment.avg_retention_rate * 100).toFixed(1)}%</SegmentMetricValue>
                      </SegmentMetric>
                      <SegmentMetric>
                        <SegmentMetricLabel>Growth</SegmentMetricLabel>
                        <SegmentMetricValue>{segment.growth_rate.toFixed(1)}%</SegmentMetricValue>
                      </SegmentMetric>
                    </SegmentMetrics>
                    
                    <CharacteristicsList>
                      {segment.characteristics.map((char, idx) => (
                        <CharacteristicTag key={idx}>{char}</CharacteristicTag>
                      ))}
                    </CharacteristicsList>
                  </SegmentCard>
                ))}
              </ChartSection>
            </VisualizationGrid>
          </TabContent>
        )}

        {activeTab === 'retention' && (
          <TabContent
            key="retention"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <ChartSection>
              <SectionTitle>Retention Rate Analysis</SectionTitle>
              <CyberpunkCard $variant="glass">
                <RetentionRateChart 
                  cohortData={analyticsData.cohort_summary.map(cohort => ({
                    ...cohort,
                    retention_rates: { 
                      '0': 1.0, 
                      '1': Math.max(0.7, 1 - cohort.churn_rate),
                      '2': Math.max(0.5, 1 - cohort.churn_rate * 1.2),
                      '3': Math.max(0.3, 1 - cohort.churn_rate * 1.5)
                    },
                    revenue_per_period: { 
                      '0': cohort.avg_ltv * 0.4,
                      '1': cohort.avg_ltv * 0.3,
                      '2': cohort.avg_ltv * 0.2,
                      '3': cohort.avg_ltv * 0.1
                    }
                  }))}
                  width={800}
                  height={500}
                  showTrendAnalysis={true}
                />
              </CyberpunkCard>
            </ChartSection>

            <div style={{ marginTop: '2rem' }}>
              <SectionTitle>Cohort Summary</SectionTitle>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                {analyticsData.cohort_summary.map((cohort, index) => (
                  <CyberpunkCard key={index} $variant="glass">
                    <h4 style={{ color: '#00ffff', margin: '0 0 1rem 0' }}>{cohort.cohort_month}</h4>
                    <SegmentMetrics>
                      <SegmentMetric>
                        <SegmentMetricLabel>Size</SegmentMetricLabel>
                        <SegmentMetricValue>{cohort.cohort_size}</SegmentMetricValue>
                      </SegmentMetric>
                      <SegmentMetric>
                        <SegmentMetricLabel>Avg LTV</SegmentMetricLabel>
                        <SegmentMetricValue>${cohort.avg_ltv.toFixed(0)}</SegmentMetricValue>
                      </SegmentMetric>
                      <SegmentMetric>
                        <SegmentMetricLabel>Churn Rate</SegmentMetricLabel>
                        <SegmentMetricValue>{(cohort.churn_rate * 100).toFixed(1)}%</SegmentMetricValue>
                      </SegmentMetric>
                      <SegmentMetric>
                        <SegmentMetricLabel>Months</SegmentMetricLabel>
                        <SegmentMetricValue>{cohort.months_tracked}</SegmentMetricValue>
                      </SegmentMetric>
                    </SegmentMetrics>
                  </CyberpunkCard>
                ))}
              </div>
            </div>
          </TabContent>
        )}

        {activeTab === 'churn' && (
          <TabContent
            key="churn"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <ChurnRiskAlert 
              churnRisks={detailedChurnRisks}
              onActionTaken={handleChurnAction}
              autoRefresh={true}
              refreshInterval={30000}
            />
          </TabContent>
        )}
      </AnimatePresence>
    </DashboardContainer>
  );
};