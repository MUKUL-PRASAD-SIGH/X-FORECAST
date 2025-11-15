import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard } from '../ui/CyberpunkCard';
import { CyberpunkButton } from '../ui/CyberpunkButton';
import { InsightCard } from './InsightCard';
import { RecommendationCard } from './RecommendationCard';
import { ExecutiveSummary } from './ExecutiveSummary';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import PriorityMatrix from './PriorityMatrix';
import { ActionTimeline } from './ActionTimeline';

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
  actionable_recommendations: BusinessInsight[];
}

interface RecommendationPortfolio {
  recommendations: TimelineBasedRecommendation[];
  priority_matrix: {
    critical: string[];
    high: string[];
    medium: string[];
    low: string[];
  };
  confidence_summary: {
    average_confidence: number;
    high_confidence_count: number;
    low_confidence_count: number;
  };
  expected_outcomes: {
    total_expected_impact: number;
    average_roi: number;
    average_success_probability: number;
  };
}

interface AIInsightsDashboardProps {
  companyId?: string;
  className?: string;
}

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
  padding: 1rem;
  min-height: 100vh;
  background: ${props => props.theme.effects.backgroundGradient};
`;

const DashboardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 1rem;
`;

const DashboardTitle = styled.h1`
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.display};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 3px;
  background: ${props => props.theme.effects.primaryGradient};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
`;

const HeaderActions = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
`;

const StatusIndicator = styled.div<{ status: 'loading' | 'ready' | 'error' }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.status === 'loading' && `
    background: rgba(255, 255, 0, 0.2);
    color: ${props.theme.colors.warning};
    border: 1px solid ${props.theme.colors.warning};
    
    &::before {
      content: '';
      width: 12px;
      height: 12px;
      border: 2px solid transparent;
      border-top: 2px solid currentColor;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
  `}
  
  ${props => props.status === 'ready' && `
    background: rgba(57, 255, 20, 0.2);
    color: ${props.theme.colors.acidGreen};
    border: 1px solid ${props.theme.colors.acidGreen};
    
    &::before {
      content: '✓';
      font-weight: bold;
    }
  `}
  
  ${props => props.status === 'error' && `
    background: rgba(255, 0, 64, 0.2);
    color: ${props.theme.colors.error};
    border: 1px solid ${props.theme.colors.error};
    
    &::before {
      content: '⚠';
      font-weight: bold;
    }
  `}
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  
  @media (min-width: ${props => props.theme.breakpoints.desktop}) {
    grid-template-columns: 2fr 1fr;
  }
`;

const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const Sidebar = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const SectionContainer = styled(CyberpunkCard)`
  position: relative;
`;

const SectionHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
`;

const SectionTitle = styled.h2`
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.xl};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 2px;
`;

const FilterContainer = styled.div`
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
`;

const FilterButton = styled(CyberpunkButton)<{ active: boolean }>`
  padding: 0.25rem 0.75rem;
  font-size: ${props => props.theme.typography.fontSize.xs};
  
  ${props => props.active && `
    background: ${props.theme.colors.neonBlue};
    color: ${props.theme.colors.darkBg};
  `}
`;

const InsightsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  
  @media (min-width: ${props => props.theme.breakpoints.tablet}) {
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  }
`;

const RecommendationsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const PrioritySection = styled.div<{ priority: string }>`
  margin-bottom: 2rem;
  
  ${props => props.priority === 'critical' && `
    border-left: 4px solid ${props.theme.colors.error};
    padding-left: 1rem;
  `}
  
  ${props => props.priority === 'high' && `
    border-left: 4px solid ${props.theme.colors.warning};
    padding-left: 1rem;
  `}
`;

const PriorityTitle = styled.h3<{ priority: string }>`
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  margin: 0 0 1rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => props.priority === 'critical' && `
    color: ${props.theme.colors.error};
  `}
  
  ${props => props.priority === 'high' && `
    color: ${props.theme.colors.warning};
  `}
  
  ${props => props.priority === 'medium' && `
    color: ${props.theme.colors.info};
  `}
  
  ${props => props.priority === 'low' && `
    color: ${props.theme.colors.success};
  `}
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  color: ${props => props.theme.colors.secondaryText};
  
  h3 {
    color: ${props => props.theme.colors.primaryText};
    margin-bottom: 1rem;
  }
`;

const LoadingOverlay = styled(motion.div)`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  border-radius: 8px;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 3px solid transparent;
  border-top: 3px solid ${props => props.theme.colors.neonBlue};
  border-radius: 50%;
  animation: spin 1s linear infinite;
`;

export const AIInsightsDashboard: React.FC<AIInsightsDashboardProps> = ({
  companyId,
  className
}) => {
  const [insights, setInsights] = useState<BusinessInsightsResult | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationPortfolio | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeInsightFilter, setActiveInsightFilter] = useState<string>('all');
  const [activePriorityFilter, setActivePriorityFilter] = useState<string>('all');
  const [selectedRecommendation, setSelectedRecommendation] = useState<TimelineBasedRecommendation | null>(null);
  const [showTimeline, setShowTimeline] = useState(false);

  useEffect(() => {
    if (companyId) {
      loadInsights();
    }
  }, [companyId]);

  const loadInsights = async () => {
    if (!companyId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Load business insights
      const insightsResponse = await fetch(`/api/company-sales/${companyId}/insights`);
      if (!insightsResponse.ok) throw new Error('Failed to load insights');
      const insightsData = await insightsResponse.json();
      setInsights(insightsData);

      // Load recommendations
      const recommendationsResponse = await fetch(`/api/company-sales/${companyId}/recommendations`);
      if (!recommendationsResponse.ok) throw new Error('Failed to load recommendations');
      const recommendationsData = await recommendationsResponse.json();
      setRecommendations(recommendationsData);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load insights');
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshInsights = () => {
    loadInsights();
  };

  const handleExportSummary = () => {
    if (!insights) return;
    
    const exportData = {
      executive_summary: insights.executive_summary,
      key_findings: insights.key_findings,
      generated_at: insights.generated_at,
      confidence_score: insights.confidence_score
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `insights-summary-${companyId}-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImplementRecommendation = (recommendationId: string) => {
    console.log('Implementing recommendation:', recommendationId);
    // Create implementation tracking
    const recommendation = recommendations?.recommendations.find(r => r.recommendation_id === recommendationId);
    if (recommendation) {
      // Show implementation timeline modal or navigate to implementation view
      alert(`Starting implementation of: ${recommendation.title}\n\nNext steps:\n${recommendation.timeline.immediate.join('\n')}`);
    }
  };

  const handleViewRecommendationDetails = (recommendationId: string) => {
    console.log('Viewing recommendation details:', recommendationId);
    const recommendation = recommendations?.recommendations.find(r => r.recommendation_id === recommendationId);
    if (recommendation) {
      setSelectedRecommendation(recommendation);
      setShowTimeline(true);
    }
  };

  const handleStartImplementation = (recommendationId: string, phase: string) => {
    console.log(`Starting implementation of ${recommendationId} - ${phase} phase`);
    // In a real implementation, this would create tasks and track progress
  };

  const handleMarkComplete = (recommendationId: string, phase: string, actionIndex: number) => {
    console.log(`Marking complete: ${recommendationId} - ${phase} - action ${actionIndex}`);
    // In a real implementation, this would update task completion status
  };

  const handlePriorityMatrixClick = (recommendationId: string) => {
    handleViewRecommendationDetails(recommendationId);
  };

  const handleActionClick = (action: string) => {
    console.log('Action clicked:', action);
    // Create action tracking and timeline
    const actionPlan = `
Action: ${action}

Suggested Timeline:
1. Immediate (0-7 days): Assess requirements and resources
2. Short-term (1-4 weeks): Begin implementation
3. Medium-term (1-3 months): Monitor progress and adjust
4. Long-term (3+ months): Evaluate results and optimize

Would you like to create an action plan for this item?
    `;
    
    if (confirm(actionPlan)) {
      // In a real implementation, this would create a task or action item
      console.log(`Creating action plan for: ${action}`);
    }
  };

  const getFilteredInsights = () => {
    if (!insights) return [];
    
    if (activeInsightFilter === 'all') {
      return insights.actionable_recommendations;
    }
    
    return insights.actionable_recommendations.filter(
      insight => insight.insight_type === activeInsightFilter
    );
  };

  const getFilteredRecommendations = () => {
    if (!recommendations) return [];
    
    if (activePriorityFilter === 'all') {
      return recommendations.recommendations;
    }
    
    return recommendations.recommendations.filter(
      rec => rec.priority === activePriorityFilter
    );
  };

  const getRecommendationsByPriority = () => {
    if (!recommendations) return {};
    
    const grouped: Record<string, TimelineBasedRecommendation[]> = {
      critical: [],
      high: [],
      medium: [],
      low: []
    };
    
    recommendations.recommendations.forEach(rec => {
      grouped[rec.priority].push(rec);
    });
    
    return grouped;
  };

  const getStatus = (): 'loading' | 'ready' | 'error' => {
    if (loading) return 'loading';
    if (error) return 'error';
    return 'ready';
  };

  const insightTypes = ['all', 'opportunity', 'risk', 'recommendation', 'performance'];
  const priorityLevels = ['all', 'critical', 'high', 'medium', 'low'];

  return (
    <DashboardContainer className={className}>
      <DashboardHeader>
        <DashboardTitle>🤖 AI Insights Dashboard</DashboardTitle>
        <HeaderActions>
          <StatusIndicator status={getStatus()}>
            {loading ? 'Analyzing' : error ? 'Error' : 'Ready'}
          </StatusIndicator>
          <CyberpunkButton
            variant="secondary"
            size="sm"
            onClick={handleRefreshInsights}
            disabled={loading}
          >
            Refresh Analysis
          </CyberpunkButton>
        </HeaderActions>
      </DashboardHeader>

      {insights && (
        <ExecutiveSummary
          insights={insights}
          onExport={handleExportSummary}
          onRefresh={handleRefreshInsights}
        />
      )}

      <ContentGrid>
        <MainContent>
          {/* Business Insights Section */}
          <SectionContainer variant="glass">
            <AnimatePresence>
              {loading && (
                <LoadingOverlay
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <LoadingSpinner />
                </LoadingOverlay>
              )}
            </AnimatePresence>
            
            <SectionHeader>
              <SectionTitle>💡 Business Insights</SectionTitle>
              <FilterContainer>
                {insightTypes.map(type => (
                  <FilterButton
                    key={type}
                    variant="ghost"
                    size="sm"
                    active={activeInsightFilter === type}
                    onClick={() => setActiveInsightFilter(type)}
                  >
                    {type.replace('_', ' ')}
                  </FilterButton>
                ))}
              </FilterContainer>
            </SectionHeader>

            {error ? (
              <EmptyState>
                <h3>Error Loading Insights</h3>
                <p>{error}</p>
                <CyberpunkButton onClick={handleRefreshInsights}>
                  Retry
                </CyberpunkButton>
              </EmptyState>
            ) : getFilteredInsights().length > 0 ? (
              <InsightsGrid>
                {getFilteredInsights().map((insight, index) => (
                  <InsightCard
                    key={index}
                    insight={insight}
                    onActionClick={handleActionClick}
                  />
                ))}
              </InsightsGrid>
            ) : (
              <EmptyState>
                <h3>No Insights Available</h3>
                <p>Upload data to generate AI-powered business insights</p>
              </EmptyState>
            )}
          </SectionContainer>

          {/* Recommendations Section */}
          <SectionContainer variant="neon">
            <SectionHeader>
              <SectionTitle>🎯 Strategic Recommendations</SectionTitle>
              <FilterContainer>
                {priorityLevels.map(priority => (
                  <FilterButton
                    key={priority}
                    variant="ghost"
                    size="sm"
                    active={activePriorityFilter === priority}
                    onClick={() => setActivePriorityFilter(priority)}
                  >
                    {priority}
                  </FilterButton>
                ))}
              </FilterContainer>
            </SectionHeader>

            {recommendations && getFilteredRecommendations().length > 0 ? (
              <RecommendationsContainer>
                {activePriorityFilter === 'all' ? (
                  Object.entries(getRecommendationsByPriority()).map(([priority, recs]) => (
                    recs.length > 0 && (
                      <PrioritySection key={priority} priority={priority}>
                        <PriorityTitle priority={priority}>
                          {priority} Priority ({recs.length})
                        </PriorityTitle>
                        {recs.map(rec => (
                          <RecommendationCard
                            key={rec.recommendation_id}
                            recommendation={rec}
                            onImplement={handleImplementRecommendation}
                            onViewDetails={handleViewRecommendationDetails}
                          />
                        ))}
                      </PrioritySection>
                    )
                  ))
                ) : (
                  getFilteredRecommendations().map(rec => (
                    <RecommendationCard
                      key={rec.recommendation_id}
                      recommendation={rec}
                      onImplement={handleImplementRecommendation}
                      onViewDetails={handleViewRecommendationDetails}
                    />
                  ))
                )}
              </RecommendationsContainer>
            ) : (
              <EmptyState>
                <h3>No Recommendations Available</h3>
                <p>Generate insights to receive strategic recommendations</p>
              </EmptyState>
            )}
          </SectionContainer>
        </MainContent>

        <Sidebar>
          {/* Priority Matrix */}
          {recommendations && (
            <PriorityMatrix
              recommendations={recommendations.recommendations}
              onRecommendationClick={handlePriorityMatrixClick}
              onPriorityFilter={setActivePriorityFilter}
            />
          )}

          {/* Confidence Summary */}
          {recommendations?.confidence_summary && (
            <CyberpunkCard variant="hologram">
              <SectionTitle>📊 Confidence Summary</SectionTitle>
              <div style={{ marginTop: '1rem' }}>
                <ConfidenceIndicator
                  confidence={recommendations.confidence_summary.average_confidence}
                  size="lg"
                  showLabel={true}
                  showPercentage={true}
                />
                <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#B0B0B0' }}>
                  <p>High Confidence: {recommendations.confidence_summary.high_confidence_count} recommendations</p>
                  <p>Low Confidence: {recommendations.confidence_summary.low_confidence_count} recommendations</p>
                </div>
              </div>
            </CyberpunkCard>
          )}

          {/* Expected Outcomes */}
          {recommendations?.expected_outcomes && (
            <CyberpunkCard variant="glass">
              <SectionTitle>🎯 Expected Outcomes</SectionTitle>
              <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#00FFFF' }}>
                    {recommendations.expected_outcomes.total_expected_impact.toFixed(1)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#B0B0B0', textTransform: 'uppercase' }}>
                    Total Impact
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#39FF14' }}>
                    {recommendations.expected_outcomes.average_roi.toFixed(1)}x
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#B0B0B0', textTransform: 'uppercase' }}>
                    Average ROI
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#FF1493' }}>
                    {(recommendations.expected_outcomes.average_success_probability * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#B0B0B0', textTransform: 'uppercase' }}>
                    Success Rate
                  </div>
                </div>
              </div>
            </CyberpunkCard>
          )}
        </Sidebar>
      </ContentGrid>

      {/* Action Timeline Modal */}
      <AnimatePresence>
        {showTimeline && selectedRecommendation && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
              padding: '2rem'
            }}
            onClick={() => setShowTimeline(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              style={{
                maxWidth: '90vw',
                maxHeight: '90vh',
                overflow: 'auto',
                position: 'relative'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                zIndex: 1001
              }}>
                <CyberpunkButton
                  variant="secondary"
                  size="sm"
                  onClick={() => setShowTimeline(false)}
                >
                  ✕ Close
                </CyberpunkButton>
              </div>
              
              <ActionTimeline
                recommendation={selectedRecommendation}
                onStartImplementation={handleStartImplementation}
                onMarkComplete={handleMarkComplete}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </DashboardContainer>
  );
};