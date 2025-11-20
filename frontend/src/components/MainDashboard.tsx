import React, { useState, useEffect } from 'react';
import styled, { useTheme } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CyberpunkCard, 
  CyberpunkButton, 
  CyberpunkNavigation, 
  CyberpunkLoader,
  NavigationItem 
} from './ui';
import { EnsembleChatInterface } from './chat/EnsembleChatInterface';
import { HolographicRenderer } from './3d/HolographicRenderer';
import { LoginForm } from './auth/LoginForm';
import { DataUpload } from './DataUpload';
import { CompanyDetails } from './CompanyDetails';
import { EnsembleStatus } from './EnsembleStatus';
import { ModelMonitoringDashboard } from './ModelMonitoringDashboard';
import { ForecastingDashboard } from './ForecastingDashboard';
import { CustomerAnalyticsDashboard } from './CustomerAnalyticsDashboard';
import { CyberpunkTheme } from '../theme/cyberpunkTheme';
import { useAuth } from '../contexts/AuthContext';
import { useCompanyMetrics } from '../hooks/useCompanyMetrics';

const DashboardContainer = styled.div<{ theme: CyberpunkTheme }>`
  min-height: 100vh;
  padding: ${(props) => props.theme.spacing.lg};
  display: flex;
  flex-direction: column;
  gap: ${(props) => props.theme.spacing.lg};
  background: ${(props) => props.theme.effects.backgroundGradient};
  position: relative;
  overflow-x: hidden;
  
  &::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, ${(props) => props.theme.colors.glowBlue} 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, ${(props) => props.theme.colors.glowPink} 0%, transparent 50%),
      radial-gradient(circle at 40% 40%, ${(props) => props.theme.colors.glowGreen} 0%, transparent 50%);
    opacity: 0.1;
    pointer-events: none;
    z-index: -1;
  }
`;

const Header = styled.header<{ theme: CyberpunkTheme }>`
  text-align: center;
  margin-bottom: ${(props) => props.theme.spacing.xl};
  position: relative;
`;

const StatusBar = styled(motion.div)<{ theme: CyberpunkTheme; connected: boolean }>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${(props) => props.theme.spacing.sm} ${(props) => props.theme.spacing.md};
  background: ${(props) => props.theme.colors.cardBg};
  border: 1px solid ${(props) => props.connected ? props.theme.colors.acidGreen : props.theme.colors.error};
  border-radius: 8px;
  margin-bottom: ${(props) => props.theme.spacing.lg};
  font-family: ${(props) => props.theme.typography.fontFamily.mono};
  font-size: ${(props) => props.theme.typography.fontSize.sm};
  box-shadow: ${(props) => props.connected ? props.theme.effects.softGlow : 'none'};
  
  @media (max-width: ${(props) => props.theme.breakpoints.mobile}) {
    flex-direction: column;
    gap: ${(props) => props.theme.spacing.sm};
  }
`;

const StatusItem = styled.div<{ theme: CyberpunkTheme }>`
  display: flex;
  align-items: center;
  gap: ${(props) => props.theme.spacing.xs};
  color: ${(props) => props.theme.colors.secondaryText};
  
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: ${(props) => props.theme.colors.acidGreen};
    animation: pulse 2s infinite;
  }
  
  .metric-value {
    color: ${(props) => props.theme.colors.neonBlue};
    font-weight: ${(props) => props.theme.typography.fontWeight.bold};
  }
`;

const MetricsGrid = styled.div<{ theme: CyberpunkTheme }>`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${(props) => props.theme.spacing.md};
  margin-bottom: ${(props) => props.theme.spacing.xl};
  
  @media (max-width: ${(props) => props.theme.breakpoints.mobile}) {
    grid-template-columns: repeat(2, 1fr);
  }
`;

const MetricCard = styled(motion.div)<{ theme: CyberpunkTheme }>`
  background: ${(props) => props.theme.colors.cardBg};
  border: 1px solid ${(props) => props.theme.colors.neonBlue};
  border-radius: 8px;
  padding: ${(props) => props.theme.spacing.md};
  text-align: center;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: ${(props) => props.theme.effects.primaryGradient};
    animation: scan 3s infinite;
  }
  
  @keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
  }
  
  .metric-label {
    font-family: ${(props) => props.theme.typography.fontFamily.mono};
    font-size: ${(props) => props.theme.typography.fontSize.xs};
    color: ${(props) => props.theme.colors.secondaryText};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: ${(props) => props.theme.spacing.xs};
  }
  
  .metric-value {
    font-family: ${(props) => props.theme.typography.fontFamily.display};
    font-size: ${(props) => props.theme.typography.fontSize.xl};
    color: ${(props) => props.theme.colors.neonBlue};
    font-weight: ${(props) => props.theme.typography.fontWeight.bold};
    text-shadow: ${(props) => props.theme.effects.softGlow};
  }
  
  .metric-change {
    font-family: ${(props) => props.theme.typography.fontFamily.mono};
    font-size: ${(props) => props.theme.typography.fontSize.xs};
    margin-top: ${(props) => props.theme.spacing.xs};
  }
  
  .positive { color: ${(props) => props.theme.colors.acidGreen}; }
  .negative { color: ${(props) => props.theme.colors.error}; }
`;

const Title = styled(motion.h1)<{ theme: CyberpunkTheme }>`
  font-family: ${(props) => props.theme.typography.fontFamily.display};
  font-size: ${(props) => props.theme.typography.fontSize.display};
  background: ${(props) => props.theme.effects.primaryGradient};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: ${(props) => props.theme.effects.neonGlow};
  margin-bottom: ${(props) => props.theme.spacing.md};
  text-align: center;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: ${(props) => props.theme.effects.primaryGradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: glitch 2s infinite;
  }
  
  @keyframes glitch {
    0%, 100% { transform: translate(0); }
    20% { transform: translate(-2px, 2px); }
    40% { transform: translate(-2px, -2px); }
    60% { transform: translate(2px, 2px); }
    80% { transform: translate(2px, -2px); }
  }
`;

const Subtitle = styled(motion.p)<{ theme: CyberpunkTheme }>`
  font-family: ${(props) => props.theme.typography.fontFamily.mono};
  color: ${(props) => props.theme.colors.secondaryText};
  font-size: ${(props) => props.theme.typography.fontSize.lg};
  text-transform: uppercase;
  letter-spacing: 2px;
  text-align: center;
  position: relative;
  
  &::before {
    content: '> ';
    color: ${(props) => props.theme.colors.neonBlue};
    animation: pulse 2s infinite;
  }
  
  &::after {
    content: ' <';
    color: ${(props) => props.theme.colors.hotPink};
    animation: pulse 2s infinite reverse;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;

const ContentGrid = styled.div<{ theme: CyberpunkTheme }>`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: ${(props) => props.theme.spacing.lg};
  margin-top: ${(props) => props.theme.spacing.xl};
  position: relative;
  z-index: 1;
  
  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-template-columns: 1fr;
    gap: ${(props) => props.theme.spacing.md};
  }
`;

const DemoCard = styled(CyberpunkCard)`
  min-height: 200px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  gap: ${props => props.theme.spacing.md};
`;

const CardTitle = styled.h3<{ theme: CyberpunkTheme }>`
  color: ${(props) => props.theme.colors.neonBlue};
  font-family: ${(props) => props.theme.typography.fontFamily.primary};
  margin-bottom: ${(props) => props.theme.spacing.sm};
  font-size: ${(props) => props.theme.typography.fontSize.xl};
  text-shadow: ${(props) => props.theme.effects.softGlow};
  transition: all 0.3s ease;
  
  &:hover {
    color: ${(props) => props.theme.colors.hotPink};
    text-shadow: ${(props) => props.theme.effects.neonGlow};
  }
`;

const CardDescription = styled.p<{ theme: CyberpunkTheme }>`
  color: ${(props) => props.theme.colors.secondaryText};
  font-family: ${(props) => props.theme.typography.fontFamily.mono};
  font-size: ${(props) => props.theme.typography.fontSize.sm};
  line-height: 1.6;
  text-align: center;
  margin-bottom: ${(props) => props.theme.spacing.md};
  opacity: 0.9;
  
  &:hover {
    opacity: 1;
    color: ${(props) => props.theme.colors.primaryText};
  }
`;

interface DashboardMetrics {
  totalCustomers: number;
  retentionRate: number;
  forecastAccuracy: number;
  systemHealth: number;
  activeAlerts: number;
  revenueGrowth: number;
}

interface SystemStatus {
  status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  uptime: string;
  lastUpdate: string;
}

export const MainDashboard: React.FC = () => {
  const { user, authToken, isAuthenticated, login, logout } = useAuth();
  const { metrics, systemStatus, loading: metricsLoading } = useCompanyMetrics();
  const [activeView, setActiveView] = useState('company');
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  // Set connection status when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      setIsConnected(true);
    } else {
      setIsConnected(false);
    }
  }, [isAuthenticated]);

  const navigationItems: NavigationItem[] = [
    {
      id: 'company',
      label: 'Company',
      active: activeView === 'company',
      onClick: () => setActiveView('company'),
      badge: metrics.ragStatus === 'initialized' ? 'READY' : 'SETUP',
    },
    {
      id: 'models',
      label: 'Model Status',
      active: activeView === 'models',
      onClick: () => setActiveView('models'),
      badge: metrics.ragStatus === 'initialized' ? '✓' : '!',
    },
    {
      id: 'forecasting',
      label: 'Forecasting',
      active: activeView === 'forecasting',
      onClick: () => setActiveView('forecasting'),
      badge: metrics.activeAlerts > 0 ? metrics.activeAlerts.toString() : undefined,
    },
    {
      id: 'analytics',
      label: 'Analytics',
      active: activeView === 'analytics',
      onClick: () => setActiveView('analytics'),
      badge: metrics.retentionRate > 0 ? `${Math.floor(metrics.retentionRate * 100)}%` : 'N/A',
    },
    {
      id: 'insights',
      label: 'AI Insights',
      active: activeView === 'insights',
      onClick: () => setActiveView('insights'),
      badge: metrics.ragStatus === 'initialized' ? 'LIVE' : 'OFF',
    },
    {
      id: 'health',
      label: 'System Health',
      active: activeView === 'health',
      onClick: () => setActiveView('health'),
      badge: systemStatus.status === 'excellent' ? '✓' : systemStatus.status === 'good' ? '○' : '!',
    },
    {
      id: 'chatbot',
      label: 'AI Assistant',
      active: chatOpen,
      onClick: () => setChatOpen(true),
      badge: metrics.ragStatus === 'initialized' ? 'AI' : 'OFF',
    },
  ];

  const handleLogin = (token: string, userData: any) => {
    login(token, userData);
  };

  const handleLogout = () => {
    logout();
  };

  const handleDemoAction = (action: string) => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      console.log(`Demo action: ${action}`);
    }, 2000);
  };

  const renderCompanyContent = () => {
    return (
      <>
        <CompanyDetails 
          metrics={{
            totalCustomers: metrics.totalCustomers,
            retentionRate: metrics.retentionRate,
            revenueGrowth: metrics.revenueGrowth,
            forecastAccuracy: metrics.forecastAccuracy,
            systemHealth: metrics.systemHealth,
            activeAlerts: metrics.activeAlerts
          }}
          companyName={user?.company_name || 'Your Company'}
        />

        {/* Integrated Data Upload Section */}
        <DataUpload 
          onUploadComplete={(result) => {
            console.log('Data uploaded successfully:', result);
            // Automatically switch to models view to show ensemble status
            if (result.success && result.models_initialized) {
              setTimeout(() => setActiveView('models'), 2000);
            }
          }}
          onAuthError={handleLogout}
        />

        {/* Ensemble Status Section */}
        <EnsembleStatus />
  
        <ContentGrid>
          <DemoCard $variant="glass" $hover>
            <CardTitle>🔮 Generate Forecasts</CardTitle>
            <CardDescription>
              Generate adaptive ensemble forecasts with confidence intervals (P10/P50/P90). 
              System automatically updates model weights based on performance.
            </CardDescription>
            <CyberpunkButton
              $variant="primary"
              onClick={() => setActiveView('forecasting')}
            >
              Generate Forecast
            </CyberpunkButton>
          </DemoCard>
  
          <DemoCard $variant="neon" $hover>
            <CardTitle>📈 Performance Analytics</CardTitle>
            <CardDescription>
              Analyze model performance evolution, weight changes over time, 
              and forecast accuracy metrics with detailed visualizations.
            </CardDescription>
            <CyberpunkButton
              $variant="secondary"
              onClick={() => setActiveView('analytics')}
            >
              View Analytics
            </CyberpunkButton>
          </DemoCard>

          <DemoCard $variant="hologram" $hover>
            <CardTitle>🧠 AI Insights Engine</CardTitle>
            <CardDescription>
              Automated business insight generation with anomaly detection and pattern recognition.
            </CardDescription>
            <CyberpunkButton
              $variant="primary"
              onClick={() => setActiveView('insights')}
            >
              Generate Insights
            </CyberpunkButton>
          </DemoCard>

          <DemoCard $variant="glass" $hover>
            <CardTitle>⚡ System Health</CardTitle>
            <CardDescription>
              Real-time system monitoring with predictive maintenance and performance metrics.
            </CardDescription>
            <CyberpunkButton
              $variant="secondary"
              onClick={() => setActiveView('health')}
            >
              View System Health
            </CyberpunkButton>
          </DemoCard>
        </ContentGrid>
      </>
    );
  };

  const renderForecastingContent = () => (
    <ForecastingDashboard />
  );

  const renderAnalyticsContent = () => (
    <CustomerAnalyticsDashboard />
  );

  const renderInsightsContent = () => (
    <ContentGrid>
      <DemoCard $variant="glass" $hover>
        <CardTitle>🧠 AI Insights Engine</CardTitle>
        <CardDescription>
          {metrics.ragStatus === 'initialized' 
            ? `Automated business insight generation for ${user?.company_name}. Documents processed: ${metrics.totalDocuments}`
            : 'Upload your data to activate AI insights and anomaly detection.'
          }
        </CardDescription>
        <CyberpunkButton 
          $variant="primary" 
          onClick={() => handleDemoAction('ai-insights')}
          disabled={metrics.ragStatus !== 'initialized'}
        >
          {metrics.ragStatus === 'initialized' ? 'Generate Insights' : 'Upload Data First'}
        </CyberpunkButton>
      </DemoCard>

      <DemoCard $variant="neon" $hover>
        <CardTitle>🔍 Anomaly Detection</CardTitle>
        <CardDescription>
          {metrics.ragStatus === 'initialized' 
            ? `Real-time outlier identification for ${user?.business_type} business patterns.`
            : 'Requires data upload to detect business anomalies.'
          }
        </CardDescription>
        <CyberpunkButton 
          $variant="secondary" 
          onClick={() => handleDemoAction('anomaly-detection')}
          disabled={metrics.ragStatus !== 'initialized'}
        >
          {metrics.ragStatus === 'initialized' ? 'Detect Anomalies' : 'Setup Required'}
        </CyberpunkButton>
      </DemoCard>

      <DemoCard $variant="hologram" $hover>
        <CardTitle>📊 Trend Analysis</CardTitle>
        <CardDescription>
          {metrics.ragStatus === 'initialized' 
            ? `Advanced trend detection for ${user?.company_name} data patterns.`
            : 'Upload historical data to enable trend analysis.'
          }
        </CardDescription>
        <CyberpunkButton 
          $variant="primary" 
          onClick={() => handleDemoAction('trend-analysis')}
          disabled={metrics.ragStatus !== 'initialized'}
        >
          {metrics.ragStatus === 'initialized' ? 'Analyze Trends' : 'Data Required'}
        </CyberpunkButton>
      </DemoCard>
    </ContentGrid>
  );

  const renderHealthContent = () => (
    <ContentGrid>
      <DemoCard $variant="neon" $hover>
        <CardTitle>⚡ System Monitor</CardTitle>
        <CardDescription>
          {user?.company_name} system health monitoring.
          Current health: {(metrics.systemHealth * 100).toFixed(0)}%
          {metrics.lastDataUpload && ` • Last upload: ${new Date(metrics.lastDataUpload).toLocaleDateString()}`}
        </CardDescription>
        <CyberpunkButton $variant="primary" onClick={() => handleDemoAction('system-monitor')}>
          View Details
        </CyberpunkButton>
      </DemoCard>

      <DemoCard $variant="hologram" $hover>
        <CardTitle>🔧 AI System Status</CardTitle>
        <CardDescription>
          RAG Status: {metrics.ragStatus === 'initialized' ? 'Active' : 'Not Initialized'}
          {metrics.activeAlerts > 0 && ` • Active alerts: ${metrics.activeAlerts}`}
          {metrics.totalDocuments > 0 && ` • Documents: ${metrics.totalDocuments}`}
        </CardDescription>
        <CyberpunkButton 
          $variant={metrics.activeAlerts > 0 ? "danger" : "secondary"} 
          onClick={() => handleDemoAction('predictive-maintenance')}
        >
          {metrics.activeAlerts > 0 ? 'Check Alerts' : 'System OK'}
        </CyberpunkButton>
      </DemoCard>

      <DemoCard $variant="glass" $hover>
        <CardTitle>📊 Performance Metrics</CardTitle>
        <CardDescription>
          {user?.company_name} performance analytics.
          Forecast accuracy: {metrics.forecastAccuracy > 0 ? `${(metrics.forecastAccuracy * 100).toFixed(1)}%` : 'N/A'}
        </CardDescription>
        <CyberpunkButton $variant="secondary" onClick={() => handleDemoAction('performance-metrics')}>
          View Metrics
        </CyberpunkButton>
      </DemoCard>
    </ContentGrid>
  );

  // Show login form if not authenticated
  if (!isAuthenticated) {
    return <LoginForm onLogin={handleLogin} />;
  }

  return (
    <DashboardContainer>
      <Header>
        <Title
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          {user?.company_name || 'Your Company'}
        </Title>
        <Subtitle
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {user?.business_type ? `${user.business_type.charAt(0).toUpperCase() + user.business_type.slice(1)} AI Intelligence` : 'AI Business Intelligence'}
        </Subtitle>
      </Header>

      <StatusBar
        connected={isConnected}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <StatusItem>
          <div className="status-dot" />
          <span>System Status: </span>
          <span className="metric-value">{systemStatus.status.toUpperCase()}</span>
        </StatusItem>
        <StatusItem>
          <span>Uptime: </span>
          <span className="metric-value">{systemStatus.uptime}</span>
        </StatusItem>
        <StatusItem>
          <span>Last Update: </span>
          <span className="metric-value">{systemStatus.lastUpdate}</span>
        </StatusItem>
        <StatusItem>
          <span>Company: </span>
          <span className="metric-value">{user?.company_name || 'N/A'}</span>
        </StatusItem>
        <StatusItem>
          <span>User: </span>
          <span className="metric-value">{user?.email || 'Guest'}</span>
        </StatusItem>
        <StatusItem>
          <button 
            onClick={handleLogout}
            style={{
              background: 'none',
              border: '1px solid #ff6b6b',
              color: '#ff6b6b',
              padding: '4px 8px',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Logout
          </button>
        </StatusItem>
      </StatusBar>

      <CyberpunkNavigation
        items={navigationItems}
        variant="floating"
      />

      {/* Tab Content */}
      <AnimatePresence mode="wait">

        
        {activeView === 'forecasting' && (
          <motion.div
            key="forecasting"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderForecastingContent()}
          </motion.div>
        )}
        
        {activeView === 'analytics' && (
          <motion.div
            key="analytics"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderAnalyticsContent()}
          </motion.div>
        )}
        
        {activeView === 'insights' && (
          <motion.div
            key="insights"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderInsightsContent()}
          </motion.div>
        )}
        
        {activeView === 'health' && (
          <motion.div
            key="health"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderHealthContent()}
          </motion.div>
        )}
        
        {activeView === 'models' && (
          <motion.div
            key="models"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <EnsembleStatus />
            <ModelMonitoringDashboard />
          </motion.div>
        )}
        
        {activeView === 'company' && (
          <motion.div
            key="company"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderCompanyContent()}
          </motion.div>
        )}
      </AnimatePresence>

      <MetricsGrid>
        <MetricCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <div className="metric-label">Total Customers</div>
          <div className="metric-value">{metrics.totalCustomers.toLocaleString()}</div>
          <div className="metric-change positive">↗ +2.3% from last month</div>
        </MetricCard>

        <MetricCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <div className="metric-label">Retention Rate</div>
          <div className="metric-value">{(metrics.retentionRate * 100).toFixed(1)}%</div>
          <div className="metric-change positive">↗ +1.2% from last week</div>
        </MetricCard>

        <MetricCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <div className="metric-label">Forecast Accuracy</div>
          <div className="metric-value">{(metrics.forecastAccuracy * 100).toFixed(1)}%</div>
          <div className="metric-change positive">↗ +0.8% improvement</div>
        </MetricCard>

        <MetricCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.9 }}
        >
          <div className="metric-label">System Health</div>
          <div className="metric-value">{(metrics.systemHealth * 100).toFixed(0)}%</div>
          <div className="metric-change positive">↗ All systems optimal</div>
        </MetricCard>

        <MetricCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 1.0 }}
        >
          <div className="metric-label">Active Alerts</div>
          <div className="metric-value">{metrics.activeAlerts}</div>
          <div className="metric-change positive">↓ -2 resolved today</div>
        </MetricCard>

        <MetricCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 1.1 }}
        >
          <div className="metric-label">Revenue Growth</div>
          <div className="metric-value">{(metrics.revenueGrowth * 100).toFixed(1)}%</div>
          <div className="metric-change positive">↗ +5.2% this quarter</div>
        </MetricCard>
      </MetricsGrid>

      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{ display: 'flex', justifyContent: 'center', margin: '2rem 0' }}
        >
          <CyberpunkLoader
            variant="hologram"
            size="lg"
            text="Initializing AI Systems"
          />
        </motion.div>
      )}

      {/* AI Chatbot Interface */}
      <AnimatePresence>
        {chatOpen && (
          <EnsembleChatInterface
            isOpen={chatOpen}
            onClose={() => setChatOpen(false)}
            companyId={user?.company_id || user?.company_name || 'default'}
          />
        )}
      </AnimatePresence>
    </DashboardContainer>
  );
};