import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CyberpunkCard, 
  CyberpunkButton, 
  CyberpunkNavigation, 
  CyberpunkLoader,
  NavigationItem 
} from './ui';
import { CyberpunkChatInterface } from './chat/CyberpunkChatInterface';
import { HolographicRenderer } from './3d/HolographicRenderer';
import { LoginForm } from './auth/LoginForm';
import { DataUpload } from './DataUpload';
import { CyberpunkTheme } from '../theme/cyberpunkTheme';

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
  const [activeView, setActiveView] = useState('dashboard');
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(true); // Always authenticated
  const [user, setUser] = useState<any>({ email: 'superx@demo.com', company_name: 'SuperX Corporation' });
  const [authToken, setAuthToken] = useState<string | null>('superx_demo_token');
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalCustomers: 0,
    retentionRate: 0,
    forecastAccuracy: 0,
    systemHealth: 0,
    activeAlerts: 0,
    revenueGrowth: 0
  });
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'good',
    uptime: '99.9%',
    lastUpdate: new Date().toLocaleTimeString()
  });
  const [isConnected, setIsConnected] = useState(false);

  // Auto-authenticate as SuperX
  useEffect(() => {
    setAuthToken('superx_demo_token');
    setIsAuthenticated(true);
    setUser({ email: 'superx@demo.com', company_name: 'SuperX Corporation' });
  }, []);

  // Simulate real-time data updates
  useEffect(() => {
    const updateMetrics = () => {
      setMetrics({
        totalCustomers: Math.floor(Math.random() * 1000) + 5000,
        retentionRate: Math.random() * 0.3 + 0.7, // 70-100%
        forecastAccuracy: Math.random() * 0.2 + 0.8, // 80-100%
        systemHealth: Math.random() * 0.3 + 0.7, // 70-100%
        activeAlerts: Math.floor(Math.random() * 5),
        revenueGrowth: Math.random() * 0.4 + 0.8 // 80-120%
      });
      
      setSystemStatus({
        status: Math.random() > 0.8 ? 'excellent' : 'good',
        uptime: `${(99.5 + Math.random() * 0.5).toFixed(1)}%`,
        lastUpdate: new Date().toLocaleTimeString()
      });
    };

    // Initial load
    updateMetrics();
    setIsConnected(true);

    // Update every 5 seconds
    const interval = setInterval(updateMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      active: activeView === 'dashboard',
      onClick: () => setActiveView('dashboard'),
    },
    {
      id: 'forecasting',
      label: 'AI Forecasting',
      onClick: () => setActiveView('forecasting'),
      badge: metrics.activeAlerts > 0 ? metrics.activeAlerts.toString() : undefined,
    },
    {
      id: 'retention',
      label: 'Customer Analytics',
      onClick: () => setActiveView('retention'),
      badge: `${Math.floor(metrics.retentionRate * 100)}%`,
    },
    {
      id: 'insights',
      label: 'AI Insights',
      onClick: () => setActiveView('insights'),
      badge: 'LIVE',
    },
    {
      id: 'health',
      label: 'System Health',
      onClick: () => setActiveView('health'),
      badge: systemStatus.status === 'excellent' ? '✓' : '!',
    },
    {
      id: 'upload',
      label: 'Upload Data',
      onClick: () => setActiveView('upload'),
      badge: 'TRAIN',
    },
    {
      id: 'chatbot',
      label: 'AI Assistant',
      onClick: () => setChatOpen(true),
      badge: 'AI',
    },
  ];

  const handleLogin = (token: string, userData: any) => {
    setAuthToken(token);
    setUser(userData);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    setAuthToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  const handleDemoAction = (action: string) => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      console.log(`Demo action: ${action}`);
    }, 2000);
  };

  const renderDashboardContent = () => (
    <>
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

      <ContentGrid>
        <DemoCard variant="neon" hover>
          <CardTitle>🤖 AI Forecasting Engine</CardTitle>
          <CardDescription>
            Advanced ensemble models combining ARIMA, ETS, and deep learning 
            for accurate demand predictions with customer behavior integration.
          </CardDescription>
          <CyberpunkButton
            variant="primary"
            onClick={() => setActiveView('forecasting')}
          >
            Launch Forecasting
          </CyberpunkButton>
        </DemoCard>

        <DemoCard variant="hologram" hover>
          <CardTitle>👥 Customer Retention Analytics</CardTitle>
          <CardDescription>
            Comprehensive churn prediction, cohort analysis, and lifetime value 
            calculations integrated with demand forecasting models.
          </CardDescription>
          <CyberpunkButton
            variant="secondary"
            onClick={() => setActiveView('retention')}
          >
            Analyze Customers
          </CyberpunkButton>
        </DemoCard>

        <DemoCard variant="glass" hover>
          <CardTitle>🧠 AI Chatbot Assistant</CardTitle>
          <CardDescription>
            Natural language interface for querying forecasts, getting insights, 
            and exploring business data with conversational AI.
          </CardDescription>
          <CyberpunkButton
            variant="primary"
            onClick={() => setChatOpen(true)}
          >
            Start Conversation
          </CyberpunkButton>
        </DemoCard>

        <DemoCard variant="neon" hover>
          <CardTitle>📊 Upload Business Data</CardTitle>
          <CardDescription>
            Upload your sales data (CSV/Excel) to train personalized AI models 
            and get insights specific to your business.
          </CardDescription>
          <CyberpunkButton
            variant="secondary"
            onClick={() => setActiveView('upload')}
          >
            Upload Data
          </CyberpunkButton>
        </DemoCard>
      </ContentGrid>
    </>
  );

  const renderForecastingContent = () => (
    <ContentGrid>
      <DemoCard variant="neon" hover>
        <CardTitle>📊 Demand Forecasting</CardTitle>
        <CardDescription>
          Multi-horizon forecasts with P10/P50/P90 confidence intervals.
          Current accuracy: {(metrics.forecastAccuracy * 100).toFixed(1)}%
        </CardDescription>
        <CyberpunkButton variant="primary" onClick={() => handleDemoAction('demand-forecast')}>
          Generate Forecast
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="hologram" hover>
        <CardTitle>🎆 NPI Forecasting</CardTitle>
        <CardDescription>
          New Product Introduction forecasting using similarity matching and Bayesian methods.
        </CardDescription>
        <CyberpunkButton variant="secondary" onClick={() => handleDemoAction('npi-forecast')}>
          Launch NPI Model
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="glass" hover>
        <CardTitle>🎉 Promotion Analytics</CardTitle>
        <CardDescription>
          Uplift modeling and promotion optimization with cannibalization effects.
        </CardDescription>
        <CyberpunkButton variant="primary" onClick={() => handleDemoAction('promotion-analytics')}>
          Optimize Promotions
        </CyberpunkButton>
      </DemoCard>
    </ContentGrid>
  );

  const renderRetentionContent = () => (
    <ContentGrid>
      <DemoCard variant="hologram" hover>
        <CardTitle>📈 Churn Prediction</CardTitle>
        <CardDescription>
          ML-powered customer churn prediction with 88% precision.
          High-risk customers: {Math.floor(metrics.totalCustomers * 0.15)}
        </CardDescription>
        <CyberpunkButton variant="danger" onClick={() => handleDemoAction('churn-prediction')}>
          Identify At-Risk
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="neon" hover>
        <CardTitle>💰 Customer LTV</CardTitle>
        <CardDescription>
          Lifetime value calculation and segmentation analysis.
          Current retention rate: {(metrics.retentionRate * 100).toFixed(1)}%
        </CardDescription>
        <CyberpunkButton variant="primary" onClick={() => handleDemoAction('ltv-analysis')}>
          Calculate LTV
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="glass" hover>
        <CardTitle>🎯 Cohort Analysis</CardTitle>
        <CardDescription>
          Customer cohort tracking and retention pattern analysis.
        </CardDescription>
        <CyberpunkButton variant="secondary" onClick={() => handleDemoAction('cohort-analysis')}>
          View Cohorts
        </CyberpunkButton>
      </DemoCard>
    </ContentGrid>
  );

  const renderInsightsContent = () => (
    <ContentGrid>
      <DemoCard variant="glass" hover>
        <CardTitle>🧠 AI Insights Engine</CardTitle>
        <CardDescription>
          Automated business insight generation with anomaly detection.
          Active insights: {Math.floor(Math.random() * 10) + 5}
        </CardDescription>
        <CyberpunkButton variant="primary" onClick={() => handleDemoAction('ai-insights')}>
          Generate Insights
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="neon" hover>
        <CardTitle>🔍 Anomaly Detection</CardTitle>
        <CardDescription>
          Real-time outlier identification with 99.5% accuracy.
        </CardDescription>
        <CyberpunkButton variant="secondary" onClick={() => handleDemoAction('anomaly-detection')}>
          Detect Anomalies
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="hologram" hover>
        <CardTitle>📊 Trend Analysis</CardTitle>
        <CardDescription>
          Advanced trend detection and pattern recognition.
        </CardDescription>
        <CyberpunkButton variant="primary" onClick={() => handleDemoAction('trend-analysis')}>
          Analyze Trends
        </CyberpunkButton>
      </DemoCard>
    </ContentGrid>
  );

  const renderHealthContent = () => (
    <ContentGrid>
      <DemoCard variant="neon" hover>
        <CardTitle>⚡ System Monitor</CardTitle>
        <CardDescription>
          Real-time system health monitoring with predictive maintenance.
          Current health: {(metrics.systemHealth * 100).toFixed(0)}%
        </CardDescription>
        <CyberpunkButton variant="primary" onClick={() => handleDemoAction('system-monitor')}>
          View Details
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="hologram" hover>
        <CardTitle>🔧 Predictive Maintenance</CardTitle>
        <CardDescription>
          95% accuracy equipment failure prediction.
          Active alerts: {metrics.activeAlerts}
        </CardDescription>
        <CyberpunkButton variant="danger" onClick={() => handleDemoAction('predictive-maintenance')}>
          Check Predictions
        </CyberpunkButton>
      </DemoCard>

      <DemoCard variant="glass" hover>
        <CardTitle>📊 Performance Metrics</CardTitle>
        <CardDescription>
          Comprehensive system performance analytics and optimization.
        </CardDescription>
        <CyberpunkButton variant="secondary" onClick={() => handleDemoAction('performance-metrics')}>
          View Metrics
        </CyberpunkButton>
      </DemoCard>
    </ContentGrid>
  );

  // Login bypassed - always show dashboard

  return (
    <DashboardContainer>
      <Header>
        <Title
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          {user?.company_name || 'X-FORECAST'}
        </Title>
        <Subtitle
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Cyberpunk AI Business Intelligence
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
        {activeView === 'dashboard' && (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderDashboardContent()}
          </motion.div>
        )}
        
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
        
        {activeView === 'retention' && (
          <motion.div
            key="retention"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderRetentionContent()}
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
        
        {activeView === 'upload' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <DataUpload 
              authToken={authToken!} 
              onUploadComplete={() => {
                // Refresh data or show success message
                console.log('Data uploaded successfully');
              }} 
            />
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
          <CyberpunkChatInterface
            isOpen={chatOpen}
            onClose={() => setChatOpen(false)}
            onSendMessage={async (message: string) => {
              // Integrate with backend API
              try {
                const response = await fetch('http://localhost:8000/api/v1/auth/chat', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${authToken}`,
                  },
                  body: JSON.stringify({ message }),
                });
                const data = await response.json();
                return {
                  id: `ai-${Date.now()}`,
                  type: 'ai' as const,
                  content: data.response_text || data.response || 'Processing your request...',
                  timestamp: new Date(),
                  confidence: data.confidence || 0.85,
                  followUpQuestions: data.follow_up_questions || []
                };
              } catch (error) {
                console.error('Chat API error:', error);
                return {
                  id: `ai-error-${Date.now()}`,
                  type: 'ai' as const,
                  content: 'I apologize, but I\'m having trouble connecting to the AI service. Please try again later.',
                  timestamp: new Date(),
                  confidence: 0.0,
                  followUpQuestions: []
                };
              }
            }}
          />
        )}
      </AnimatePresence>
    </DashboardContainer>
  );
};