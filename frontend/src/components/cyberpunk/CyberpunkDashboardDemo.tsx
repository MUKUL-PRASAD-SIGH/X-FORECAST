import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import {
  CyberpunkAlertNotification,
  RealTimeDashboard,
  ModelWeightAdjustment,
  PerformanceGauge,
  type AlertNotificationProps,
  type DashboardMetric,
  type ActivityItem,
  type ModelWeight
} from './index';
import { CyberpunkCard, CyberpunkButton } from '../ui';

const DemoContainer = styled.div`
  padding: 2rem;
  background: ${props => props.theme.colors.darkBg};
  min-height: 100vh;
`;

const DemoHeader = styled.div`
  text-align: center;
  margin-bottom: 3rem;
  
  h1 {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.display};
    color: ${props => props.theme.colors.neonBlue};
    text-shadow: ${props => props.theme.effects.neonGlow};
    margin: 0 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 3px;
  }
  
  p {
    font-family: ${props => props.theme.typography.fontFamily.mono};
    color: ${props => props.theme.colors.secondaryText};
    font-size: ${props => props.theme.typography.fontSize.md};
    margin: 0;
  }
`;

const DemoSection = styled(CyberpunkCard)`
  margin-bottom: 3rem;
  
  h2 {
    font-family: ${props => props.theme.typography.fontFamily.display};
    font-size: ${props => props.theme.typography.fontSize.xl};
    color: ${props => props.theme.colors.hotPink};
    text-shadow: ${props => props.theme.effects.softGlow};
    margin: 0 0 2rem 0;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
`;

const AlertsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-height: 400px;
  overflow-y: auto;
`;

const GaugeGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  justify-items: center;
  margin-bottom: 2rem;
`;

const ControlsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const ControlButton = styled(CyberpunkButton)`
  margin: 0.25rem;
`;

export const CyberpunkDashboardDemo: React.FC = () => {
  const [alerts, setAlerts] = useState<AlertNotificationProps[]>([]);
  const [dashboardMetrics, setDashboardMetrics] = useState<DashboardMetric[]>([]);
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [modelWeights, setModelWeights] = useState<ModelWeight[]>([]);
  const [weightMode, setWeightMode] = useState<'manual' | 'auto' | 'hybrid'>('manual');
  const [gaugeValues, setGaugeValues] = useState({
    accuracy: 85,
    performance: 92,
    health: 78,
    efficiency: 88
  });

  // Initialize demo data
  useEffect(() => {
    // Initialize alerts
    const initialAlerts: AlertNotificationProps[] = [
      {
        id: 'alert-1',
        severity: 'critical',
        title: 'Model Performance Degraded',
        message: 'LSTM model accuracy dropped below 70% threshold',
        details: {
          model: 'LSTM',
          current_accuracy: 0.68,
          threshold: 0.70,
          recommendation: 'Immediate retraining required'
        },
        timestamp: new Date().toISOString(),
        actions: [
          { label: 'Retrain Model', action: 'retrain', variant: 'primary' },
          { label: 'View Details', action: 'details', variant: 'secondary' }
        ]
      },
      {
        id: 'alert-2',
        severity: 'warning',
        title: 'High Response Time',
        message: 'XGBoost model response time exceeds 500ms',
        details: 'Average response time: 650ms\nTarget: <500ms',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        actions: [
          { label: 'Optimize', action: 'optimize', variant: 'primary' }
        ]
      },
      {
        id: 'alert-3',
        severity: 'info',
        title: 'Training Complete',
        message: 'ARIMA model training completed successfully',
        details: 'New accuracy: 89.5%\nImprovement: +3.2%',
        timestamp: new Date(Date.now() - 600000).toISOString(),
        actions: [
          { label: 'Deploy', action: 'deploy', variant: 'primary' }
        ]
      }
    ];
    setAlerts(initialAlerts);

    // Initialize dashboard metrics
    const initialMetrics: DashboardMetric[] = [
      { label: 'Active Models', value: '5/5', status: 'good', trend: '+0 from last hour' },
      { label: 'Avg Accuracy', value: '87.3%', status: 'good', trend: '+2.1% from yesterday' },
      { label: 'Response Time', value: '245ms', status: 'good', trend: '-15ms from last hour' },
      { label: 'Predictions/Hour', value: '1,247', status: 'warning', trend: '+156 from last hour' }
    ];
    setDashboardMetrics(initialMetrics);

    // Initialize activities
    const initialActivities: ActivityItem[] = [
      { id: 'act-1', timestamp: new Date().toISOString(), message: 'LSTM model retrained with new data' },
      { id: 'act-2', timestamp: new Date(Date.now() - 120000).toISOString(), message: 'Weight optimization completed' },
      { id: 'act-3', timestamp: new Date(Date.now() - 240000).toISOString(), message: 'XGBoost model performance alert triggered' },
      { id: 'act-4', timestamp: new Date(Date.now() - 360000).toISOString(), message: 'New forecast request processed' },
      { id: 'act-5', timestamp: new Date(Date.now() - 480000).toISOString(), message: 'ARIMA model weights adjusted' }
    ];
    setActivities(initialActivities);

    // Initialize model weights
    const initialWeights: ModelWeight[] = [
      {
        modelName: 'ARIMA',
        weight: 0.25,
        performance: 'good',
        accuracy: 0.89,
        mape: 12.3,
        responseTime: 180,
        isActive: true
      },
      {
        modelName: 'ETS',
        weight: 0.20,
        performance: 'average',
        accuracy: 0.82,
        mape: 15.7,
        responseTime: 220,
        isActive: true
      },
      {
        modelName: 'XGBoost',
        weight: 0.30,
        performance: 'excellent',
        accuracy: 0.94,
        mape: 8.9,
        responseTime: 650,
        isActive: true
      },
      {
        modelName: 'LSTM',
        weight: 0.15,
        performance: 'poor',
        accuracy: 0.68,
        mape: 22.1,
        responseTime: 890,
        isActive: false
      },
      {
        modelName: 'Croston',
        weight: 0.10,
        performance: 'average',
        accuracy: 0.76,
        mape: 18.4,
        responseTime: 310,
        isActive: true
      }
    ];
    setModelWeights(initialWeights);
  }, []);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update gauge values
      setGaugeValues(prev => ({
        accuracy: Math.max(60, Math.min(100, prev.accuracy + (Math.random() - 0.5) * 4)),
        performance: Math.max(60, Math.min(100, prev.performance + (Math.random() - 0.5) * 3)),
        health: Math.max(50, Math.min(100, prev.health + (Math.random() - 0.5) * 5)),
        efficiency: Math.max(70, Math.min(100, prev.efficiency + (Math.random() - 0.5) * 2))
      }));

      // Add new activity
      const newActivity: ActivityItem = {
        id: `act-${Date.now()}`,
        timestamp: new Date().toISOString(),
        message: `System update: ${Math.random() > 0.5 ? 'Model prediction completed' : 'Performance metrics updated'}`
      };
      
      setActivities(prev => [newActivity, ...prev.slice(0, 9)]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleAlertClose = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const handleAlertAction = (alertId: string, action: string) => {
    console.log(`Alert ${alertId} action: ${action}`);
    // Handle specific actions here
  };

  const handleWeightChange = (modelName: string, weight: number) => {
    setModelWeights(prev => prev.map(model => 
      model.modelName === modelName ? { ...model, weight } : model
    ));
  };

  const handleApplyWeights = (weights: Record<string, number>) => {
    console.log('Applying weights:', weights);
    // Apply weights to backend
  };

  const handleResetWeights = () => {
    console.log('Resetting weights');
    // Reset to original weights
  };

  const handleOptimizeWeights = () => {
    console.log('Optimizing weights');
    // Trigger auto-optimization
  };

  const addRandomAlert = () => {
    const severities: ('critical' | 'warning' | 'info')[] = ['critical', 'warning', 'info'];
    const titles = [
      'Model Drift Detected',
      'Performance Threshold Exceeded',
      'Training Completed',
      'Data Quality Issue',
      'System Health Check'
    ];
    const messages = [
      'Automatic monitoring detected anomaly',
      'Real-time performance metrics updated',
      'Background process completed successfully',
      'Validation check triggered alert',
      'Scheduled maintenance notification'
    ];

    const newAlert: AlertNotificationProps = {
      id: `alert-${Date.now()}`,
      severity: severities[Math.floor(Math.random() * severities.length)],
      title: titles[Math.floor(Math.random() * titles.length)],
      message: messages[Math.floor(Math.random() * messages.length)],
      timestamp: new Date().toISOString(),
      actions: [
        { label: 'Acknowledge', action: 'ack', variant: 'secondary' }
      ]
    };

    setAlerts(prev => [newAlert, ...prev]);
  };

  return (
    <DemoContainer>
      <DemoHeader>
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          Cyberpunk Dashboard Components
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          Interactive cyberpunk-styled dashboard components with real-time updates and glitch effects
        </motion.p>
      </DemoHeader>

      {/* Performance Gauges Section */}
      <DemoSection variant="hologram">
        <h2>🎯 Performance Gauges with Neon Glow Effects</h2>
        <GaugeGrid>
          <PerformanceGauge
            value={gaugeValues.accuracy}
            label="Model Accuracy"
            unit="%"
            size={220}
            thresholds={{ excellent: 90, good: 80, average: 70, poor: 60 }}
          />
          <PerformanceGauge
            value={gaugeValues.performance}
            label="System Performance"
            unit="%"
            size={220}
            thresholds={{ excellent: 95, good: 85, average: 75, poor: 65 }}
          />
          <PerformanceGauge
            value={gaugeValues.health}
            label="System Health"
            unit="%"
            size={220}
            thresholds={{ excellent: 90, good: 80, average: 70, poor: 60 }}
          />
          <PerformanceGauge
            value={gaugeValues.efficiency}
            label="Efficiency"
            unit="%"
            size={220}
            thresholds={{ excellent: 95, good: 85, average: 75, poor: 65 }}
          />
        </GaugeGrid>
      </DemoSection>

      {/* Real-time Dashboard Section */}
      <DemoSection variant="neon">
        <h2>📊 Real-time Dashboard with Scan-line Effects</h2>
        <RealTimeDashboard
          title="Ensemble Model Monitoring"
          status="active"
          metrics={dashboardMetrics}
          activities={activities}
          refreshInterval={5000}
          enableEffects={true}
        />
      </DemoSection>

      {/* Model Weight Adjustment Section */}
      <DemoSection variant="glass">
        <h2>⚖️ Interactive Model Weight Adjustment Controls</h2>
        <ModelWeightAdjustment
          models={modelWeights}
          mode={weightMode}
          onWeightChange={handleWeightChange}
          onModeChange={setWeightMode}
          onApplyWeights={handleApplyWeights}
          onResetWeights={handleResetWeights}
          onOptimizeWeights={handleOptimizeWeights}
        />
      </DemoSection>

      {/* Alert Notifications Section */}
      <DemoSection variant="default">
        <h2>🚨 Cyberpunk Alert Notifications with Glitch Animations</h2>
        <ControlsGrid>
          <div>
            <h3 style={{ color: '#00FFFF', marginBottom: '1rem' }}>Controls</h3>
            <ControlButton variant="primary" onClick={addRandomAlert}>
              Add Random Alert
            </ControlButton>
            <ControlButton variant="secondary" onClick={() => setAlerts([])}>
              Clear All Alerts
            </ControlButton>
          </div>
        </ControlsGrid>
        <AlertsContainer>
          {alerts.map(alert => (
            <CyberpunkAlertNotification
              key={alert.id}
              {...alert}
              onClose={handleAlertClose}
              onAction={handleAlertAction}
            />
          ))}
        </AlertsContainer>
      </DemoSection>
    </DemoContainer>
  );
};

export default CyberpunkDashboardDemo;