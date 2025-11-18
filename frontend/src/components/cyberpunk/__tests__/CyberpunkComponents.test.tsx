import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { cyberpunkTheme } from '../../../theme/cyberpunkTheme';
import {
  CyberpunkAlertNotification,
  RealTimeDashboard,
  ModelWeightAdjustment,
  PerformanceGauge
} from '../index';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={cyberpunkTheme}>
      {component}
    </ThemeProvider>
  );
};

describe('Cyberpunk Dashboard Components', () => {
  describe('CyberpunkAlertNotification', () => {
    const mockAlert = {
      id: 'test-alert',
      severity: 'critical' as const,
      title: 'Test Alert',
      message: 'This is a test alert message',
      timestamp: new Date().toISOString(),
      onClose: jest.fn(),
      onAction: jest.fn(),
      actions: [
        { label: 'Test Action', action: 'test', variant: 'primary' as const }
      ]
    };

    it('renders alert notification correctly', () => {
      renderWithTheme(<CyberpunkAlertNotification {...mockAlert} />);
      
      expect(screen.getByText('Test Alert')).toBeInTheDocument();
      expect(screen.getByText('This is a test alert message')).toBeInTheDocument();
      expect(screen.getByText('Test Action')).toBeInTheDocument();
      expect(screen.getByText('Dismiss')).toBeInTheDocument();
    });

    it('calls onClose when dismiss button is clicked', () => {
      renderWithTheme(<CyberpunkAlertNotification {...mockAlert} />);
      
      fireEvent.click(screen.getByText('Dismiss'));
      expect(mockAlert.onClose).toHaveBeenCalledWith('test-alert');
    });

    it('calls onAction when action button is clicked', () => {
      renderWithTheme(<CyberpunkAlertNotification {...mockAlert} />);
      
      fireEvent.click(screen.getByText('Test Action'));
      expect(mockAlert.onAction).toHaveBeenCalledWith('test-alert', 'test');
    });
  });

  describe('PerformanceGauge', () => {
    const mockGaugeProps = {
      value: 85,
      label: 'Test Gauge',
      unit: '%',
      size: 200
    };

    it('renders performance gauge correctly', () => {
      renderWithTheme(<PerformanceGauge {...mockGaugeProps} />);
      
      expect(screen.getByText('85%')).toBeInTheDocument();
      expect(screen.getByText('Test Gauge')).toBeInTheDocument();
    });

    it('displays correct status based on value', () => {
      const { rerender } = renderWithTheme(<PerformanceGauge {...mockGaugeProps} value={95} />);
      expect(screen.getByText('Excellent')).toBeInTheDocument();

      rerender(
        <ThemeProvider theme={cyberpunkTheme}>
          <PerformanceGauge {...mockGaugeProps} value={75} />
        </ThemeProvider>
      );
      expect(screen.getByText('Good')).toBeInTheDocument();

      rerender(
        <ThemeProvider theme={cyberpunkTheme}>
          <PerformanceGauge {...mockGaugeProps} value={50} />
        </ThemeProvider>
      );
      expect(screen.getByText('Average')).toBeInTheDocument();
    });
  });

  describe('RealTimeDashboard', () => {
    const mockDashboardProps = {
      title: 'Test Dashboard',
      status: 'active' as const,
      metrics: [
        { label: 'Test Metric', value: '100', status: 'good' as const, trend: '+5%' }
      ],
      activities: [
        { id: 'act-1', timestamp: new Date().toISOString(), message: 'Test activity' }
      ]
    };

    it('renders dashboard correctly', () => {
      renderWithTheme(<RealTimeDashboard {...mockDashboardProps} />);
      
      expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Test Metric')).toBeInTheDocument();
      expect(screen.getByText('100')).toBeInTheDocument();
      expect(screen.getByText('Test activity')).toBeInTheDocument();
    });

    it('displays status indicator correctly', () => {
      renderWithTheme(<RealTimeDashboard {...mockDashboardProps} />);
      
      expect(screen.getByText('ACTIVE')).toBeInTheDocument();
    });
  });

  describe('ModelWeightAdjustment', () => {
    const mockWeightProps = {
      models: [
        {
          modelName: 'ARIMA',
          weight: 0.25,
          performance: 'good' as const,
          accuracy: 0.89,
          mape: 12.3,
          responseTime: 180,
          isActive: true
        },
        {
          modelName: 'XGBoost',
          weight: 0.30,
          performance: 'excellent' as const,
          accuracy: 0.94,
          mape: 8.9,
          responseTime: 220,
          isActive: true
        }
      ],
      mode: 'manual' as const,
      onWeightChange: jest.fn(),
      onModeChange: jest.fn(),
      onApplyWeights: jest.fn(),
      onResetWeights: jest.fn(),
      onOptimizeWeights: jest.fn()
    };

    it('renders model weight adjustment correctly', () => {
      renderWithTheme(<ModelWeightAdjustment {...mockWeightProps} />);
      
      expect(screen.getByText('🎛️ Model Weight Control')).toBeInTheDocument();
      expect(screen.getByText('ARIMA')).toBeInTheDocument();
      expect(screen.getByText('XGBoost')).toBeInTheDocument();
      expect(screen.getByText('Manual')).toBeInTheDocument();
      expect(screen.getByText('Auto')).toBeInTheDocument();
      expect(screen.getByText('Hybrid')).toBeInTheDocument();
    });

    it('displays model performance badges correctly', () => {
      renderWithTheme(<ModelWeightAdjustment {...mockWeightProps} />);
      
      expect(screen.getByText('good')).toBeInTheDocument();
      expect(screen.getByText('excellent')).toBeInTheDocument();
    });

    it('calls onModeChange when mode button is clicked', () => {
      renderWithTheme(<ModelWeightAdjustment {...mockWeightProps} />);
      
      fireEvent.click(screen.getByText('Auto'));
      expect(mockWeightProps.onModeChange).toHaveBeenCalledWith('auto');
    });
  });
});