import React from 'react';
import { render, screen } from '@testing-library/react';
import { CustomerSegmentChart } from '../CustomerSegmentChart';
import { ThemeProvider } from 'styled-components';

// Mock theme for testing
const mockTheme = {
  colors: {
    acidGreen: '#39ff14',
    neonBlue: '#00ffff',
    secondaryText: '#888',
  },
  typography: {
    fontFamily: {
      display: 'Arial',
      mono: 'Courier New',
    },
    fontSize: {
      xs: '12px',
      sm: '14px',
      md: '16px',
      lg: '18px',
    },
  },
  effects: {
    softGlow: '0 0 10px rgba(57, 255, 20, 0.5)',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
  },
  breakpoints: {
    mobile: '768px',
    tablet: '1024px',
    desktop: '1200px',
  },
} as any;

// Mock recharts to avoid canvas issues in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  RadarChart: ({ children, data }: any) => (
    <div data-testid="radar-chart" data-chart-data={JSON.stringify(data)}>
      {children}
    </div>
  ),
  PolarGrid: () => <div data-testid="polar-grid" />,
  PolarAngleAxis: ({ dataKey }: any) => (
    <div data-testid="polar-angle-axis" data-key={dataKey} />
  ),
  PolarRadiusAxis: ({ angle, domain }: any) => (
    <div data-testid="polar-radius-axis" data-angle={angle} data-domain={JSON.stringify(domain)} />
  ),
  Radar: ({ name, dataKey, stroke, fill }: any) => (
    <div 
      data-testid="radar" 
      data-name={name} 
      data-key={dataKey} 
      data-stroke={stroke} 
      data-fill={fill} 
    />
  ),
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Cell: () => <div data-testid="cell" />,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Pie: ({ children }: any) => <div data-testid="pie">{children}</div>,
}));

describe('RadarChart Integration', () => {
  const mockSegments = [
    {
      segment_id: '1',
      segment_name: 'Premium Customers',
      customer_count: 1000,
      avg_ltv: 5000,
      avg_retention_rate: 0.85,
      revenue_contribution: 45.2,
      growth_rate: 12.5,
      health_score: 92,
      characteristics: ['high-value', 'loyal'],
    },
    {
      segment_id: '2',
      segment_name: 'Regular Customers',
      customer_count: 2500,
      avg_ltv: 2000,
      avg_retention_rate: 0.65,
      revenue_contribution: 35.8,
      growth_rate: 8.2,
      health_score: 78,
      characteristics: ['moderate-value', 'stable'],
    },
    {
      segment_id: '3',
      segment_name: 'New Customers',
      customer_count: 800,
      avg_ltv: 800,
      avg_retention_rate: 0.45,
      revenue_contribution: 19.0,
      growth_rate: 25.1,
      health_score: 65,
      characteristics: ['low-value', 'growing'],
    },
  ];

  const renderWithTheme = (component: React.ReactElement) => {
    return render(
      <ThemeProvider theme={mockTheme}>
        {component}
      </ThemeProvider>
    );
  };

  describe('RadarChart Children Prop Handling', () => {
    it('renders radar chart with multiple children components', () => {
      renderWithTheme(
        <CustomerSegmentChart 
          segments={mockSegments} 
          chartType="radar" 
          showComparison={false}
        />
      );

      // Check that radar chart is rendered
      expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
      
      // Check that all radar chart children are rendered
      expect(screen.getByTestId('polar-grid')).toBeInTheDocument();
      expect(screen.getByTestId('polar-angle-axis')).toBeInTheDocument();
      expect(screen.getByTestId('polar-radius-axis')).toBeInTheDocument();
      
      // Check that radar components are rendered for each segment
      const radarComponents = screen.getAllByTestId('radar');
      expect(radarComponents.length).toBeGreaterThan(0);
    });

    it('handles empty data gracefully in radar chart', () => {
      renderWithTheme(
        <CustomerSegmentChart 
          segments={[]} 
          chartType="radar" 
          showComparison={false}
        />
      );

      // Should show empty state message
      expect(screen.getByText('No Segment Data Available')).toBeInTheDocument();
    });

    it('validates radar chart data structure', () => {
      renderWithTheme(
        <CustomerSegmentChart 
          segments={mockSegments} 
          chartType="radar" 
          showComparison={false}
        />
      );

      const radarChart = screen.getByTestId('radar-chart');
      const chartData = JSON.parse(radarChart.getAttribute('data-chart-data') || '[]');
      
      // Verify data structure for radar chart
      expect(Array.isArray(chartData)).toBe(true);
      if (chartData.length > 0) {
        expect(chartData[0]).toHaveProperty('subject');
      }
    });

    it('renders PolarRadiusAxis with correct props', () => {
      renderWithTheme(
        <CustomerSegmentChart 
          segments={mockSegments} 
          chartType="radar" 
          showComparison={false}
        />
      );

      const polarRadiusAxis = screen.getByTestId('polar-radius-axis');
      expect(polarRadiusAxis.getAttribute('data-angle')).toBe('90');
      expect(JSON.parse(polarRadiusAxis.getAttribute('data-domain') || '[]')).toEqual([0, 100]);
    });

    it('renders PolarAngleAxis with correct dataKey', () => {
      renderWithTheme(
        <CustomerSegmentChart 
          segments={mockSegments} 
          chartType="radar" 
          showComparison={false}
        />
      );

      const polarAngleAxis = screen.getByTestId('polar-angle-axis');
      expect(polarAngleAxis.getAttribute('data-key')).toBe('subject');
    });
  });

  describe('Chart Type Switching', () => {
    it('switches between chart types correctly', () => {
      const { rerender } = renderWithTheme(
        <CustomerSegmentChart 
          segments={mockSegments} 
          chartType="bar" 
          showComparison={false}
        />
      );

      // Initially shows bar chart
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument();

      // Switch to radar chart
      rerender(
        <ThemeProvider theme={mockTheme}>
          <CustomerSegmentChart 
            segments={mockSegments} 
            chartType="radar" 
            showComparison={false}
          />
        </ThemeProvider>
      );

      expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
      expect(screen.queryByTestId('bar-chart')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles malformed segment data', () => {
      const malformedSegments = [
        {
          segment_id: '1',
          segment_name: 'Test',
          // Missing required fields
        } as any,
      ];

      renderWithTheme(
        <CustomerSegmentChart 
          segments={malformedSegments} 
          chartType="radar" 
          showComparison={false}
        />
      );

      // Should still render without crashing
      expect(screen.getByText('Customer Segment Performance')).toBeInTheDocument();
    });
  });
});