import React from 'react';
import { render, screen } from '@testing-library/react';
import { ChartWrapper, EnhancedChartWrapper, validateChartData, normalizeChartData } from '../ChartWrapper';
import { BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

// Mock recharts components
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  BarChart: ({ children }: any) => (
    <div data-testid="bar-chart">{children}</div>
  ),
  Bar: () => <div data-testid="bar" />,
  RadarChart: ({ children }: any) => (
    <div data-testid="radar-chart">{children}</div>
  ),
  Radar: () => <div data-testid="radar" />,
  PolarGrid: () => <div data-testid="polar-grid" />,
  PolarAngleAxis: () => <div data-testid="polar-angle-axis" />,
  PolarRadiusAxis: () => <div data-testid="polar-radius-axis" />,
}));

describe('ChartWrapper', () => {
  const mockData = [
    { name: 'A', value: 10 },
    { name: 'B', value: 20 },
  ];

  describe('Basic ChartWrapper', () => {
    it('renders children when data is provided', () => {
      render(
        <ChartWrapper data={mockData}>
          <BarChart data={mockData}>
            <Bar dataKey="value" />
          </BarChart>
        </ChartWrapper>
      );

      expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.getByTestId('bar')).toBeInTheDocument();
    });

    it('renders empty state when no data provided', () => {
      render(
        <ChartWrapper data={[]}>
          <BarChart data={[]}>
            <Bar dataKey="value" />
          </BarChart>
        </ChartWrapper>
      );

      expect(screen.getByText('No data available')).toBeInTheDocument();
      expect(screen.queryByTestId('responsive-container')).not.toBeInTheDocument();
    });

    it('renders custom empty message', () => {
      render(
        <ChartWrapper data={[]} emptyMessage="Custom empty message">
          <BarChart data={[]}>
            <Bar dataKey="value" />
          </BarChart>
        </ChartWrapper>
      );

      expect(screen.getByText('Custom empty message')).toBeInTheDocument();
    });

    it('handles multiple children (ReactNode array)', () => {
      render(
        <ChartWrapper data={mockData}>
          <RadarChart data={mockData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="name" />
            <PolarRadiusAxis />
            <Radar dataKey="value" />
          </RadarChart>
        </ChartWrapper>
      );

      expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
      expect(screen.getByTestId('polar-grid')).toBeInTheDocument();
      expect(screen.getByTestId('polar-angle-axis')).toBeInTheDocument();
      expect(screen.getByTestId('polar-radius-axis')).toBeInTheDocument();
      expect(screen.getByTestId('radar')).toBeInTheDocument();
    });

    it('handles undefined data gracefully', () => {
      render(
        <ChartWrapper data={undefined}>
          <BarChart data={[]}>
            <Bar dataKey="value" />
          </BarChart>
        </ChartWrapper>
      );

      expect(screen.getByText('No data available')).toBeInTheDocument();
    });

    it('handles null children gracefully', () => {
      render(
        <ChartWrapper data={mockData}>
          {null}
        </ChartWrapper>
      );

      expect(screen.getByText('Chart component not provided')).toBeInTheDocument();
    });
  });

  describe('EnhancedChartWrapper', () => {
    it('renders children normally when no error occurs', () => {
      render(
        <EnhancedChartWrapper data={mockData}>
          <BarChart data={mockData}>
            <Bar dataKey="value" />
          </BarChart>
        </EnhancedChartWrapper>
      );

      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    });

    it('calls onError callback when error occurs', () => {
      const onError = jest.fn();
      const ThrowingComponent = () => {
        throw new Error('Test error');
      };

      render(
        <EnhancedChartWrapper data={mockData} onError={onError}>
          <ThrowingComponent />
        </EnhancedChartWrapper>
      );

      expect(screen.getByText(/Chart rendering error/)).toBeInTheDocument();
    });
  });

  describe('Utility Functions', () => {
    describe('validateChartData', () => {
      it('returns true for valid data array', () => {
        expect(validateChartData(mockData)).toBe(true);
      });

      it('returns false for empty array', () => {
        expect(validateChartData([])).toBe(false);
      });

      it('returns false for non-array input', () => {
        expect(validateChartData(null as any)).toBe(false);
        expect(validateChartData(undefined as any)).toBe(false);
        expect(validateChartData('string' as any)).toBe(false);
      });

      it('returns false for array with invalid items', () => {
        expect(validateChartData([null, undefined, 'string'])).toBe(false);
      });
    });

    describe('normalizeChartData', () => {
      it('filters data to include only items with required fields', () => {
        const data = [
          { name: 'A', value: 10, extra: 'field' },
          { name: 'B' }, // missing value
          { value: 20 }, // missing name
          { name: 'C', value: 30 },
        ];

        const result = normalizeChartData(data, ['name', 'value']);
        expect(result).toHaveLength(2);
        expect(result[0]).toEqual({ name: 'A', value: 10, extra: 'field' });
        expect(result[1]).toEqual({ name: 'C', value: 30 });
      });

      it('returns empty array for invalid input', () => {
        expect(normalizeChartData([], ['name'])).toEqual([]);
        expect(normalizeChartData(null as any, ['name'])).toEqual([]);
      });

      it('handles undefined field values', () => {
        const data = [
          { name: 'A', value: undefined },
          { name: 'B', value: 0 },
          { name: 'C', value: null },
        ];

        const result = normalizeChartData(data, ['name', 'value']);
        expect(result).toHaveLength(1);
        expect(result[0].name).toBe('B');
      });
    });
  });

  describe('Prop Types and TypeScript Integration', () => {
    it('accepts string width and height', () => {
      render(
        <ChartWrapper data={mockData} width="100%" height="400px">
          <BarChart data={mockData}>
            <Bar dataKey="value" />
          </BarChart>
        </ChartWrapper>
      );

      expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    });

    it('accepts number width and height', () => {
      render(
        <ChartWrapper data={mockData} width={800} height={400}>
          <BarChart data={mockData}>
            <Bar dataKey="value" />
          </BarChart>
        </ChartWrapper>
      );

      expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    });
  });
});