import React from 'react';
import { ResponsiveContainer } from 'recharts';
import styled from 'styled-components';

// Chart wrapper component with proper children typing
interface ChartWrapperProps {
  children: React.ReactNode | React.ReactNode[];
  data?: any[];
  width?: number | `${number}%` | string;
  height?: number | `${number}%` | string;
  emptyMessage?: string;
}

const EmptyStateContainer = styled.div<{ width: number | `${number}%` | string; height: number | `${number}%` | string }>`
  width: ${props => typeof props.width === 'string' ? props.width : `${props.width}px`};
  height: ${props => typeof props.height === 'string' ? props.height : `${props.height}px`};
  display: flex;
  align-items: center;
  justify-content: center;
  color: #39ff14;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(57, 255, 20, 0.3);
  border-radius: 8px;
  text-align: center;
  padding: 20px;
`;

export const ChartWrapper: React.FC<ChartWrapperProps> = ({ 
  children, 
  data, 
  width = "100%", 
  height = "70%",
  emptyMessage = "No data available"
}) => {
  // Handle empty data scenarios with proper validation
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return (
      <EmptyStateContainer width={width} height={height}>
        {emptyMessage}
      </EmptyStateContainer>
    );
  }

  // Validate that children is a valid React element for chart rendering
  if (!children) {
    return (
      <EmptyStateContainer width={width} height={height}>
        Chart component not provided
      </EmptyStateContainer>
    );
  }

  return (
    <ResponsiveContainer 
      width={width as any} 
      height={height as any}
    >
      {children}
    </ResponsiveContainer>
  );
};

// Enhanced chart wrapper with error boundary
interface EnhancedChartWrapperProps extends ChartWrapperProps {
  onError?: (error: Error) => void;
}

interface ErrorBoundaryProps {
  children: React.ReactNode; 
  onError?: (error: Error) => void; 
  width: number | `${number}%` | string; 
  height: number | `${number}%` | string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ChartErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Chart rendering error:', error, errorInfo);
    if (this.props.onError) {
      this.props.onError(error);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <EmptyStateContainer width={this.props.width} height={this.props.height}>
          Chart rendering error: {this.state.error?.message || 'Unknown error'}
        </EmptyStateContainer>
      );
    }

    return this.props.children;
  }
}

export const EnhancedChartWrapper: React.FC<EnhancedChartWrapperProps> = ({ 
  children, 
  data, 
  width = "100%", 
  height = "70%",
  emptyMessage = "No data available",
  onError
}) => {
  return (
    <ChartErrorBoundary width={width} height={height} onError={onError}>
      <ChartWrapper 
        data={data} 
        width={width} 
        height={height} 
        emptyMessage={emptyMessage}
      >
        {children}
      </ChartWrapper>
    </ChartErrorBoundary>
  );
};

// Type definitions for common chart props
export interface BaseChartProps {
  data: any[];
  width?: string | number;
  height?: string | number;
  loading?: boolean;
  error?: string | null;
}

// Utility function to validate chart data
export const validateChartData = (data: any[]): boolean => {
  return Array.isArray(data) && data.length > 0 && data.every(item => item && typeof item === 'object');
};

// Utility function to normalize chart data for different chart types
export const normalizeChartData = (data: any[], requiredFields: string[]): any[] => {
  if (!validateChartData(data)) {
    return [];
  }

  return data.filter(item => 
    requiredFields.every(field => item.hasOwnProperty(field) && item[field] !== undefined)
  );
};