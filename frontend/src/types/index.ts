// Core TypeScript interfaces for the Cyberpunk UI system

// Base component interfaces
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

// Common styled component props
export interface StyledComponentProps {
  $variant?: string;
  $size?: 'sm' | 'md' | 'lg';
  $padding?: 'sm' | 'md' | 'lg';
  $hover?: boolean;
  $glitch?: boolean;
  $loading?: boolean;
}

// Button specific interfaces
export interface ButtonVariants {
  $variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
}

export interface ButtonSizes {
  $size?: 'sm' | 'md' | 'lg';
}

// Card specific interfaces
export interface CardVariants {
  $variant?: 'default' | 'glass' | 'neon' | 'hologram';
}

// Navigation interfaces
export interface NavigationItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
  badge?: string | number;
}

// Data interfaces
export interface CompanyMetrics {
  totalCustomers: number;
  retentionRate: number;
  forecastAccuracy: number;
  systemHealth: number;
  activeAlerts: number;
  revenueGrowth: number;
  totalDocuments: number;
  ragStatus: string;
  lastDataUpload: string | null;
}

export interface SystemStatus {
  status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  uptime: string;
  lastUpdate: string;
}

export interface User {
  user_id: string;
  email: string;
  company_name: string;
  company_id: string;
  business_type: string;
  rag_initialized: boolean;
}

// API interfaces
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Upload interfaces
export interface DetectedColumn {
  name: string;
  type: string;
  sample_values: string[];
  confidence: number;
}

export interface ColumnMapping {
  required_field: string;
  detected_column: string | null;
  confidence: number;
  status: 'mapped' | 'unmapped' | 'uncertain';
}

export interface DataQuality {
  overall_score: number;
  completeness: number;
  consistency: number;
  validity: number;
  issues: string[];
  recommendations: string[];
}

export interface ProcessingStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  message?: string;
}

export interface FileUploadStatus {
  file: File;
  status: 'pending' | 'uploading' | 'completed' | 'failed' | 'retrying';
  progress: number;
  result?: any;
  error?: string;
  retryCount: number;
}

export interface BatchUploadSummary {
  totalFiles: number;
  completedFiles: number;
  failedFiles: number;
  processingFiles: number;
  overallProgress: number;
}

export interface LiveAnalysis {
  fileSize: number;
  fileType: string;
  estimatedRows: number;
  detectedColumns: number;
  processingTime: number;
  dataQualityScore: number;
}

// Export interfaces
export interface ExportOptions {
  format: 'pdf' | 'excel' | 'json' | 'csv';
  includeCharts: boolean;
  includeMetrics: boolean;
  includeConfidenceIntervals: boolean;
  includeScenarios: boolean;
  includeRecommendations: boolean;
  dateRange: {
    start: string;
    end: string;
  };
  customTitle?: string;
  customNotes?: string;
}

// Forecast interfaces
export interface ForecastData {
  point_forecast: Record<string, number>;
  confidence_intervals?: {
    p10: Record<string, number>;
    p50: Record<string, number>;
    p90: Record<string, number>;
  };
}

export interface ScenarioData {
  id: string;
  name: string;
  description: string;
  forecast: ForecastData;
  probability: number;
}

// Ensemble interfaces
export interface EnsembleMetrics {
  model_count: number;
  active_models: string[];
  performance_scores: Record<string, number>;
  last_training: string;
  status: 'training' | 'ready' | 'error' | 'idle';
  accuracy: number;
  confidence: number;
}

export interface ModelWeight {
  model_name: string;
  weight: number;
  performance: number;
  last_updated: string;
}

// Chart interfaces
export interface ChartDataPoint {
  date: string;
  value: number;
  category?: string;
  [key: string]: any;
}

export interface ChartConfig {
  type: '2d' | '3d';
  showConfidenceIntervals: boolean;
  showTrendLines: boolean;
  timeRange: string;
}

// Insight interfaces
export interface InsightData {
  id: string;
  title: string;
  description: string;
  type: 'trend' | 'anomaly' | 'recommendation' | 'alert';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  timestamp: string;
  data?: any;
}

export interface RecommendationData {
  id: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  impact: string;
  effort: string;
  category: string;
  actions: string[];
  timestamp: string;
}

// Theme interfaces
export interface CyberpunkTheme {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    error: string;
    warning: string;
    success: string;
    info: string;
    text: {
      primary: string;
      secondary: string;
      disabled: string;
    };
    neon: {
      blue: string;
      pink: string;
      green: string;
      purple: string;
    };
  };
  typography: {
    fontFamily: {
      primary: string;
      secondary: string;
      mono: string;
    };
    fontSize: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
      xxl: string;
    };
    fontWeight: {
      light: number;
      normal: number;
      medium: number;
      bold: number;
    };
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    xxl: string;
  };
  effects: {
    glow: string;
    shadow: string;
    blur: string;
  };
}

// Event handler types
export type ClickHandler = (event: React.MouseEvent<HTMLElement>) => void;
export type ChangeHandler<T = string> = (value: T) => void;
export type SubmitHandler = (event: React.FormEvent<HTMLFormElement>) => void;

// Utility types
export type UploadMode = 'single' | 'batch' | 'streaming';
export type LoaderVariant = 'spinner' | 'matrix' | 'pulse' | 'glitch' | 'hologram';
export type LoaderSize = 'sm' | 'md' | 'lg';
export type LoaderColor = 'blue' | 'pink' | 'green' | 'purple';

// Component prop types
export interface ComponentWithVariant {
  variant?: string;
}

export interface ComponentWithSize {
  size?: 'sm' | 'md' | 'lg';
}

export interface ComponentWithLoading {
  loading?: boolean;
}

export interface ComponentWithDisabled {
  disabled?: boolean;
}

// Re-export component-specific types
export * from './components';

// Re-export missing interfaces
export * from './missing';

// Re-export fixes for compilation issues
export * from './fixes';

// Re-export commonly used React types
export type { ReactNode, ReactElement, ComponentProps, FC } from 'react';