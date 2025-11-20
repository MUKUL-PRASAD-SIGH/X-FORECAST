// Component-specific TypeScript interfaces

import { ReactNode, ButtonHTMLAttributes, InputHTMLAttributes } from 'react';
import { MotionProps } from 'framer-motion';

// Base component interfaces
export interface BaseComponentProps {
  className?: string;
  children?: ReactNode;
}

// CyberpunkButton interfaces
export interface CyberpunkButtonStyledProps {
  $variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  $size?: 'sm' | 'md' | 'lg';
  $loading?: boolean;
  $glitch?: boolean;
}

export interface CyberpunkButtonFunctionalProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'size'> {
  children: ReactNode;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
  className?: string;
}

export interface CyberpunkButtonMotionProps {
  whileHover?: any;
  whileTap?: any;
  whileFocus?: any;
  initial?: any;
  animate?: any;
  exit?: any;
  transition?: any;
}

export interface CyberpunkButtonProps extends CyberpunkButtonFunctionalProps, CyberpunkButtonStyledProps, CyberpunkButtonMotionProps {}

// CyberpunkCard interfaces
export interface CyberpunkCardStyledProps {
  $variant?: 'default' | 'glass' | 'neon' | 'hologram';
  $padding?: 'sm' | 'md' | 'lg';
  $glitch?: boolean;
  $hover?: boolean;
}

export interface CyberpunkCardFunctionalProps extends Omit<MotionProps, 'variant' | 'padding'> {
  children: ReactNode;
  className?: string;
  as?: keyof JSX.IntrinsicElements | React.ComponentType<any>;
}

export interface CyberpunkCardProps extends CyberpunkCardFunctionalProps, CyberpunkCardStyledProps {}

// CyberpunkInput interfaces
export interface CyberpunkInputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'search';
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  error?: string;
  label?: string;
  icon?: ReactNode;
  glitch?: boolean;
  className?: string;
  variant?: string;
}

// CyberpunkLoader interfaces
export interface CyberpunkLoaderProps {
  variant?: 'spinner' | 'matrix' | 'pulse' | 'glitch' | 'hologram';
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'pink' | 'green' | 'purple';
  text?: string;
  className?: string;
}

// CyberpunkNavigation interfaces
export interface NavigationItem {
  id: string;
  label: string;
  icon?: ReactNode;
  active?: boolean;
  onClick?: () => void;
  badge?: string | number;
}

export interface CyberpunkNavigationProps {
  items: NavigationItem[];
  orientation?: 'horizontal' | 'vertical';
  variant?: 'primary' | 'minimal' | 'floating';
  className?: string;
}

// DataUpload interfaces
export interface DataUploadProps {
  authToken?: string;
  onUploadComplete: (result: any) => void;
  onAuthError?: () => void;
}

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

// ForecastExport interfaces
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

export interface ForecastExportProps {
  forecastData?: any;
  scenarioData?: any[];
  onExportComplete?: (result: { success: boolean; downloadUrl?: string; error?: string }) => void;
}

// Dashboard interfaces
export interface DashboardMetrics {
  totalCustomers: number;
  retentionRate: number;
  forecastAccuracy: number;
  systemHealth: number;
  activeAlerts: number;
  revenueGrowth: number;
}

export interface SystemStatus {
  status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
  uptime: string;
  lastUpdate: string;
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

// Utility types
export type UploadMode = 'single' | 'batch' | 'streaming';
export type LoaderVariant = 'spinner' | 'matrix' | 'pulse' | 'glitch' | 'hologram';
export type LoaderSize = 'sm' | 'md' | 'lg';
export type LoaderColor = 'blue' | 'pink' | 'green' | 'purple';

// Event handler types
export type ClickHandler = (event: React.MouseEvent<HTMLElement>) => void;
export type ChangeHandler<T = string> = (value: T) => void;
export type SubmitHandler = (event: React.FormEvent<HTMLFormElement>) => void;

// Legacy prop compatibility types
export interface LegacyButtonProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  glitch?: boolean;
}

export interface LegacyCardProps {
  variant?: 'default' | 'glass' | 'neon' | 'hologram';
  padding?: 'sm' | 'md' | 'lg';
  hover?: boolean;
  glitch?: boolean;
}

// Component factory types
export interface ComponentFactoryProps<T = {}> {
  baseComponent: string | React.ComponentType<any>;
  styledProps: T;
  functionalProps: Record<string, any>;
}

// Theme-aware component props
export interface ThemeAwareProps {
  theme?: any; // Will be injected by styled-components
}

// Animation props
export interface AnimationProps {
  initial?: any;
  animate?: any;
  exit?: any;
  transition?: any;
  whileHover?: any;
  whileTap?: any;
  whileFocus?: any;
}

// Responsive props
export interface ResponsiveProps {
  mobile?: any;
  tablet?: any;
  desktop?: any;
}

// Accessibility props
export interface AccessibilityProps {
  'aria-label'?: string;
  'aria-describedby'?: string;
  'aria-expanded'?: boolean;
  'aria-hidden'?: boolean;
  role?: string;
  tabIndex?: number;
}

// Form component interfaces
export interface FormFieldProps extends BaseComponentProps {
  label?: string;
  error?: string;
  required?: boolean;
  disabled?: boolean;
  helperText?: string;
}

// Modal/Dialog interfaces
export interface ModalProps extends BaseComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
}

// Notification interfaces
export interface NotificationProps {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  actions?: Array<{
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary';
  }>;
}

// Loading state interfaces
export interface LoadingStateProps {
  loading: boolean;
  error?: string | null;
  retry?: () => void;
}

// Pagination interfaces
export interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  showFirstLast?: boolean;
  showPrevNext?: boolean;
  maxVisiblePages?: number;
}

// Search interfaces
export interface SearchProps {
  value: string;
  onChange: (value: string) => void;
  onSearch?: (value: string) => void;
  placeholder?: string;
  suggestions?: string[];
  loading?: boolean;
}

// Filter interfaces
export interface FilterOption {
  label: string;
  value: string | number;
  count?: number;
}

export interface FilterProps {
  options: FilterOption[];
  selected: (string | number)[];
  onChange: (selected: (string | number)[]) => void;
  multiple?: boolean;
  searchable?: boolean;
}

// Table interfaces
export interface TableColumn<T = any> {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (value: any, row: T) => ReactNode;
  width?: string | number;
}

export interface TableProps<T = any> {
  columns: TableColumn<T>[];
  data: T[];
  loading?: boolean;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  onSort?: (key: string, direction: 'asc' | 'desc') => void;
  onRowClick?: (row: T) => void;
  selectedRows?: string[];
  onSelectionChange?: (selected: string[]) => void;
}

// Chart component interfaces
export interface BaseChartProps {
  data: ChartDataPoint[];
  width?: number | string;
  height?: number | string;
  loading?: boolean;
  error?: string;
  config?: Partial<ChartConfig>;
}

export interface LineChartProps extends BaseChartProps {
  showDots?: boolean;
  strokeWidth?: number;
  smooth?: boolean;
}

export interface BarChartProps extends BaseChartProps {
  orientation?: 'horizontal' | 'vertical';
  stacked?: boolean;
}

export interface PieChartProps extends BaseChartProps {
  showLabels?: boolean;
  showLegend?: boolean;
  innerRadius?: number;
}

// Export all interfaces
export type {
  // Re-export from React
  ReactNode,
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  // Re-export from framer-motion
  MotionProps,
};