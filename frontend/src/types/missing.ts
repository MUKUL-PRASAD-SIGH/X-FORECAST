// Missing TypeScript interfaces identified from compilation errors

// EnsembleMetrics interface for ModelMonitoringDashboard
export interface EnsembleMetrics {
  model_count: number;
  active_models: string[];
  performance_scores: Record<string, number>;
  last_training: string;
  status: 'training' | 'ready' | 'error' | 'idle';
  accuracy: number;
  confidence: number;
  total_predictions?: number;
  avg_response_time?: number;
  error_rate?: number;
  // Properties expected by ModelMonitoringDashboard
  overall_accuracy?: number;
  model_performances?: Array<{
    model_name: string;
    weight: number;
    performance: number;
    status: string;
    last_updated: string;
  }>;
  confidence_score?: number;
  system_health?: number;
  last_updated?: string;
  models?: Array<{
    name: string;
    weight: number;
    performance: number;
    status: string;
    last_updated: string;
  }>;
}

// Chart-related interfaces
export interface ChartDataPoint {
  date: string;
  value: number;
  category?: string;
  forecast?: number;
  actual?: number;
  confidence_lower?: number;
  confidence_upper?: number;
  [key: string]: any;
}

export interface ChartProps {
  data: ChartDataPoint[];
  width?: number | string;
  height?: number | string;
  loading?: boolean;
  error?: string;
  showLegend?: boolean;
  showTooltip?: boolean;
  responsive?: boolean;
}

// Customer Analytics interfaces
export interface CustomerSegment {
  segment_id: string;
  name: string;
  description: string;
  customer_count: number;
  revenue_contribution: number;
  growth_rate: number;
  churn_risk: 'low' | 'medium' | 'high';
  characteristics: string[];
  recommendations: string[];
}

export interface CustomerAnalyticsData {
  segments: CustomerSegment[];
  total_customers: number;
  total_revenue: number;
  churn_rate: number;
  acquisition_cost: number;
  lifetime_value: number;
  satisfaction_score: number;
}

// Insight and Recommendation interfaces
export interface InsightData {
  id: string;
  title: string;
  description: string;
  type: 'trend' | 'anomaly' | 'recommendation' | 'alert' | 'opportunity';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  impact: 'low' | 'medium' | 'high';
  category: string;
  timestamp: string;
  data?: any;
  actions?: string[];
  metrics?: Record<string, number>;
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
  expected_outcome?: string;
  success_metrics?: string[];
  timeline?: string;
  resources_required?: string[];
}

// Alert and Notification interfaces
export interface AlertData {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'warning' | 'error' | 'success';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  source: string;
  actions?: Array<{
    label: string;
    action: () => void;
    variant?: 'primary' | 'secondary' | 'danger';
  }>;
  dismissible?: boolean;
  autoClose?: boolean;
  duration?: number;
}

// Executive Summary interfaces
export interface ExecutiveSummaryData {
  period: string;
  key_metrics: {
    revenue: { value: number; change: number; trend: 'up' | 'down' | 'stable' };
    customers: { value: number; change: number; trend: 'up' | 'down' | 'stable' };
    satisfaction: { value: number; change: number; trend: 'up' | 'down' | 'stable' };
    efficiency: { value: number; change: number; trend: 'up' | 'down' | 'stable' };
  };
  highlights: string[];
  concerns: string[];
  recommendations: string[];
  forecast_summary: {
    next_quarter: string;
    confidence: number;
    key_drivers: string[];
  };
}

// Action Timeline interfaces
export interface ActionItem {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'blocked' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assigned_to?: string;
  due_date: string;
  created_date: string;
  completed_date?: string;
  category: string;
  tags: string[];
  dependencies?: string[];
  progress?: number;
  notes?: string[];
}

export interface TimelineData {
  actions: ActionItem[];
  milestones: Array<{
    id: string;
    title: string;
    date: string;
    description: string;
    status: 'upcoming' | 'current' | 'completed';
  }>;
  summary: {
    total_actions: number;
    completed_actions: number;
    overdue_actions: number;
    completion_rate: number;
  };
}

// Model Weight Adjustment interfaces
export interface ModelWeight {
  model_name: string;
  weight: number;
  performance: number;
  last_updated: string;
  status: 'active' | 'inactive' | 'training' | 'error';
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}

export interface WeightAdjustmentData {
  models: ModelWeight[];
  ensemble_performance: {
    overall_accuracy: number;
    confidence: number;
    last_evaluation: string;
  };
  adjustment_history: Array<{
    timestamp: string;
    changes: Array<{
      model: string;
      old_weight: number;
      new_weight: number;
      reason: string;
    }>;
    performance_impact: number;
  }>;
}

// Real-time Dashboard interfaces
export interface RealTimeMetrics {
  timestamp: string;
  active_users: number;
  system_load: number;
  response_time: number;
  error_rate: number;
  throughput: number;
  memory_usage: number;
  cpu_usage: number;
  network_io: number;
  disk_io: number;
}

export interface SystemHealth {
  overall_status: 'healthy' | 'warning' | 'critical' | 'maintenance';
  components: Array<{
    name: string;
    status: 'healthy' | 'warning' | 'critical' | 'offline';
    last_check: string;
    response_time?: number;
    error_count?: number;
  }>;
  alerts: AlertData[];
  uptime: number;
  last_incident?: {
    timestamp: string;
    description: string;
    duration: number;
    resolved: boolean;
  };
}

// Forecast Comparison interfaces
export interface ForecastComparison {
  scenarios: Array<{
    id: string;
    name: string;
    description: string;
    probability: number;
    forecast_data: ChartDataPoint[];
    key_assumptions: string[];
    risk_factors: string[];
    confidence_interval: {
      lower: number;
      upper: number;
    };
  }>;
  baseline: {
    name: string;
    forecast_data: ChartDataPoint[];
  };
  comparison_metrics: {
    variance: number;
    correlation: number;
    best_case_uplift: number;
    worst_case_downside: number;
  };
}

// 3D Visualization interfaces
export interface Visualization3DProps {
  data: Array<{
    x: number;
    y: number;
    z: number;
    value: number;
    label?: string;
    color?: string;
  }>;
  type: 'scatter' | 'surface' | 'mesh' | 'volume';
  width?: number;
  height?: number;
  interactive?: boolean;
  showAxes?: boolean;
  showGrid?: boolean;
  cameraPosition?: { x: number; y: number; z: number };
  lighting?: {
    ambient: number;
    directional: number;
    position: { x: number; y: number; z: number };
  };
}

// Performance Monitoring interfaces
export interface PerformanceMetrics {
  timestamp: string;
  response_time: number;
  throughput: number;
  error_rate: number;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: number;
  active_connections: number;
  queue_size: number;
}

export interface PerformanceThresholds {
  response_time: { warning: number; critical: number };
  throughput: { warning: number; critical: number };
  error_rate: { warning: number; critical: number };
  cpu_usage: { warning: number; critical: number };
  memory_usage: { warning: number; critical: number };
}

// Chat Interface interfaces
export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant' | 'system';
  timestamp: string;
  type?: 'text' | 'code' | 'chart' | 'file' | 'error';
  metadata?: {
    model?: string;
    confidence?: number;
    processing_time?: number;
    sources?: string[];
  };
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
  context?: {
    company_id: string;
    user_id: string;
    session_type: 'analysis' | 'forecasting' | 'general';
  };
}

// File Upload and Processing interfaces
export interface FileProcessingResult {
  success: boolean;
  message: string;
  file_id?: string;
  records_processed?: number;
  errors?: string[];
  warnings?: string[];
  processing_time?: number;
  data_quality_score?: number;
  detected_format?: string;
  column_mappings?: ColumnMapping[];
}

export interface UploadProgress {
  file_name: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  current_step: string;
  estimated_time_remaining?: number;
  bytes_uploaded?: number;
  total_bytes?: number;
}

// Import interfaces from components.ts to avoid duplication
import type {
  DetectedColumn,
  ColumnMapping,
  DataQuality,
  ProcessingStep,
  FileUploadStatus,
  BatchUploadSummary,
  LiveAnalysis,
} from './components';

// Export all interfaces
export type {
  DetectedColumn,
  ColumnMapping,
  DataQuality,
  ProcessingStep,
  FileUploadStatus,
  BatchUploadSummary,
  LiveAnalysis,
};