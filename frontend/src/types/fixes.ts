// Temporary fixes for remaining TypeScript compilation issues

// Extended EnsembleMetrics interface that includes all properties used across components
export interface ExtendedEnsembleMetrics {
  // Core properties
  model_count: number;
  active_models: string[];
  performance_scores: Record<string, number>;
  last_training: string;
  status: 'training' | 'ready' | 'error' | 'idle';
  accuracy: number;
  confidence: number;
  
  // Optional properties used by ModelMonitoringDashboard
  overall_accuracy?: number;
  model_performances?: Array<{
    model_name: string;
    weight: number;
    performance: number;
    status: 'healthy' | 'warning' | 'degraded' | 'failed';
    last_updated: string;
    trend?: 'stable' | 'improving' | 'declining';
    accuracy?: number;
    mape?: number;
    mae?: number;
  }>;
  confidence_score?: number;
  system_health?: number;
  last_updated?: string;
  total_predictions?: number;
  avg_response_time?: number;
  error_rate?: number;
  
  // Additional properties for compatibility
  models?: Array<{
    name: string;
    weight: number;
    performance: number;
    status: string;
    last_updated: string;
  }>;
  weight_evolution?: any[];
  prediction_reliability?: number;
}

// Recharts component type fixes - using interface augmentation instead
declare module 'recharts' {
  interface PolarAngleAxisProps {
    dataKey?: string;
    tick?: any;
    [key: string]: any;
  }
}

// Window interface extensions for test files
declare global {
  interface Window {
    testMetrics?: any[];
  }
  
  interface Performance {
    memory?: {
      usedJSHeapSize: number;
      jsHeapSizeLimit: number;
    };
  }
}

// Prop transformation utility types
export type LegacyToModernProps<T> = {
  [K in keyof T as K extends 'variant' ? '$variant' :
                   K extends 'size' ? '$size' :
                   K extends 'padding' ? '$padding' :
                   K extends 'loading' ? '$loading' :
                   K extends 'glitch' ? '$glitch' :
                   K extends 'hover' ? '$hover' : K]: T[K];
};

// Helper type for components that accept both legacy and modern props
export type CompatibleProps<T> = T & {
  variant?: string;
  size?: string;
  padding?: string;
  loading?: boolean;
  glitch?: boolean;
  hover?: boolean;
};

// Motion props helper type
export type MotionPropsHelper = {
  whileHover?: any;
  whileTap?: any;
  whileFocus?: any;
  initial?: any;
  animate?: any;
  exit?: any;
  transition?: any;
};

export default {};