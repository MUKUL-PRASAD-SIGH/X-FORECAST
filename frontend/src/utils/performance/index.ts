/**
 * Performance Optimization Utilities Index
 * Exports all performance optimization classes and utilities
 */

export { default as LODManager } from './LODManager';
export type { LODLevel, LODConfiguration, GeometryLOD } from './LODManager';

export { default as DeviceCapabilityDetector } from './DeviceCapabilityDetector';
export type { DeviceCapabilities, PerformanceSettings } from './DeviceCapabilityDetector';

export { default as ParticleSystemManager } from './ParticleSystemManager';
export type { 
  ParticleSystemConfig, 
  ParticleEmitter, 
  ParticlePool 
} from './ParticleSystemManager';

export { default as WebGLShaderOptimizer } from './WebGLShaderOptimizer';
export type { 
  ShaderPerformanceMetrics, 
  OptimizedShaderConfig 
} from './WebGLShaderOptimizer';

export { default as usePerformanceOptimization } from '../../hooks/usePerformanceOptimization';
export type { PerformanceStats } from '../../hooks/usePerformanceOptimization';

export { default as PerformanceMonitor } from '../../components/performance/PerformanceMonitor';