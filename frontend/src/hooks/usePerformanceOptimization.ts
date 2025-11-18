/**
 * Performance Optimization Hook
 * Integrates LOD, device detection, particle management, and shader optimization
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import LODManager from '../utils/performance/LODManager';
import DeviceCapabilityDetector from '../utils/performance/DeviceCapabilityDetector';
import ParticleSystemManager from '../utils/performance/ParticleSystemManager';
import WebGLShaderOptimizer from '../utils/performance/WebGLShaderOptimizer';

export interface PerformanceStats {
  fps: number;
  frameTime: number;
  memoryUsage: number;
  drawCalls: number;
  triangles: number;
  particles: number;
  lodLevel: number;
  performanceTier: 'low' | 'medium' | 'high';
}

export interface PerformanceSettings {
  enableLOD: boolean;
  enableParticleOptimization: boolean;
  enableShaderOptimization: boolean;
  maxParticles: number;
  targetFPS: number;
  adaptiveQuality: boolean;
}

export const usePerformanceOptimization = (
  scene?: THREE.Scene,
  camera?: THREE.Camera,
  renderer?: THREE.WebGLRenderer,
  initialSettings?: Partial<PerformanceSettings>
) => {
  const [performanceStats, setPerformanceStats] = useState<PerformanceStats>({
    fps: 60,
    frameTime: 16.67,
    memoryUsage: 0,
    drawCalls: 0,
    triangles: 0,
    particles: 0,
    lodLevel: 0,
    performanceTier: 'medium'
  });

  const [settings, setSettings] = useState<PerformanceSettings>({
    enableLOD: true,
    enableParticleOptimization: true,
    enableShaderOptimization: true,
    maxParticles: 1000,
    targetFPS: 60,
    adaptiveQuality: true,
    ...initialSettings
  });

  const [isInitialized, setIsInitialized] = useState(false);
  const [performanceTier, setPerformanceTier] = useState<'low' | 'medium' | 'high'>('medium');

  // Managers
  const lodManager = useRef<LODManager>(LODManager.getInstance());
  const deviceDetector = useRef<DeviceCapabilityDetector>(DeviceCapabilityDetector.getInstance());
  const particleManager = useRef<ParticleSystemManager>(ParticleSystemManager.getInstance());
  const shaderOptimizer = useRef<WebGLShaderOptimizer>(WebGLShaderOptimizer.getInstance());

  // Performance monitoring
  const frameTimeHistory = useRef<number[]>([]);
  const lastFrameTime = useRef<number>(performance.now());
  const frameCount = useRef<number>(0);
  const lastStatsUpdate = useRef<number>(performance.now());

  // Initialize performance optimization
  useEffect(() => {
    const initializePerformance = async () => {
      try {
        // Detect device capabilities
        const capabilities = await deviceDetector.current.detectCapabilities();
        const optimalSettings = deviceDetector.current.getOptimalSettings();
        
        const tier = capabilities.performance.tier;
        setPerformanceTier(tier);

        // Configure managers based on device capabilities
        if (scene) {
          particleManager.current.setScene(scene);
          particleManager.current.setPerformanceTier(tier);
        }

        if (camera) {
          lodManager.current.setCamera(camera);
        }

        shaderOptimizer.current.setPerformanceLevel(tier);

        // Create LOD configurations for different object types
        lodManager.current.createLODConfiguration('forecast', tier);
        lodManager.current.createLODConfiguration('particle', tier);
        lodManager.current.createLODConfiguration('holographic', tier);

        // Update settings based on device capabilities
        setSettings(prev => ({
          ...prev,
          maxParticles: optimalSettings.particleCount,
          targetFPS: optimalSettings.maxFPS,
          enableLOD: optimalSettings.lodLevels > 1,
          enableParticleOptimization: true,
          enableShaderOptimization: true
        }));

        setIsInitialized(true);
      } catch (error) {
        console.warn('Performance optimization initialization failed:', error);
        setIsInitialized(true); // Continue with default settings
      }
    };

    initializePerformance();
  }, [scene, camera]);

  // Performance monitoring loop
  useEffect(() => {
    if (!isInitialized) return;

    let animationFrameId: number;

    const updatePerformanceStats = () => {
      const currentTime = performance.now();
      const deltaTime = currentTime - lastFrameTime.current;
      
      // Update frame time history
      frameTimeHistory.current.push(deltaTime);
      if (frameTimeHistory.current.length > 60) {
        frameTimeHistory.current.shift();
      }

      // Update managers
      if (settings.enableLOD) {
        lodManager.current.updateLOD(deltaTime);
      }

      if (settings.enableParticleOptimization) {
        particleManager.current.update(deltaTime);
      }

      frameCount.current++;
      lastFrameTime.current = currentTime;

      // Update stats every second
      if (currentTime - lastStatsUpdate.current >= 1000) {
        const avgFrameTime = frameTimeHistory.current.reduce((a, b) => a + b, 0) / frameTimeHistory.current.length;
        const fps = 1000 / avgFrameTime;

        // Get memory usage if available
        const memoryInfo = (performance as any).memory;
        const memoryUsage = memoryInfo ? memoryInfo.usedJSHeapSize / 1024 / 1024 : 0;

        // Get LOD stats
        const lodStats = lodManager.current.getStats();
        
        // Get particle stats
        const particleStats = particleManager.current.getStats();

        // Get shader performance metrics
        const shaderMetrics = shaderOptimizer.current.getPerformanceMetrics();

        setPerformanceStats({
          fps: Math.round(fps),
          frameTime: Math.round(avgFrameTime * 100) / 100,
          memoryUsage: Math.round(memoryUsage),
          drawCalls: shaderMetrics.drawCalls,
          triangles: shaderMetrics.trianglesRendered,
          particles: particleStats.activeParticles,
          lodLevel: lodStats.averageLODLevel,
          performanceTier
        });

        // Adaptive quality adjustment
        if (settings.adaptiveQuality) {
          adjustQualityBasedOnPerformance(fps);
        }

        lastStatsUpdate.current = currentTime;
        
        // Reset shader metrics
        shaderOptimizer.current.resetMetrics();
      }

      animationFrameId = requestAnimationFrame(updatePerformanceStats);
    };

    updatePerformanceStats();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isInitialized, settings, performanceTier]);

  // Adaptive quality adjustment
  const adjustQualityBasedOnPerformance = useCallback((currentFPS: number) => {
    const targetFPS = settings.targetFPS;
    const fpsRatio = currentFPS / targetFPS;

    if (fpsRatio < 0.8) {
      // Performance is poor, reduce quality
      if (performanceTier === 'high') {
        setPerformanceTier('medium');
        shaderOptimizer.current.setPerformanceLevel('medium');
        particleManager.current.setPerformanceTier('medium');
      } else if (performanceTier === 'medium') {
        setPerformanceTier('low');
        shaderOptimizer.current.setPerformanceLevel('low');
        particleManager.current.setPerformanceTier('low');
      }
    } else if (fpsRatio > 1.2 && currentFPS > targetFPS) {
      // Performance is good, can increase quality
      if (performanceTier === 'low') {
        setPerformanceTier('medium');
        shaderOptimizer.current.setPerformanceLevel('medium');
        particleManager.current.setPerformanceTier('medium');
      } else if (performanceTier === 'medium') {
        setPerformanceTier('high');
        shaderOptimizer.current.setPerformanceLevel('high');
        particleManager.current.setPerformanceTier('high');
      }
    }
  }, [settings.targetFPS, performanceTier]);

  // Create optimized LOD object
  const createLODObject = useCallback((
    objectId: string,
    geometries: { high: THREE.BufferGeometry; medium: THREE.BufferGeometry; low: THREE.BufferGeometry },
    materials: { high: THREE.Material; medium: THREE.Material; low: THREE.Material },
    objectType: string = 'default'
  ) => {
    if (!settings.enableLOD) {
      return new THREE.Mesh(geometries.high, materials.high);
    }
    
    return lodManager.current.createLODObject(objectId, geometries, materials, objectType);
  }, [settings.enableLOD]);

  // Create optimized particle system
  const createParticleSystem = useCallback((
    emitterId: string,
    position: THREE.Vector3,
    config: any
  ) => {
    if (!settings.enableParticleOptimization) {
      return null;
    }

    // Adjust particle count based on performance
    const adjustedConfig = {
      ...config,
      maxParticles: Math.min(config.maxParticles, settings.maxParticles)
    };

    return particleManager.current.createEmitter(emitterId, position, adjustedConfig);
  }, [settings.enableParticleOptimization, settings.maxParticles]);

  // Create optimized shader material
  const createOptimizedShader = useCallback((
    shaderType: 'holographic' | 'particle' | 'glow' | 'scan'
  ) => {
    if (!settings.enableShaderOptimization) {
      return null;
    }

    const shaderConfig = shaderOptimizer.current.getOptimizedCyberpunkShader(shaderType);
    return shaderOptimizer.current.createOptimizedMaterial(shaderConfig);
  }, [settings.enableShaderOptimization]);

  // Update settings
  const updateSettings = useCallback((newSettings: Partial<PerformanceSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  // Get current LOD level for an object
  const getCurrentLODLevel = useCallback((objectId: string) => {
    return lodManager.current.getCurrentLODLevel(objectId);
  }, []);

  // Check if frame should be skipped for an object
  const shouldSkipFrame = useCallback((objectId: string) => {
    return lodManager.current.shouldSkipFrame(objectId);
  }, []);

  // Get optimized particle count
  const getOptimizedParticleCount = useCallback((objectId: string, baseCount: number) => {
    return lodManager.current.getParticleCount(objectId, baseCount);
  }, []);

  // Check if effects should be enabled
  const shouldEnableEffects = useCallback((objectId: string) => {
    return lodManager.current.shouldEnableEffects(objectId);
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      lodManager.current.cleanup();
      particleManager.current.cleanup();
      shaderOptimizer.current.cleanup();
    };
  }, []);

  return {
    // State
    performanceStats,
    settings,
    performanceTier,
    isInitialized,

    // Functions
    createLODObject,
    createParticleSystem,
    createOptimizedShader,
    updateSettings,
    getCurrentLODLevel,
    shouldSkipFrame,
    getOptimizedParticleCount,
    shouldEnableEffects,

    // Managers (for advanced usage)
    lodManager: lodManager.current,
    particleManager: particleManager.current,
    shaderOptimizer: shaderOptimizer.current
  };
};

export default usePerformanceOptimization;