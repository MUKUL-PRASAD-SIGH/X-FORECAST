/**
 * Performance Optimization Tests
 * Tests for 3D rendering performance optimizations
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import * as THREE from 'three';
import LODManager from '../../../utils/performance/LODManager';
import DeviceCapabilityDetector from '../../../utils/performance/DeviceCapabilityDetector';
import ParticleSystemManager from '../../../utils/performance/ParticleSystemManager';
import WebGLShaderOptimizer from '../../../utils/performance/WebGLShaderOptimizer';
import { usePerformanceOptimization } from '../../../hooks/usePerformanceOptimization';
import PerformanceMonitor from '../../performance/PerformanceMonitor';

// Mock Three.js Canvas
jest.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="three-canvas">{children}</div>,
  useFrame: (callback: any) => {
    // Mock useFrame hook
    React.useEffect(() => {
      const interval = setInterval(() => {
        callback({ clock: { elapsedTime: Date.now() / 1000 } });
      }, 16);
      return () => clearInterval(interval);
    }, [callback]);
  },
  useThree: () => ({
    camera: new THREE.PerspectiveCamera(),
    scene: new THREE.Scene(),
    gl: {
      getParameter: jest.fn().mockReturnValue(4096),
      getSupportedExtensions: jest.fn().mockReturnValue(['WEBGL_debug_renderer_info'])
    }
  })
}));

// Mock performance API
Object.defineProperty(window, 'performance', {
  value: {
    now: jest.fn(() => Date.now()),
    memory: {
      usedJSHeapSize: 50 * 1024 * 1024 // 50MB
    }
  }
});

// Mock navigator
Object.defineProperty(navigator, 'deviceMemory', {
  value: 8,
  writable: true
});

Object.defineProperty(navigator, 'hardwareConcurrency', {
  value: 8,
  writable: true
});

describe('Performance Optimization System', () => {
  beforeEach(() => {
    // Reset singletons
    jest.clearAllMocks();
  });

  describe('LODManager', () => {
    it('should create LOD configurations for different performance tiers', () => {
      const lodManager = LODManager.getInstance();
      
      const lowConfig = lodManager.createLODConfiguration('test', 'low');
      const mediumConfig = lodManager.createLODConfiguration('test', 'medium');
      const highConfig = lodManager.createLODConfiguration('test', 'high');
      
      expect(lowConfig.levels).toHaveLength(2);
      expect(mediumConfig.levels).toHaveLength(3);
      expect(highConfig.levels).toHaveLength(4);
      
      // Low tier should have fewer particles
      expect(lowConfig.levels[0].particleCount).toBeLessThan(mediumConfig.levels[0].particleCount);
      expect(mediumConfig.levels[0].particleCount).toBeLessThan(highConfig.levels[0].particleCount);
    });

    it('should create geometry LOD with different complexity levels', () => {
      const lodManager = LODManager.getInstance();
      const baseGeometry = new THREE.BoxGeometry(1, 1, 1);
      
      const geometryLOD = lodManager.createGeometryLOD(baseGeometry);
      
      expect(geometryLOD.high).toBeDefined();
      expect(geometryLOD.medium).toBeDefined();
      expect(geometryLOD.low).toBeDefined();
      
      // Low detail should have fewer vertices than high detail
      const highVertexCount = geometryLOD.high.getAttribute('position').count;
      const lowVertexCount = geometryLOD.low.getAttribute('position').count;
      expect(lowVertexCount).toBeLessThanOrEqual(highVertexCount);
    });

    it('should provide frame skipping based on LOD settings', () => {
      const lodManager = LODManager.getInstance();
      lodManager.createLODConfiguration('test-object', 'low');
      
      // Mock camera
      const camera = new THREE.PerspectiveCamera();
      lodManager.setCamera(camera);
      
      const shouldSkip = lodManager.shouldSkipFrame('test-object');
      expect(typeof shouldSkip).toBe('boolean');
    });
  });

  describe('DeviceCapabilityDetector', () => {
    it('should detect device capabilities', async () => {
      const detector = DeviceCapabilityDetector.getInstance();
      
      // Mock WebGL context
      const mockCanvas = document.createElement('canvas');
      const mockGL = {
        getParameter: jest.fn().mockImplementation((param) => {
          switch (param) {
            case 'VENDOR': return 'Mock Vendor';
            case 'RENDERER': return 'Mock Renderer';
            case 'MAX_TEXTURE_SIZE': return 4096;
            default: return 1024;
          }
        }),
        getSupportedExtensions: jest.fn().mockReturnValue(['WEBGL_debug_renderer_info']),
        createShader: jest.fn().mockReturnValue({}),
        shaderSource: jest.fn(),
        compileShader: jest.fn(),
        getShaderParameter: jest.fn().mockReturnValue(true),
        createProgram: jest.fn().mockReturnValue({}),
        attachShader: jest.fn(),
        linkProgram: jest.fn(),
        getProgramParameter: jest.fn().mockReturnValue(true),
        useProgram: jest.fn(),
        getUniformLocation: jest.fn().mockReturnValue({}),
        uniform1f: jest.fn(),
        drawArrays: jest.fn()
      };
      
      jest.spyOn(mockCanvas, 'getContext').mockReturnValue(mockGL as any);
      jest.spyOn(document, 'createElement').mockReturnValue(mockCanvas);
      
      const capabilities = await detector.detectCapabilities();
      
      expect(capabilities).toBeDefined();
      expect(capabilities.gpu).toBeDefined();
      expect(capabilities.performance).toBeDefined();
      expect(capabilities.display).toBeDefined();
      expect(capabilities.webgl).toBeDefined();
      
      expect(['low', 'medium', 'high']).toContain(capabilities.performance.tier);
    });

    it('should provide optimal settings based on capabilities', async () => {
      const detector = DeviceCapabilityDetector.getInstance();
      
      // Mock high-performance device
      const mockCanvas = document.createElement('canvas');
      const mockGL = {
        getParameter: jest.fn().mockReturnValue(4096),
        getSupportedExtensions: jest.fn().mockReturnValue(['WEBGL_debug_renderer_info']),
        createShader: jest.fn().mockReturnValue({}),
        shaderSource: jest.fn(),
        compileShader: jest.fn(),
        getShaderParameter: jest.fn().mockReturnValue(true),
        createProgram: jest.fn().mockReturnValue({}),
        attachShader: jest.fn(),
        linkProgram: jest.fn(),
        getProgramParameter: jest.fn().mockReturnValue(true),
        useProgram: jest.fn(),
        getUniformLocation: jest.fn().mockReturnValue({}),
        uniform1f: jest.fn(),
        drawArrays: jest.fn()
      };
      
      jest.spyOn(mockCanvas, 'getContext').mockReturnValue(mockGL as any);
      jest.spyOn(document, 'createElement').mockReturnValue(mockCanvas);
      
      await detector.detectCapabilities();
      const settings = detector.getOptimalSettings();
      
      expect(settings).toBeDefined();
      expect(settings.particleCount).toBeGreaterThan(0);
      expect(settings.lodLevels).toBeGreaterThan(0);
      expect(['off', 'low', 'medium', 'high']).toContain(settings.shadowQuality);
      expect(typeof settings.antialiasing).toBe('boolean');
      expect(settings.maxFPS).toBeGreaterThan(0);
    });
  });

  describe('ParticleSystemManager', () => {
    it('should create particle pools with performance-based limits', () => {
      const particleManager = ParticleSystemManager.getInstance();
      const scene = new THREE.Scene();
      particleManager.setScene(scene);
      
      const config = {
        maxParticles: 1000,
        particleSize: 0.1,
        lifetime: 2000,
        emissionRate: 50,
        velocity: new THREE.Vector3(0, 1, 0),
        acceleration: new THREE.Vector3(0, -9.8, 0),
        color: new THREE.Color(0x00ffff),
        opacity: 0.8,
        blending: THREE.AdditiveBlending
      };
      
      particleManager.setPerformanceTier('low');
      const lowPool = particleManager.createParticlePool('test-low', config);
      
      particleManager.setPerformanceTier('high');
      const highPool = particleManager.createParticlePool('test-high', config);
      
      expect(lowPool.particles.length).toBeLessThan(highPool.particles.length);
    });

    it('should create and manage particle emitters', () => {
      const particleManager = ParticleSystemManager.getInstance();
      const scene = new THREE.Scene();
      particleManager.setScene(scene);
      
      const position = new THREE.Vector3(0, 0, 0);
      const config = {
        maxParticles: 100,
        particleSize: 0.1,
        lifetime: 1000,
        emissionRate: 10,
        velocity: new THREE.Vector3(0, 1, 0),
        acceleration: new THREE.Vector3(0, -9.8, 0),
        color: new THREE.Color(0x00ffff),
        opacity: 0.8,
        blending: THREE.AdditiveBlending
      };
      
      const emitter = particleManager.createEmitter('test-emitter', position, config);
      
      expect(emitter).toBeDefined();
      expect(emitter.id).toBe('test-emitter');
      expect(emitter.active).toBe(true);
      expect(emitter.position).toEqual(position);
    });

    it('should provide performance statistics', () => {
      const particleManager = ParticleSystemManager.getInstance();
      const stats = particleManager.getStats();
      
      expect(stats).toBeDefined();
      expect(typeof stats.totalEmitters).toBe('number');
      expect(typeof stats.totalParticles).toBe('number');
      expect(typeof stats.activeParticles).toBe('number');
      expect(['low', 'medium', 'high']).toContain(stats.performanceTier);
    });
  });

  describe('WebGLShaderOptimizer', () => {
    it('should create optimized shaders for different performance levels', () => {
      const shaderOptimizer = WebGLShaderOptimizer.getInstance();
      
      shaderOptimizer.setPerformanceLevel('low');
      const lowShader = shaderOptimizer.getOptimizedCyberpunkShader('holographic');
      
      shaderOptimizer.setPerformanceLevel('high');
      const highShader = shaderOptimizer.getOptimizedCyberpunkShader('holographic');
      
      expect(lowShader.performanceLevel).toBe('low');
      expect(highShader.performanceLevel).toBe('high');
      
      // High quality should have more defines/features
      expect(Object.keys(highShader.defines).length).toBeGreaterThanOrEqual(
        Object.keys(lowShader.defines).length
      );
    });

    it('should create different shader types', () => {
      const shaderOptimizer = WebGLShaderOptimizer.getInstance();
      
      const holographicShader = shaderOptimizer.getOptimizedCyberpunkShader('holographic');
      const particleShader = shaderOptimizer.getOptimizedCyberpunkShader('particle');
      const glowShader = shaderOptimizer.getOptimizedCyberpunkShader('glow');
      const scanShader = shaderOptimizer.getOptimizedCyberpunkShader('scan');
      
      expect(holographicShader.vertexShader).toBeDefined();
      expect(particleShader.vertexShader).toBeDefined();
      expect(glowShader.vertexShader).toBeDefined();
      expect(scanShader.vertexShader).toBeDefined();
      
      // Each shader should have different content
      expect(holographicShader.vertexShader).not.toBe(particleShader.vertexShader);
    });

    it('should track performance metrics', () => {
      const shaderOptimizer = WebGLShaderOptimizer.getInstance();
      
      shaderOptimizer.updateMetrics(10, 1000);
      const metrics = shaderOptimizer.getPerformanceMetrics();
      
      expect(metrics.drawCalls).toBe(10);
      expect(metrics.trianglesRendered).toBe(1000);
      
      shaderOptimizer.resetMetrics();
      const resetMetrics = shaderOptimizer.getPerformanceMetrics();
      expect(resetMetrics.drawCalls).toBe(0);
      expect(resetMetrics.trianglesRendered).toBe(0);
    });
  });

  describe('PerformanceMonitor Component', () => {
    it('should render performance statistics', async () => {
      render(<PerformanceMonitor />);
      
      await waitFor(() => {
        expect(screen.getByText('Performance Monitor')).toBeInTheDocument();
      });
    });

    it('should render in minimized mode', async () => {
      render(<PerformanceMonitor minimized={true} />);
      
      await waitFor(() => {
        expect(screen.getByText('EXPAND')).toBeInTheDocument();
      });
    });

    it('should show controls when enabled', async () => {
      render(<PerformanceMonitor showControls={true} />);
      
      await waitFor(() => {
        expect(screen.getByText('CTRL')).toBeInTheDocument();
      });
    });
  });

  describe('Integration Tests', () => {
    it('should integrate all performance systems', async () => {
      const TestComponent = () => {
        const {
          performanceStats,
          settings,
          performanceTier,
          isInitialized,
          createLODObject,
          createParticleSystem,
          createOptimizedShader
        } = usePerformanceOptimization();

        if (!isInitialized) {
          return <div>Loading...</div>;
        }

        return (
          <div>
            <div data-testid="performance-tier">{performanceTier}</div>
            <div data-testid="fps">{performanceStats.fps}</div>
            <div data-testid="particles">{performanceStats.particles}</div>
            <div data-testid="lod-enabled">{settings.enableLOD.toString()}</div>
          </div>
        );
      };

      render(<TestComponent />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      expect(screen.getByTestId('performance-tier')).toBeInTheDocument();
      expect(screen.getByTestId('fps')).toBeInTheDocument();
      expect(screen.getByTestId('particles')).toBeInTheDocument();
      expect(screen.getByTestId('lod-enabled')).toBeInTheDocument();
    });
  });
});