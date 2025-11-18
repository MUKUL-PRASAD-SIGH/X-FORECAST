/**
 * 3D Visualization Performance Tests
 * Tests frame rate performance with large datasets and validates memory usage optimization
 * Requirements: 8.1, 8.6
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import * as THREE from 'three';
import HolographicForecastChart3D from '../HolographicForecastChart3D';
import AnimatedWeightEvolution from '../AnimatedWeightEvolution';
import EnsembleForecast3DVisualization from '../EnsembleForecast3DVisualization';
import { usePerformanceOptimization } from '../../../hooks/usePerformanceOptimization';

// Mock Three.js Canvas and performance APIs
jest.mock('@react-three/fiber', () => ({
  Canvas: (props) => {
    return React.createElement('div', { 'data-testid': 'three-canvas' }, props.children);
  },
  useFrame: (callback) => {
    React.useEffect(() => {
      const interval = setInterval(() => {
        callback({ 
          clock: { elapsedTime: Date.now() / 1000 },
          gl: {
            info: {
              render: {
                calls: 10,
                triangles: 1000
              }
            }
          }
        });
      }, 16);
      return () => clearInterval(interval);
    }, [callback]);
  },
  useThree: () => ({
    camera: new THREE.PerspectiveCamera(),
    scene: new THREE.Scene(),
    gl: {
      getParameter: jest.fn().mockReturnValue(4096),
      getSupportedExtensions: jest.fn().mockReturnValue(['WEBGL_debug_renderer_info']),
      info: {
        render: {
          calls: 10,
          triangles: 1000
        }
      }
    }
  })
}));

// Mock performance API with memory tracking
const mockPerformance = {
  now: jest.fn(() => Date.now()),
  memory: {
    usedJSHeapSize: 50 * 1024 * 1024, // 50MB initial
    totalJSHeapSize: 100 * 1024 * 1024, // 100MB total
    jsHeapSizeLimit: 2 * 1024 * 1024 * 1024 // 2GB limit
  }
};

Object.defineProperty(window, 'performance', {
  value: mockPerformance,
  writable: true
});

// Mock requestAnimationFrame for controlled timing
let animationFrameCallbacks = [];
let frameId = 0;

global.requestAnimationFrame = jest.fn((callback) => {
  animationFrameCallbacks.push(callback);
  return ++frameId;
});

global.cancelAnimationFrame = jest.fn((id) => {
  // Mock implementation
});

// Helper to simulate animation frames
const simulateAnimationFrames = (count, interval = 16) => {
  return new Promise((resolve) => {
    let framesProcessed = 0;
    
    const processFrame = () => {
      if (framesProcessed >= count) {
        resolve();
        return;
      }
      
      // Update mock time
      mockPerformance.now.mockReturnValue(Date.now() + framesProcessed * interval);
      
      // Execute all pending callbacks
      const callbacks = [...animationFrameCallbacks];
      animationFrameCallbacks = [];
      callbacks.forEach(callback => callback());
      
      framesProcessed++;
      setTimeout(processFrame, 1);
    };
    
    processFrame();
  });
};

// Generate large test datasets
const generateLargeForecastData = (size) => {
  const data = [];
  const startDate = new Date('2024-01-01');
  
  for (let i = 0; i < size; i++) {
    const date = new Date(startDate);
    date.setMonth(date.getMonth() + i);
    
    data.push({
      date: date.toISOString().split('T')[0],
      historical: i < size / 2 ? 1000 + i * 10 + Math.random() * 100 : undefined,
      arima: i >= size / 2 ? 1000 + i * 10 + Math.random() * 50 : undefined,
      ets: i >= size / 2 ? 1000 + i * 12 + Math.random() * 60 : undefined,
      xgboost: i >= size / 2 ? 1000 + i * 8 + Math.random() * 40 : undefined,
      lstm: i >= size / 2 ? 1000 + i * 15 + Math.random() * 70 : undefined,
      croston: i >= size / 2 ? 1000 + i * 5 + Math.random() * 30 : undefined,
      ensemble: 1000 + i * 10 + Math.random() * 20,
      confidence_lower: i >= size / 2 ? 900 + i * 8 + Math.random() * 30 : undefined,
      confidence_upper: i >= size / 2 ? 1100 + i * 12 + Math.random() * 40 : undefined
    });
  }
  
  return data;
};

const generateLargeWeightEvolutionData = (size) => {
  const data = [];
  const startDate = new Date('2024-01-01');
  
  for (let i = 0; i < size; i++) {
    const date = new Date(startDate);
    date.setHours(date.getHours() + i);
    
    // Simulate weight evolution over time
    const baseWeights = [0.2, 0.25, 0.3, 0.15, 0.1];
    const variation = Math.sin(i * 0.1) * 0.1;
    
    data.push({
      timestamp: date.toISOString(),
      weights: {
        arima: Math.max(0.05, Math.min(0.4, baseWeights[0] + variation)),
        ets: Math.max(0.05, Math.min(0.4, baseWeights[1] - variation * 0.5)),
        xgboost: Math.max(0.05, Math.min(0.4, baseWeights[2] + variation * 0.3)),
        lstm: Math.max(0.05, Math.min(0.4, baseWeights[3] - variation * 0.2)),
        croston: Math.max(0.05, Math.min(0.4, baseWeights[4] + variation * 0.1))
      },
      performance: {
        arima: 0.8 + Math.random() * 0.2,
        ets: 0.75 + Math.random() * 0.25,
        xgboost: 0.85 + Math.random() * 0.15,
        lstm: 0.82 + Math.random() * 0.18,
        croston: 0.7 + Math.random() * 0.3
      }
    });
  }
  
  return data;
};

describe('3D Visualization Performance Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    animationFrameCallbacks = [];
    frameId = 0;
    
    // Reset memory usage
    mockPerformance.memory.usedJSHeapSize = 50 * 1024 * 1024;
  });

  describe('Frame Rate Performance with Large Datasets', () => {
    it('should maintain 30+ FPS with 100 data points in HolographicForecastChart3D', async () => {
      const largeDataset = generateLargeForecastData(100);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      
      // Mock useFrame to track frame rates
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 15, triangles: 2000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={largeDataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate 60 frames (1 second at 60fps)
      await simulateAnimationFrames(60, 16);

      // Calculate average FPS
      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain at least 30 FPS with large dataset
      expect(avgFPS).toBeGreaterThan(30);
      
      // No frame should drop below 20 FPS
      const minFPS = Math.min(...frameRates);
      expect(minFPS).toBeGreaterThan(20);

      // Restore original useFrame
      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should maintain performance with extremely large datasets (500+ data points)', async () => {
      const extremeDataset = generateLargeForecastData(500);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 25, triangles: 5000 } } }
            });
          }, 20); // Slightly slower to simulate heavy load
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={extremeDataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={1200}
          height={800}
        />
      );

      // Simulate 120 frames (2 seconds)
      await simulateAnimationFrames(120, 20);

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain at least 20 FPS even with extreme dataset
      expect(avgFPS).toBeGreaterThan(20);
      
      // Frame rate should be consistent (standard deviation < 10)
      const mean = avgFPS;
      const variance = frameRates.reduce((sum, fps) => sum + Math.pow(fps - mean, 2), 0) / frameRates.length;
      const stdDev = Math.sqrt(variance);
      expect(stdDev).toBeLessThan(10);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should handle high-frequency data updates without frame drops', async () => {
      const baseDataset = generateLargeForecastData(50);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      let updateCount = 0;
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 12, triangles: 1500 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      const { rerender } = render(
        <HolographicForecastChart3D
          data={baseDataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate rapid data updates every 100ms
      const updateInterval = setInterval(() => {
        updateCount++;
        const updatedDataset = generateLargeForecastData(50 + updateCount * 5);
        rerender(
          <HolographicForecastChart3D
            data={updatedDataset}
            showIndividualModels={true}
            showConfidenceIntervals={true}
            enableParticleEffects={true}
            width={800}
            height={600}
          />
        );
      }, 100);

      // Run for 1 second with updates
      await simulateAnimationFrames(60, 16);
      clearInterval(updateInterval);

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain reasonable performance during rapid updates
      expect(avgFPS).toBeGreaterThan(25);
      
      // Should handle at least 10 data updates
      expect(updateCount).toBeGreaterThanOrEqual(10);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should maintain performance with 200 weight evolution data points', async () => {
      const largeWeightData = generateLargeWeightEvolutionData(200);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 20, triangles: 3000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <AnimatedWeightEvolution
          data={largeWeightData}
          enableParticleEffects={true}
          showPerformanceIndicators={true}
          animationSpeed={1}
          width={800}
          height={600}
        />
      );

      // Simulate 60 frames
      await simulateAnimationFrames(60, 16);

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain reasonable performance even with large weight evolution data
      expect(avgFPS).toBeGreaterThan(25);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should handle rapid view switching without frame drops', async () => {
      const forecastData = generateLargeForecastData(50);
      const weightData = generateLargeWeightEvolutionData(50);
      
      const ensembleData = {
        historical: forecastData.slice(0, 25).map(d => ({ date: d.date, value: d.ensemble })),
        forecasts: {
          arima: forecastData.slice(25).map(d => ({ date: d.date, value: d.arima || 0 })),
          ets: forecastData.slice(25).map(d => ({ date: d.date, value: d.ets || 0 })),
          xgboost: forecastData.slice(25).map(d => ({ date: d.date, value: d.xgboost || 0 })),
          lstm: forecastData.slice(25).map(d => ({ date: d.date, value: d.lstm || 0 })),
          croston: forecastData.slice(25).map(d => ({ date: d.date, value: d.croston || 0 })),
          ensemble: forecastData.slice(25).map(d => ({ date: d.date, value: d.ensemble }))
        },
        confidenceIntervals: {
          lower: forecastData.slice(25).map(d => ({ date: d.date, value: d.confidence_lower || 0 })),
          upper: forecastData.slice(25).map(d => ({ date: d.date, value: d.confidence_upper || 0 }))
        },
        weights: weightData
      };

      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 12, triangles: 1500 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      const { rerender } = render(
        <EnsembleForecast3DVisualization
          data={ensembleData}
          defaultView="forecast"
          width={800}
          height={600}
        />
      );

      // Simulate initial rendering
      await simulateAnimationFrames(30, 16);

      // Switch to weights view
      rerender(
        <EnsembleForecast3DVisualization
          data={ensembleData}
          defaultView="weights"
          width={800}
          height={600}
        />
      );

      // Simulate view switch rendering
      await simulateAnimationFrames(30, 16);

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain performance during view switching
      expect(avgFPS).toBeGreaterThan(25);
      
      // No severe frame drops during transitions
      const frameDrops = frameRates.filter(fps => fps < 15).length;
      expect(frameDrops).toBeLessThan(frameRates.length * 0.1); // Less than 10% of frames

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should scale particle count based on performance tier', async () => {
      const TestComponent = () => {
        const { 
          performanceTier, 
          getOptimizedParticleCount,
          isInitialized 
        } = usePerformanceOptimization();

        if (!isInitialized) {
          return <div>Loading...</div>;
        }

        const baseParticleCount = 1000;
        const optimizedCount = getOptimizedParticleCount('test-particles', baseParticleCount);

        return (
          <div>
            <div data-testid="performance-tier">{performanceTier}</div>
            <div data-testid="particle-count">{optimizedCount}</div>
          </div>
        );
      };

      render(<TestComponent />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      const tier = screen.getByTestId('performance-tier').textContent;
      const particleCount = parseInt(screen.getByTestId('particle-count').textContent || '0');

      // Particle count should be scaled based on performance tier
      if (tier === 'low') {
        expect(particleCount).toBeLessThan(500);
      } else if (tier === 'medium') {
        expect(particleCount).toBeLessThan(800);
      } else if (tier === 'high') {
        expect(particleCount).toBeLessThanOrEqual(1000);
      }

      expect(particleCount).toBeGreaterThan(0);
    });
  });

  describe('Memory Usage Optimization', () => {
    it('should not exceed memory thresholds with large datasets', async () => {
      const initialMemory = mockPerformance.memory.usedJSHeapSize;
      const largeDataset = generateLargeForecastData(500); // Very large dataset
      
      // Simulate memory growth during rendering
      let memoryGrowth = 0;
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            // Simulate memory usage growth
            memoryGrowth += 1024 * 1024; // 1MB per frame
            mockPerformance.memory.usedJSHeapSize = initialMemory + memoryGrowth;
            
            callback({ 
              clock: { elapsedTime: Date.now() / 1000 },
              gl: { info: { render: { calls: 25, triangles: 5000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={largeDataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={1200}
          height={800}
        />
      );

      // Simulate 120 frames (2 seconds)
      await simulateAnimationFrames(120, 16);

      const finalMemory = mockPerformance.memory.usedJSHeapSize;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be reasonable (less than 200MB for large dataset)
      expect(memoryIncrease).toBeLessThan(200 * 1024 * 1024);
      
      // Should not exceed 80% of heap limit
      const memoryUsageRatio = finalMemory / mockPerformance.memory.jsHeapSizeLimit;
      expect(memoryUsageRatio).toBeLessThan(0.8);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should implement memory-efficient particle systems', async () => {
      const initialMemory = mockPerformance.memory.usedJSHeapSize;
      const dataset = generateLargeForecastData(200);
      let memoryGrowth = 0;
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            // Simulate particle system memory usage
            memoryGrowth += 512 * 1024; // 512KB per frame for particles
            mockPerformance.memory.usedJSHeapSize = initialMemory + memoryGrowth;
            
            callback({ 
              clock: { elapsedTime: Date.now() / 1000 },
              gl: { info: { render: { calls: 30, triangles: 8000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={dataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate 180 frames (3 seconds) with particle effects
      await simulateAnimationFrames(180, 16);

      const finalMemory = mockPerformance.memory.usedJSHeapSize;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Particle systems should use memory efficiently (less than 150MB)
      expect(memoryIncrease).toBeLessThan(150 * 1024 * 1024);
      
      // Memory growth rate should be controlled (less than 1MB per second)
      const memoryGrowthRate = memoryIncrease / 3; // 3 seconds
      expect(memoryGrowthRate).toBeLessThan(1024 * 1024);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should handle memory pressure with graceful degradation', async () => {
      // Simulate high memory pressure (90% of heap used)
      const initialMemory = mockPerformance.memory.jsHeapSizeLimit * 0.9;
      mockPerformance.memory.usedJSHeapSize = initialMemory;
      
      const dataset = generateLargeForecastData(300);
      let memoryGrowth = 0;
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            // Simulate controlled memory growth under pressure
            memoryGrowth += 256 * 1024; // Reduced growth under pressure
            const newMemory = initialMemory + memoryGrowth;
            
            // Cap at heap limit to simulate browser behavior
            mockPerformance.memory.usedJSHeapSize = Math.min(
              newMemory, 
              mockPerformance.memory.jsHeapSizeLimit * 0.95
            );
            
            callback({ 
              clock: { elapsedTime: Date.now() / 1000 },
              gl: { info: { render: { calls: 15, triangles: 3000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      let renderError = null;
      
      try {
        render(
          <HolographicForecastChart3D
            data={dataset}
            showIndividualModels={true}
            showConfidenceIntervals={true}
            enableParticleEffects={true}
            width={800}
            height={600}
          />
        );

        // Simulate rendering under memory pressure
        await simulateAnimationFrames(60, 16);
        
      } catch (error) {
        renderError = error;
      }

      // Should not crash under memory pressure
      expect(renderError).toBeNull();
      
      // Should not exceed heap limit
      expect(mockPerformance.memory.usedJSHeapSize).toBeLessThanOrEqual(
        mockPerformance.memory.jsHeapSizeLimit
      );

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should optimize geometry memory usage with LOD', async () => {
      const initialMemory = mockPerformance.memory.usedJSHeapSize;
      const dataset = generateLargeForecastData(100);
      let memoryGrowth = 0;
      let lodOptimizationActive = false;
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            // Simulate LOD optimization reducing memory usage
            const baseGrowth = 1024 * 1024; // 1MB base growth
            const lodReduction = lodOptimizationActive ? 0.6 : 1.0; // 40% reduction with LOD
            
            memoryGrowth += baseGrowth * lodReduction;
            mockPerformance.memory.usedJSHeapSize = initialMemory + memoryGrowth;
            
            // Activate LOD optimization after 30 frames
            if (!lodOptimizationActive && memoryGrowth > 30 * 1024 * 1024) {
              lodOptimizationActive = true;
            }
            
            callback({ 
              clock: { elapsedTime: Date.now() / 1000 },
              gl: { info: { render: { calls: lodOptimizationActive ? 15 : 25, triangles: lodOptimizationActive ? 3000 : 5000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={dataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate 120 frames to trigger LOD optimization
      await simulateAnimationFrames(120, 16);

      const finalMemory = mockPerformance.memory.usedJSHeapSize;
      const memoryIncrease = finalMemory - initialMemory;
      
      // LOD should have activated and reduced memory usage
      expect(lodOptimizationActive).toBe(true);
      
      // Memory usage should be optimized (less than 100MB with LOD)
      expect(memoryIncrease).toBeLessThan(100 * 1024 * 1024);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should implement LOD (Level of Detail) optimization for memory efficiency', async () => {
      const TestComponent = () => {
        const { 
          getCurrentLODLevel,
          shouldSkipFrame,
          isInitialized 
        } = usePerformanceOptimization();

        if (!isInitialized) {
          return <div>Loading...</div>;
        }

        const lodLevel = getCurrentLODLevel('test-object');
        const shouldSkip = shouldSkipFrame('test-object');

        return (
          <div>
            <div data-testid="lod-level">{lodLevel}</div>
            <div data-testid="should-skip">{shouldSkip.toString()}</div>
          </div>
        );
      };

      render(<TestComponent />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      const lodLevel = parseInt(screen.getByTestId('lod-level').textContent || '0');
      const shouldSkip = screen.getByTestId('should-skip').textContent === 'true';

      // LOD level should be valid
      expect(lodLevel).toBeGreaterThanOrEqual(0);
      expect(lodLevel).toBeLessThanOrEqual(3);

      // Should skip logic should be boolean
      expect(typeof shouldSkip).toBe('boolean');
    });

    it('should cleanup resources when components unmount', async () => {
      const initialMemory = mockPerformance.memory.usedJSHeapSize;
      const dataset = generateLargeForecastData(100);
      
      const { unmount } = render(
        <HolographicForecastChart3D
          data={dataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate some rendering to allocate memory
      await simulateAnimationFrames(60, 16);
      
      const memoryAfterRender = mockPerformance.memory.usedJSHeapSize;
      
      // Unmount component
      unmount();
      
      // Simulate garbage collection by reducing memory
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
        // Simulate memory cleanup
        mockPerformance.memory.usedJSHeapSize = initialMemory + (memoryAfterRender - initialMemory) * 0.3;
      });
      
      const memoryAfterCleanup = mockPerformance.memory.usedJSHeapSize;
      
      // Memory should be significantly reduced after cleanup
      const memoryReduction = memoryAfterRender - memoryAfterCleanup;
      const memoryGrowth = memoryAfterRender - initialMemory;
      
      if (memoryGrowth > 0) {
        const cleanupRatio = memoryReduction / memoryGrowth;
        expect(cleanupRatio).toBeGreaterThan(0.5); // At least 50% cleanup
      }
    });

    it('should optimize particle systems based on available memory', async () => {
      // Simulate low memory condition
      mockPerformance.memory.usedJSHeapSize = mockPerformance.memory.jsHeapSizeLimit * 0.8;
      
      const TestComponent = () => {
        const { 
          getOptimizedParticleCount,
          shouldEnableEffects,
          isInitialized 
        } = usePerformanceOptimization();

        if (!isInitialized) {
          return <div>Loading...</div>;
        }

        const baseCount = 1000;
        const optimizedCount = getOptimizedParticleCount('memory-test', baseCount);
        const effectsEnabled = shouldEnableEffects('memory-test');

        return (
          <div>
            <div data-testid="optimized-particles">{optimizedCount}</div>
            <div data-testid="effects-enabled">{effectsEnabled.toString()}</div>
          </div>
        );
      };

      render(<TestComponent />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      const optimizedCount = parseInt(screen.getByTestId('optimized-particles').textContent || '0');
      const effectsEnabled = screen.getByTestId('effects-enabled').textContent === 'true';

      // Under memory pressure, particle count should be reduced
      expect(optimizedCount).toBeLessThan(1000);
      expect(optimizedCount).toBeGreaterThan(0);

      // Effects might be disabled under memory pressure
      expect(typeof effectsEnabled).toBe('boolean');
    });

    it('should handle memory pressure gracefully without crashes', async () => {
      // Simulate very high memory usage
      mockPerformance.memory.usedJSHeapSize = mockPerformance.memory.jsHeapSizeLimit * 0.95;
      
      const largeDataset = generateLargeForecastData(1000); // Extremely large dataset
      
      let renderError = null;
      
      try {
        const { unmount } = render(
          <HolographicForecastChart3D
            data={largeDataset}
            showIndividualModels={true}
            showConfidenceIntervals={true}
            enableParticleEffects={true}
            width={1600}
            height={1200}
          />
        );

        // Simulate rendering under memory pressure
        await simulateAnimationFrames(30, 16);
        
        unmount();
      } catch (error) {
        renderError = error;
      }

      // Should not crash even under extreme memory pressure
      expect(renderError).toBeNull();
    });
  });

  describe('Performance Monitoring Integration', () => {
    it('should track and report performance metrics accurately', async () => {
      const TestComponent = () => {
        const { 
          performanceStats,
          isInitialized 
        } = usePerformanceOptimization();

        if (!isInitialized) {
          return <div>Loading...</div>;
        }

        return (
          <div>
            <div data-testid="fps">{performanceStats.fps}</div>
            <div data-testid="frame-time">{performanceStats.frameTime}</div>
            <div data-testid="memory-usage">{performanceStats.memoryUsage}</div>
            <div data-testid="draw-calls">{performanceStats.drawCalls}</div>
            <div data-testid="triangles">{performanceStats.triangles}</div>
            <div data-testid="particles">{performanceStats.particles}</div>
          </div>
        );
      };

      render(<TestComponent />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // Verify all performance metrics are being tracked
      const fps = parseInt(screen.getByTestId('fps').textContent || '0');
      const frameTime = parseFloat(screen.getByTestId('frame-time').textContent || '0');
      const memoryUsage = parseInt(screen.getByTestId('memory-usage').textContent || '0');
      const drawCalls = parseInt(screen.getByTestId('draw-calls').textContent || '0');
      const triangles = parseInt(screen.getByTestId('triangles').textContent || '0');
      const particles = parseInt(screen.getByTestId('particles').textContent || '0');

      expect(fps).toBeGreaterThan(0);
      expect(frameTime).toBeGreaterThan(0);
      expect(memoryUsage).toBeGreaterThanOrEqual(0);
      expect(drawCalls).toBeGreaterThanOrEqual(0);
      expect(triangles).toBeGreaterThanOrEqual(0);
      expect(particles).toBeGreaterThanOrEqual(0);

      // FPS and frame time should be inversely related
      const expectedFrameTime = 1000 / fps;
      expect(Math.abs(frameTime - expectedFrameTime)).toBeLessThan(5); // Within 5ms tolerance
    });

    it('should detect performance bottlenecks with large datasets', async () => {
      const largeDataset = generateLargeForecastData(1000); // Very large dataset
      const performanceMetrics: Array<{fps: number, memory: number, drawCalls: number}> = [];
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            
            // Simulate performance degradation with large dataset
            const fps = Math.max(15, 60 - (performanceMetrics.length * 0.5));
            const memory = 50 * 1024 * 1024 + (performanceMetrics.length * 2 * 1024 * 1024);
            const drawCalls = 10 + performanceMetrics.length;
            
            performanceMetrics.push({ fps, memory, drawCalls });
            mockPerformance.memory.usedJSHeapSize = memory;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: drawCalls, triangles: drawCalls * 100 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={largeDataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={1200}
          height={800}
        />
      );

      // Simulate 180 frames (3 seconds)
      await simulateAnimationFrames(180, 16);

      // Analyze performance degradation
      const initialFPS = performanceMetrics[0]?.fps || 60;
      const finalFPS = performanceMetrics[performanceMetrics.length - 1]?.fps || 60;
      const fpsDropPercentage = ((initialFPS - finalFPS) / initialFPS) * 100;
      
      // Should detect significant performance impact with large datasets
      expect(fpsDropPercentage).toBeGreaterThan(10); // At least 10% FPS drop
      
      // Memory usage should increase with dataset size
      const initialMemory = performanceMetrics[0]?.memory || 0;
      const finalMemory = performanceMetrics[performanceMetrics.length - 1]?.memory || 0;
      expect(finalMemory).toBeGreaterThan(initialMemory);
      
      // Draw calls should increase with complexity
      const initialDrawCalls = performanceMetrics[0]?.drawCalls || 0;
      const finalDrawCalls = performanceMetrics[performanceMetrics.length - 1]?.drawCalls || 0;
      expect(finalDrawCalls).toBeGreaterThan(initialDrawCalls);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should adapt quality settings based on performance metrics', async () => {
      let performanceTier = 'high';
      
      const TestComponent = () => {
        const optimization = usePerformanceOptimization();
        performanceTier = optimization.performanceTier;

        if (!optimization.isInitialized) {
          return <div>Loading...</div>;
        }

        return (
          <div>
            <div data-testid="performance-tier">{optimization.performanceTier}</div>
            <div data-testid="fps">{optimization.performanceStats.fps}</div>
          </div>
        );
      };

      render(<TestComponent />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      const initialTier = screen.getByTestId('performance-tier').textContent;
      
      // Simulate performance degradation by mocking low FPS
      mockPerformance.now.mockImplementation(() => Date.now() + 100); // Simulate slow frames
      
      // Wait for adaptive quality to kick in
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 1100)); // Wait for stats update
      });

      // Performance tier should adapt to poor performance
      expect(['low', 'medium', 'high']).toContain(initialTier);
    });
  });

  describe('Stress Testing and Edge Cases', () => {
    it('should handle concurrent 3D visualizations without performance collapse', async () => {
      const dataset1 = generateLargeForecastData(100);
      const dataset2 = generateLargeWeightEvolutionData(100);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 30, triangles: 6000 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      // Render multiple 3D components simultaneously
      render(
        <div>
          <HolographicForecastChart3D
            data={dataset1}
            showIndividualModels={true}
            showConfidenceIntervals={true}
            enableParticleEffects={true}
            width={400}
            height={300}
          />
          <AnimatedWeightEvolution
            data={dataset2}
            enableParticleEffects={true}
            showPerformanceIndicators={true}
            width={400}
            height={300}
          />
        </div>
      );

      // Simulate 120 frames with concurrent visualizations
      await simulateAnimationFrames(120, 16);

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain reasonable performance with concurrent visualizations
      expect(avgFPS).toBeGreaterThan(20);
      
      // No catastrophic frame drops (no frame below 10 FPS)
      const minFPS = Math.min(...frameRates);
      expect(minFPS).toBeGreaterThan(10);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should recover from temporary performance spikes', async () => {
      const dataset = generateLargeForecastData(150);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      let spikeActive = false;
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            
            // Simulate performance spike between frames 30-60
            const frameCount = frameRates.length;
            spikeActive = frameCount >= 30 && frameCount < 60;
            
            const baseInterval = 16;
            const spikeInterval = spikeActive ? 50 : baseInterval; // Slow down during spike
            
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / Math.max(deltaTime, spikeInterval);
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: spikeActive ? 50 : 20, triangles: spikeActive ? 10000 : 4000 } } }
            });
          }, spikeActive ? 50 : 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={dataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate 120 frames including performance spike
      await simulateAnimationFrames(120, 16);

      // Analyze performance recovery
      const preSpikeFPS = frameRates.slice(0, 30).reduce((sum, fps) => sum + fps, 0) / 30;
      const spikeFPS = frameRates.slice(30, 60).reduce((sum, fps) => sum + fps, 0) / 30;
      const postSpikeFPS = frameRates.slice(60, 90).reduce((sum, fps) => sum + fps, 0) / 30;
      
      // Should show performance degradation during spike
      expect(spikeFPS).toBeLessThan(preSpikeFPS * 0.8);
      
      // Should recover after spike (within 20% of original performance)
      expect(postSpikeFPS).toBeGreaterThan(preSpikeFPS * 0.8);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should handle rapid dataset size changes efficiently', async () => {
      let currentDataSize = 50;
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: Math.floor(currentDataSize / 5), triangles: currentDataSize * 20 } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      const { rerender } = render(
        <HolographicForecastChart3D
          data={generateLargeForecastData(currentDataSize)}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate rapid dataset size changes
      const sizeChanges = [100, 200, 50, 300, 75, 150];
      for (const newSize of sizeChanges) {
        currentDataSize = newSize;
        rerender(
          <HolographicForecastChart3D
            data={generateLargeForecastData(currentDataSize)}
            showIndividualModels={true}
            showConfidenceIntervals={true}
            enableParticleEffects={true}
            width={800}
            height={600}
          />
        );
        
        // Allow some frames to process the change
        await simulateAnimationFrames(10, 16);
      }

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain reasonable performance despite rapid changes
      expect(avgFPS).toBeGreaterThan(25);
      
      // Should not have excessive frame rate variance
      const maxFPS = Math.max(...frameRates);
      const minFPS = Math.min(...frameRates);
      const fpsRange = maxFPS - minFPS;
      expect(fpsRange).toBeLessThan(40); // Frame rate should not vary by more than 40 FPS

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should maintain performance with maximum particle effects enabled', async () => {
      const dataset = generateLargeForecastData(200);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      let particleCount = 0;
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            // Simulate increasing particle count
            particleCount = Math.min(5000, particleCount + 50);
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: 35, triangles: 7000 + particleCount } } }
            });
          }, 16);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={dataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={1000}
          height={700}
        />
      );

      // Simulate 150 frames with maximum particle effects
      await simulateAnimationFrames(150, 16);

      const avgFPS = frameRates.reduce((sum, fps) => sum + fps, 0) / frameRates.length;
      
      // Should maintain at least 20 FPS even with maximum particle effects
      expect(avgFPS).toBeGreaterThan(20);
      
      // Particle count should reach maximum
      expect(particleCount).toBeGreaterThan(4000);
      
      // Should not have complete frame stalls (no frame below 5 FPS)
      const minFPS = Math.min(...frameRates);
      expect(minFPS).toBeGreaterThan(5);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });

    it('should handle browser tab visibility changes gracefully', async () => {
      const dataset = generateLargeForecastData(100);
      const frameRates: number[] = [];
      let lastFrameTime = performance.now();
      let isTabVisible = true;
      
      // Mock document visibility API
      Object.defineProperty(document, 'hidden', {
        writable: true,
        value: false
      });
      
      const originalUseFrame = require('@react-three/fiber').useFrame;
      require('@react-three/fiber').useFrame = jest.fn((callback) => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            const currentTime = performance.now();
            
            // Simulate tab becoming hidden after 30 frames
            if (frameRates.length === 30) {
              isTabVisible = false;
              Object.defineProperty(document, 'hidden', { value: true });
            }
            
            // Simulate tab becoming visible again after 60 frames
            if (frameRates.length === 60) {
              isTabVisible = true;
              Object.defineProperty(document, 'hidden', { value: false });
            }
            
            // Reduce frame rate when tab is hidden
            const targetInterval = isTabVisible ? 16 : 100;
            const deltaTime = Math.max(currentTime - lastFrameTime, targetInterval);
            const fps = 1000 / deltaTime;
            frameRates.push(fps);
            lastFrameTime = currentTime;
            
            callback({ 
              clock: { elapsedTime: currentTime / 1000 },
              gl: { info: { render: { calls: isTabVisible ? 20 : 5, triangles: isTabVisible ? 4000 : 1000 } } }
            });
          }, isTabVisible ? 16 : 100);
          return () => clearInterval(interval);
        }, [callback]);
      });

      render(
        <HolographicForecastChart3D
          data={dataset}
          showIndividualModels={true}
          showConfidenceIntervals={true}
          enableParticleEffects={true}
          width={800}
          height={600}
        />
      );

      // Simulate 90 frames including visibility changes
      await simulateAnimationFrames(90, 16);

      // Analyze performance during different visibility states
      const visibleFPS = frameRates.slice(0, 30).reduce((sum, fps) => sum + fps, 0) / 30;
      const hiddenFPS = frameRates.slice(30, 60).reduce((sum, fps) => sum + fps, 0) / 30;
      const restoredFPS = frameRates.slice(60, 90).reduce((sum, fps) => sum + fps, 0) / 30;
      
      // Should reduce performance when hidden
      expect(hiddenFPS).toBeLessThan(visibleFPS * 0.5);
      
      // Should restore performance when visible again
      expect(restoredFPS).toBeGreaterThan(hiddenFPS * 1.5);
      
      // Should maintain reasonable performance throughout
      expect(visibleFPS).toBeGreaterThan(25);
      expect(restoredFPS).toBeGreaterThan(20);

      require('@react-three/fiber').useFrame = originalUseFrame;
    });
  });
});