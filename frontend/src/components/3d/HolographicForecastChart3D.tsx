import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box, Plane } from '@react-three/drei';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { usePerformanceOptimization } from '../../hooks/usePerformanceOptimization';

interface ForecastData {
  date: string;
  historical?: number;
  arima?: number;
  ets?: number;
  xgboost?: number;
  lstm?: number;
  croston?: number;
  ensemble: number;
  confidence_lower?: number;
  confidence_upper?: number;
}

interface HolographicForecastChart3DProps {
  data: ForecastData[];
  showIndividualModels?: boolean;
  showConfidenceIntervals?: boolean;
  enableParticleEffects?: boolean;
  autoRotate?: boolean;
  width?: number;
  height?: number;
}

const ChartContainer = styled(motion.div)<{ width: number; height: number }>`
  width: ${props => props.width}px;
  height: ${props => props.height}px;
  position: relative;
  background: 
    radial-gradient(circle at 30% 30%, rgba(0, 255, 255, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 70% 70%, rgba(255, 20, 147, 0.06) 0%, transparent 50%);
  border: 2px solid transparent;
  border-radius: 12px;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, 
      rgba(0, 255, 255, 0.6) 0%, 
      rgba(255, 20, 147, 0.6) 50%, 
      rgba(57, 255, 20, 0.6) 100%
    );
    border-radius: 12px;
    z-index: -1;
    animation: borderGlow 3s ease-in-out infinite;
  }
  
  @keyframes borderGlow {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }
`;

const ControlsOverlay = styled.div`
  position: absolute;
  top: 15px;
  right: 15px;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const ControlButton = styled.button<{ active?: boolean }>`
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.8)'};
  border: 1px solid ${props => props.active ? '#00ffff' : 'rgba(0, 255, 255, 0.3)'};
  color: ${props => props.active ? '#00ffff' : 'rgba(255, 255, 255, 0.7)'};
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-family: 'Courier New', monospace;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(4px);
  
  &:hover {
    border-color: #ff1493;
    color: #ff1493;
    box-shadow: 0 0 10px rgba(255, 20, 147, 0.3);
  }
`;

// 3D Holographic Forecast Line Component
const HolographicForecastLine: React.FC<{
  points: THREE.Vector3[];
  color: string;
  glowing?: boolean;
  animated?: boolean;
}> = ({ points, color, glowing = false, animated = false }) => {
  const lineRef = useRef<any>(null);
  const materialRef = useRef<THREE.LineBasicMaterial>(null);
  
  useFrame((state) => {
    if (animated && materialRef.current) {
      const time = state.clock.elapsedTime;
      materialRef.current.opacity = 0.7 + 0.3 * Math.sin(time * 2);
    }
    
    if (animated && lineRef.current) {
      lineRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.02;
    }
  });
  
  return (
    <Line
      ref={lineRef}
      points={points}
      color={color}
      lineWidth={glowing ? 4 : 2}
      transparent
      opacity={0.8}
    >
      <lineBasicMaterial
        ref={materialRef}
        color={color}
        transparent
        opacity={0.8}
      />
    </Line>
  );
};

// 3D Probability Cloud Component with Performance Optimization
const ProbabilityCloud: React.FC<{
  data: ForecastData[];
  startIndex: number;
  performanceOptimization?: any;
}> = ({ data, startIndex, performanceOptimization }) => {
  const cloudRef = useRef<THREE.Group>(null);
  const particlesRef = useRef<THREE.Points>(null);
  
  const cloudGeometry = useMemo(() => {
    // Get optimized particle count based on performance
    const baseParticleCount = 100;
    const particlesPerPoint = performanceOptimization?.getOptimizedParticleCount('probability-cloud', baseParticleCount) || baseParticleCount;
    
    const positions = new Float32Array(data.length * particlesPerPoint * 3);
    const colors = new Float32Array(data.length * particlesPerPoint * 3);
    const sizes = new Float32Array(data.length * particlesPerPoint);
    
    data.forEach((point, dataIndex) => {
      if (!point.confidence_lower || !point.confidence_upper) return;
      
      const x = (dataIndex + startIndex) * 0.5 - data.length * 0.25;
      const y = point.ensemble * 0.1;
      const confidenceRange = point.confidence_upper - point.confidence_lower;
      
      for (let i = 0; i < particlesPerPoint; i++) {
        const particleIndex = dataIndex * particlesPerPoint + i;
        const baseIndex = particleIndex * 3;
        
        // Position particles within confidence interval
        positions[baseIndex] = x + (Math.random() - 0.5) * 2;
        positions[baseIndex + 1] = y + (Math.random() - 0.5) * confidenceRange * 0.1;
        positions[baseIndex + 2] = (Math.random() - 0.5) * 1;
        
        // Color based on confidence level
        const confidence = 1 - (confidenceRange / (point.ensemble * 2));
        colors[baseIndex] = 0.2 + confidence * 0.8; // R
        colors[baseIndex + 1] = 1; // G (cyan)
        colors[baseIndex + 2] = 1; // B (cyan)
        
        sizes[particleIndex] = Math.random() * 0.05 + 0.02;
      }
    });
    
    return { positions, colors, sizes, particlesPerPoint };
  }, [data, startIndex, performanceOptimization]);
  
  useFrame((state) => {
    if (particlesRef.current) {
      // Check if we should skip this frame for performance
      if (performanceOptimization?.shouldSkipFrame('probability-cloud')) {
        return;
      }
      
      particlesRef.current.rotation.y += 0.001;
      const time = state.clock.elapsedTime;
      
      // Animate particle positions with LOD consideration
      const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
      const lodLevel = performanceOptimization?.getCurrentLODLevel('probability-cloud') || 0;
      const animationStep = lodLevel === 0 ? 1 : lodLevel === 1 ? 2 : 4; // Skip more particles at higher LOD levels
      
      for (let i = 0; i < positions.length; i += 3 * animationStep) {
        positions[i + 1] += Math.sin(time + i * 0.01) * 0.001;
      }
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <group ref={cloudRef}>
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={cloudGeometry.positions.length / 3}
            array={cloudGeometry.positions}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={cloudGeometry.colors.length / 3}
            array={cloudGeometry.colors}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-size"
            count={cloudGeometry.sizes.length}
            array={cloudGeometry.sizes}
            itemSize={1}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.03}
          vertexColors
          transparent
          opacity={performanceOptimization?.shouldEnableEffects('probability-cloud') ? 0.6 : 0.4}
          sizeAttenuation
          blending={THREE.AdditiveBlending}
        />
      </points>
    </group>
  );
};

// Animated Data Points Component
const AnimatedDataPoints: React.FC<{
  data: ForecastData[];
  startIndex: number;
  color: string;
  type: 'historical' | 'forecast';
}> = ({ data, startIndex, color, type }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      const time = state.clock.elapsedTime;
      groupRef.current.children.forEach((child, index) => {
        if (child instanceof THREE.Mesh) {
          child.rotation.y = time + index * 0.1;
          child.scale.setScalar(1 + Math.sin(time * 2 + index * 0.2) * 0.1);
        }
      });
    }
  });
  
  return (
    <group ref={groupRef}>
      {data.map((point, index) => {
        const x = (index + startIndex) * 0.5 - data.length * 0.25;
        const y = point.ensemble * 0.1;
        const z = 0;
        
        return (
          <Sphere
            key={`${type}-${index}`}
            position={[x, y, z]}
            args={[type === 'historical' ? 0.03 : 0.04]}
          >
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.8}
            />
          </Sphere>
        );
      })}
    </group>
  );
};

// Holographic Grid Component
const HolographicGrid: React.FC = () => {
  const gridRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (gridRef.current) {
      const time = state.clock.elapsedTime;
      gridRef.current.children.forEach((child, index) => {
        if (child instanceof THREE.Line) {
          const material = child.material as THREE.LineBasicMaterial;
          material.opacity = 0.2 + 0.1 * Math.sin(time + index * 0.5);
        }
      });
    }
  });
  
  return (
    <group ref={gridRef}>
      {/* Grid lines */}
      {Array.from({ length: 21 }, (_, i) => (
        <Line
          key={`grid-x-${i}`}
          points={[[-5, 0, i - 10], [5, 0, i - 10]]}
          color="#00ffff"
          transparent
          opacity={0.2}
        />
      ))}
      {Array.from({ length: 21 }, (_, i) => (
        <Line
          key={`grid-z-${i}`}
          points={[[i - 10, 0, -5], [i - 10, 0, 5]]}
          color="#00ffff"
          transparent
          opacity={0.2}
        />
      ))}
      
      {/* Holographic scan lines */}
      <Plane args={[20, 0.1]} position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <meshBasicMaterial
          color="#39ff14"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </Plane>
    </group>
  );
};

// Cyberpunk Lighting Setup
const CyberpunkLighting: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.2} color="#0a0a0a" />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#00ffff" />
      <pointLight position={[-5, -5, -5]} intensity={0.6} color="#ff1493" />
      <pointLight position={[0, 8, -5]} intensity={0.4} color="#39ff14" />
      <spotLight
        position={[0, 10, 0]}
        angle={0.3}
        penumbra={1}
        intensity={0.5}
        color="#00ffff"
        castShadow
      />
    </>
  );
};

// Main Component
export const HolographicForecastChart3D: React.FC<HolographicForecastChart3DProps> = ({
  data,
  showIndividualModels = true,
  showConfidenceIntervals = true,
  enableParticleEffects = true,
  autoRotate = false,
  width = 800,
  height = 600
}) => {
  const [showGrid, setShowGrid] = useState(true);
  const [showParticles, setShowParticles] = useState(enableParticleEffects);
  const [rotateEnabled, setRotateEnabled] = useState(autoRotate);
  
  // Initialize performance optimization
  const performanceOptimization = usePerformanceOptimization();
  
  // Separate historical and forecast data
  const historicalData = data.filter(d => d.historical !== undefined);
  const forecastData = data.filter(d => d.historical === undefined);
  
  // Create 3D points for different model lines
  const createPoints = (dataArray: ForecastData[], field: keyof ForecastData, startIndex: number = 0) => {
    return dataArray
      .filter(d => d[field] !== undefined)
      .map((point, index) => {
        const x = (index + startIndex) * 0.5 - dataArray.length * 0.25;
        const y = (point[field] as number) * 0.1;
        const z = 0;
        return new THREE.Vector3(x, y, z);
      });
  };
  
  const modelColors = {
    historical: '#00ffff',
    arima: '#ff6b6b',
    ets: '#4ecdc4',
    xgboost: '#45b7d1',
    lstm: '#f9ca24',
    croston: '#6c5ce7',
    ensemble: '#ff1493'
  };
  
  return (
    <ChartContainer
      width={width}
      height={height}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8 }}
    >
      <ControlsOverlay>
        <ControlButton
          active={showGrid}
          onClick={() => setShowGrid(!showGrid)}
        >
          Grid: {showGrid ? 'ON' : 'OFF'}
        </ControlButton>
        <ControlButton
          active={showParticles}
          onClick={() => setShowParticles(!showParticles)}
        >
          Particles: {showParticles ? 'ON' : 'OFF'}
        </ControlButton>
        <ControlButton
          active={rotateEnabled}
          onClick={() => setRotateEnabled(!rotateEnabled)}
        >
          Rotate: {rotateEnabled ? 'ON' : 'OFF'}
        </ControlButton>
        <ControlButton
          active={showIndividualModels}
          onClick={() => {}} // Controlled by parent
        >
          Models: {showIndividualModels ? 'ALL' : 'ENSEMBLE'}
        </ControlButton>
      </ControlsOverlay>
      
      <Canvas
        camera={{ position: [8, 6, 8], fov: 60 }}
        style={{ background: 'transparent' }}
      >
        <CyberpunkLighting />
        
        {showGrid && <HolographicGrid />}
        
        {/* Historical data line */}
        {historicalData.length > 0 && (
          <>
            <HolographicForecastLine
              points={createPoints(historicalData, 'historical')}
              color={modelColors.historical}
              glowing
              animated
            />
            <AnimatedDataPoints
              data={historicalData}
              startIndex={0}
              color={modelColors.historical}
              type="historical"
            />
          </>
        )}
        
        {/* Individual model forecast lines */}
        {showIndividualModels && forecastData.length > 0 && (
          <>
            {(['arima', 'ets', 'xgboost', 'lstm', 'croston'] as const).map(model => (
              <HolographicForecastLine
                key={model}
                points={createPoints(forecastData, model, historicalData.length)}
                color={modelColors[model]}
                animated
              />
            ))}
          </>
        )}
        
        {/* Ensemble forecast line */}
        {forecastData.length > 0 && (
          <>
            <HolographicForecastLine
              points={createPoints(forecastData, 'ensemble', historicalData.length)}
              color={modelColors.ensemble}
              glowing
              animated
            />
            <AnimatedDataPoints
              data={forecastData}
              startIndex={historicalData.length}
              color={modelColors.ensemble}
              type="forecast"
            />
          </>
        )}
        
        {/* 3D Probability Clouds for confidence intervals */}
        {showConfidenceIntervals && showParticles && forecastData.length > 0 && 
         performanceOptimization.shouldEnableEffects('probability-cloud') && (
          <ProbabilityCloud
            data={forecastData}
            startIndex={historicalData.length}
            performanceOptimization={performanceOptimization}
          />
        )}
        
        {/* Axis labels */}
        <Text
          position={[0, -2, 0]}
          fontSize={0.3}
          color="#00ffff"
          anchorX="center"
          anchorY="middle"
        >
          ENSEMBLE FORECAST VISUALIZATION
        </Text>
        
        <Text
          position={[6, 0, 0]}
          fontSize={0.2}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          rotation={[0, -Math.PI / 2, 0]}
        >
          TIME →
        </Text>
        
        <Text
          position={[0, 3, 0]}
          fontSize={0.2}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          rotation={[0, 0, Math.PI / 2]}
        >
          VALUE ↑
        </Text>
        
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={rotateEnabled}
          autoRotateSpeed={0.5}
          maxPolarAngle={Math.PI * 0.8}
          minPolarAngle={Math.PI * 0.2}
        />
      </Canvas>
    </ChartContainer>
  );
};

export default HolographicForecastChart3D;