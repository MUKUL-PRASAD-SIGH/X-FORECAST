import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { usePerformanceOptimization } from '../../hooks/usePerformanceOptimization';

interface WeightEvolutionData {
  timestamp: string;
  weights: {
    arima: number;
    ets: number;
    xgboost: number;
    lstm: number;
    croston: number;
  };
  performance: {
    arima: number;
    ets: number;
    xgboost: number;
    lstm: number;
    croston: number;
  };
}

interface AnimatedWeightEvolutionProps {
  data: WeightEvolutionData[];
  enableParticleEffects?: boolean;
  showPerformanceIndicators?: boolean;
  autoRotate?: boolean;
  animationSpeed?: number;
  width?: number;
  height?: number;
}

const EvolutionContainer = styled(motion.div)<{ width: number; height: number }>`
  width: ${props => props.width}px;
  height: ${props => props.height}px;
  position: relative;
  background: 
    radial-gradient(circle at 20% 80%, rgba(255, 20, 147, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 20%, rgba(57, 255, 20, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 40% 40%, rgba(0, 255, 255, 0.06) 0%, transparent 50%);
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
      rgba(255, 20, 147, 0.8) 0%, 
      rgba(57, 255, 20, 0.8) 25%,
      rgba(0, 255, 255, 0.8) 50%,
      rgba(255, 20, 147, 0.8) 75%,
      rgba(57, 255, 20, 0.8) 100%
    );
    border-radius: 12px;
    z-index: -1;
    animation: weightBorderGlow 4s ease-in-out infinite;
  }
  
  @keyframes weightBorderGlow {
    0%, 100% { opacity: 0.6; filter: blur(1px); }
    50% { opacity: 1; filter: blur(0px); }
  }
`;

const InfoOverlay = styled.div`
  position: absolute;
  top: 15px;
  left: 15px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 6px;
  padding: 10px;
  font-family: 'Courier New', monospace;
  font-size: 11px;
  color: #00ffff;
  backdrop-filter: blur(4px);
`;

const ModelLegend = styled.div`
  position: absolute;
  bottom: 15px;
  left: 15px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(255, 20, 147, 0.3);
  border-radius: 6px;
  padding: 10px;
  font-family: 'Courier New', monospace;
  font-size: 10px;
  backdrop-filter: blur(4px);
`;

const LegendItem = styled.div<{ color: string }>`
  display: flex;
  align-items: center;
  margin-bottom: 4px;
  color: ${props => props.color};
  
  &::before {
    content: '';
    width: 12px;
    height: 12px;
    background: ${props => props.color};
    margin-right: 8px;
    border-radius: 2px;
    filter: drop-shadow(0 0 2px ${props => props.color});
  }
  
  &:last-child {
    margin-bottom: 0;
  }
`;

// Particle System for Weight Changes with Performance Optimization
const WeightParticleSystem: React.FC<{
  position: [number, number, number];
  color: string;
  intensity: number;
  enabled: boolean;
  performanceOptimization?: any;
}> = ({ position, color, intensity, enabled, performanceOptimization }) => {
  const particlesRef = useRef<THREE.Points>(null);
  const [particles] = useState(() => {
    // Get optimized particle count based on performance
    const baseCount = Math.floor(intensity * 50) + 10;
    const count = performanceOptimization?.getOptimizedParticleCount('weight-particles', baseCount) || baseCount;
    
    const positions = new Float32Array(count * 3);
    const velocities = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    
    const colorObj = new THREE.Color(color);
    
    for (let i = 0; i < count; i++) {
      // Initial positions around the weight point
      positions[i * 3] = position[0] + (Math.random() - 0.5) * 0.5;
      positions[i * 3 + 1] = position[1] + (Math.random() - 0.5) * 0.5;
      positions[i * 3 + 2] = position[2] + (Math.random() - 0.5) * 0.5;
      
      // Random velocities
      velocities[i * 3] = (Math.random() - 0.5) * 0.02;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
      
      // Colors
      colors[i * 3] = colorObj.r;
      colors[i * 3 + 1] = colorObj.g;
      colors[i * 3 + 2] = colorObj.b;
      
      // Sizes
      sizes[i] = Math.random() * 0.03 + 0.01;
    }
    
    return { positions, velocities, colors, sizes, count };
  });
  
  useFrame((state) => {
    if (!enabled || !particlesRef.current) return;
    
    // Check if we should skip this frame for performance
    if (performanceOptimization?.shouldSkipFrame('weight-particles')) {
      return;
    }
    
    const time = state.clock.elapsedTime;
    const positionAttribute = particlesRef.current.geometry.attributes.position;
    const positions = positionAttribute.array as Float32Array;
    
    // Get LOD level to determine update frequency
    const lodLevel = performanceOptimization?.getCurrentLODLevel('weight-particles') || 0;
    const updateStep = lodLevel === 0 ? 1 : lodLevel === 1 ? 2 : 4;
    
    for (let i = 0; i < particles.count; i += updateStep) {
      const i3 = i * 3;
      
      // Update positions with velocities
      positions[i3] += particles.velocities[i3];
      positions[i3 + 1] += particles.velocities[i3 + 1];
      positions[i3 + 2] += particles.velocities[i3 + 2];
      
      // Add some orbital motion
      positions[i3] += Math.sin(time + i * 0.1) * 0.001;
      positions[i3 + 1] += Math.cos(time + i * 0.1) * 0.001;
      
      // Reset particles that drift too far
      const distance = Math.sqrt(
        Math.pow(positions[i3] - position[0], 2) +
        Math.pow(positions[i3 + 1] - position[1], 2) +
        Math.pow(positions[i3 + 2] - position[2], 2)
      );
      
      if (distance > 2) {
        positions[i3] = position[0] + (Math.random() - 0.5) * 0.5;
        positions[i3 + 1] = position[1] + (Math.random() - 0.5) * 0.5;
        positions[i3 + 2] = position[2] + (Math.random() - 0.5) * 0.5;
      }
    }
    
    positionAttribute.needsUpdate = true;
  });
  
  if (!enabled) return null;
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.count}
          array={particles.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particles.count}
          array={particles.colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={particles.count}
          array={particles.sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

// Animated Weight Sphere Component
const AnimatedWeightSphere: React.FC<{
  position: [number, number, number];
  weight: number;
  performance: number;
  color: string;
  model: string;
  showPerformance: boolean;
  animationSpeed: number;
}> = ({ position, weight, performance, color, model, showPerformance, animationSpeed }) => {
  const sphereRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  
  const radius = Math.max(weight * 0.3, 0.05);
  const performanceRadius = radius + performance * 0.2;
  
  useFrame((state) => {
    if (sphereRef.current) {
      const time = state.clock.elapsedTime * animationSpeed;
      
      // Pulsing based on weight
      const scale = 1 + Math.sin(time * 2) * weight * 0.2;
      sphereRef.current.scale.setScalar(scale);
      
      // Rotation based on performance
      sphereRef.current.rotation.y = time * performance;
      sphereRef.current.rotation.x = Math.sin(time) * 0.2;
    }
    
    if (ringRef.current && showPerformance) {
      const time = state.clock.elapsedTime * animationSpeed;
      ringRef.current.rotation.z = time * 2;
      
      // Performance ring opacity
      const material = ringRef.current.material as THREE.MeshBasicMaterial;
      material.opacity = 0.3 + performance * 0.4;
    }
  });
  
  return (
    <group position={position}>
      {/* Main weight sphere */}
      <Sphere ref={sphereRef} args={[radius, 16, 16]}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.8}
        />
      </Sphere>
      
      {/* Performance ring */}
      {showPerformance && (
        <mesh ref={ringRef}>
          <ringGeometry args={[performanceRadius, performanceRadius + 0.02, 32]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.5}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
      
      {/* Model label */}
      <Text
        position={[0, radius + 0.2, 0]}
        fontSize={0.1}
        color={color}
        anchorX="center"
        anchorY="middle"
      >
        {model.toUpperCase()}
      </Text>
      
      {/* Weight value */}
      <Text
        position={[0, -radius - 0.15, 0]}
        fontSize={0.08}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {(weight * 100).toFixed(1)}%
      </Text>
    </group>
  );
};

// Weight Evolution Timeline Component
const WeightEvolutionTimeline: React.FC<{
  data: WeightEvolutionData[];
  currentIndex: number;
  modelColors: Record<string, string>;
}> = ({ data, currentIndex, modelColors }) => {
  const timelineRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (timelineRef.current) {
      const time = state.clock.elapsedTime;
      timelineRef.current.rotation.y = Math.sin(time * 0.5) * 0.1;
    }
  });
  
  const models = ['arima', 'ets', 'xgboost', 'lstm', 'croston'] as const;
  
  return (
    <group ref={timelineRef}>
      {models.map((model, modelIndex) => {
        const points = data.map((point, dataIndex) => {
          const x = (dataIndex - data.length / 2) * 0.5;
          const y = point.weights[model] * 2 - 1;
          const z = (modelIndex - 2) * 0.5;
          return new THREE.Vector3(x, y, z);
        });
        
        return (
          <Line
            key={model}
            points={points}
            color={modelColors[model]}
            lineWidth={3}
            transparent
            opacity={0.7}
          />
        );
      })}
      
      {/* Current time indicator */}
      <Box
        position={[(currentIndex - data.length / 2) * 0.5, 0, 0]}
        args={[0.05, 4, 2]}
      >
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.3}
        />
      </Box>
    </group>
  );
};

// Cyberpunk Lighting for Weight Evolution
const WeightEvolutionLighting: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.3} color="#1a1a2e" />
      <pointLight position={[3, 3, 3]} intensity={0.8} color="#ff1493" />
      <pointLight position={[-3, -3, -3]} intensity={0.6} color="#39ff14" />
      <pointLight position={[0, 5, -3]} intensity={0.4} color="#00ffff" />
      <spotLight
        position={[0, 8, 0]}
        angle={0.4}
        penumbra={1}
        intensity={0.6}
        color="#ff1493"
        castShadow
      />
    </>
  );
};

// Main Component
export const AnimatedWeightEvolution: React.FC<AnimatedWeightEvolutionProps> = ({
  data,
  enableParticleEffects = true,
  showPerformanceIndicators = true,
  autoRotate = false,
  animationSpeed = 1,
  width = 800,
  height = 600
}) => {
  const [currentTimeIndex, setCurrentTimeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  
  // Initialize performance optimization
  const performanceOptimization = usePerformanceOptimization();
  
  const modelColors = {
    arima: '#ff6b6b',
    ets: '#4ecdc4',
    xgboost: '#45b7d1',
    lstm: '#f9ca24',
    croston: '#6c5ce7'
  };
  
  // Auto-advance timeline
  useEffect(() => {
    if (!isPlaying || data.length === 0) return;
    
    const interval = setInterval(() => {
      setCurrentTimeIndex(prev => (prev + 1) % data.length);
    }, 2000 / animationSpeed);
    
    return () => clearInterval(interval);
  }, [isPlaying, data.length, animationSpeed]);
  
  const currentData = data[currentTimeIndex] || data[0];
  const models = ['arima', 'ets', 'xgboost', 'lstm', 'croston'] as const;
  
  // Calculate positions for weight spheres in a circle
  const getModelPosition = (index: number, total: number): [number, number, number] => {
    const angle = (index / total) * Math.PI * 2;
    const radius = 2;
    return [
      Math.cos(angle) * radius,
      0,
      Math.sin(angle) * radius
    ];
  };
  
  return (
    <EvolutionContainer
      width={width}
      height={height}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8 }}
    >
      <InfoOverlay>
        <div>Weight Evolution Timeline</div>
        <div style={{ marginTop: '5px', fontSize: '10px' }}>
          Time: {currentData?.timestamp || 'N/A'}
        </div>
        <div style={{ marginTop: '5px', fontSize: '10px' }}>
          Frame: {currentTimeIndex + 1} / {data.length}
        </div>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={{
            marginTop: '8px',
            background: 'rgba(0, 255, 255, 0.2)',
            border: '1px solid #00ffff',
            color: '#00ffff',
            padding: '4px 8px',
            borderRadius: '3px',
            fontSize: '10px',
            cursor: 'pointer'
          }}
        >
          {isPlaying ? 'PAUSE' : 'PLAY'}
        </button>
      </InfoOverlay>
      
      <ModelLegend>
        {models.map(model => (
          <LegendItem key={model} color={modelColors[model]}>
            {model.toUpperCase()}: {currentData ? (currentData.weights[model] * 100).toFixed(1) : '0.0'}%
          </LegendItem>
        ))}
      </ModelLegend>
      
      <Canvas
        camera={{ position: [6, 4, 6], fov: 60 }}
        style={{ background: 'transparent' }}
      >
        <WeightEvolutionLighting />
        
        {/* Weight spheres for current time */}
        {currentData && models.map((model, index) => (
          <React.Fragment key={`${model}-${currentTimeIndex}`}>
            <AnimatedWeightSphere
              position={getModelPosition(index, models.length)}
              weight={currentData.weights[model]}
              performance={currentData.performance[model]}
              color={modelColors[model]}
              model={model}
              showPerformance={showPerformanceIndicators}
              animationSpeed={animationSpeed}
            />
            
            {/* Particle effects */}
            {enableParticleEffects && performanceOptimization.shouldEnableEffects('weight-particles') && (
              <WeightParticleSystem
                position={getModelPosition(index, models.length)}
                color={modelColors[model]}
                intensity={currentData.weights[model]}
                enabled={enableParticleEffects}
                performanceOptimization={performanceOptimization}
              />
            )}
          </React.Fragment>
        ))}
        
        {/* Weight evolution timeline */}
        {data.length > 1 && (
          <WeightEvolutionTimeline
            data={data}
            currentIndex={currentTimeIndex}
            modelColors={modelColors}
          />
        )}
        
        {/* Center title */}
        <Text
          position={[0, -2.5, 0]}
          fontSize={0.2}
          color="#00ffff"
          anchorX="center"
          anchorY="middle"
        >
          MODEL WEIGHT EVOLUTION
        </Text>
        
        {/* Connection lines between models */}
        {currentData && models.map((model, index) => {
          const nextIndex = (index + 1) % models.length;
          const pos1 = getModelPosition(index, models.length);
          const pos2 = getModelPosition(nextIndex, models.length);
          
          return (
            <Line
              key={`connection-${index}`}
              points={[pos1, pos2]}
              color="rgba(255, 255, 255, 0.2)"
              transparent
              opacity={0.3}
            />
          );
        })}
        
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={autoRotate}
          autoRotateSpeed={0.3}
          maxPolarAngle={Math.PI * 0.8}
          minPolarAngle={Math.PI * 0.2}
        />
      </Canvas>
    </EvolutionContainer>
  );
};

export default AnimatedWeightEvolution;